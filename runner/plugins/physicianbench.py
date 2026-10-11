#!/usr/bin/env python3
"""PhysicianBench (clinical_data_ops, FHIR) BenchmarkPlugin.

ALL PhysicianBench-specific knowledge lives here: the FHIR tool->role/milestone semantics and the
RESULT-CONDITIONAL resolvers that read the actual FHIR body (OperationOutcome error / Bundle total /
server-assigned id) to decide which milestone was truly earned. The substrate core names no FHIR concept;
this file does. A 4th dataset drops a sibling module + a spec/registry.json entry; no core edit.

Consumes ONLY substrate's shared helpers (_errored, _result_output, _hash8); the plugin->substrate import
is the single, one-directional dependency.

TYPED CONTEXT + SOURCE PROVENANCE (shared cross-benchmark contract):
  * dimension_policy.required_context_units are TYPED {id, type} over the PB vocabulary
    patient_identity / current_medication_list / allergy_status.
  * every EvidenceUnit is tagged with context_type (the semantic KIND the FHIR read obtained, derived from
    the resourceType actually returned -- Patient->patient_identity, MedicationRequest/Statement/
    Medication->current_medication_list, AllergyIntolerance->allergy_status), source_channel
    'fhir_patient_record', source_instance_id 'Patient/<id>' or '<Type>/<id>' (the specific resource read),
    and extractor 'fhir_read'/'fhir_search'.
  * A SUBJECT token subject:Patient/<patient_id> is emitted on every PB evidence unit DISTINCT from the
    per-resource id (an Observation has its OWN id 190335, but its SUBJECT is the patient) so Patient–Evidence Alignment
    binding converges on the patient SUBJECT, not a scatter of resource-own ids."""
import re as _re
import substrate as _S

_errored = _S._errored
_result_output = _S._result_output
_hash8 = _S._hash8


# --- usable_for_context (shared cross-benchmark) ---
def _usable_fields(sem_event):
    """Derive (semantic_status, usable_for_context) for an EvidenceUnit from its PRODUCING
    SemanticEvent (the resolver's resolved status, NOT the tool name). usable_for_context = (status=='
    success' AND a non-empty progress_token). An EMPTY Bundle (total 0) / OperationOutcome / id-less read is
    delivered+partial with NO token -> usable_for_context=False: it still feeds Clinical Auditability/delivery but
    is NOT acquired patient context."""
    st = (sem_event or {}).get("status")
    pt = (sem_event or {}).get("progress_token")
    return st, bool(st == "success" and pt)


_CH_FHIR = "fhir_patient_record"

# resourceType (lower) -> the semantic context TYPE it supplies (PB vocabulary). Resources outside this map
# are real record detail but carry no REQUIRED context type (-> context_type None).
_RTYPE_CONTEXT = {
    "patient": "patient_identity",
    "medicationrequest": "current_medication_list",
    "medicationstatement": "current_medication_list",
    "medication": "current_medication_list",
    "allergyintolerance": "allergy_status",
}

# TOOL-level context_type override: the granular Observation searches (labs / vitals / social_history) all
# share resourceType "Observation", so _RTYPE_CONTEXT cannot distinguish them. The LABS search specifically
# supplies the benchmark-declared 'relevant_labs' necessary-evidence type (policy.minimum_necessary_evidence).
_TOOL_CONTEXT = {
    "fhir_observation_search_labs": "relevant_labs",
}

# The REAL PhysicianBench environment (runner/environments.py _FHIR_GRANULAR) does NOT expose the generic
# fhir_search/read/create to the agent; it exposes a GRANULAR tool per FHIR domain. Each granular SEARCH
# routes to fhir_search(resourceType[, category]) and returns the upstream {"entries":[...], "total", ...}
# shape; each granular CREATE routes to fhir_create(resource) and returns the raw created resource. This
# table is the PB-specific knowledge that lets the plugin map those real tool names to the right role /
# milestone / resource type. (Benchmark-agnostic core names none of it; mirrors environments._FHIR_GRANULAR.)
_GRANULAR_SEARCH = {
    "fhir_patient_search_demographics": "Patient",
    "fhir_condition_search_problems": "Condition",
    "fhir_observation_search_labs": "Observation",
    "fhir_observation_search_vitals": "Observation",
    "fhir_observation_search_social_history": "Observation",
    "fhir_medication_request_search_orders": "MedicationRequest",
    "fhir_procedure_search_orders": "Procedure",
    "fhir_document_reference_search_clinical_notes": "DocumentReference",
    "fhir_service_request_search": "ServiceRequest",
}
_GRANULAR_CREATE = {
    "fhir_medication_request_create": "MedicationRequest",
    "fhir_service_request_create": "ServiceRequest",
    "fhir_communication_create_message": "Communication",
    "fhir_appointment_create": "Appointment",
}

_PAT_REF_RE = _re.compile(r"Patient/([A-Za-z0-9_\-.]+)")
_IDENT_URL_RE = _re.compile(r"patient(?:\.identifier)?=([A-Za-z0-9_\-.]+)")


# ============================================================================= RESOLVERS
def _is_operation_outcome_error(obj):
    if not isinstance(obj, dict):
        return False
    if obj.get("resourceType") == "OperationOutcome":
        for iss in (obj.get("issue") or []):
            if str(iss.get("severity", "")).lower() in ("error", "fatal"):
                return True
        return True   # an OperationOutcome with no graded issue is still not a created resource
    return False


def _resolve_fhir_create(event, prev_state):
    """PB fhir_create: an HTTP-success that returned an OperationOutcome error OR a body with NO created
    resource id is NOT a real creation -> partial, no resource_created. A body with a server-assigned id
    (and a real resourceType) -> success with a resource:<type>/<id>:created token."""
    if _errored(event):
        return {"role": "commit", "status": "failure", "milestones_added": [],
                "obligation_id": "resource_created", "state_changed": False, "progress_token": None}
    out = _result_output(event)
    if _is_operation_outcome_error(out):
        return {"role": "commit", "status": "partial", "milestones_added": [],
                "obligation_id": "resource_created", "state_changed": False, "progress_token": None}
    rid = out.get("id") if isinstance(out, dict) else None
    rtype = out.get("resourceType") if isinstance(out, dict) else None
    if rid and rtype and rtype != "OperationOutcome":
        return {"role": "commit", "status": "success", "milestones_added": ["resource_created"],
                "obligation_id": "resource_created",
                "progress_token": "resource:%s/%s:created" % (rtype, rid)}
    # accepted call but no created id surfaced -> not a real creation
    return {"role": "commit", "status": "partial", "milestones_added": [],
            "obligation_id": "resource_created", "state_changed": False, "progress_token": None}


def _search_entries(out):
    """The list of matched resources in a FHIR SEARCH result, normalized across the two shapes PB emits:
    the upstream granular shape {"entries":[resource,...], "total"} (what the REAL environment returns) and
    a raw FHIR Bundle {"resourceType":"Bundle","entry":[{"resource":...}]} (generic fhir_search path).
    Returns a list of resource dicts, or None when `out` is not a recognizable search result."""
    if not isinstance(out, dict):
        return None
    if isinstance(out.get("entries"), list):
        return [r for r in out["entries"] if isinstance(r, dict)]
    if out.get("resourceType") == "Bundle":
        return [e["resource"] for e in (out.get("entry") or [])
                if isinstance(e, dict) and isinstance(e.get("resource"), dict)]
    return None


def _bundle_count(out):
    """Number of matched resources in a FHIR search result; None if not a recognizable search result. Reads
    the granular {"entries":[...],"total"} shape and the raw Bundle .total/.entry shape."""
    ents = _search_entries(out)
    if ents is None:
        return None
    if isinstance(out, dict) and isinstance(out.get("total"), int):
        return out["total"]
    return len(ents)


def _resolve_fhir_search(event, prev_state):
    """PB fhir_search: an empty result set (Bundle total 0 / no entry) -> partial, no patient_record_loaded
    (nothing was actually loaded). A non-empty Bundle -> success with a state token keyed by the matched-id
    set so a re-run of the SAME search repeats its token (no progress) but a search hitting new records
    advances state."""
    if _errored(event):
        return {"role": "acquire", "status": "failure", "milestones_added": [],
                "obligation_id": "patient_record_loaded", "state_changed": False, "progress_token": None}
    out = _result_output(event)
    n = _bundle_count(out)
    if n is None:
        # not a recognizable bundle but no error: treat as a delivered single resource if it has an id
        rid = out.get("id") if isinstance(out, dict) else None
        if rid:
            return {"role": "acquire", "status": "success", "milestones_added": ["patient_record_loaded"],
                    "obligation_id": "patient_record_loaded",
                    "progress_token": "state:search=%s" % _hash8(str(rid))}
        return {"role": "acquire", "status": "partial", "milestones_added": [],
                "obligation_id": "patient_record_loaded", "state_changed": False, "progress_token": None}
    if n <= 0:
        return {"role": "acquire", "status": "partial", "milestones_added": [],
                "obligation_id": "patient_record_loaded", "state_changed": False, "progress_token": None}
    ents = _search_entries(out) or []
    ids = ",".join(sorted(str(r.get("id") or "") for r in ents)[:50])
    return {"role": "acquire", "status": "success", "milestones_added": ["patient_record_loaded"],
            "obligation_id": "patient_record_loaded",
            "progress_token": "state:search=%s" % _hash8(ids or str(n))}


def _resolve_fhir_read(event, prev_state):
    """PB fhir_read: an empty / OperationOutcome / id-less body -> partial, no record_detail_loaded. A real
    resource body -> success with a resource:<type>/<id>:read state token (re-reading the SAME id repeats
    the token; a new id advances state)."""
    if _errored(event):
        return {"role": "acquire", "status": "failure", "milestones_added": [],
                "obligation_id": "record_detail_loaded", "state_changed": False, "progress_token": None}
    out = _result_output(event)
    if _is_operation_outcome_error(out):
        return {"role": "acquire", "status": "partial", "milestones_added": [],
                "obligation_id": "record_detail_loaded", "state_changed": False, "progress_token": None}
    rid = out.get("id") if isinstance(out, dict) else None
    rtype = out.get("resourceType") if isinstance(out, dict) else None
    if rid and rtype:
        return {"role": "acquire", "status": "success", "milestones_added": ["record_detail_loaded"],
                "obligation_id": "record_detail_loaded",
                "progress_token": "state:read=%s/%s" % (rtype, rid)}
    return {"role": "acquire", "status": "partial", "milestones_added": [],
            "obligation_id": "record_detail_loaded", "state_changed": False, "progress_token": None}


def _resolve_write_file(event, prev_state):
    """PB write_file: the required deliverable. The REAL environment returns {"written": <abs path>} on a
    successful save; an error body (or no written path) -> partial, no deliverable_written. A real written
    path -> success with a deliverable:<path> token so re-writing the SAME path repeats the token (no new
    progress) but a NEW deliverable advances state."""
    if _errored(event):
        return {"role": "commit", "status": "failure", "milestones_added": [],
                "obligation_id": "deliverable_written", "state_changed": False, "progress_token": None}
    out = _result_output(event)
    path = out.get("written") if isinstance(out, dict) else None
    if path:
        return {"role": "commit", "status": "success", "milestones_added": ["deliverable_written"],
                "obligation_id": "deliverable_written",
                "progress_token": "deliverable:%s" % _hash8(str(path))}
    return {"role": "commit", "status": "partial", "milestones_added": [],
            "obligation_id": "deliverable_written", "state_changed": False, "progress_token": None}


# ----------------------------------------------------------------------------- SUBJECT identity + provenance
def _patient_subject(event, fallback=None):
    """The PATIENT SUBJECT id this FHIR call concerns, DISTINCT from a resource's own id. Read (in order)
    from the request args (patient / params.identifier), the returned bundle/resource subject reference
    (Patient/<id>), the self-link URL (patient.identifier=<id>), or a directly-read Patient body. None when
    no subject is determinable. fallback supplies the trace-wide subject when a single event is anonymous."""
    args = event.get("args") if isinstance(event.get("args"), dict) else {}
    pid = args.get("patient") or args.get("patient_id") or args.get("subject")
    if not pid:
        params = args.get("params") if isinstance(args.get("params"), dict) else {}
        pid = params.get("identifier") or params.get("patient")
    if pid:
        m = _PAT_REF_RE.search(str(pid))
        return m.group(1) if m else str(pid)
    out = _result_output(event)
    if isinstance(out, dict):
        # a directly-read Patient resource: its OWN id IS the subject
        if out.get("resourceType") == "Patient" and out.get("id"):
            return str(out.get("id"))
        # a resource body / bundle entry subject reference
        subj = out.get("subject")
        if isinstance(subj, dict) and subj.get("reference"):
            m = _PAT_REF_RE.search(str(subj["reference"]))
            if m:
                return m.group(1)
        # SEARCH results (both shapes): a Patient hit -> its id IS the subject; any other resource ->
        # its subject reference Patient/<id>.
        for res in (_search_entries(out) or []):
            if res.get("resourceType") == "Patient" and res.get("id"):
                return str(res.get("id"))
            subj = res.get("subject")
            if isinstance(subj, dict) and subj.get("reference"):
                m = _PAT_REF_RE.search(str(subj["reference"]))
                if m:
                    return m.group(1)
        for link in (out.get("link") or []):
            m = _IDENT_URL_RE.search(str(link.get("url") or ""))
            if m:
                return m.group(1)
    return fallback


def _provenance(event):
    """(context_type, source_channel, source_instance_id, extractor) for a PB FHIR tool_call. context_type
    is the PB vocabulary type of the resourceType actually returned (Patient->patient_identity, Medication*
    ->current_medication_list, AllergyIntolerance->allergy_status); a search is typed by the resourceType
    it queried for. source_instance_id is the specific resource read (<Type>/<id>) or the queried type for a
    search. Errors carry no context_type (nothing was obtained)."""
    tool = event.get("tool")
    extractor = tool
    out = _result_output(event)
    args = event.get("args") if isinstance(event.get("args"), dict) else {}
    # the queried resourceType: explicit arg (generic fhir_search) OR implied by the granular tool name.
    qtype = (args.get("resourceType") or _GRANULAR_SEARCH.get(tool) or _GRANULAR_CREATE.get(tool) or "").strip()
    if _errored(event) or _is_operation_outcome_error(out):
        # what was REQUESTED (for provenance) but nothing usable obtained -> no context type
        inst = ("%s/%s" % (qtype, args.get("id")) if qtype and args.get("id")
                else (qtype or tool))
        return None, _CH_FHIR, inst, extractor
    # a SEARCH (both the granular {"entries":[...]} shape and a raw Bundle): typed by the resourceType of
    # the matched resources (so a category-less Observation search is typed by what it actually returned),
    # falling back to the queried/granular resourceType. Instance keyed by the patient subject.
    ents = _search_entries(out)
    if ents is not None:
        rtype_seen = next((r.get("resourceType") for r in ents if r.get("resourceType")), None)
        ctype = _TOOL_CONTEXT.get(tool) or _RTYPE_CONTEXT.get(str(rtype_seen or "").lower()) or _RTYPE_CONTEXT.get(qtype.lower())
        subj = _patient_subject(event)
        inst = "%s?patient=%s" % (rtype_seen or qtype or "search", subj or _hash8(str(args)))
        return ctype, _CH_FHIR, inst, extractor
    # a single resource body (fhir_read, or a create echoing the created resource)
    rtype = out.get("resourceType") if isinstance(out, dict) else None
    rid = out.get("id") if isinstance(out, dict) else None
    ctype = _TOOL_CONTEXT.get(tool) or _RTYPE_CONTEXT.get(str(rtype or "").lower()) or _RTYPE_CONTEXT.get(qtype.lower())
    inst = "%s/%s" % (rtype, rid) if rtype and rid else (rtype or qtype or tool)
    return ctype, _CH_FHIR, inst, extractor


# ----------------------------------------------------------------------------- evidence extractor
def _pb_evidence(trace):
    """PhysicianBench EvidenceView: real delivery refined with FHIR source provenance. Each unit carries
    context_type (typed by resourceType), source_channel/source_instance_id/extractor, the resolver's
    semantic progress_token, AND a SUBJECT token subject:Patient/<patient_id> distinct from the resource's
    own id so Patient–Evidence Alignment binding converges on the patient subject, not a scatter of resource ids."""
    sem = _S.map_trace(trace, PLUGIN)
    sem_by_step = {id(s.get("raw") or {}): s for s in sem}
    # trace-wide subject: the dominant patient across all calls, used to backfill an anonymous event
    from collections import Counter
    seen = Counter()
    for e in trace:
        if e.get("event_type") != "tool_call":
            continue
        s = _patient_subject(e)
        if s:
            seen[s] += 1
    trace_subject = seen.most_common(1)[0][0] if seen else None

    units = []
    for i, e in enumerate(trace):
        if e.get("event_type") != "tool_call": continue
        d = _S._real_delivery(e)
        ctype, channel, instance, extractor = _provenance(e)
        subj = _patient_subject(e, fallback=trace_subject)
        subject_token = "subject:Patient/%s" % subj if subj else None
        sm = sem_by_step.get(id(e)) or {}
        sem_status, usable = _usable_fields(sm)
        units.append({"id": "%s#%d" % (e.get("tool"), i), "delivered_to_agent": d["delivered"],
                      "delivery_fidelity": d["fidelity"], "error_visible": d["error_visible"],
                      "payload": _S._source_text(e)[:300],
                      "context_type": ctype, "source_channel": channel,
                      "source_instance_id": instance, "extractor": extractor,
                      "subject_token": subject_token, "progress_token": sm.get("progress_token"),
                      "semantic_status": sem_status, "usable_for_context": usable})
    return units


# ----------------------------------------------------------------------------- registration
# Build the full tool_semantics / resolver maps. The generic fhir_search/read/create stay (the conformance
# fixtures + any generic path use them); EVERY granular tool the REAL PhysicianBench environment exposes is
# added so a genuine FHIR read maps to role='acquire' (+ patient_record_loaded), a create to role='commit'
# (+ resource_created), and write_file to role='commit' (+ deliverable_written) — instead of silently
# falling to the default_tool_role='act' with zero milestones.
_TOOL_SEMANTICS = {
    "fhir_search": {"role": "acquire", "success_milestones": ["patient_record_loaded"]},
    "fhir_read": {"role": "acquire", "success_milestones": ["record_detail_loaded"]},
    "fhir_create": {"role": "commit", "success_milestones": ["resource_created"]},
    "write_file": {"role": "commit", "success_milestones": ["deliverable_written"]},
}
_RESOLVERS = {
    "fhir_create": _resolve_fhir_create,
    "fhir_search": _resolve_fhir_search,
    "fhir_read": _resolve_fhir_read,
    "write_file": _resolve_write_file,
}
for _t in _GRANULAR_SEARCH:                       # granular reads -> acquire + patient_record_loaded
    _TOOL_SEMANTICS[_t] = {"role": "acquire", "success_milestones": ["patient_record_loaded"]}
    _RESOLVERS[_t] = _resolve_fhir_search
for _t in _GRANULAR_CREATE:                        # granular writes -> commit + resource_created
    _TOOL_SEMANTICS[_t] = {"role": "commit", "success_milestones": ["resource_created"]}
    _RESOLVERS[_t] = _resolve_fhir_create


# ============================================================ Canonical Evaluation Contract (PB)
# In THIS environment two declared milestones name the SAME capability, so requiring both double-counts it
# and makes the second unreachable. `fhir_read` declares record_detail_loaded and `fhir_search` declares
# patient_record_loaded -- but a PB search does NOT return a summary: its entries carry the FULL resource
# body (a Patient entry is a complete FHIR resource with birthDate/gender/identifier/meta), which
# is exactly what a read would return. An agent that works entirely through search HAS loaded record
# detail. PB tasks list fhir_read in reference.required_tool_groups, while the real environment exposes
# only the granular search/create tools (no fhir_read) -> without the collapse, record_detail_loaded would
# enter every contract and stay unreachable, capping required_operation_completion below 1.0 for EVERY
# agent, a constant handicap that also compresses the dimension's spread.
#
# NOTE ON THE JUSTIFICATION: the reason to collapse is SEMANTIC EQUIVALENCE (search already delivers the
# detail), NOT "nobody used the tool". Those are different, and only the first licenses the collapse -- if
# a PB search returned only summaries, the honest reading would be "the agents should have used fhir_read
# and did not", and the milestone would have to stay. PB search entries do carry full resource bodies.
_EQUIVALENT_MILESTONES = {"record_detail_loaded": "patient_record_loaded"}


def _collapse_equivalent_milestones(req):
    """Collapse milestones that name the same capability in this environment, preserving order and keeping
    the surviving representative only when it is genuinely required."""
    out = []
    for m in req:
        canon = _EQUIVALENT_MILESTONES.get(m, m)
        if canon not in out:
            out.append(canon)
    return out

def _pb_task_contract(task):
    """Compile a PB task into the Canonical Evaluation Contract. PROCESS-ONLY -- never reads the outcome
    gold (which drug/test/diagnosis). Care Action Fulfillment: the UNION of the milestones the task's required_tool_groups
    demand, as ONE CONJUNCTIVE group, so required_operation_completion measures completing ALL required
    operations (retrieve + create-order + write-deliverable) rather than the max-covered single path, which
    would reduce to a trivial 'loaded the record' = 1.0. Clinical Resource Competence.argument_targets carries opaque process flags."""
    ref = task.get("reference") or {}
    rtg = ref.get("required_tool_groups") or []
    ts = PLUGIN.get("tool_semantics", {})
    req = []
    creates_required = False
    for grp in rtg:
        for tool in (grp or []):
            sem = ts.get(tool) or {}
            mss = sem.get("success_milestones") or []
            if sem.get("role") == "commit" and "resource_created" in mss:
                creates_required = True
            for ms in mss:
                if ms not in req:
                    req.append(ms)
    req = _collapse_equivalent_milestones(req)
    execution = {"required_milestone_groups": [req]} if req else {}
    tooling = {"argument_targets": {"order_create_required": creates_required,
                                    "deliverable_required": "deliverable_written" in req}}
    return {"execution": execution, "tooling": tooling, "contract_version": "pb-1"}


def _pb_argument_semantic_score(call, task=None, policy=None, sem_event=None):
    """PROCESS-only argument semantics for ONE PB tool call. Judges whether a CREATE targeted the
    ASSIGNED patient (ANCHORED id match, not substring) and is well-formed, and whether a SEARCH/READ
    scoped to the assigned patient. NEVER judges clinical correctness (drug/dose/test) -- that is the
    Outcome axis. Consumes contract argument_targets (order_create_required)."""
    raw = call if isinstance(call, dict) else {}
    tool = (raw.get("tool") or "")
    args = raw.get("args")
    if isinstance(args, str):
        import json as _j
        try: args = _j.loads(args)
        except Exception: args = {}
    args = args if isinstance(args, dict) else {}
    _targets = ((policy or {}).get("evaluation_contract") or {}).get("tooling", {}).get("argument_targets", {}) or {}
    exp = (policy or {}).get("expected_subject") or {}
    exp_id = exp.get("id")
    def _anchored(ref):
        if not (exp_id and ref): return None
        rs = str(ref).strip(); tail = rs.split("/")[-1].strip()
        return (tail == str(exp_id)) or (rs == str(exp_id))
    is_create = ("create" in tool) or (tool == "fhir_create")
    is_read = ("search" in tool) or ("read" in tool)
    if is_create:
        res = args.get("resource") if isinstance(args.get("resource"), dict) else args
        rtype = (res or {}).get("resourceType")
        # PB uses TYPED create tools (fhir_service_request_create / fhir_medication_request_create / ...)
        # whose resource type is carried in the TOOL NAME, not a nested resource.resourceType. Only a
        # GENERIC fhir_create must supply resourceType; a typed *_create is well-formed by its name.
        is_typed_create = (tool != "fhir_create") and tool.endswith("_create")
        if not rtype and not is_typed_create:
            return {"status": "valid", "score": 0.0, "axis": "resource_wellformed", "reason": "create_without_resourceType"}
        # subject reference: nested FHIR resource.subject, OR the flat *_reference/patient/subject arg the
        # typed create tools take (e.g. patient_reference="Patient/MRN...").
        subj = res.get("subject") if isinstance(res, dict) else None
        subj_ref = None
        if isinstance(subj, dict):
            subj_ref = subj.get("reference")
        elif isinstance(subj, str) and subj:
            subj_ref = subj
        if not subj_ref:
            for _k in ("patient_reference", "subject_reference", "patient", "subject", "beneficiary_reference"):
                _v = args.get(_k)
                if isinstance(_v, str) and _v:
                    subj_ref = _v; break
                if isinstance(_v, dict) and _v.get("reference"):
                    subj_ref = _v["reference"]; break
        _is_appt = (rtype == "Appointment") or ("appointment" in tool)
        if _is_appt and not subj_ref:
            return {"status": "valid", "score": 1.0, "axis": "resource_wellformed", "note": "appointment_uses_participant_not_subject"}
        ok = _anchored(subj_ref)
        if ok is None:
            return {"status": "valid", "score": 1.0, "axis": "resource_wellformed"}
        return {"status": "valid", "score": 1.0 if ok else 0.0, "axis": "subject_match",
                "expected": exp_id, "got": str(subj_ref)[:60]}
    if is_read and exp_id:
        import re as _re
        blob = " ".join(str(v) for v in args.values())
        ids = set(_re.findall(r"MRN\d+", blob))
        if not ids:
            return {"status": "not_applicable", "reason": "read_without_patient_anchor"}
        return {"status": "valid", "score": 1.0 if str(exp_id) in ids else 0.0, "axis": "read_subject_scope",
                "expected": exp_id, "got": sorted(ids)[:3]}
    return {"status": "not_applicable", "reason": "no_process_arg_opportunity"}

PLUGIN = {
    "benchmark": "PhysicianBench", "default_tool_role": "act",
    # SYSTEM-PROFILE effect requirement: when the agent COMMITS an order in its deliverable but does not land
    # it, that order is a REQUIRED operation (the graded outcome IS the effect). effect_milestone is added to
    # the Care Action Fulfillment requirement iff the deliverable contains a committed order (markers = the agent's OWN order
    # language, oracle-blind -- NOT gold). So a run whose committed order is missing scores below one in which
    # the order was actually created.
    "effect_milestone": "resource_created",
    "effect_commitment_markers": ["order ", "order:", "orders:", "will order", "place an order", "obtain ",
                                  "request ", "refer to", "referral", "prescrib", "initiate ", "administer ",
                                  "schedule ", "ultrasound", "imaging", "workup", "servicerequest",
                                  "medicationrequest"],
    "tool_semantics": _TOOL_SEMANTICS,
    "evidence_extractor": _pb_evidence,
    "resolvers": _RESOLVERS,
    "dimension_policy": {"required_milestones": ["patient_record_loaded"],
                         "required_context_units": [
                             {"id": "patient_identity", "type": "patient_identity"},
                             {"id": "current_medication_list", "type": "current_medication_list"},
                             {"id": "allergy_status", "type": "allergy_status"},
                             {"id": "relevant_labs", "type": "relevant_labs"}],
                         # DEFAULT verification_policy. A single authoritative FHIR resource
                         # (one Patient / one MedicationRequest / one AllergyIntolerance from the patient's
                         # OWN record) is a direct single-source fact and is NOT forced to two sources. Only
                         # claims that genuinely warrant corroboration are gated: a high-risk recommendation
                         # (a drug interaction / contraindication / dose-change call), an external medical
                         # fact (a guideline / literature claim beyond the record), or a claim already in
                         # conflict across the record.
                         "verification_policy": {
                             "cross_source_required_for": [{"type": "high_risk_recommendation", "patterns": ["dose", "dosage", "increase", "decrease", "initiate", "discontinue", "anticoagul", "administer", "mg ", "prescrib", "titrate", "start ", "stop ", "hold "]}, {"type": "external_medical_fact", "patterns": ["guideline", "studies show", "literature", "typically causes", "is known to", "according to", "per uptodate", "recommended per", "class effect"]}, {"type": "conflicting_evidence", "patterns": ["conflict", "inconsistent", "discrepan", "contradict", "disagree", "does not match", "mismatch"]}]},
                         "governance_policy_id": "PhysicianBench"}}

PLUGIN["task_contract_resolver"] = _pb_task_contract
PLUGIN["argument_semantic_score"] = _pb_argument_semantic_score
PLUGIN["contract_version"] = "pb-1"
_S.register_plugin(PLUGIN)


# ============================================================ Clinical Auditability contract
# Telemetry families for the benchmark-agnostic Clinical Auditability scorer. FHIR work splits into reads, writes and
# the deliverable; the terminal is its own family. Families are coarse ON PURPOSE: a run that issues 30 lab
# searches must not out-weigh one that issues 3, so all reads collapse into one family.
def _pb_observability_family(sem_event):
    if sem_event.get("terminal") in ("final", "escalate"):
        return "terminal"
    cap = str(sem_event.get("capability_id") or "")
    role = sem_event.get("event_role")
    if "write_file" in cap:
        return "deliverable"
    if "_create" in cap or role == "commit":
        return "clinical_write"
    if "_search" in cap or "_read" in cap or role == "acquire":
        return "record_read"
    return "other:%s" % (role or "unknown")


def _pb_target_resolved(sem_event):
    """Can an operator say UNIQUELY what this operation acted on? Not "args has some key" -- a FHIR write
    must name both the resource type and the subject, a read must name what was queried for whom."""
    e = sem_event.get("raw") or {}
    a = e.get("args") if isinstance(e.get("args"), dict) else {}
    cap = str(sem_event.get("capability_id") or "")
    if "write_file" in cap:
        return bool(str(a.get("file_path") or "").strip())
    subject = any(str(a.get(k) or "").strip() for k in ("patient_reference", "patient", "subject", "mrn"))
    rtype = bool(str(a.get("resourceType") or "").strip()) or bool(_GRANULAR_SEARCH.get(cap)) \
        or bool(_GRANULAR_CREATE.get(cap))
    if "_create" in cap or sem_event.get("event_role") == "commit":
        return bool(rtype and subject)          # a write must identify resource AND subject
    if a.get("id"):
        return True                             # a direct read by id is unambiguous
    return bool(rtype and (subject or a.get("query") or a.get("category")))


def _pb_receipt(sem_event):
    """FHIR receipt = the server ECHOED the affected resource back. A create/update must return a body with
    resourceType+id; a read is witnessed by the returned resource itself. The substrate's own progress_token
    is NOT a receipt -- it is generated locally whether or not the server confirmed anything."""
    e = sem_event.get("raw") or {}
    out = _result_output(e)
    if not isinstance(out, dict):
        return False
    if out.get("resourceType") and out.get("id"):
        return True
    ents = _search_entries(out)
    if ents:
        return any(isinstance(r, dict) and r.get("resourceType") for r in ents)
    return bool(out.get("written"))          # write_file echoes the path it actually wrote


PLUGIN["observability_contract"] = {
    "target_resolver": _pb_target_resolved,
    "receipt_resolver": _pb_receipt,
    "family_of": _pb_observability_family,
    # no span-attributable token/latency/retry telemetry is recorded for PB -> ResourceAccountability = N/A
    "cost_signals": [],
}
