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
    per-resource id (an Observation has its OWN id 190335, but its SUBJECT is the patient) so Context
    binding converges on the patient SUBJECT, not a scatter of resource-own ids."""
import re as _re
import substrate as _S

_errored = _S._errored
_result_output = _S._result_output
_hash8 = _S._hash8


# --- CONTRACT-A: usable_for_context (shared cross-benchmark) ---
def _usable_fields(sem_event):
    """CONTRACT-A: derive (semantic_status, usable_for_context) for an EvidenceUnit from its PRODUCING
    SemanticEvent (the resolver's resolved status, NOT the tool name). usable_for_context = (status=='
    success' AND a non-empty progress_token). An EMPTY Bundle (total 0) / OperationOutcome / id-less read is
    delivered+partial with NO token -> usable_for_context=False: it still feeds Observability/delivery but
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


def _bundle_count(out):
    """Number of matched resources in a FHIR search result (Bundle.total or len(entry)); None if not a
    bundle."""
    if not isinstance(out, dict):
        return None
    if out.get("resourceType") != "Bundle":
        return None
    if isinstance(out.get("total"), int):
        return out["total"]
    return len(out.get("entry") or [])


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
    ids = ",".join(sorted(str((en.get("resource") or {}).get("id") or "")
                          for en in (out.get("entry") or []))[:50])
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
        for en in (out.get("entry") or []):
            res = en.get("resource") or {}
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
    if _errored(event) or _is_operation_outcome_error(out):
        # what was REQUESTED (for provenance) but nothing usable obtained -> no context type
        rtype = (args.get("resourceType") or "").strip()
        inst = ("%s/%s" % (rtype, args.get("id")) if rtype and args.get("id")
                else (rtype or tool))
        return None, _CH_FHIR, inst, extractor
    if isinstance(out, dict) and out.get("resourceType") == "Bundle":
        # a search: typed by the queried resourceType; instance keyed by the query target
        qtype = (args.get("resourceType") or "").strip()
        ctype = _RTYPE_CONTEXT.get(qtype.lower())
        subj = _patient_subject(event)
        inst = "%s?patient=%s" % (qtype or "search", subj or _hash8(str(args)))
        return ctype, _CH_FHIR, inst, extractor
    rtype = out.get("resourceType") if isinstance(out, dict) else None
    rid = out.get("id") if isinstance(out, dict) else None
    ctype = _RTYPE_CONTEXT.get(str(rtype or "").lower())
    inst = "%s/%s" % (rtype, rid) if rtype and rid else (rtype or tool)
    return ctype, _CH_FHIR, inst, extractor


# ----------------------------------------------------------------------------- evidence extractor
def _pb_evidence(trace):
    """PhysicianBench EvidenceView: real delivery refined with FHIR source provenance. Each unit carries
    context_type (typed by resourceType), source_channel/source_instance_id/extractor, the resolver's
    semantic progress_token, AND a SUBJECT token subject:Patient/<patient_id> distinct from the resource's
    own id so Context binding converges on the patient subject, not a scatter of resource ids."""
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
PLUGIN = {
    "benchmark": "PhysicianBench", "default_tool_role": "act",
    "tool_semantics": {
        "fhir_search": {"role": "acquire", "success_milestones": ["patient_record_loaded"]},
        "fhir_read": {"role": "acquire", "success_milestones": ["record_detail_loaded"]},
        "fhir_create": {"role": "commit", "success_milestones": ["resource_created"]}},
    "evidence_extractor": _pb_evidence,
    "resolvers": {"fhir_create": _resolve_fhir_create,
                  "fhir_search": _resolve_fhir_search,
                  "fhir_read": _resolve_fhir_read},
    "dimension_policy": {"required_milestones": ["patient_record_loaded"],
                         "required_context_units": [
                             {"id": "patient_identity", "type": "patient_identity"},
                             {"id": "current_medication_list", "type": "current_medication_list"},
                             {"id": "allergy_status", "type": "allergy_status"}],
                         # V1 / CONTRACT-B DEFAULT verification_policy. A single authoritative FHIR resource
                         # (one Patient / one MedicationRequest / one AllergyIntolerance from the patient's
                         # OWN record) is a direct single-source fact and is NOT forced to two sources. Only
                         # claims that genuinely warrant corroboration are gated: a high-risk recommendation
                         # (a drug interaction / contraindication / dose-change call), an external medical
                         # fact (a guideline / literature claim beyond the record), or a claim already in
                         # conflict across the record.
                         "verification_policy": {
                             "cross_source_required_for": [{"type": "high_risk_recommendation", "patterns": ["dose", "dosage", "increase", "decrease", "initiate", "discontinue", "anticoagul", "administer", "mg ", "prescrib", "titrate", "start ", "stop ", "hold "]}, {"type": "external_medical_fact", "patterns": ["guideline", "studies show", "literature", "typically causes", "is known to", "according to", "per uptodate", "recommended per", "class effect"]}, {"type": "conflicting_evidence", "patterns": ["conflict", "inconsistent", "discrepan", "contradict", "disagree", "does not match", "mismatch"]}]},
                         "governance_policy_id": "PhysicianBench"}}

_S.register_plugin(PLUGIN)
