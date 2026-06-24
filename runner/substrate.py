#!/usr/bin/env python3
"""Universal substrate for benchmark-AGNOSTIC dimension scoring.

The 7 dimension evaluators are meant to consume ONLY these structures and NEVER a benchmark name, tool
name, image, FHIR resource, or DOM selector:
  CanonicalTask     - the unified task dict (spec/task.schema.json)
  SemanticTrace     - list[SemanticEvent]  (event_role / milestones / progress / failure_attribution)
  CapabilityManifest- four-state {implemented, available, authorized, healthy} per capability
  DimensionPolicy   - per-dimension declarative policy (required_milestones / required_context_units /
                      lifecycle_policy / governance_policy_id)
  EvidenceView      - list[EvidenceUnit] (id / delivered / fidelity / error_visible / acknowledged)

A BenchmarkPlugin supplies the ONLY benchmark-specific knowledge: how a tool maps to a semantic role and
which milestones it produces, how to extract evidence units, the dimension policy, and the source outcome
evaluator. A 4th dataset registers a plugin; the dimension evaluators never change.

SemanticEvent v2 CONTRACT (every tool_call event carries these; consumers read via .get so adding fields
stays back-compatible):
  capability_id : the invoked tool/capability id. SET ON EVERY tool_call INCLUDING failures (Execution /
                  attribution joins it to the manifest healthy/authorized state). Never dropped.
  obligation_id : the PRIMARY milestone this tool is declared to produce (binds a FAILURE to its RECOVERY:
                  a failed obligation is recovered ONLY by a later event producing the SAME obligation_id,
                  i.e. the same tool retried or an alt tool in the same required-tool group -- not by "any
                  later progress"). None when the tool serves no declared milestone.
  progress_token: a SEMANTIC token of the RESULT, never the tool name
                  (evidence:<source>:<hash8> / state:<k>=<v> / resource:<type>/<id>:created /
                  region:<id>:resolved). Derived from the rendered/source CONTENT so OCR(page1) and
                  OCR(page2) differ; a repeated identical call repeats its token (no progress). None when
                  no new evidence/state.
  status        : success | partial | failure  (partial = ran without error but did NOT meet its semantic
                  goal -> milestone withheld).
  state_changed : True ONLY if this event yielded a NEW milestone OR a NEW progress_token."""
import re, json, hashlib

# ----------------------------------------------------------------------------- SemanticEvent
ROLES = ("acquire", "act", "verify", "commit", "final", "escalate", "other")
FAILURE_ATTRIB = ("agent", "environment", "external_service", "harness", "unknown", None)
STATUSES = ("success", "partial", "failure")

def semantic_event(role, status="success", capability_id=None, obligation_id=None, progress_token=None,
                   milestones_added=None, state_changed=False, terminal=None, failure_attribution=None,
                   raw=None):
    """v2 SemanticEvent. capability_id/obligation_id default None for non-tool events (final/escalate);
    map_trace sets them on every tool_call. Consumers read every field via .get(), so unset fields are
    simply absent-valued, never schema-breaking."""
    assert role in ROLES, "unknown role %r" % role
    assert status in STATUSES, "unknown status %r" % status
    return {"event_role": role, "status": status, "capability_id": capability_id,
            "obligation_id": obligation_id, "progress_token": progress_token,
            "milestones_added": list(milestones_added or []), "state_changed": bool(state_changed),
            "terminal": terminal, "failure_attribution": failure_attribution, "raw": raw}

# ----------------------------------------------------------------------------- BenchmarkPlugin registry
_PLUGINS = {}
def register_plugin(p): _PLUGINS[p["benchmark"]] = p
def get_plugin(name):
    """FAIL-CLOSED: returns None for an unknown/unregistered benchmark (no silent default plugin). Callers
    that must score MUST go through require_plugin() so a missing plugin flips score_eligible off rather
    than being scored against an empty/default policy."""
    return _PLUGINS.get(name)
def list_plugins(): return sorted(_PLUGINS)

def require_plugin(name):
    """Resolve a plugin or FLAG the gap. Returns (plugin, problem):
       - (plugin, None)                          when registered,
       - (None, 'missing_benchmark_plugin:<n>')  when not -- the caller sets score_eligible=False instead
         of scoring against a vacuous default. Never raises silently-defaulting state."""
    p = _PLUGINS.get(name)
    if p is None:
        return None, "missing_benchmark_plugin:%s" % name
    return p, None

# ----------------------------------------------------------------------------- helpers
def _errored(e):
    try:
        import proxy_verifiers as _pv
        return _pv._errored(e)
    except Exception:
        return str(e.get("status", "")).lower() == "error" or bool(e.get("error") or e.get("error_type"))

def _attr(e):
    try:
        import lifecycle_exec as _le
        return _le.error_attribution(e)
    except Exception:
        return "unknown"

def _hash8(s):
    return hashlib.sha1((s or "").encode("utf-8", "replace")).hexdigest()[:8]

def _result_output(e):
    """The raw tool OUTPUT object. Most MedCTA tools nest it under result.output; PB/HAB put the resource /
    page dict directly at result. Returns whatever the tool produced (dict|str|None)."""
    r = e.get("result")
    if isinstance(r, dict) and "output" in r:
        return r.get("output")
    return r

def _no_payload(e):
    """A degenerate event with NO observable output (no result, no observation, no canonical_observation):
    the resolver has nothing to read, so it cannot prove the semantic goal was UNMET. Real traces always
    carry a payload; only synthetic stubs are empty. In that case the mapper falls back to the static
    optimistic milestone instead of asserting partial -- so result-conditional grading applies exactly when
    there IS a result to condition on."""
    if e.get("result") not in (None, "", {}):
        return False
    ob = e.get("observation")
    if ob not in (None, "", {}):
        return False
    co = e.get("canonical_observation")
    if isinstance(co, dict) and (co.get("modalities") or {}).get("text"):
        return False
    return True

def _primary_obligation(meta):
    """The obligation_id a tool is declared responsible for = its PRIMARY success milestone (explicit
    'obligation' override, else the first declared success_milestone). None when the tool earns none."""
    if not isinstance(meta, dict):
        return None
    if meta.get("obligation"):
        return meta.get("obligation")
    ms = meta.get("success_milestones") or []
    return ms[0] if ms else None

# ----------------------------------------------------------------------------- SemanticEventMapper
def map_trace(trace, plugin):
    """SemanticEventMapper: raw canonical trace -> [SemanticEvent] under the v2 contract. A tool's meaning
    comes from the plugin: a RESULT-CONDITIONAL resolver(event, prev_state) if declared (reads the actual
    OUTPUT to decide real state change / which milestones were truly achieved), else the static
    tool_semantics (optimistic).

    Every tool_call event carries capability_id (= the tool name, INCLUDING failures), obligation_id (= the
    tool's primary declared milestone, or None), a SEMANTIC progress_token (content-hashed, never the tool
    name), a v2 status (success|partial|failure), and a NOVELTY-gated state_changed (a success that repeats
    a prior token/milestone is state_changed=False so stagnation can see it). The core names no
    benchmark/tool; the plugin does."""
    tool_sem = (plugin or {}).get("tool_semantics", {})
    resolvers = (plugin or {}).get("resolvers", {})
    default_role = (plugin or {}).get("default_tool_role", "act")
    out = []
    seen_ms, seen_pt = set(), set()
    prev_state = {"milestones": seen_ms, "tokens": seen_pt}
    for e in trace:
        et = e.get("event_type")
        if et == "tool_call":
            tool = e.get("tool")
            meta = tool_sem.get(tool, {"role": default_role, "success_milestones": []})
            cap_id = tool                                   # capability_id ALWAYS set, even on failure
            obligation = _primary_obligation(meta)
            ok = not _errored(e)
            res = resolvers.get(tool)
            if res and _no_payload(e) and not e.get("semantic_assume_success"):
                # P6: a resolver exists (this tool HAS a semantic completion condition) but there is NO output
                # to prove it was met -> CONSERVATIVE partial, milestone WITHHELD (never optimistic). Test
                # fixtures that legitimately assume success set semantic_assume_success=True.
                role, status, ms, pt, explicit_changed = meta.get("role", default_role), "partial", [], None, False
            else:
                if res and _no_payload(e):
                    res = None      # flagged test stub -> static optimistic semantics
                if res:
                    r = res(e, prev_state) or {}
                    role = r.get("role") or meta.get("role", default_role)
                    status = r.get("status", "success" if ok else "failure")
                    ms = list(r.get("milestones_added") or [])
                    pt = r.get("progress_token")
                    obligation = r.get("obligation_id", obligation)
                    explicit_changed = r.get("state_changed")
                else:
                    role = meta.get("role", default_role)
                    status = "success" if ok else "failure"
                    ms = list(meta.get("success_milestones", [])) if ok else []
                    pt = _default_token(e, tool) if ok else None     # content-hashed evidence, never tool name
                    explicit_changed = None
            if status not in STATUSES:
                status = "success" if ok else "failure"
            attr = None if status != "failure" else _attr(e)
            new_ms = [m for m in ms if m not in seen_ms]
            new_pt = pt is not None and pt not in seen_pt
            _sr = e.get("state_record") if isinstance(e.get("state_record"), dict) else None
            _rec_changed = _sr.get("state_changed") if _sr else None
            if _rec_changed is not None:                         # authoritative real state diff (review 3.4)
                changed = bool(_rec_changed) and status == "success"
            elif explicit_changed is not None:
                changed = bool(explicit_changed) and status == "success"
            else:
                changed = (status == "success") and (bool(new_ms) or new_pt)
            out.append(semantic_event(role, status=status, capability_id=cap_id, obligation_id=obligation,
                       progress_token=pt, milestones_added=ms, state_changed=changed,
                       failure_attribution=attr, raw=e))
            seen_ms.update(ms)
            if pt is not None: seen_pt.add(pt)
        elif et == "final_answer":
            out.append(semantic_event("final", terminal="final", raw=e))
        elif et and ("escalat" in str(et).lower() or et == "deliverable_budget_warning"):
            out.append(semantic_event("escalate", terminal="escalate", raw=e))
    return out

def _default_token(e, tool):
    """Content-hashed evidence token for a tool with no declared resolver: evidence:<tool>:<hash8 of the
    rendered/source content>. Two different outputs of the same tool get DIFFERENT tokens (new evidence); a
    byte-identical repeat repeats the token (no progress). None when there is no content."""
    content = _rendered_text(e) or _source_text(e)
    content = (content or "").strip()
    if not content:
        return None
    return "evidence:%s:%s" % (tool, _hash8(content))

def milestones_reached(sem_trace):
    m = set()
    for s in sem_trace:
        m.update(s.get("milestones_added") or [])
    return m

# ----------------------------------------------------------------------------- EvidenceView
def _rendered_text(e):
    """What the agent ACTUALLY saw. Priority: runner-recorded agent_visible_text (exact string put into the
    agent context) > raw observation > canonical_observation (audit mirror, compat fallback only)."""
    av = e.get("agent_visible_text")
    if isinstance(av, str) and av:
        return av
    ob = e.get("observation")
    if isinstance(ob, str) and ob:
        return ob
    co = e.get("canonical_observation")
    if isinstance(co, dict):
        t = (co.get("modalities") or {}).get("text")
        if t: return str(t)
    ob = e.get("observation")
    if isinstance(ob, str): return ob
    if isinstance(ob, dict): return json.dumps(ob, ensure_ascii=False)
    return ""

def _source_text(e):
    r = e.get("result")
    out = r.get("output") if isinstance(r, dict) else r
    if isinstance(out, dict): out = out.get("text") or json.dumps(out, ensure_ascii=False)
    return str(out or "")

def _real_delivery(e):
    """Prefer the RECORDED delivery_record (run.py renderer-level instrumentation, review 4.2); fall back to
    deriving from canonical_observation when an older trace carries no record."""
    dr = e.get("delivery_record")
    if isinstance(dr, dict):
        errored = _errored(e)
        delivered = bool(dr.get("rendered_to_agent")) and bool(dr.get("produced")) and not errored
        fid = 0.0 if errored else (0.5 if dr.get("truncated") else (1.0 if dr.get("rendered_to_agent") else 0.0))
        return {"delivered": delivered, "fidelity": fid, "error_visible": bool(dr.get("error_state_rendered"))}
    # --- legacy fallback (no delivery_record): derive from canonical_observation ---

    """REAL info-flow from the RECORDED rendering, NOT inferred from tool status. delivered = content was
    actually rendered into the agent context; fidelity = how much of the source survived (truncation/drop);
    error_visible = the failure actually appears in the agent-visible surface."""
    errored = _errored(e)
    rendered = _rendered_text(e); src = _source_text(e)
    if errored:
        ev = any(k in rendered.lower() for k in ("error", "fail", "exception", "operationoutcome"))
        return {"delivered": False, "fidelity": 0.0, "error_visible": ev}
    if not rendered.strip():
        return {"delivered": False, "fidelity": 0.0, "error_visible": False}   # renderer dropped the result
    if src.strip():
        sl = src.strip().lower(); rl = rendered.lower()
        fid = 1.0 if (sl[:200] in rl or len(rl) >= 0.8 * len(sl)) else round(min(1.0, len(rl) / max(1, len(sl))), 3)
    else:
        fid = 1.0
    return {"delivered": True, "fidelity": fid, "error_visible": False}

def evidence_view(trace, plugin):
    """EvidenceUnit = {id, delivered_to_agent, delivery_fidelity, error_visible, payload}. Delivery is read
    from the RECORDED agent-visible rendering (canonical_observation/observation), NOT inferred from status.
    A plugin extractor may refine fidelity with modality-specific signals (e.g. localization)."""
    ext = (plugin or {}).get("evidence_extractor")
    if ext: return ext(trace)
    units = []
    for i, e in enumerate(trace):
        if e.get("event_type") != "tool_call": continue
        d = _real_delivery(e)
        units.append({"id": "%s#%d" % (e.get("tool"), i), "delivered_to_agent": d["delivered"],
                      "delivery_fidelity": d["fidelity"], "error_visible": d["error_visible"],
                      "payload": _source_text(e)[:300]})
    return units

# ----------------------------------------------------------------------------- CapabilityManifest accessor
def capability_manifest(provenance):
    caps = (provenance or {}).get("capabilities") or {}
    return {k: {"implemented": v.get("implemented"), "available": v.get("available"),
                "authorized": v.get("authorized"), "healthy": v.get("healthy")}
            for k, v in caps.items() if isinstance(v, dict)}

# ----------------------------------------------------------------------------- DimensionPolicy accessor
def dimension_policy(task, plugin=None):
    """Merge the task's declared policy with the benchmark plugin defaults. Dimension evaluators read
    required_milestones / required_context_units / lifecycle_policy / governance_policy_id from HERE.

    FAIL-CLOSED: if the task's benchmark has NO registered plugin, the returned policy carries
    score_eligible=False + plugin_problem=missing_benchmark_plugin so the dimension evaluators do NOT
    silently score against a vacuous default policy."""
    name = task.get("source_benchmark")
    if plugin is None:
        plugin, problem = require_plugin(name)
        if plugin is None:
            return {"score_eligible": False, "plugin_problem": problem,
                    "required_milestones": [], "required_context_units": [],
                    "governance_policy_id": None}
    base = dict(plugin.get("dimension_policy") or {})
    ref = task.get("reference") or {}
    if ref.get("required_tool_groups"):
        base.setdefault("required_tool_groups", ref["required_tool_groups"])
        # PER-TASK readiness/completion: map each required tool GROUP to the milestone set its tools emit,
        # so |reached_milestones ∩ group_milestones| / |group_milestones| reproduces the old tool-path
        # completion fraction -- expressed in milestones (the evaluator never sees a tool name). This keeps
        # per-task discrimination (a 4-tool task the agent half-finished scores ~0.5), not a flat 1.0.
        tsem = (plugin or {}).get("tool_semantics", {})
        groups_ms = []
        for grp in ref["required_tool_groups"]:
            ms = set()
            for tool in (grp or []):
                ms.update((tsem.get(tool) or {}).get("success_milestones") or [])
            if ms: groups_ms.append(sorted(ms))
        if groups_ms:
            base["required_milestone_groups"] = groups_ms
            base["required_milestones"] = max(groups_ms, key=len)   # the most complete required path
    lifecycle_policy = task.get("lifecycle_policy") or {}
    base["lifecycle_policy"] = lifecycle_policy                      # P3: SINGLE normalization entry --
    for _f in ("ordering_constraints", "terminal_policy", "escalation_conditions", "stagnation_window",
               "irrecoverable_evidence_gap", "non_recoverable_obligations"):
        if _f in lifecycle_policy:                                  # lift sub-fields to top level so the
            base[_f] = lifecycle_policy[_f]                         # evaluator never guesses where they live
    base.setdefault("governance_policy_id", task.get("source_benchmark"))
    return base

# ============================================================================= RESOLVERS
# Each resolver reads the REAL output to decide success/partial and which milestones are truly earned.
# success -> milestone(s) granted; partial -> ran without error but goal unmet, milestone WITHHELD;
# failure -> errored. Resolvers may return obligation_id to override the static primary milestone.

# ---- MedCTA ----------------------------------------------------------------
def _resolve_region(event, prev_state):
    """RegionAttributeDescription: a successful HTTP call that FELL BACK to the whole image (resolved=False)
    did NOT examine the targeted region -> it must NOT be credited target_region_examined. Reads the real
    localization status from result.output.localization."""
    out = _result_output(event)
    loc = out.get("localization") if isinstance(out, dict) else None
    if _errored(event):
        return {"role": "acquire", "status": "failure", "milestones_added": [], "state_changed": False,
                "obligation_id": "target_region_examined"}
    if isinstance(loc, dict) and loc.get("resolved") is True:
        return {"role": "acquire", "status": "success", "state_changed": True,
                "milestones_added": ["target_region_examined", "relevant_image_evidence_obtained"],
                "obligation_id": "target_region_examined",
                "progress_token": "region:%s:resolved" % _hash8(str(loc.get("requested") or "region"))}
    # fell back to the full image: general image evidence only, the targeted region was NOT examined
    return {"role": "acquire", "status": "partial", "state_changed": False,
            "milestones_added": ["image_overview_obtained"], "obligation_id": "target_region_examined",
            "progress_token": None}

def _resolve_ocr(event, prev_state):
    """MedCTA OCR: empty/blank rendered text -> partial, NO text_evidence_obtained, progress_token=None
    (the page carried no readable text -> no evidence). Non-empty text -> success with a CONTENT-hashed
    evidence token so OCR(page1) and OCR(page2) earn DIFFERENT tokens (new evidence) while a repeated
    identical OCR repeats its token (no progress)."""
    if _errored(event):
        return {"role": "acquire", "status": "failure", "milestones_added": [],
                "obligation_id": "text_evidence_obtained", "state_changed": False, "progress_token": None}
    out = _result_output(event)
    text = out if isinstance(out, str) else (out.get("text") if isinstance(out, dict) else "")
    text = (text or "").strip()
    if not text:
        return {"role": "acquire", "status": "partial", "milestones_added": [],
                "obligation_id": "text_evidence_obtained", "state_changed": False, "progress_token": None}
    return {"role": "acquire", "status": "success", "milestones_added": ["text_evidence_obtained"],
            "obligation_id": "text_evidence_obtained",
            "progress_token": "evidence:ocr:%s" % _hash8(text)}

_GS_EMPTY = ("[no offline result", "no result", "no results found", "[no result")
def _resolve_googlesearch(event, prev_state):
    """MedCTA GoogleSearch: a '[no offline result]' / empty / irrelevant snippet -> partial, no
    external_reference_obtained milestone (the search returned nothing usable). A real snippet -> success
    with a content-hashed external-reference token."""
    if _errored(event):
        return {"role": "acquire", "status": "failure", "milestones_added": [],
                "obligation_id": "external_reference_obtained", "state_changed": False, "progress_token": None}
    out = _result_output(event)
    snippet = out if isinstance(out, str) else (out.get("text") if isinstance(out, dict) else "")
    snippet = (snippet or "").strip()
    low = snippet.lower()
    if (not snippet) or any(low.startswith(m) or m in low for m in _GS_EMPTY):
        return {"role": "acquire", "status": "partial", "milestones_added": [],
                "obligation_id": "external_reference_obtained", "state_changed": False, "progress_token": None}
    return {"role": "acquire", "status": "success", "milestones_added": ["external_reference_obtained"],
            "obligation_id": "external_reference_obtained",
            "progress_token": "evidence:search:%s" % _hash8(snippet)}

# ---- PhysicianBench (FHIR) -------------------------------------------------
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

# ---- HealthAdminBench (browser) --------------------------------------------
def _hab_page(event):
    """The rendered page surface for a HAB action: prefer the recorded agent-visible text, else the tool
    result's embedded 'observation' page text. Returns (url, page_text)."""
    out = _result_output(event)
    url = out.get("url") if isinstance(out, dict) else None
    page = ""
    if isinstance(out, dict):
        page = out.get("observation") or ""
    if not page:
        page = _rendered_text(event)
    return url, str(page or "")

_SUBMIT_CONFIRM = ("submitted", "success", "confirmation", "thank you", "has been submitted",
                   "received", "your appeal", "appeal submitted", "saved", "confirmed",
                   "submission complete", "successfully")
def _resolve_submit(event, prev_state):
    """HAB submit: accepted but NO confirmation in the RENDERED observation -> partial, no form_submitted
    (a button press with no confirmation surface is not a completed submission). A rendered confirmation ->
    success with a state:submitted token keyed by the confirming page."""
    if _errored(event):
        return {"role": "commit", "status": "failure", "milestones_added": [],
                "obligation_id": "form_submitted", "state_changed": False, "progress_token": None}
    url, page = _hab_page(event)
    low = page.lower()
    if any(k in low for k in _SUBMIT_CONFIRM):
        return {"role": "commit", "status": "success", "milestones_added": ["form_submitted"],
                "obligation_id": "form_submitted",
                "progress_token": "state:submitted=%s" % _hash8((url or "") + "|" + page)}
    return {"role": "commit", "status": "partial", "milestones_added": [],
            "obligation_id": "form_submitted", "state_changed": False, "progress_token": None}

def _resolve_dom_action(event, prev_state):
    """HAB click/type: state_changed ONLY if the rendered page state actually DIFFERS from the last page the
    agent saw (a click/type that left the page unchanged made no progress -> partial, no token). The page
    surface is content-hashed into a state:page token so a real navigation/expansion advances state while a
    no-op repeats the prior page token."""
    if _errored(event):
        return {"role": "act", "status": "failure", "milestones_added": [], "state_changed": False,
                "progress_token": None}
    url, page = _hab_page(event)
    surface = (url or "") + "|" + page.strip()
    if not page.strip():
        return {"role": "act", "status": "partial", "milestones_added": [], "state_changed": False,
                "progress_token": None}
    token = "state:page=%s" % _hash8(surface)
    seen = prev_state.get("tokens") or set()
    if token in seen:
        # the page is identical to one already seen -> the action produced no new state
        return {"role": "act", "status": "partial", "milestones_added": [], "state_changed": False,
                "progress_token": token}
    return {"role": "act", "status": "success", "milestones_added": [], "state_changed": True,
            "progress_token": token}

# ----------------------------------------------------------------------------- evidence extractor (MedCTA)
def _medcta_evidence(trace):
    """MedCTA EvidenceView: real delivery (canonical_observation) refined by localization — a delivered region
    result that FELL BACK to the whole image is delivered but low-fidelity (targeted region not localized)."""
    units = []
    for i, e in enumerate(trace):
        if e.get("event_type") != "tool_call": continue
        d = _real_delivery(e)
        out = _result_output(e)
        loc = out.get("localization") if isinstance(out, dict) else None
        fid = d["fidelity"]
        if d["delivered"] and isinstance(loc, dict) and loc.get("resolved") is False:
            fid = min(fid, 0.5)
        txt = out.get("text") if isinstance(out, dict) else _source_text(e)
        units.append({"id": "%s#%d" % (e.get("tool"), i), "delivered_to_agent": d["delivered"],
                      "delivery_fidelity": fid, "error_visible": d["error_visible"], "payload": str(txt)[:300]})
    return units

# ----------------------------------------------------------------------------- plugin registrations
register_plugin({
    "benchmark": "MedCTA", "default_tool_role": "acquire",
    "tool_semantics": {
        "ImageDescription": {"role": "acquire", "success_milestones": ["image_overview_obtained", "relevant_image_evidence_obtained"]},
        "RegionAttributeDescription": {"role": "acquire", "success_milestones": ["target_region_examined", "relevant_image_evidence_obtained"]},
        "OCR": {"role": "acquire", "success_milestones": ["text_evidence_obtained"]},
        "GoogleSearch": {"role": "acquire", "success_milestones": ["external_reference_obtained"]},
        "Calculator": {"role": "act", "success_milestones": []}},
    "evidence_extractor": _medcta_evidence,
    "resolvers": {"RegionAttributeDescription": _resolve_region,
                  "OCR": _resolve_ocr,
                  "GoogleSearch": _resolve_googlesearch},
    "dimension_policy": {"required_milestones": ["relevant_image_evidence_obtained"],
                         "required_context_units": ["target_image_evidence"],
                         "governance_policy_id": "MedCTA"}})
register_plugin({
    "benchmark": "PhysicianBench", "default_tool_role": "act",
    "tool_semantics": {
        "fhir_search": {"role": "acquire", "success_milestones": ["patient_record_loaded"]},
        "fhir_read": {"role": "acquire", "success_milestones": ["record_detail_loaded"]},
        "fhir_create": {"role": "commit", "success_milestones": ["resource_created"]}},
    "resolvers": {"fhir_create": _resolve_fhir_create,
                  "fhir_search": _resolve_fhir_search,
                  "fhir_read": _resolve_fhir_read},
    "dimension_policy": {"required_milestones": ["patient_record_loaded"],
                         "required_context_units": ["correct_patient", "current_medications", "allergy_status"],
                         "governance_policy_id": "PhysicianBench"}})
register_plugin({
    "benchmark": "HealthAdminBench", "default_tool_role": "act",
    "tool_semantics": {
        "snapshot": {"role": "verify", "success_milestones": ["page_state_observed"]},
        "navigate": {"role": "acquire", "success_milestones": ["target_page_reached"]},
        "click": {"role": "act", "success_milestones": []},
        "type": {"role": "act", "success_milestones": []},
        "submit": {"role": "commit", "success_milestones": ["form_submitted"]}},
    "resolvers": {"submit": _resolve_submit,
                  "click": _resolve_dom_action,
                  "type": _resolve_dom_action},
    "dimension_policy": {"required_milestones": ["form_submitted"],
                         "required_context_units": ["correct_case", "current_form_state", "submission_requirements"],
                         "governance_policy_id": "HealthAdminBench"}})
