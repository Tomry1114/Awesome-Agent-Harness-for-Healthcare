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
evaluator. A 4th dataset registers a plugin; the dimension evaluators never change."""
import re

# ----------------------------------------------------------------------------- SemanticEvent
ROLES = ("acquire", "act", "verify", "commit", "final", "escalate", "other")
FAILURE_ATTRIB = ("agent", "environment", "external_service", "harness", "unknown", None)

def semantic_event(role, status="success", failure_attribution=None, milestones_added=None,
                   progress_token=None, state_changed=False, terminal=None, raw=None):
    assert role in ROLES, "unknown role %r" % role
    return {"event_role": role, "status": status, "failure_attribution": failure_attribution,
            "milestones_added": list(milestones_added or []), "progress_token": progress_token,
            "state_changed": bool(state_changed), "terminal": terminal, "raw": raw}

# ----------------------------------------------------------------------------- BenchmarkPlugin registry
_PLUGINS = {}
def register_plugin(p): _PLUGINS[p["benchmark"]] = p
def get_plugin(name): return _PLUGINS.get(name)
def list_plugins(): return sorted(_PLUGINS)

# ----------------------------------------------------------------------------- SemanticEventMapper
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

def map_trace(trace, plugin):
    """SemanticEventMapper: raw canonical trace -> [SemanticEvent]. A tool's meaning comes from the plugin:
    a RESULT-CONDITIONAL resolver(event, prev_state) if declared (reads the actual OUTPUT to decide real
    state change / which milestones were truly achieved), else the static tool_semantics (optimistic).
    state_changed is NOVELTY-gated: a successful call that yields no milestone/progress not already seen does
    NOT advance state -> repeated no-progress calls become visible to stagnation, and a tool that 'returns OK'
    without doing real work is not auto-credited. The core still names no benchmark/tool; the plugin does."""
    tool_sem = (plugin or {}).get("tool_semantics", {})
    resolvers = (plugin or {}).get("resolvers", {})
    default_role = (plugin or {}).get("default_tool_role", "act")
    out = []
    seen_ms, seen_pt = set(), set()
    prev_state = {"milestones": seen_ms, "tokens": seen_pt}
    for e in trace:
        et = e.get("event_type")
        if et == "tool_call":
            ok = not _errored(e)
            res = resolvers.get(e.get("tool"))
            if res:
                r = res(e, prev_state) or {}
                role = r.get("role") or tool_sem.get(e.get("tool"), {}).get("role", default_role)
                status = r.get("status", "success" if ok else "failure")
                ms = list(r.get("milestones_added") or [])
                pt = r.get("progress_token")
                explicit_changed = r.get("state_changed")
            else:
                meta = tool_sem.get(e.get("tool"), {"role": default_role, "success_milestones": []})
                role = meta.get("role", default_role)
                status = "success" if ok else "failure"
                ms = list(meta.get("success_milestones", [])) if ok else []
                pt = (meta.get("progress_token") or e.get("tool")) if ok else None
                explicit_changed = None
            attr = None if status != "failure" else _attr(e)
            new_ms = [m for m in ms if m not in seen_ms]
            new_pt = pt is not None and pt not in seen_pt
            if explicit_changed is not None:
                changed = bool(explicit_changed) and status != "failure"
            else:
                changed = (status != "failure") and (bool(new_ms) or new_pt)
            out.append(semantic_event(role, status=status, failure_attribution=attr,
                       milestones_added=ms, progress_token=pt, state_changed=changed, raw=e))
            seen_ms.update(ms)
            if pt is not None: seen_pt.add(pt)
        elif et == "final_answer":
            out.append(semantic_event("final", terminal="final", raw=e))
        elif et and ("escalat" in str(et).lower() or et == "deliverable_budget_warning"):
            out.append(semantic_event("escalate", terminal="escalate", raw=e))
    return out

def milestones_reached(sem_trace):
    m = set()
    for s in sem_trace:
        m.update(s.get("milestones_added") or [])
    return m

# ----------------------------------------------------------------------------- EvidenceView
def evidence_view(trace, plugin):
    """EvidenceUnit = {id, delivered_to_agent, delivery_fidelity, error_visible, acknowledged?}. The plugin
    extracts units; default = each successful tool output is one delivered unit."""
    ext = (plugin or {}).get("evidence_extractor")
    if ext: return ext(trace)
    units = []
    for i, e in enumerate(trace):
        if e.get("event_type") != "tool_call": continue
        ok = not _errored(e)
        out = (e.get("result") or {}).get("output") if isinstance(e.get("result"), dict) else e.get("result")
        units.append({"id": "%s#%d" % (e.get("tool"), i), "delivered_to_agent": ok,
                      "delivery_fidelity": 1.0 if ok else 0.0, "error_visible": (not ok),
                      "payload": str(out)[:300]})
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
    required_milestones / required_context_units / lifecycle_policy / governance_policy_id from HERE."""
    plugin = plugin or get_plugin(task.get("source_benchmark")) or {}
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
    base.update(task.get("lifecycle_policy") and {"lifecycle_policy": task["lifecycle_policy"]} or {})
    base.setdefault("governance_policy_id", task.get("source_benchmark"))
    return base

# ----------------------------------------------------------------------------- plugin registrations
def _resolve_region(event, prev_state):
    """RegionAttributeDescription: a successful HTTP call that FELL BACK to the whole image (resolved=False)
    did NOT examine the targeted region -> it must NOT be credited target_region_examined. Reads the real
    localization status from result.output.localization."""
    out = (event.get("result") or {}).get("output") if isinstance(event.get("result"), dict) else None
    loc = out.get("localization") if isinstance(out, dict) else None
    if _errored(event):
        return {"role": "acquire", "status": "failure", "milestones_added": [], "state_changed": False}
    if isinstance(loc, dict) and loc.get("resolved") is True:
        return {"role": "acquire", "status": "success", "state_changed": True,
                "milestones_added": ["target_region_examined", "relevant_image_evidence_obtained"],
                "progress_token": str(loc.get("requested") or "region")}
    # fell back to the full image: general image evidence only, the targeted region was NOT examined
    return {"role": "acquire", "status": "partial", "state_changed": False,
            "milestones_added": ["image_overview_obtained"], "progress_token": None}


def _medcta_evidence(trace):
    """measurements + finding terms actually delivered to the agent (reuses Observability evidence units)."""
    units = []
    for i, e in enumerate(trace):
        if e.get("event_type") != "tool_call": continue
        ok = not _errored(e)
        out = (e.get("result") or {}).get("output") if isinstance(e.get("result"), dict) else e.get("result")
        txt = out.get("text") if isinstance(out, dict) else str(out or "")
        loc = out.get("localization") if isinstance(out, dict) else None
        units.append({"id": "%s#%d" % (e.get("tool"), i), "delivered_to_agent": ok,
                      "delivery_fidelity": (1.0 if (loc or {}).get("resolved", True) else 0.5) if ok else 0.0,
                      "error_visible": (not ok), "payload": str(txt)[:300]})
    return units

register_plugin({
    "benchmark": "MedCTA", "default_tool_role": "acquire",
    "tool_semantics": {
        "ImageDescription": {"role": "acquire", "success_milestones": ["image_overview_obtained", "relevant_image_evidence_obtained"]},
        "RegionAttributeDescription": {"role": "acquire", "success_milestones": ["target_region_examined", "relevant_image_evidence_obtained"]},
        "OCR": {"role": "acquire", "success_milestones": ["text_evidence_obtained"]},
        "GoogleSearch": {"role": "acquire", "success_milestones": ["external_reference_obtained"]},
        "Calculator": {"role": "act", "success_milestones": []}},
    "evidence_extractor": _medcta_evidence,
    "resolvers": {"RegionAttributeDescription": _resolve_region},
    "dimension_policy": {"required_milestones": ["relevant_image_evidence_obtained"],
                         "required_context_units": ["target_image_evidence"],
                         "governance_policy_id": "MedCTA"}})
register_plugin({
    "benchmark": "PhysicianBench", "default_tool_role": "act",
    "tool_semantics": {
        "fhir_search": {"role": "acquire", "success_milestones": ["patient_record_loaded"]},
        "fhir_read": {"role": "acquire", "success_milestones": ["record_detail_loaded"]},
        "fhir_create": {"role": "commit", "success_milestones": ["resource_created"]}},
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
    "dimension_policy": {"required_milestones": ["form_submitted"],
                         "required_context_units": ["correct_case", "current_form_state", "submission_requirements"],
                         "governance_policy_id": "HealthAdminBench"}})
