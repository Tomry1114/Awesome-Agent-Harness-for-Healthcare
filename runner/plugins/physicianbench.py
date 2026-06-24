#!/usr/bin/env python3
"""PhysicianBench (clinical_data_ops, FHIR) BenchmarkPlugin.

ALL PhysicianBench-specific knowledge lives here: the FHIR tool->role/milestone semantics and the
RESULT-CONDITIONAL resolvers that read the actual FHIR body (OperationOutcome error / Bundle total /
server-assigned id) to decide which milestone was truly earned. The substrate core names no FHIR concept;
this file does. A 4th dataset drops a sibling module + a spec/registry.json entry; no core edit.

Consumes ONLY substrate's shared helpers (_errored, _result_output, _hash8); the plugin->substrate import
is the single, one-directional dependency."""
import substrate as _S

_errored = _S._errored
_result_output = _S._result_output
_hash8 = _S._hash8


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


# ----------------------------------------------------------------------------- registration
PLUGIN = {
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
                         "governance_policy_id": "PhysicianBench"}}

_S.register_plugin(PLUGIN)
