"""Effect completion (Phase 4b) -- the AUTHORIZED half of INCOMPLETE_EFFECT -> COMPLETE.

Given a committed-but-unrealized order (from effect_reconciliation over ROOT AGENT content only), REALIZE
exactly what the AGENT decided: build the state mutation from the AGENT'S order text and place it, under a
scoped deterministic_gap authorization that the OperationalGuard actually validates, after ensuring the
policy's required read-evidence is RESOLVED (read-only). The harness never chooses a clinical action.

Integrity guardrails (all fail-CLOSED):
  - provenance: the caller extracts committed orders from ROOT agent content, never harness-added spans, so a
    COMPLETE-text patch can never be mistaken for an agent decision.
  - governance-first + EvidenceState: `resolve_read_evidence` classifies each policy-required read as
    PRESENT/ABSENT/UNKNOWN/FAILED; only PRESENT|ABSENT (obligation actually checked, subject-bound) permits a
    mutation -- UNKNOWN/FAILED BLOCK it.
  - existing-effect fail-closed: `inspect_existing_effect` returns PRESENT/ABSENT/UNKNOWN; a failed probe is
    UNKNOWN (NOT ABSENT), so a probe error can NEVER trigger a spurious create.
  - authorization is an EXECUTION boundary: the caller routes the fhir_create through the normal
    before_action/after_action pipeline; verify_commit must find the exact-scope authorization or the mutation
    is refused.
  - verification: server read-back confirms persistence.

Substrate note: the FHIR shapes below are the hapi_fhir ADAPTER's concern (an EffectAdapter would own
inspect/compile/verify). Kept here with a clear mapping until the adapter manifest carries them. No benchmark
names, no checkpoint knowledge.
"""
import re
from .evidence_state import classify_evidence_state, PRESENT, ABSENT, UNKNOWN, FAILED, is_resolved

# adapter result-semantics for a FHIR search Bundle-as-flat-list
_FHIR_SEM = {"collection_paths": ["entries"], "absence_when_empty": True}

# category -> (resourceType, date_field). imaging/procedure/lab/referral -> ServiceRequest.
_CATEGORY_RESOURCE = {
    "imaging": ("ServiceRequest", "authoredOn"),
    "procedure": ("ServiceRequest", "authoredOn"),
    "lab": ("ServiceRequest", "authoredOn"),
    "referral": ("ServiceRequest", "authoredOn"),
    "medication": ("MedicationRequest", "authoredOn"),
    "other": ("ServiceRequest", "authoredOn"),
}


def resource_type_for_category(category):
    """The FHIR resourceType a committed order of this category is realized as (adapter mapping)."""
    return _CATEGORY_RESOURCE.get(str(category or "other"), _CATEGORY_RESOURCE["other"])[0]


def context_refs(task):
    """Resolve subject/practitioner/authoredOn from the PUBLIC task context only. Returns dict or {} if the
    subject cannot be resolved (=> no completion). this record substrate: the FHIR Patient.id equals the MRN equals context.patient_ref."""
    ctx = (task or {}).get("context") or {}
    ref = ctx.get("patient_ref")
    if not ref:
        return {}
    text = str(ctx.get("text") or "")
    m_when = re.search(r"current date and time is\s*([0-9][0-9T:+\-]{9,40})", text)
    m_prac = re.search(r"Practitioner ID:\s*([A-Za-z0-9._-]+)", text)
    return {
        "subject": "Patient/%s" % ref,
        "authoredOn": (m_when.group(1) if m_when else None),
        "requester": ("Practitioner/%s" % m_prac.group(1)) if m_prac else None,
    }


def required_evidence_resource_types(policy):
    """resourceTypes the policy requires be READ before an action (from required_tool_before_action like
    'fhir_search(AllergyIntolerance)'). Public policy only. Returns [str]."""
    out = []
    for item in ((policy or {}).get("required_tool_before_action") or []):
        m = re.search(r"\(([A-Za-z]+)\)", str(item))
        if m:
            out.append(m.group(1))
    return out


def resolve_read_evidence(env, resource_type, subject_ref):
    """Subject-bound governance read. Returns {state, resource, subject_bound}. state PRESENT/ABSENT = the
    obligation was actually CHECKED for this subject (resolved); UNKNOWN/FAILED = caller MUST BLOCK the
    downstream mutation. Read-only. The query is subject-bound by construction (subject=subject_ref)."""
    try:
        out = env.call_tool("fhir_search", {"resourceType": resource_type, "subject": subject_ref})
    except Exception as ex:
        return {"state": FAILED, "resource": {"error": repr(ex)}, "subject_bound": True}
    return {"state": classify_evidence_state(out, _FHIR_SEM), "resource": out, "subject_bound": True}


def classify_effect_inspection(out):
    """PURE existing-effect classifier over a raw search result (no I/O). Returns {state, texts, matched_ids}.
    Kept separate from the env call so the probe can be driven through the unified ActionExecutor (a recovery
    INSPECT_EFFECT read) instead of a private env.call_tool -- see RunDriver.inspect_effect."""
    st = classify_evidence_state(out, _FHIR_SEM)
    if not is_resolved(st):                       # FAILED / UNKNOWN -> unknown (never ABSENT)
        return {"state": UNKNOWN, "texts": [], "matched_ids": []}
    texts, ids = [], []
    if isinstance(out, dict):
        for item in (out.get("entries") or []):
            r = item.get("resource", item) if isinstance(item, dict) else {}
            if isinstance(r, dict) and r.get("id"):
                ids.append(r["id"])
            code = (r or {}).get("code") or {} if isinstance(r, dict) else {}
            if code.get("text"):
                texts.append(code["text"])
            for c in (code.get("coding") or []):
                if c.get("display"):
                    texts.append(c["display"])
    # #7 FAIL-CLOSED: PRESENT but no COMPARABLE representation (resources exist yet none expose a code.text /
    # coding.display we can match on) -> we cannot tell realized-vs-not -> UNKNOWN, so the caller does NOT create.
    if st == PRESENT and not texts:
        return {"state": UNKNOWN, "texts": [], "matched_ids": ids, "reason": "present_no_comparable_representation"}
    return {"state": st, "texts": texts, "matched_ids": ids}


def inspect_existing_effect(env, resource_type, subject_ref):
    """Fail-CLOSED existing-effect probe (legacy direct path; prefer RunDriver.inspect_effect which routes the
    same read through the executor). Returns {state, texts, matched_ids}."""
    try:
        out = env.call_tool("fhir_search", {"resourceType": resource_type, "subject": subject_ref})
    except Exception:
        return {"state": UNKNOWN, "texts": [], "matched_ids": []}
    return classify_effect_inspection(out)


def build_order_resource(order, refs):
    """Construct the FHIR order resource from the AGENT'S order text + public refs. Returns (resourceType, dict)
    or (None, None) if unresolvable. status=active, intent=order (what the eval requires); code.text = the
    agent's order phrase (the clinical decision the AGENT made)."""
    if not refs or not refs.get("subject"):
        return (None, None)
    text = str((order or {}).get("text") or "").strip()
    if not text:
        return (None, None)
    rt, date_field = _CATEGORY_RESOURCE.get(str((order or {}).get("category") or "other"), _CATEGORY_RESOURCE["other"])
    res = {"resourceType": rt, "status": "active", "intent": "order",
           "subject": {"reference": refs["subject"]}, "code": {"text": text[:180]}}
    if refs.get("authoredOn"):
        res[date_field] = refs["authoredOn"]
    if refs.get("requester"):
        res["requester"] = {"reference": refs["requester"]}
    return (rt, res)
