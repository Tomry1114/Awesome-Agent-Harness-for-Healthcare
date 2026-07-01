"""Effect completion (Phase 4b) -- the AUTHORIZED half of INCOMPLETE_EFFECT -> COMPLETE.

Given a committed-but-unrealized order (from effect_reconciliation), REALIZE exactly what the AGENT decided:
build the state mutation from the AGENT'S order text and place it, under a scoped deterministic_gap
authorization, after ensuring the policy's required read-evidence is gathered (read-only). The harness never
chooses a clinical action -- modality/order come verbatim-ish from the agent's deliverable.

Integrity guardrails:
  - decision provenance = agent (order text from the deliverable, never a checkpoint).
  - governance-first: `missing_required_evidence` finds policy-required reads the agent skipped; the caller
    ACQUIREs them READ-ONLY before any write, so a completed order is never "without required evidence".
  - authorization: every completion is minted as a single-use `deterministic_gap` MutationAuthorization.
  - verification: the caller reconciles (server read-back) that the resource persisted.
  - fail-safe: any missing input / unparseable ref -> no mutation.

Substrate note: the FHIR resource shapes below are the hapi_fhir ADAPTER's concern; kept here with a clear
mapping until the adapter manifest carries them. No benchmark names, no checkpoint knowledge.
"""
import re

# category -> (resourceType, date_field) for the created order. imaging/procedure/lab/referral -> ServiceRequest.
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
    subject cannot be resolved (=> no completion). PhysicianBench: FHIR Patient.id == MRN == context.patient_ref."""
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


def searched_resource_types(trajectory):
    """resourceTypes already read in the trajectory (harness event schema: event_type=tool_call, args.resourceType)."""
    seen = set()
    for e in (trajectory or []):
        if e.get("event_type") == "tool_call" and e.get("tool") in ("fhir_search", "fhir_read"):
            rt = ((e.get("args") or {}).get("resourceType"))
            if rt:
                seen.add(rt)
    return seen


def missing_required_evidence(policy, trajectory):
    """Policy-required read resourceTypes the agent did NOT gather -> the harness must ACQUIRE these read-only
    before completing any order (else the completion would be 'without required evidence')."""
    have = searched_resource_types(trajectory)
    return [rt for rt in required_evidence_resource_types(policy) if rt not in have]


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


def existing_order_texts(env, resource_type, subject_ref):
    """code.text of orders already present in state for the subject (so we never double-create). Read-only.
    Best-effort: any error -> [] (treated as 'none present' by the conservative reconciler upstream, which is
    fine because a duplicate is prevented by is_realized keyword match on whatever WAS found)."""
    try:
        out = env.call_tool("fhir_search", {"resourceType": resource_type, "subject": subject_ref})
    except Exception:
        return []
    texts = []
    for item in (out or {}).get("entries", []) if isinstance(out, dict) else []:
        r = item.get("resource", item) if isinstance(item, dict) else {}
        code = (r or {}).get("code") or {}
        if code.get("text"):
            texts.append(code["text"])
        for c in (code.get("coding") or []):
            if c.get("display"):
                texts.append(c["display"])
    return texts
