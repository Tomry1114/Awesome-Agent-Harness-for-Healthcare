#!/usr/bin/env python3
"""HealthAdminBench (healthcare_admin, browser/GUI) BenchmarkPlugin.

ALL HealthAdminBench-specific knowledge lives here: the DOM tool->role/milestone semantics and the
RESULT-CONDITIONAL resolvers that read the RENDERED page surface (confirmation text / page-state diff) to
decide which milestone was truly earned. The substrate core names no DOM/page concept; this file does. A 4th
dataset drops a sibling module + a spec/registry.json entry; no core edit.

Consumes ONLY substrate's shared helpers (_errored, _result_output, _hash8, _rendered_text); the
plugin->substrate import is the single, one-directional dependency.

TYPED CONTEXT + SOURCE PROVENANCE (shared cross-benchmark contract):
  * dimension_policy.required_context_units are TYPED {id, type} over the HAB vocabulary
    case_identity / form_state / submission_requirements.
  * every EvidenceUnit is tagged with context_type (case_identity when the rendered page is a specific
    case/denial route, submission_requirements on a submit-confirmation page, else form_state for a generic
    portal page), source_channel 'gui_portal', source_instance_id the case-scoped URL/case-id (so repeated
    reads of the SAME case route are ONE source), and extractor 'browser'."""
import re as _re
import substrate as _S

_errored = _S._errored
_result_output = _S._result_output
_hash8 = _S._hash8
_rendered_text = _S._rendered_text

_CH_GUI = "gui_portal"
# a case/denial route segment, e.g. .../denied/DEN-001 or .../case/CASE-9 -> the case identity
_CASE_RE = _re.compile(r"/(?:denied|denials|case|cases|appeal|appeals|claim|claims)/([A-Za-z0-9_\-]+)")


# ============================================================================= RESOLVERS
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


# ----------------------------------------------------------------------------- SUBJECT identity + provenance
def _case_id(url):
    """The case/denial id segment of a portal URL (DEN-001), or None for a generic portal page."""
    m = _CASE_RE.search(str(url or ""))
    return m.group(1) if m else None


def _provenance(event, page_text=None):
    """(context_type, source_channel, source_instance_id, extractor) for a HAB action. context_type:
    case_identity when the page is a specific case/denial route, submission_requirements when the rendered
    surface is a submit confirmation, else form_state for a generic portal page. source_instance_id is the
    case-scoped url (case-id route stripped of query) so repeated reads of the SAME case are ONE source; a
    non-case page keys on its url path. Errors / blank pages carry no context_type."""
    url, page = _hab_page(event)
    if page_text is not None:
        page = page_text
    if _errored(event) or not str(page or "").strip():
        return None, _CH_GUI, (str(url).split("?")[0] if url else "gui:page"), "browser"
    cid = _case_id(url)
    low = str(page).lower()
    if any(k in low for k in _SUBMIT_CONFIRM):
        ctype = "submission_requirements"
    elif cid:
        ctype = "case_identity"
    else:
        ctype = "form_state"
    if cid:
        instance = "case:%s" % cid            # all reads of the SAME case route share ONE source instance
    else:
        instance = (str(url).split("?")[0] if url else "gui:page")
    return ctype, _CH_GUI, instance, "browser"


# ----------------------------------------------------------------------------- evidence extractor
def _hab_evidence(trace):
    """HealthAdminBench EvidenceView: real delivery refined with GUI source provenance. Each unit carries
    context_type (case_identity / form_state / submission_requirements), source_channel/source_instance_id/
    extractor, and the resolver's semantic progress_token. Repeated reads of the SAME case route share one
    source_instance_id (one GUI source)."""
    sem = _S.map_trace(trace, PLUGIN)
    sem_by_step = {id(s.get("raw") or {}): s for s in sem}
    units = []
    for i, e in enumerate(trace):
        if e.get("event_type") != "tool_call": continue
        d = _S._real_delivery(e)
        url, page = _hab_page(e)
        ctype, channel, instance, extractor = _provenance(e, page_text=page)
        sm = sem_by_step.get(id(e)) or {}
        units.append({"id": "%s#%d" % (e.get("tool"), i), "delivered_to_agent": d["delivered"],
                      "delivery_fidelity": d["fidelity"], "error_visible": d["error_visible"],
                      "payload": str(page)[:300],
                      "context_type": ctype, "source_channel": channel,
                      "source_instance_id": instance, "extractor": extractor,
                      "progress_token": sm.get("progress_token")})
    return units


# ----------------------------------------------------------------------------- registration
PLUGIN = {
    "benchmark": "HealthAdminBench", "default_tool_role": "act",
    "tool_semantics": {
        "snapshot": {"role": "verify", "success_milestones": ["page_state_observed"]},
        "navigate": {"role": "acquire", "success_milestones": ["target_page_reached"]},
        "click": {"role": "act", "success_milestones": []},
        "type": {"role": "act", "success_milestones": []},
        "submit": {"role": "commit", "success_milestones": ["form_submitted"]}},
    "evidence_extractor": _hab_evidence,
    "resolvers": {"submit": _resolve_submit,
                  "click": _resolve_dom_action,
                  "type": _resolve_dom_action},
    "dimension_policy": {"required_milestones": ["form_submitted"],
                         "required_context_units": [
                             {"id": "case_identity", "type": "case_identity"},
                             {"id": "form_state", "type": "form_state"},
                             {"id": "submission_requirements", "type": "submission_requirements"}],
                         "governance_policy_id": "HealthAdminBench"}}

_S.register_plugin(PLUGIN)
