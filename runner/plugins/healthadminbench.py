#!/usr/bin/env python3
"""HealthAdminBench (healthcare_admin, browser/GUI) BenchmarkPlugin.

ALL HealthAdminBench-specific knowledge lives here: the DOM tool->role/milestone semantics and the
RESULT-CONDITIONAL resolvers that read the RENDERED page surface (confirmation text / page-state diff) to
decide which milestone was truly earned. The substrate core names no DOM/page concept; this file does. A 4th
dataset drops a sibling module + a spec/registry.json entry; no core edit.

Consumes ONLY substrate's shared helpers (_errored, _result_output, _hash8, _rendered_text); the
plugin->substrate import is the single, one-directional dependency."""
import substrate as _S

_errored = _S._errored
_result_output = _S._result_output
_hash8 = _S._hash8
_rendered_text = _S._rendered_text


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


# ----------------------------------------------------------------------------- registration
PLUGIN = {
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
                         "governance_policy_id": "HealthAdminBench"}}

_S.register_plugin(PLUGIN)
