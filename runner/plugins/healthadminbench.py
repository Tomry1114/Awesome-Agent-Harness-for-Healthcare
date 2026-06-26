#!/usr/bin/env python3
"""HealthAdminBench (healthcare_admin, browser/GUI) BenchmarkPlugin.

ALL HealthAdminBench-specific knowledge lives here: the DOM tool->role/milestone semantics and the
RESULT-CONDITIONAL resolvers that read the RENDERED page surface (confirmation text / page-state diff) to
decide which milestone was truly earned. The substrate core names no DOM/page concept; this file does. A 4th
dataset drops a sibling module + a spec/registry.json entry; no core edit.

Consumes ONLY substrate's shared helpers (_errored, _result_output, _hash8, _rendered_text); the
plugin->substrate import is the single, one-directional dependency.

TYPED CONTEXT + SOURCE PROVENANCE (shared cross-benchmark contract):
  * dimension_policy.required_context_units are TYPED {id, type} over the HAB vocabulary (V2/CONTRACT-F)
    case_identity / pre_submit_form_state / submission_requirements. submission_confirmation is a DISTINCT
    post-submit OUTCOME type that is NOT a required unit (it cannot back-fill submission_requirements).
  * every EvidenceUnit is tagged with context_type (V2: submission_confirmation on a POST-submit OUTCOME
    page, submission_requirements on a PRE-submit rules/fields page, case_identity on a specific case/denial
    route, else pre_submit_form_state for a generic pre-submit portal page), source_channel 'gui_portal',
    source_instance_id the case-scoped URL/case-id (so repeated reads of the SAME case route are ONE
    source), extractor 'browser', plus CONTRACT-A semantic_status + usable_for_context."""
import re as _re
import substrate as _S

_errored = _S._errored
_result_output = _S._result_output
_hash8 = _S._hash8
_rendered_text = _S._rendered_text


# --- CONTRACT-A: usable_for_context (shared cross-benchmark) ---
def _usable_fields(sem_event):
    """CONTRACT-A: derive (semantic_status, usable_for_context) for an EvidenceUnit from its PRODUCING
    SemanticEvent (the resolver's resolved status, NOT the tool name). usable_for_context = (status=='
    success' AND a non-empty progress_token). A click/type that left the page UNCHANGED (no new page token)
    or a submit with NO confirmation surface is delivered+partial with NO token -> usable_for_context=False:
    it still feeds Observability/delivery but is NOT acquired GUI context."""
    st = (sem_event or {}).get("status")
    pt = (sem_event or {}).get("progress_token")
    return st, bool(st == "success" and pt)


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


# V2 / CONTRACT-F: a SUBMISSION CONFIRMATION is a POST-submit OUTCOME surface. The cue must denote the
# OUTCOME of a submit, NOT any page that merely contains the word "submitted"/"received"/"success" -- those
# bare tokens appear inside a DENIAL-REASON detail page (e.g. "Claim submitted to incorrect payer",
# remittance error text), which is a PRE-submit case-detail page, not a confirmation. So we require an
# OUTCOME phrase (the submit succeeded), not a bare substring, to avoid mis-typing case detail as a
# confirmation. These phrases ALSO type a page as submission_confirmation in _provenance.
_CONFIRM_OUTCOME = (
    "has been submitted", "appeal submitted", "appeal has been", "successfully submitted",
    "submission complete", "submission successful", "submission received", "your appeal has",
    "appeal was submitted", "thank you for your submission", "appeal received",
    "request has been submitted", "request submitted", "form submitted", "successfully saved",
    "saved successfully", "confirmation number", "submission confirmed", "appeal confirmed")

# V2 / CONTRACT-F: a SUBMISSION-REQUIREMENTS surface is a PRE-submit page describing the rules/fields the
# agent must satisfy BEFORE submitting (required fields, deadlines, what to attach). It is NOT a confirmation
# and NOT a bare case-detail page. These cues are matched ONLY when the page is NOT a confirmation.
_REQUIREMENT_CUES = (
    "required field", "is required", "must include", "must provide", "must attach", "must submit by",
    "appeal deadline", "deadline", "submission requirement", "supporting document", "supporting documentation",
    "in order to appeal", "to file an appeal", "to submit an appeal", "reason for appeal",
    "appeal form", "required to", "please provide", "please attach", "fields marked", "mandatory")


def _is_confirmation(page):
    """True iff the rendered page is a POST-submit CONFIRMATION outcome (an OUTCOME phrase, not a bare
    'submitted' token that also occurs in denial-reason text)."""
    low = str(page or "").lower()
    return any(k in low for k in _CONFIRM_OUTCOME)


def _has_requirements(page):
    """True iff the rendered PRE-submit page describes submission requirements/rules/fields."""
    low = str(page or "").lower()
    return any(k in low for k in _REQUIREMENT_CUES)


def _resolve_submit(event, prev_state):
    """HAB submit: accepted but NO confirmation OUTCOME in the RENDERED observation -> partial, no
    form_submitted (a button press whose page shows no completion surface -- e.g. it still shows the
    pre-submit case detail with a denial-reason 'submitted' mention -- is NOT a completed submission). A
    rendered confirmation OUTCOME -> success with a state:submitted token keyed by the confirming page."""
    if _errored(event):
        return {"role": "commit", "status": "failure", "milestones_added": [],
                "obligation_id": "form_submitted", "state_changed": False, "progress_token": None}
    url, page = _hab_page(event)
    if _is_confirmation(page):
        return {"role": "commit", "status": "success", "milestones_added": ["form_submitted"],
                "obligation_id": "form_submitted",
                "progress_token": "state:submitted=%s" % _hash8((url or "") + "|" + page)}
    # REAL PORTAL: the NextJS SPA often re-renders the case page IN PLACE after a submit (no textual
    # confirmation surface), but the authoritative localStorage state diff (state_record.state_changed --
    # the same signal substrate.map_trace trusts) proves the submission COMMITTED (disposition/appeal
    # recorded). Honor it as a completed submission so a real submit is not mis-scored partial.
    _sr = event.get("state_record") or {}
    if _sr.get("state_changed"):
        return {"role": "commit", "status": "success", "milestones_added": ["form_submitted"],
                "obligation_id": "form_submitted", "state_changed": True,
                "progress_token": "state:submitted=%s" % _hash8(str(_sr.get("state_after_hash") or url or "x"))}
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
    """(context_type, source_channel, source_instance_id, extractor) for a HAB action. V2 / CONTRACT-F
    typing over the HAB vocabulary:
      * submission_confirmation -- a POST-submit OUTCOME page (a real confirmation phrase). This is a
        DISTINCT type that does NOT satisfy a Context required_unit (a confirmation cannot back-fill the
        PRE-submit submission_requirements unit -- CONTRACT-D/F).
      * submission_requirements  -- a PRE-submit page describing the rules/fields to satisfy BEFORE
        submitting (required fields / deadlines / what to attach).
      * case_identity            -- a specific case/denial route page.
      * pre_submit_form_state    -- a generic PRE-submit portal/form page (the working state).
    source_instance_id is the case-scoped url (case-id route stripped of query) so repeated reads of the
    SAME case are ONE source; a non-case page keys on its url path. Errors / blank pages carry no
    context_type. CONFIRMATION is tested FIRST so a confirmation on a case route is typed
    submission_confirmation, not case_identity."""
    url, page = _hab_page(event)
    if page_text is not None:
        page = page_text
    if _errored(event) or not str(page or "").strip():
        return None, _CH_GUI, (str(url).split("?")[0] if url else "gui:page"), "browser"
    cid = _case_id(url)
    # PRECEDENCE (V2/CONTRACT-F): confirmation (post-submit OUTCOME) FIRST; then a case/denial route IS the
    # case identity (a case-detail page that merely MENTIONS a deadline/required field is still the case
    # page, NOT a dedicated requirements page -- so case_identity wins over the soft requirement cues on a
    # case route); a dedicated PRE-submit requirements page applies only OFF a case route; else generic
    # pre-submit form state.
    if _is_confirmation(page):
        ctype = "submission_confirmation"         # POST-submit OUTCOME -> NOT a required Context unit
    elif cid:
        ctype = "case_identity"
    elif _has_requirements(page):
        ctype = "submission_requirements"         # PRE-submit rules/fields on a non-case page
    else:
        ctype = "pre_submit_form_state"           # generic PRE-submit working page
    if cid:
        instance = "case:%s" % cid            # all reads of the SAME case route share ONE source instance
    else:
        instance = (str(url).split("?")[0] if url else "gui:page")
    return ctype, _CH_GUI, instance, "browser"


# ----------------------------------------------------------------------------- evidence extractor
def _hab_evidence(trace):
    """HealthAdminBench EvidenceView: real delivery refined with GUI source provenance. Each unit carries
    context_type (V2/CONTRACT-F: case_identity / pre_submit_form_state / submission_requirements /
    submission_confirmation), source_channel/source_instance_id/extractor, the resolver's semantic
    progress_token, and (CONTRACT-A) semantic_status + usable_for_context. Repeated reads of the SAME case
    route share one source_instance_id (one GUI source)."""
    sem = _S.map_trace(trace, PLUGIN)
    sem_by_step = {id(s.get("raw") or {}): s for s in sem}
    units = []
    for i, e in enumerate(trace):
        if e.get("event_type") != "tool_call": continue
        d = _S._real_delivery(e)
        url, page = _hab_page(e)
        ctype, channel, instance, extractor = _provenance(e, page_text=page)
        sm = sem_by_step.get(id(e)) or {}
        sem_status, usable = _usable_fields(sm)
        units.append({"id": "%s#%d" % (e.get("tool"), i), "delivered_to_agent": d["delivered"],
                      "delivery_fidelity": d["fidelity"], "error_visible": d["error_visible"],
                      "payload": str(page)[:300],
                      "context_type": ctype, "source_channel": channel,
                      "source_instance_id": instance, "extractor": extractor,
                      "progress_token": sm.get("progress_token"),
                      "semantic_status": sem_status, "usable_for_context": usable})
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
                         # V2 / CONTRACT-F vocabulary. submission_requirements is a PRE-submit unit; a
                         # POST-submit submission_confirmation is a DIFFERENT context_type that is NOT a
                         # required unit and therefore CANNOT back-fill it.
                         "required_context_units": [
                             {"id": "case_identity", "type": "case_identity"},
                             {"id": "pre_submit_form_state", "type": "pre_submit_form_state"},
                             {"id": "submission_requirements", "type": "submission_requirements"}],
                         # V1 / CONTRACT-B DEFAULT verification_policy. A single authoritative GUI page (one
                         # case route, one form, one confirmation) is a direct single-source fact and is NOT
                         # forced to two sources. Only a high-risk recommendation, an external medical fact,
                         # or a conflicting-evidence claim warrants cross-source corroboration.
                         "verification_policy": {
                             "cross_source_required_for": [{"type": "high_risk_recommendation", "patterns": ["dose", "dosage", "increase", "decrease", "initiate", "discontinue", "anticoagul", "administer", "mg ", "prescrib", "titrate", "start ", "stop ", "hold "]}, {"type": "external_medical_fact", "patterns": ["guideline", "studies show", "literature", "typically causes", "is known to", "according to", "per uptodate", "recommended per", "class effect"]}, {"type": "conflicting_evidence", "patterns": ["conflict", "inconsistent", "discrepan", "contradict", "disagree", "does not match", "mismatch"]}]},
                         "governance_policy_id": "HealthAdminBench"}}

_S.register_plugin(PLUGIN)
