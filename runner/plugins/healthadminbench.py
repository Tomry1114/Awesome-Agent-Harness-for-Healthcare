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
    case_identity / pre_submit_form_state / submission_requirements. submission_confirmation is a DISTINCT
    post-submit OUTCOME type that is NOT a required unit (it cannot back-fill submission_requirements).
  * every EvidenceUnit is tagged with context_type (submission_confirmation on a POST-submit OUTCOME
    page, submission_requirements on a PRE-submit rules/fields page, case_identity on a specific case/denial
    route, else pre_submit_form_state for a generic pre-submit portal page), source_channel 'gui_portal',
    source_instance_id the case-scoped URL/case-id (so repeated reads of the SAME case route are ONE
    source), extractor 'browser', plus semantic_status + usable_for_context."""
import re as _re
import substrate as _S

_errored = _S._errored
_result_output = _S._result_output
_hash8 = _S._hash8
_rendered_text = _S._rendered_text


# --- usable_for_context (shared cross-benchmark) ---
def _usable_fields(sem_event):
    """Derive (semantic_status, usable_for_context) for an EvidenceUnit from its PRODUCING
    SemanticEvent (the resolver's resolved status, NOT the tool name). usable_for_context = (status=='
    success' AND a non-empty progress_token). A click/type that left the page UNCHANGED (no new page token)
    or a submit with NO confirmation surface is delivered+partial with NO token -> usable_for_context=False:
    it still feeds Clinical Auditability/delivery but is NOT acquired GUI context."""
    st = (sem_event or {}).get("status")
    pt = (sem_event or {}).get("progress_token")
    return st, bool(st == "success" and pt)


_CH_GUI = "gui_portal"
# a case/denial route segment, e.g. .../denied/DEN-001 or .../case/CASE-9 -> the case identity
_CASE_RE = _re.compile(r"/(?:denied|denials|case|cases|appeal|appeals|claim|claims|referral|referrals)/([A-Za-z0-9_\-]+)")


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


# A SUBMISSION CONFIRMATION is a POST-submit OUTCOME surface. The cue must denote the
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

# A SUBMISSION-REQUIREMENTS surface is a PRE-submit page describing the rules/fields the
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
    # recorded). It is honored as a completed submission so a real submit is not scored partial.
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


# ----------------------------------------------------------------------------- REAL portal vocabulary
# The navigate/type/submit tool_semantics above describe a mock GUI vocabulary. The REAL Playwright portal
# (GuiEnvReal) emits click/fill/scroll/back/download/done -- 'navigate' and 'submit' never occur there, so
# their milestones (target_page_reached / form_submitted) would be unearnable and
# Care Action Fulfillment.required_operation_completion would be structurally 0 on every task. The mock names stay
# registered (conformance fixtures use them); the real ones are registered alongside them.
#
# COMMIT cues are taken from the state-changing ref vocabulary the portal exposes across its tasks
# (save-note-button, submit-disposition-button, submit-auth-button, submit-appeal-button,
# eligibility-submit-button). Navigation-ish refs that also change state
# (start-appeal-button, search-appeals-button, tab-*, back-*, *-nav-link) are deliberately NOT commit cues,
# so merely moving around the portal can never earn form_submitted.
_COMMIT_REF_RE = _re.compile(r"(^|[-_])(submit|save)([-_]|$)", _re.I)


def _click_ref(event):
    """The DOM target of a click/fill, over the arg-key vocabulary the real portal uses."""
    a = event.get("args")
    if isinstance(a, dict):
        for k in _HAB_TARGET_KEYS:
            v = a.get(k)
            if isinstance(v, str) and v:
                return v
    return ""


def _resolve_click(event, prev_state):
    """REAL portal: a submission is a CLICK on a submit/save control, not a distinct tool. Route
    those to the commit resolver -- which still demands REAL evidence (a rendered confirmation OUTCOME or an
    authoritative localStorage state diff), so a click that committed nothing stays partial and earns no
    milestone. Every other click is a DOM action; one that lands on a case-scoped route ALSO earns
    target_page_reached, which is exactly the contract's 'reach the case/form page before submitting'."""
    if _COMMIT_REF_RE.search(_click_ref(event)):
        return _resolve_submit(event, prev_state)
    out = _resolve_dom_action(event, prev_state)
    url, _page = _hab_page(event)
    cid = _case_id(url)
    if out.get("status") == "success" and cid:
        ms = list(out.get("milestones_added") or [])
        if "target_page_reached" not in ms:
            ms = ms + ["target_page_reached"]
        out = dict(out)
        out["milestones_added"] = ms
    # A read-back of a case the agent has ALREADY COMMITTED ON is a self-check, so it is scored as
    # role=verify. This mirrors the core's own harness-actor rule ("a recovery server read-back IS a
    # verification", substrate.map_trace). Without it Closed-Loop Verification's verification_action_completion would be
    # structurally 0 on every HAB run: the only other verify-role tool the plugin declares is `snapshot`,
    # which the real GuiEnvReal toolset never emits.
    #
    # DELIBERATELY NARROW -- keyed on the SPECIFIC case the agent submitted, not on "anything after a
    # commit". Many HAB tasks require two or more submissions, so crediting every post-commit action as a
    # self-check would reward simply moving on to the next sub-task. Only `click` is routed here (back and
    # scroll keep their existing roles).
    # The core still applies its own genuine-check filter afterwards: a no-progress repeat earns nothing,
    # so a thrashing agent cannot buy verification credit from this.
    if cid:
        for _c in (prev_state.get("commits") or []):
            if _case_id(_hab_page(_c)[0]) == cid:
                out = dict(out)
                out["role"] = "verify"
                break
    return out

# ----------------------------------------------------------------------------- SUBJECT identity + provenance
def _case_id(url):
    """The case/denial id segment of a portal URL (DEN-001), or None for a generic portal page."""
    m = _CASE_RE.search(str(url or ""))
    return m.group(1) if m else None


def _provenance(event, page_text=None):
    """(context_type, source_channel, source_instance_id, extractor) for a HAB action. Patient–Evidence Alignment
    typing over the HAB vocabulary:
      * submission_confirmation -- a POST-submit OUTCOME page (a real confirmation phrase). This is a
        DISTINCT type that does NOT satisfy a Patient–Evidence Alignment required_unit (a confirmation cannot back-fill the
        PRE-submit submission_requirements unit).
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
    # PRECEDENCE: confirmation (post-submit OUTCOME) FIRST; then a case/denial route IS the
    # case identity (a case-detail page that merely MENTIONS a deadline/required field is still the case
    # page, NOT a dedicated requirements page -- so case_identity wins over the soft requirement cues on a
    # case route); a dedicated PRE-submit requirements page applies only OFF a case route; else generic
    # pre-submit form state.
    if _is_confirmation(page):
        ctype = "submission_confirmation"         # POST-submit OUTCOME -> NOT a required Patient–Evidence Alignment unit
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


def _hab_expected_subject(task):
    """Expected subject for a HAB task = the denial / referral case the task is about.

    Read from the STRUCTURED upstream config (environment.config.native_config), never parsed out of the
    instruction prose: denial_id covers the 60 denial tasks, patient_referral_id the 75 emr+fax tasks
    (60+75 = all 135). Returns None when neither key is present, so expected_subject_match stays
    not_applicable instead of binding to a guessed subject.

    The OBSERVED counterpart is _case_id(url) over the portal routes /emr/denied/DEN-xxx and
    /emr/referral/REF-xxx -- both are in _CASE_RE, so a declared subject is always observable in principle
    and a 0 means the agent really worked the wrong case.
    """
    nc = (((task or {}).get("environment") or {}).get("config") or {}).get("native_config") or {}
    # The TYPE must mirror the evidence layer's subject vocabulary: _provenance keys every GUI source
    # instance as "case:<id>", so the declared type is Case for both families (the id itself distinguishes
    # DEN-xxx from REF-xxx). Declaring Denial/Referral here would never match and would zero every task.
    for key, typ in (("denial_id", "Case"), ("patient_referral_id", "Case")):
        v = nc.get(key)
        if v:
            return {"type": typ, "id": str(v)}
    return None

# ----------------------------------------------------------------------------- evidence extractor
def _hab_evidence(trace):
    """HealthAdminBench EvidenceView: real delivery refined with GUI source provenance. Each unit carries
    context_type (case_identity / pre_submit_form_state / submission_requirements /
    submission_confirmation), source_channel/source_instance_id/extractor, the resolver's semantic
    progress_token, and semantic_status + usable_for_context. Repeated reads of the SAME case
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

# ============================================================ Canonical Evaluation Contract (HAB)
def _hab_task_contract(task):
    """Compile a HAB task into the Canonical Evaluation Contract. PROCESS-ONLY. Care Action Fulfillment: a HAB task that
    requires a submit implicitly requires REACHING the target page first, so the required group is the
    conjunction [target_page_reached, form_submitted]."""
    ref = task.get("reference") or {}
    rtg = ref.get("required_tool_groups") or []
    ts = PLUGIN.get("tool_semantics", {})
    req = []
    submit_required = False
    for grp in rtg:
        for tool in (grp or []):
            sem = ts.get(tool) or {}
            if "form_submitted" in (sem.get("success_milestones") or []):
                submit_required = True
            for ms in sem.get("success_milestones") or []:
                if ms not in req:
                    req.append(ms)
    if submit_required and "target_page_reached" not in req:
        req = ["target_page_reached"] + req      # reaching the case/form page precedes a valid submit
    execution = {"required_milestone_groups": [req]} if req else {}
    tooling = {"argument_targets": {"submit_required": submit_required}}
    return {"execution": execution, "tooling": tooling, "contract_version": "hab-1"}


_HAB_TARGET_KEYS = ("ref", "selector", "target", "element", "element_id", "id", "field", "name")
_HAB_VALUE_KEYS = ("text", "value", "content", "input")


def _hab_argument_semantic_score(call, task=None, policy=None, sem_event=None):
    """PROCESS-only argument semantics for a HAB GUI action: did type/select/upload/click/submit target
    a REAL element (well-formed field/DOM target) and, for value actions, carry a value? Judges the
    MECHANICAL correctness of the GUI op -- NEVER whether the entered clinical content is right (Outcome).
    (Deeper case/field CORRECTNESS is bounded by what HAB tasks expose at the process level.)"""
    raw = call if isinstance(call, dict) else {}
    tool = (raw.get("tool") or "")
    args = raw.get("args")
    if isinstance(args, str):
        import json as _j
        try: args = _j.loads(args)
        except Exception: args = {}
    args = args if isinstance(args, dict) else {}
    tool = tool.lower()
    if tool not in ("type", "click", "submit", "select", "upload"):
        return {"status": "not_applicable", "reason": "not_a_gui_action"}
    has_target = any(str(args.get(k) or "").strip() for k in _HAB_TARGET_KEYS)
    has_value = any(str(args.get(k) or "").strip() for k in _HAB_VALUE_KEYS)
    if tool in ("type", "select", "upload"):
        ok = has_target and has_value
        return {"status": "valid", "score": 1.0 if ok else 0.0, "axis": "field_target",
                "reason": None if ok else ("missing_target" if not has_target else "missing_value")}
    if tool == "click":
        return {"status": "valid", "score": 1.0 if has_target else 0.0, "axis": "click_target",
                "reason": None if has_target else "click_missing_target"}
    # submit: a well-formed submit must target a form/case element.
    return {"status": "valid", "score": 1.0 if has_target else 0.0, "axis": "submit_target",
            "reason": None if has_target else "submit_missing_target"}

PLUGIN = {
    "benchmark": "HealthAdminBench", "default_tool_role": "act",
    "tool_semantics": {
        "snapshot": {"role": "verify", "success_milestones": ["page_state_observed"]},
        "navigate": {"role": "acquire", "success_milestones": ["target_page_reached"]},
        "click": {"role": "act", "success_milestones": []},
        "type": {"role": "act", "success_milestones": []},
        "submit": {"role": "commit", "success_milestones": ["form_submitted"]},
        # REAL GuiEnvReal vocabulary (see _resolve_click): these are the tools the portal actually emits.
        "fill": {"role": "act", "success_milestones": []},
        "scroll": {"role": "acquire", "success_milestones": []},
        "back": {"role": "act", "success_milestones": []},
        "download": {"role": "acquire", "success_milestones": []}},
    "evidence_extractor": _hab_evidence,
    "expected_subject_resolver": _hab_expected_subject,
    "resolvers": {"submit": _resolve_submit,
                  "click": _resolve_click,
                  "type": _resolve_dom_action,
                  "fill": _resolve_dom_action},
    "dimension_policy": {"required_milestones": ["form_submitted"],
                         # submission_requirements is a PRE-submit unit; a
                         # POST-submit submission_confirmation is a DIFFERENT context_type that is NOT a
                         # required unit and therefore CANNOT back-fill it.
                         "required_context_units": [
                             {"id": "case_identity", "type": "case_identity"},
                             {"id": "pre_submit_form_state", "type": "pre_submit_form_state"},
                             {"id": "submission_requirements", "type": "submission_requirements"}],
                         # DEFAULT verification_policy. A single authoritative GUI page (one
                         # case route, one form, one confirmation) is a direct single-source fact and is NOT
                         # forced to two sources. Only a high-risk recommendation, an external medical fact,
                         # or a conflicting-evidence claim warrants cross-source corroboration.
                         "verification_policy": {
                             "cross_source_required_for": [{"type": "high_risk_recommendation", "patterns": ["dose", "dosage", "increase", "decrease", "initiate", "discontinue", "anticoagul", "administer", "mg ", "prescrib", "titrate", "start ", "stop ", "hold "]}, {"type": "external_medical_fact", "patterns": ["guideline", "studies show", "literature", "typically causes", "is known to", "according to", "per uptodate", "recommended per", "class effect"]}, {"type": "conflicting_evidence", "patterns": ["conflict", "inconsistent", "discrepan", "contradict", "disagree", "does not match", "mismatch"]}]},
                         "governance_policy_id": "HealthAdminBench"}}


PLUGIN["task_contract_resolver"] = _hab_task_contract
PLUGIN["argument_semantic_score"] = _hab_argument_semantic_score
PLUGIN["contract_version"] = "hab-1"
_S.register_plugin(PLUGIN)


# ============================================================ Clinical Auditability contract
# GUI telemetry families. Deliberately NOT one family per click: state acquisition, UI action, commit,
# navigation and document retrieval are the meaningful families, so 40 clicks and 10 clicks weigh the same.
def _hab_observability_family(sem_event):
    if sem_event.get("terminal") in ("final", "escalate"):
        return "terminal"
    cap = str(sem_event.get("capability_id") or "")
    role = sem_event.get("event_role")
    if cap in ("download", "view_doc_click_link"):
        return "document_retrieval"
    if cap in ("scroll", "snapshot"):
        return "state_acquisition"
    if cap in ("back", "navigate", "navigate_to"):
        return "navigation"
    if role == "commit":
        return "form_commit"
    if role == "verify":
        return "state_readback"
    if cap in ("click", "fill", "type", "select", "upload"):
        return "ui_action"
    return "other:%s" % (role or "unknown")


def _hab_target_resolved(sem_event):
    """A GUI operation is target-resolved when the element it acted on can be located again: a concrete
    ref/selector, or a URL for a navigation. "args is non-empty" is not enough."""
    e = sem_event.get("raw") or {}
    a = e.get("args") if isinstance(e.get("args"), dict) else {}
    cap = str(sem_event.get("capability_id") or "")
    ref = str(a.get("ref") or a.get("selector") or a.get("element") or a.get("element_id") or "").strip()
    if ref:
        return True
    if cap in ("back", "navigate", "navigate_to", "scroll"):
        url, _pg = _hab_page(e)
        return bool(str(url or "").strip())
    return False


def _hab_receipt(sem_event):
    """GUI receipt = the authoritative portal state diff. A rendered page alone is not a receipt; the
    localStorage state_record is what proves the effect landed."""
    e = sem_event.get("raw") or {}
    sr = e.get("state_record") if isinstance(e.get("state_record"), dict) else None
    if not sr:
        return False
    return (sr.get("state_changed") is not None) or bool(sr.get("state_after_hash"))


PLUGIN["observability_contract"] = {
    "target_resolver": _hab_target_resolved,
    "receipt_resolver": _hab_receipt,
    "family_of": _hab_observability_family,
    "cost_signals": [],
}
