"""Closed-Loop Verification returns a STRUCTURED gap (satisfied paths, plus unmet paths each with a resolver_type and
target_hints), and the GUI completion planner acquires evidence toward the first unmet evidence subgoal. This pins
the generic verify -> resolve loop that every task family relies on."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from workspace import H0, Workspace
W = Workspace(H0["gui"]).module("recovery/gui_completion.py").GenericGuiCompletionWorkflow


class _G:
    committed_fields = {}
    goal_type = "gui"
    raw = {"verify_all": [
        {"path": "full_state.agentActions.viewedRemittanceImage", "check": "truthy"},
        {"path": "full_state.signals.clicked_diagnoses_tab", "check": "truthy"},
        {"path": "full_state.faxPortal.attachmentNames", "check": "nonempty"},
    ], "verify_spec": {"path": "full_state.agentActions.viewedRemittanceImage", "check": "truthy"}}


def test_verify_effect_returns_structured_gap():
    state = {"full_state": {"agentActions": {"viewedRemittanceImage": True}, "signals": {}, "faxPortal": {}}}
    eff = W()._verify_effect(_G, state)
    assert "full_state.agentActions.viewedRemittanceImage" in eff["satisfied"]
    unmet = {u["path"]: u for u in eff["unmet"]}
    assert "full_state.signals.clicked_diagnoses_tab" in unmet
    tab = unmet["full_state.signals.clicked_diagnoses_tab"]
    assert tab["resolver_type"] == "acquire" and "diagnoses" in tab["target_hints"]
    assert unmet["full_state.faxPortal.attachmentNames"]["resolver_type"] == "attach"


def test_planner_acquires_toward_unmet_evidence_control():
    class _G1:
        committed_fields = {}
        goal_type = "gui"
        raw = {"verify_all": [{"path": "full_state.agentActions.viewedRemittanceImage", "check": "truthy"}],
               "verify_spec": {"path": "full_state.agentActions.viewedRemittanceImage", "check": "truthy"}}
    controls = [{"ref": "tab-remittance", "role": "tab", "label": "Remittance Image"}]
    d = W().next_action(_G1, {}, "", controls, {"full_state": {"agentActions": {}}}, [])
    assert d.get("kind") == "step" and d["step"].name == "acquire_evidence"
    assert d["step"].affordance_target["ref"] == "tab-remittance"


def test_planner_never_acquires_a_commit_control_for_evidence():
    class _G2:
        committed_fields = {}
        goal_type = "gui"
        raw = {"verify_all": [{"path": "full_state.agentActions.viewedSubmitReceipt", "check": "truthy"}],
               "verify_spec": {"path": "full_state.agentActions.viewedSubmitReceipt", "check": "truthy"}}
    # a control matching the hint 'submit' but which is a COMMIT must NOT be acquired
    controls = [{"ref": "submit-btn", "role": "button", "label": "Submit Receipt", "commit": True}]
    d = W().next_action(_G2, {}, "", controls, {"full_state": {"agentActions": {}}}, [])
    assert not (d.get("kind") == "step" and d["step"].name == "acquire_evidence")


def test_r6_download_resolver_fetches_the_document():
    """An unmet downloaded_* subgoal with a matching download control present makes the harness take the mechanical
    download_document step on that control."""
    class _G:
        committed_fields = {}
        goal_type = "gui"
        raw = {"verify_all": [{"path": "full_state.signals.downloaded_auth_letter", "check": "truthy"}],
               "verify_spec": {"path": "full_state.signals.downloaded_auth_letter", "check": "truthy"}}
    ctrls = [{"ref": "dl", "role": "button", "label": "Download Auth Letter"}]
    d = W().next_action(_G, {}, "", ctrls, {"full_state": {"signals": {}}}, [])
    assert d.get("kind") == "step" and d["step"].name == "download_document"
    assert d["step"].affordance_target["ref"] == "dl"


def test_r6_clear_resolver_only_when_everything_else_satisfied():
    """Clear-from-worklist is a mechanical TERMINAL step: the planner fires it only when no other obligation is unmet,
    so the task is not closed before its other required effects exist."""
    class _Gc:
        committed_fields = {}
        goal_type = "gui"
        raw = {"verify_all": [{"path": "full_state.clearedReferrals", "check": "nonempty"}],
               "verify_spec": {"path": "full_state.clearedReferrals", "check": "nonempty"}}
    ctrls = [{"ref": "clr", "role": "button", "label": "Clear from Worklist"}]
    d = W().next_action(_Gc, {}, "", ctrls, {"full_state": {"clearedReferrals": []}}, [])
    assert d.get("kind") == "step" and d["step"].name == "clear_worklist"

    # but if ANOTHER obligation is still unmet, do NOT clear yet (premature terminal step)
    class _Gc2:
        committed_fields = {}
        goal_type = "gui"
        raw = {"verify_all": [{"path": "full_state.clearedReferrals", "check": "nonempty"},
                              {"path": "full_state.signals.clicked_go_to_portal", "check": "truthy"}],
               "verify_spec": {"path": "full_state.clearedReferrals", "check": "nonempty"}}
    d2 = W().next_action(_Gc2, {}, "", ctrls, {"full_state": {"clearedReferrals": [], "signals": {}}}, [])
    assert not (d2.get("kind") == "step" and d2["step"].name == "clear_worklist")


def test_rb5_recovery_refuses_to_submit_empty_form_or_queue_link():
    """Recovery must NOT submit a blank surface: pressing a queue/nav link (submit-authorizations-link) or an empty
    payload form would create an empty submission that overwrites the agent's real one. The planner blocks with
    reason empty_form_no_submit, while a form with filled fields stays submittable."""
    class _G:
        committed_fields = {}
        goal_type = "gui"
        raw = {"verify_spec": {"path": "payer_a_state.submissions", "check": "nonempty"},
               "verify_all": [{"path": "payer_a_state.submissions", "check": "nonempty", "decision": "prior_auth_submit"}],
               "agent_commit_refs": ["submit-authorizations-link", "sub"],   # controls the agent itself pressed
               "operational_commitment": True}
    wreck = [{"ref": "submit-authorizations-link", "role": "link", "label": "Submit Authorizations", "commit": True}]
    d = W().next_action(_G, {}, "", wreck, {"payer_a_state": {}}, [])
    assert d.get("kind") == "block" and d.get("reason") == "empty_form_no_submit"
    # a FILLED form is still submittable
    filled = [{"ref": "rt", "role": "input", "label": "Request Type", "value": "Outpatient"},
              {"ref": "sub", "role": "button", "label": "Submit Authorization Request", "commit": True}]
    d2 = W().next_action(_G, {}, "", filled, {"payer_a_state": {}}, [])
    assert d2.get("kind") == "step" and d2["step"].name == "submit"
