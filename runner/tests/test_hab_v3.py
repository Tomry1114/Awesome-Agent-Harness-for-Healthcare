"""Bounded Clinical Recovery v3 - HAB generic GUI-completion unit tests. Validates GenericGuiCompletionWorkflow drives
entirely off the live control model + a verify_spec (no task-name workflows). Hard asserts (pytest)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from workspace import H0, Workspace

_WF = Workspace(H0["gui"]).module("recovery/gui_completion.py")
GenericGuiCompletionWorkflow, GOAL_TYPE = _WF.GenericGuiCompletionWorkflow, _WF.GOAL_TYPE
HabBenchmarkAdapter = Workspace(H0["gui"]).module("recovery/hab.py").HabBenchmarkAdapter
C = Workspace(H0["gui"]).module("recovery/contracts.py")


def _goal(verify_spec, committed=None):
    return C.CommittedGoal(goal_id="gui-0", goal_type=GOAL_TYPE,
                           committed_fields=committed or {},
                           raw={"verify_spec": verify_spec})


def test_generic_gui_verify_effect():
    wf = GenericGuiCompletionWorkflow()
    g = _goal({"path": "payer_a_state.differences.priorAuth.added", "check": "nonempty"})
    # landed submission -> True (a read-back: payer_a_state is the raw payer slice; the view exposes its submissions)
    sv_ok = {"payer_a_state": {"submissions": [{"requestType": "x"}]}}
    assert wf.verify_effect(g, sv_ok) is True
    # empty -> False (we committed; the read-back shows nothing)
    sv_no = {"payer_a_state": {"submissions": []}}
    assert wf.verify_effect(g, sv_no) is False
    # unreadable -> None
    assert wf.verify_effect(g, {}) is None


_SUBMIT_AUTH = [{"event_type": "tool_call", "origin": "agent", "tool": "click", "status": "ok",
                 "args": {"ref": "submit-auth-button"}}]


def test_adapter_emits_generic_goal_and_declines_unknown():
    """Recovery's conjunction is the shell's composition from the public goal and the agent's own decisions in the
    trajectory (completion obligations never copy hidden checkpoints); the task type is not read."""
    ad = HabBenchmarkAdapter()
    goal = "Submit a prior authorization request to the Payer A portal. Record the authorization number in EMR."
    task = {"environment": {"config": {"task_type": "submit_auth_aetna"}}, "goal": goal, "authoritative_state": {}}
    # nothing the agent committed and nothing the goal requires before a submission -> decline
    assert ad.resolve_commitments(task, [], "", None, {"authoritative_state": {}}) == []
    # the agent's own Submit (not landed) -> one generic goal: its effect first, then the goal's authorization note
    goals = ad.resolve_commitments(task, _SUBMIT_AUTH, "", None, {"authoritative_state": {}})
    assert len(goals) == 1 and goals[0].goal_type == GOAL_TYPE, goals
    assert goals[0].raw.get("verify_spec") == {"id": "prior_auth_submitted", "decision": "prior_auth_submit",
                                               "path": "payer_a_state.submissions", "check": "nonempty",
                                               "alt_paths": ["payer_a_state.differences.priorAuth.added"]}
    assert [c["id"] for c in goals[0].raw["verify_all"]] == ["prior_auth_submitted", "auth_note_added"]
    assert goals[0].raw["decisions"] == ["prior_auth_submit"] and goals[0].raw["agent_commit_refs"] == ["submit-auth-button"]
    # any other task type with the same goal and the same agent actions -> the same goal
    task2 = dict(task, environment={"config": {"task_type": "totally_unknown_type"}})
    assert len(ad.resolve_commitments(task2, _SUBMIT_AUTH, "", None, {"authoritative_state": {}})) == 1
    # a goal-derived requirement alone engages recovery when unmet
    task_dt = {"environment": {"config": {"task_type": "denial_triage"}}, "goal": "Document your reasoning in a triage note"}
    assert len(ad.resolve_commitments(task_dt, [], "", None, {"authoritative_state": {}})) == 1
    # already realized -> decline when the composed conjunction holds on the shell's obligation view of recovery's
    # read-back (context: recovery_readback); a residual requirement (the note) -> engage
    live = ad.context(dict(task, recovery_readback={
        "payer_a_state": {"submissions": [{"x": 1}]}, "full_state": {"agentActions": {"addedAuthNote": True}}}))
    assert ad.resolve_commitments(task, _SUBMIT_AUTH, "", None, live) == []
    live2 = ad.context(dict(task, recovery_readback={"payer_a_state": {"submissions": [{"x": 1}]}}))
    goals2 = ad.resolve_commitments(task, _SUBMIT_AUTH, "", None, live2)
    assert len(goals2) == 1 and goals2[0].raw["agent_commit_refs"] == []     # the submit landed: nothing to complete


def test_rb1_early_decline_uses_goal_conditional_obligations():
    """The early 'already-complete' check uses the SAME composition as verify_all (the goal's requirements and the
    agent's own decisions). The submission landed but a goal-required conjunct that depends on it (the
    authorization note) is unmet -> ENGAGE; the auth-letter and tab flags are not required (not in the goal) and the
    referral clear is not composed (the agent did not clear) -> with the note, DECLINE."""
    ad = HabBenchmarkAdapter()
    task = {"environment": {"config": {"task_type": "submit_auth_aetna"}},
            "goal": "Submit the prior auth, record the authorization number in EMR, and clear the referral from the worklist."}
    readback = {   # recovery's read-back of the portal; the decline reads the shell's obligation view of it
        "payer_a_state": {"submissions": [{"x": 1}]},
        "full_state": {"agentActions": {"viewedAuthLetter": True, "clickedDiagnosesTab": True}},
    }
    live = ad.context(dict(task, recovery_readback=readback))
    assert len(ad.resolve_commitments(task, _SUBMIT_AUTH, "", None, live)) == 1
    noted = dict(readback, full_state={"agentActions": {"addedAuthNote": True}})
    assert ad.resolve_commitments(task, _SUBMIT_AUTH, "", None, ad.context(dict(task, recovery_readback=noted))) == []

    # and when the goal does NOT ask for the note, the landed submission alone is complete -> decline
    task2 = {"environment": {"config": {"task_type": "submit_auth_aetna"}},
             "goal": "Submit a prior authorization request to Payer A."}
    assert ad.resolve_commitments(task2, _SUBMIT_AUTH, "", None, ad.context(dict(task2, recovery_readback=readback))) == []
