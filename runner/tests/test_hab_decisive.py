"""Decisive HAB recovery test: a disposition SELECTED (draft) + note written but NOT submitted must NOT be
read as "submit realized"; the recovery's next action must be the Submit, not a skip-to-documentation. Guards
against a draft selectedDisposition (in the UI / a DOM-merged value) satisfying the submit verify_spec and making
the workflow skip the real Submit."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from workspace import H0, Workspace
_WF = Workspace(H0["gui"]).module("recovery/gui_completion.py")
GenericGuiCompletionWorkflow, GOAL_TYPE = _WF.GenericGuiCompletionWorkflow, _WF.GOAL_TYPE
C = Workspace(H0["gui"]).module("recovery/contracts.py")


def _goal(refs=(9,)):
    return C.CommittedGoal(goal_id="gui-0", goal_type=GOAL_TYPE, committed_fields={},
        raw={"verify_all": [   # paths into the shell's obligation view of the read-back (obligation_gate)
                {"path": "full_state.agentActions.selectedDisposition", "check": "nonempty",
                 "decision": "disposition_submit"},                                                # SUBMIT effect
                {"path": "full_state.agentActions.documentedAppealInEpic", "check": "nonempty"}],   # documentation
             "agent_commit_refs": refs,          # the agent itself pressed this Submit (hab.agent_commit_refs)
             "operational_commitment": True})


_SUBMIT_BTN = {"ref": 9, "role": "button", "label": "Submit Disposition", "name": "",
               "required": False, "value": "", "options": [], "commit": True}
_DOC = {"ref": 12, "role": "button", "label": "Save to Epic", "name": "", "commit": True}


def test_drafted_disposition_not_submitted_drives_submit():
    wf = GenericGuiCompletionWorkflow()
    g = _goal()
    # NOT submitted -> persisted agentActions EMPTY (the draft lives only in the UI, not the authoritative state)
    persisted_unsubmitted = {"full_state": {"agentActions": {}}}   # the read-back: full_state = the EMR slice
    assert wf._submit_subgoal_realized(g, persisted_unsubmitted) is False, \
        "draft (unpersisted) selectedDisposition must NOT count as submit-realized"
    dec = wf.next_action(g, {}, "", [_SUBMIT_BTN], persisted_unsubmitted, [])
    assert dec.get("kind") == "step" and getattr(dec.get("step"), "kind", None) == C.IRREVERSIBLE_COMMIT, \
        "next_action must DRIVE the Submit when the disposition is not yet persisted, got %r" % (dec,)


def test_persisted_disposition_not_resubmitted():
    wf = GenericGuiCompletionWorkflow()
    g = _goal()
    persisted_submitted = {"full_state": {"agentActions": {"selectedDisposition": "Corrected Claim - Resubmit"}}}
    assert wf._submit_subgoal_realized(g, persisted_submitted) is True
    dec = wf.next_action(g, {}, "", [_SUBMIT_BTN, _DOC], persisted_submitted, [])
    is_resubmit = (dec.get("kind") == "step"
                   and getattr(dec.get("step"), "kind", None) == C.IRREVERSIBLE_COMMIT
                   and str(dec.get("tag", "")).startswith("commit:9"))
    assert not is_resubmit, "must not re-submit an already-persisted disposition"


def test_d2_no_submit_without_the_agents_own_attempt():
    """The same unsubmitted draft, but the agent never pressed Submit (no agent_commit_refs): the harness never makes
    the commit on its own initiative."""
    wf = GenericGuiCompletionWorkflow()
    g = _goal(refs=())
    dec = wf.next_action(g, {}, "", [_SUBMIT_BTN], {"full_state": {"agentActions": {}}}, [])
    assert not (dec.get("kind") == "step" and getattr(dec.get("step"), "kind", None) == C.IRREVERSIBLE_COMMIT), dec


if __name__ == "__main__":
    test_drafted_disposition_not_submitted_drives_submit()
    test_persisted_disposition_not_resubmitted()
    print("PASS: draft!=submitted; unsubmitted drives Submit; persisted not re-submitted.")
