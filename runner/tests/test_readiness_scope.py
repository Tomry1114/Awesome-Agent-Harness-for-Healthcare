"""The decision-readiness preemption / commit gate applies ONLY to disposition-CHOICE tasks. A SUBMISSION task
(submit_auth/dme/appeal) with an unresolved readiness gap must COMPLETE the agent's committed submission (not
abandon it with re_entry_requested); a DISPOSITION task must still block. Pins that the gate does not discard a
submission the agent already started."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from workspace import H0, Workspace
_WF = Workspace(H0["gui"]).module("recovery/gui_completion.py")
GenericGuiCompletionWorkflow, GOAL_TYPE = _WF.GenericGuiCompletionWorkflow, _WF.GOAL_TYPE
C = Workspace(H0["gui"]).module("recovery/contracts.py")

_SUBMIT = {"ref": 9, "role": "button", "label": "Submit Authorization", "name": "",
           "required": False, "value": "", "options": [], "commit": True}


def _goal(authoring, gap=("clickedDiagnosesTab",)):
    return C.CommittedGoal(goal_id="gui-0", goal_type=GOAL_TYPE, committed_fields={},
        raw={"verify_all": [{"path": "agentActions.selectedDisposition", "check": "nonempty",
                             "decision": "disposition_submit"},
                            {"path": "agentActions.documentedAppealInEpic", "check": "nonempty"}],
             "agent_commit_refs": [9],                     # the agent itself pressed Submit
             "operational_commitment": True,               # agent began the submission
             "decision_readiness_gap": list(gap),          # evidence not fully observed
             "decision_readiness_sources": [],             # no reachable locator -> would re-entry-block
             "authoring_disposition": authoring})


def test_submission_readiness_gap_completes_not_blocks():
    wf = GenericGuiCompletionWorkflow()
    dec = wf.next_action(_goal(authoring=False), {}, "", [_SUBMIT], {"agentActions": {}}, [])
    assert dec.get("kind") == "step" and getattr(dec.get("step"), "kind", None) == C.IRREVERSIBLE_COMMIT, \
        "submission task with readiness gap must COMPLETE the submit, got %r" % (dec,)


def test_disposition_readiness_gap_still_blocks():
    wf = GenericGuiCompletionWorkflow()
    dec = wf.next_action(_goal(authoring=True), {}, "", [_SUBMIT], {"agentActions": {}}, [])
    assert dec.get("kind") == "block" and "readiness" in str(dec.get("reason", "")), \
        "disposition task with readiness gap must still BLOCK, got %r" % (dec,)


def test_decision_authoring_orthogonal_to_workflow_family():
    """denial-medium-1: its public challenge type is denial_triage. It must be recognized as a DECISION-AUTHORING task
    (recovery cannot author the disposition), and its workflow family is appeal -- the two properties are orthogonal.
    Both are read from the public task: the hidden task_type (electronic_appeals here) is not given to workspace
    code and changes neither."""
    _hab = Workspace(H0["gui"]).module("recovery/hab.py")
    _is_disposition_task, _family = _hab._is_disposition_task, _hab._family
    med = {"environment": {"config": {"challenge_type": "denial_triage"}}}
    assert _family(med) == "appeal"                      # workflow-family
    assert _is_disposition_task(med) is True             # decision-authoring
    prior = {"environment": {"config": {"challenge_type": "submit_auth_aetna"}}}
    assert _is_disposition_task(prior) is False and _family(prior) == "prior_auth"   # a submission task
    assert _family({"environment": {"config": {"challenge_type": "workflow"}}}) == "dme"
    hidden_only = {"environment": {"config": {"task_type": "denial_triage"}}}
    assert _is_disposition_task(hidden_only) is False and _family(hidden_only) == ""
