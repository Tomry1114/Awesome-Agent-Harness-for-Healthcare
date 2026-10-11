"""VERIFIED granularity: payload_covered is True only when EVERY agent-committed value is reflected in the
landed state (VERIFIED_PAYLOAD), False if a committed value is missing (VERIFIED_EFFECT only), None if
nothing was committed. Never asserts task correctness (scorer's domain)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from workspace import H0, Workspace
_WF = Workspace(H0["gui"]).module("recovery/gui_completion.py")
GenericGuiCompletionWorkflow, GOAL_TYPE = _WF.GenericGuiCompletionWorkflow, _WF.GOAL_TYPE
C = Workspace(H0["gui"]).module("recovery/contracts.py")

WF = GenericGuiCompletionWorkflow()
def goal(committed):
    return C.CommittedGoal(goal_id="gui-0", goal_type=GOAL_TYPE, committed_fields=committed, raw={})

def test_payload_covered_true_when_all_committed_reflected():
    g = goal({"fax-number-input": "18005551234", "attach": "Rx_Oxygen.pdf"})
    state = {"faxPortal": {"recipient": "18005551234", "attachmentNames": ["Rx_Oxygen.pdf"]}}
    assert WF.payload_covered(g, state) is True

def test_payload_covered_false_when_a_committed_value_missing():
    g = goal({"fax-number-input": "18005551234", "attach": "History_and_Physical.pdf"})
    state = {"faxPortal": {"recipient": "18005551234", "attachmentNames": ["Rx_Oxygen.pdf"]}}
    assert WF.payload_covered(g, state) is False

def test_payload_covered_none_when_nothing_committed():
    assert WF.payload_covered(goal({}), {"faxPortal": {"attachmentNames": ["x"]}}) is None
