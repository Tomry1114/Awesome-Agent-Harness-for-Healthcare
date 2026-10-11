"""The GUI obligation model: "met" is the whole composed conjunction, from one persisted-state read. The conjunction comes
from the public goal and the agent's own decisions; the task type is not read."""
from .obligation_gate import Snapshot, gui_obligations, lock_inputs, obligation_snapshot
from workspace import H0

HAB = H0["gui"]   # the workspace whose completion content these calls read (a run hands its own to every reader)

GOAL = ("Open denial DEN-001 for Martinez, Carlos. Review all available information about this denial and determine the "
        "appropriate triage disposition. Document your reasoning in a triage note.")
SUBMIT_CLICK = {"event_type": "tool_call", "origin": "agent", "tool": "click", "status": "ok",
                "args": {"ref": "submit-disposition-button"}}


class _Env:
    def __init__(self, es): self._es = es
    def eval_state(self): return self._es


def test_gui_obligations_met_reads_the_composed_conjunction():
    task = {"goal": GOAL, "environment": {"config": {"task_type": "denial_triage"}}}
    met = {"full_state": {"agentActions": {"selectedDisposition": "Route to Appeals", "documentedAppealInEpic": True,
                                           "viewedDenialDetails": True}}}
    doc_only = {"full_state": {"agentActions": {"documentedAppealInEpic": True, "viewedDenialDetails": True}}}
    # before the agent submits anything: the goal's two requirements (the denial opened, the triage note)
    assert gui_obligations(task, _Env(met), HAB) == (True, (), (True, True))
    # the agent's own Submit Disposition composes its effect first: the disposition it chose must land
    assert gui_obligations(task, _Env(met), HAB, actions=[SUBMIT_CLICK]) == (True, ("disposition_submit",), (True, True, True))
    assert gui_obligations(task, _Env(doc_only), HAB, actions=[SUBMIT_CLICK]) == (
        False, ("disposition_submit",), (False, True, True))
    snap = obligation_snapshot(task, _Env(doc_only), HAB, actions=[SUBMIT_CLICK])
    assert snap == (False, True, True) and snap.submissions == 1
    assert obligation_snapshot(task, _Env(doc_only), HAB, pending={"type": "tool_call", "tool": "click",
                                                              "args": {"ref": "[submit-disposition-button]"}}) == snap
    # a harness read or a harness-origin call is not the agent's decision
    harness = [dict(SUBMIT_CLICK, origin="recovery")]
    assert gui_obligations(task, _Env(doc_only), HAB, actions=harness) == (True, (), (True, True))
    assert gui_obligations(task, _Env(doc_only), HAB, pending=dict(SUBMIT_CLICK, _harness_acquire=True))[1] == ()
    # the task type is not read
    other = dict(task, environment={"config": {"task_type": "zzz"}})
    assert gui_obligations(other, _Env(doc_only), HAB, actions=[SUBMIT_CLICK]) == gui_obligations(
        task, _Env(doc_only), HAB, actions=[SUBMIT_CLICK])
    # nothing composed: met for ctx.obligations(), an empty snapshot the lock never reads as met
    bare = {"goal": "look around", "environment": {"config": {}}}
    assert gui_obligations(bare, _Env({}), HAB) == (True, (), None)
    assert obligation_snapshot(bare, _Env({}), HAB) == () and lock_inputs(None, obligation_snapshot(bare, _Env({}), HAB)) == (
        False, False)


def test_lock_inputs():
    assert lock_inputs(None, None) == (None, None)                                   # no obligation model
    assert lock_inputs(Snapshot((False, True), 1), Snapshot((True, True), 1)) == (True, True)
    assert lock_inputs(Snapshot((True, False), 1), Snapshot((True, True), 1)) == (True, False)   # not newly landed
    assert lock_inputs(Snapshot((False,), 0), Snapshot((True,), 0)) == (True, False)             # not a submission
    assert lock_inputs(None, Snapshot((True, True), 1)) == (True, True)
    assert lock_inputs(Snapshot((True,), 1), Snapshot((True, False), 1)) == (False, True)        # other shape
    assert lock_inputs(Snapshot(()), Snapshot(())) == (False, False)
