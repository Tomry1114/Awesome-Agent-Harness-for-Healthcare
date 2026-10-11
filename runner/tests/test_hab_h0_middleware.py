# -*- coding: utf-8 -*-
"""The HealthAdminBench workspace's host middleware at its seam: obligation_nudge, grounding_audit and read_acquire,
each driven with a MiddlewareContext whose shell state the test sets. What the loop does with their results (order at
the final, the Nudge and Read semantics, the exact events and texts in a run) is pinned by test_inline_hab_gates.py
and test_inline_hab_recovery.py."""
import json
import os
import sys

import pytest

_RUNNER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RUNNER not in sys.path:
    sys.path.insert(0, _RUNNER)

import middleware_host as MH                     # noqa: E402
from workspace import H0, Workspace              # noqa: E402

_WS = Workspace(H0["gui"])
ObligationNudge = _WS.load("middleware/obligation_nudge.py", "ObligationNudge")
GroundingAudit = _WS.load("middleware/grounding_audit.py", "GroundingAudit")
ReadAcquire = _WS.load("middleware/read_acquire.py", "ReadAcquire")
ON = sys.modules[ObligationNudge.__module__]
GA = sys.modules[GroundingAudit.__module__]


def _ctx(env=None, task=None, step=0, max_steps=50, used=0, recovery=0, spent=False, in_repair=False,
         obligations=None, observed=(), errors=None, traj=None):
    shell = {"budget_spent_at_turn_start": lambda: spent, "agent_env_actions": lambda: used,
             "recovery_env_actions": lambda: recovery, "harness_env_actions": lambda: recovery,
             "in_repair": lambda: in_repair,
             "observed_history": lambda: list(observed), "obligations": obligations}
    return MH.MiddlewareContext(env=env, task=task if task is not None else {},
                                trajectory=traj if traj is not None else [], step=step, max_steps=max_steps,
                                used_actions=used + recovery, judge_factory=lambda: None,
                                errors=errors if errors is not None else [], shell=shell)


def test_the_hab_manifest_runs_the_obligation_nudge_before_the_grounding_audit():
    names = [c.__name__ for c in _WS.middleware_classes()]
    assert names.index("ObligationNudge") < names.index("GroundingAudit") and "ReadAcquire" in names


# ------------------------------------------------------------------------------------------------ obligation_nudge
def test_obligation_nudge_is_bounded_and_counts_agent_and_recovery_actions():
    m, traj = ObligationNudge({}), []
    unmet = lambda: (False, ("disposition_submit",), (True, False))
    out = [m.before_final(_ctx(step=s, used=2, recovery=3, obligations=unmet, traj=traj), "done") for s in range(5)]
    fired = [o for o in out if o is not None]
    assert len(fired) == 3 and out[3:] == [None, None]
    for o in fired:
        assert isinstance(o, MH.Nudge) and (o.stage, o.obs, o.label) == ("obligation", "obligation_unmet", "CONTINUE")
        assert o.feedback == ON.MESSAGE % 45
    assert traj == [{"step": s, "event_type": "obligation_nudge", "unmet": True, "decisions": ["disposition_submit"],
                     "attempt": s + 1, "status": "ok"} for s in range(3)]


def test_obligation_nudge_silent_when_met_short_of_budget_spent_or_the_check_fails():
    unmet = lambda: (False, "x", None)

    def fails():
        raise RuntimeError("state read failed")
    assert ObligationNudge({}).before_final(_ctx(obligations=lambda: (True, "x", None)), "a") is None
    assert ObligationNudge({}).before_final(_ctx(obligations=None), "a") is None     # no obligation model: met
    assert ObligationNudge({}).before_final(_ctx(used=45, obligations=unmet), "a") is None   # 5 left
    assert ObligationNudge({}).before_final(_ctx(used=44, obligations=unmet), "a") is not None   # 6 left
    assert ObligationNudge({}).before_final(_ctx(spent=True, obligations=unmet), "a") is None
    assert ObligationNudge({}).before_final(_ctx(obligations=fails), "a") is None


# ------------------------------------------------------------------------------------------------ grounding_audit
class _StateEnv(object):
    def __init__(self, state):
        self.state = state

    def eval_state(self):
        return self.state


_TRIAGE = {"environment": {"config": {"task_type": "denial_triage"}}}


def test_grounding_audit_names_observed_codes_the_note_lacks_for_two_rounds():
    m, traj = GroundingAudit(_TRIAGE), []
    env = _StateEnv({"full_state": {"triageNotes": ["Denial CO-50, appeal."]}})
    seen = ["Denial CO-50 remark N386", "diagnosis E11.9 documented"]
    outs = [m.before_final(_ctx(env=env, task=_TRIAGE, step=s, observed=seen, traj=traj), "a") for s in (7, 8, 9)]
    assert outs[2] is None
    for o in outs[:2]:
        assert isinstance(o, MH.Nudge) and (o.stage, o.obs, o.label) == ("grounding", "grounding_gap", "REVISE")
        assert o.feedback == GA.feedback(["E11.9", "N386"])
    assert traj == [{"event_type": "grounding_gap", "missing": ["E11.9", "N386"], "status": "ok", "step": s,
                     "round": r} for s, r in ((7, 1), (8, 2))]


def test_grounding_audit_runs_with_the_budget_spent_and_is_quiet_when_nothing_is_missing():
    env = _StateEnv({})
    assert GroundingAudit(_TRIAGE).before_final(_ctx(env=env, task=_TRIAGE, spent=True, observed=["code E11.9"]),
                                                "a") is not None
    assert GroundingAudit(_TRIAGE).before_final(_ctx(env=env, task=_TRIAGE, observed=["nothing salient"]), "a") is None
    noted = _StateEnv({"full_state": {"triageNotes": ["E11.9 reviewed"]}})
    assert GroundingAudit(_TRIAGE).before_final(_ctx(env=noted, task=_TRIAGE, observed=["code E11.9"]), "a") is None


# ------------------------------------------------------------------------------------------------ read_acquire
def _prior_auth_task():
    path = os.path.join(os.path.dirname(_RUNNER), "benchmark_dataprocess", "HealthAdminBench", "tasks_unified.jsonl")
    for line in open(path, encoding="utf-8"):
        t = json.loads(line)
        if t.get("task_id") == "HAB-emr-medium-1":
            return t
    pytest.fail("HAB-emr-medium-1 is not in %s" % path)


class _TabEnv(object):
    def list_controls(self):
        return [{"role": "tab", "label": "Diagnoses", "ref": "tab-dx", "visible": True}]

    def eval_state(self):
        return {}                    # no read-evidence flag is set: clickedDiagnosesTab is unmet


def test_read_acquire_reads_the_unmet_flags_control_at_most_twice():
    task, traj, errors = _prior_auth_task(), [], []
    m = ReadAcquire(task)
    outs = [m.next_read(_ctx(env=_TabEnv(), task=task, step=s, traj=traj, errors=errors)) for s in range(3)]
    args = {"ref": "tab-dx", "provenance": "harness-acquired-evidence", "evidence_flag": "clickedDiagnosesTab"}
    assert [MH.queued_read(o) for o in outs] == [
        {"type": "tool_call", "tool": "click", "args": args, "_harness_acquire": True,
         "_requester": "next_read"}] * 2 + [None]
    assert traj == [{"step": s, "event_type": "read_acquire_dispatch", "evidence_flag": "clickedDiagnosesTab",
                     "ref": "tab-dx", "op": "click", "attempt": s + 1, "status": "ok"} for s in range(2)]
    assert errors == []


def test_read_acquire_waits_while_the_budget_is_spent_or_repair_has_started():
    task = _prior_auth_task()
    assert ReadAcquire(task).next_read(_ctx(env=_TabEnv(), task=task, spent=True)) is None
    assert ReadAcquire(task).next_read(_ctx(env=_TabEnv(), task=task, in_repair=True)) is None
    assert ReadAcquire({}).next_read(_ctx(env=_TabEnv(), task=_TRIAGE)) is None       # not a prior-auth task


def test_read_acquire_reports_an_env_failure_as_a_runtime_error():
    class Broken(_TabEnv):
        def list_controls(self):
            raise RuntimeError("page gone")
    task, errors = _prior_auth_task(), []
    assert ReadAcquire(task).next_read(_ctx(env=Broken(), task=task, errors=errors)) is None
    assert errors == ["read_acquire: RuntimeError('page gone')"]
