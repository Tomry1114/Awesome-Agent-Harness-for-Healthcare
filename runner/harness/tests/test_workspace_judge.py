# -*- coding: utf-8 -*-
"""Workspace code reaches the harness judge only through the shell's metered entry points.

Kernel middleware from a workspace gets capability.WorkspaceContextView: no raw judge_fn, no verification reserve,
nothing settable. It judges through ctx.semantic_check() (one check, the judge valid for that hook call) or
ctx.budgeted_judge() (one unit per call), both drawn by the semantic meter. The shell's own capabilities keep the full
context."""
import os
import shutil
import sys

import pytest

_RUNNER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _RUNNER not in sys.path:
    sys.path.insert(0, _RUNNER)

import harness as H                                     # noqa: E402
import workspace as W                                   # noqa: E402
from harness import decision as D                       # noqa: E402
from harness.capability import HarnessContext, WorkspaceContextView   # noqa: E402
from harness.state import Ledger                        # noqa: E402


@pytest.fixture(autouse=True)
def _no_harness_judge(monkeypatch):
    """build_kernel binds the harness judge from configuration and refuses without it (runner/judges.py). These
    tests exercise the kernel with no judge wired, so the judge builder is replaced by one that returns none."""
    import harness as _H
    monkeypatch.setattr(_H, "_build_judge", lambda: (None, None))

TASK = {"task_id": "J", "goal": "Work up suspected osteomyelitis.", "context": {"patient_ref": "Patient/1", "text": "c"},
        "available_tools": [{"name": "fhir_condition_search_problems"}], "environment": {"type": "fhir", "config": {}},
        "checkpoints": []}


def _ctx(budget=10):
    return HarnessContext(Ledger(), None, {}, judge_fn=lambda p: "answer:" + p, semantic_budget=budget)


def test_one_semantic_check_pays_once_for_the_calls_it_makes():
    ctx = _ctx()
    judge = ctx.semantic_check("kernel_middleware")
    assert ctx.semantic_meter.spent_total() == 1
    assert judge("a") == "answer:a" and judge("b") == "answer:b"
    assert ctx.semantic_meter.spent_total() == 1


def test_a_check_judge_kept_for_a_later_hook_is_refused():
    ctx = _ctx()
    judge = ctx.semantic_check("kernel_middleware")
    ctx.hook_serial += 1
    assert judge("a") is None and ctx.semantic_meter.refused_total() == 1


def test_no_judge_or_no_budget_means_no_check_and_no_spend():
    assert HarnessContext(Ledger(), None, {}, semantic_budget=10).semantic_check("kernel_middleware") is None
    ctx = _ctx(budget=2)          # half is reserved for verification, one unit is general
    assert ctx.semantic_check("kernel_middleware") is not None and ctx.semantic_check("kernel_middleware") is None


def test_the_view_hides_the_raw_judge_and_the_reserve_and_cannot_be_written():
    view = WorkspaceContextView(_ctx())
    with pytest.raises(AttributeError):
        view.judge_fn
    with pytest.raises(AttributeError):
        view.spend_verify_semantic()
    with pytest.raises(AttributeError):
        view.semantic_remaining = 99
    assert view.has_judge() and view.semantic_check() is not None


def _edited(tmp_path, body):
    path = str(tmp_path / "ws")
    shutil.copytree(W.H0["fhir"], path)
    p = os.path.join(path, "middleware", "commitment_execution.py")
    src = open(p).read()
    head = "    def before_final(self, answer, ctx):\n"
    assert src.count(head) == 1
    open(p, "w").write(src.replace(head, head + body))
    return path


@pytest.mark.parametrize("body", ["        ctx.judge_fn('free call')\n", "        ctx.semantic_remaining = 99\n",
                                  "        ctx.spend_verify_semantic()\n",
                                  "        ctx.semantic_meter.draw('goal_spec', 'call')\n"],
                         ids=["raw_judge", "budget_write", "reserve", "meter"])
def test_a_workspace_middleware_that_reaches_past_the_meter_fails_closed(tmp_path, monkeypatch, body):
    k = H.build_kernel(TASK, env_type="fhir", workspace=_edited(tmp_path, body))
    spent = k.ctx.semantic_meter.spent_total()
    eff = k.before_final("done", step=1)
    assert eff.type == D.ESCALATE
    assert any("commitment_execution.before_final" in e for e in k._capability_errors), k._capability_errors
    assert k.ctx.semantic_meter.spent_total() == spent


def test_the_shells_capabilities_keep_the_full_context(monkeypatch):
    k = H.build_kernel(TASK, env_type="fhir", workspace=W.H0["fhir"])
    for cap in k.capabilities:
        view = k._ctx_for(cap)
        if W.component_path(cap) is None:
            assert view is k.ctx, cap.name
        else:
            assert isinstance(view, WorkspaceContextView), cap.name
