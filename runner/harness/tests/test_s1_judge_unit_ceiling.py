# -*- coding: utf-8 -*-
"""A semantic_check unit answers at most MAX_JUDGE_CALLS_PER_UNIT judge calls (runner/harness/semantic_budget.py).

A semantic_check unit pays for every call of one decision in the hook call that took it, so without a ceiling one
granted unit could buy any number of judge calls. The ceiling is a shell constant set above what the default (H0)
workspaces use."""
from harness import capability as C
from harness import semantic_budget as SB
from harness.state import Ledger


def _ctx(calls):
    def judge_fn(prompt):
        calls.append(prompt)
        return "YES"
    return C.HarnessContext(Ledger(), None, {}, judge_fn=judge_fn, semantic_budget=5)


def test_a_unit_answers_calls_up_to_the_ceiling_and_refuses_the_next_once():
    calls = []
    ctx = _ctx(calls)
    ceiling = getattr(SB, "MAX_JUDGE_CALLS_PER_UNIT", 12)
    judge = ctx.semantic_check("middleware")
    answers = [judge("q%d" % i) for i in range(ceiling + 2)]
    assert answers == ["YES"] * ceiling + [None, None]
    assert len(calls) == ceiling
    rec = ctx.semantic_meter.record()
    assert rec["calls"] == ceiling and rec["spent"] == 1
    assert rec["refused"] == 2 and ctx.semantic_meter.refused_total() == 2          # each refused call counted once
    assert rec["consumers"]["middleware"]["refused"] == 2
    assert rec["calls_per_unit"] == {"ceiling": ceiling,
                                     "consumers": {"middleware": {"max_calls": ceiling, "refused_calls": 2}}}


def test_the_ceiling_is_per_unit_and_leaves_per_call_judges_alone():
    calls = []
    ctx = _ctx(calls)
    ceiling = getattr(SB, "MAX_JUDGE_CALLS_PER_UNIT", 12)
    first, second = ctx.semantic_check("middleware"), ctx.semantic_check("middleware")
    assert [first("a") for _ in range(ceiling)] == ["YES"] * ceiling
    assert second("b") == "YES"                                          # a second unit has its own calls
    budgeted = ctx.budgeted_judge("middleware")
    assert budgeted("c") == "YES"                                        # one unit per call: never over the ceiling
    rec = ctx.semantic_meter.record()
    assert first("over") is None and len(calls) == ceiling + 2            # the first unit is spent
    assert rec["refused"] == 0 and rec["calls_per_unit"]["consumers"]["middleware"]["max_calls"] == ceiling
