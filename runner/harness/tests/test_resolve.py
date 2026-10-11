# -*- coding: utf-8 -*-
"""Conflict is a property of the HOOK, not a global severity rank.

resolve() picks the winning decision per hook (stage): the same pair of decisions may resolve differently at
different hooks, losers are reported rather than dropped, ALLOW never wins over a decision that acts, and ties keep
the first capability that raised that level.

Run:  python -m pytest runner/harness/tests/test_resolve.py -q
"""
import os
import sys

_RUNNER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _RUNNER not in sys.path:
    sys.path.insert(0, _RUNNER)

from harness import decision as D                                     # noqa: E402


def _d(kind, cap):
    return D.HarnessDecision(kind, capability=cap, rule_id=cap)


def test_an_empty_hook_allows():
    r = D.resolve([], stage="before_action")
    assert r.winner.type == D.ALLOW and r.also == []


def test_a_veto_at_before_action_beats_everything_that_would_still_run_the_action():
    r = D.resolve([_d(D.ACQUIRE, "a"), _d(D.BLOCK, "b")], stage="before_action")
    assert r.winner.capability == "b", "BLOCK is a veto: the action must not run"


def test_before_final_is_about_accepting_the_final_so_defer_leads_there():
    """The same two decisions resolve DIFFERENTLY at a different hook -- which is the point of making conflict
    a property of the hook. A single global rank (DEFER_FINAL above ACQUIRE) would order them the same everywhere."""
    at_final = D.resolve([_d(D.ACQUIRE, "a"), _d(D.DEFER_FINAL, "d")], stage="before_final")
    assert at_final.winner.capability == "d"

    at_action = D.resolve([_d(D.ACQUIRE, "a"), _d(D.DEFER_FINAL, "d")], stage="before_action")
    assert at_action.winner.capability == "a", \
        "before an action there is no final to defer; the read is what shapes control flow"


def test_the_loser_is_reported_rather_than_dropped():
    """Delivery does not conflict even when control flow does. If only a global max were kept, a second capability
    with something to say would leave no trace, so `it never fired` and `it fired and was outranked` would be the
    same record -- and only one of those is a fact about the harness."""
    r = D.resolve([_d(D.ACQUIRE, "a"), _d(D.REVISE, "b")], stage="after_action")
    assert r.winner.capability == "a"
    assert [x.capability for x in r.also] == ["b"]


def test_allow_is_never_a_winner_over_something_that_acted():
    r = D.resolve([_d(D.ALLOW, "quiet"), _d(D.REVISE, "loud")], stage="after_action")
    assert r.winner.capability == "loud" and r.also == []


def test_ties_keep_the_first_capability_that_raised_that_level():
    r = D.resolve([_d(D.REVISE, "first"), _d(D.REVISE, "second")], stage="after_action")
    assert r.winner.capability == "first"
    assert [x.capability for x in r.also] == ["second"]


def test_combine_is_the_one_value_form_and_agrees_with_resolve():
    ds = [_d(D.ACQUIRE, "a"), _d(D.REVISE, "b")]
    for stage in ("before_action", "after_action", "before_final"):
        assert D.combine(ds, stage=stage) is D.resolve(ds, stage=stage).winner


def test_there_is_no_global_rank_left_deciding_across_hooks():
    """A single order applied everywhere assumes that the action CLASS decides who matters. Two hooks must be able
    to disagree about the same pair."""
    ds = [_d(D.ACQUIRE, "a"), _d(D.DEFER_FINAL, "d")]
    winners = {stage: D.resolve(ds, stage=stage).winner.capability
               for stage in ("before_action", "before_final")}
    assert len(set(winners.values())) == 2, winners


def test_resolve_itself_actuates_nothing():
    """resolve() is a pure function over a decision list. If it ever grows a side effect, the outranked
    decisions would start doing something while still being reported as having done nothing."""
    import inspect
    src = inspect.getsource(D.resolve)
    for effect in ("ledger", "actuat", "deliver", "self."):
        assert effect not in src, "resolve() gained a side effect: %r" % effect
