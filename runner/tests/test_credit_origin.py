# -*- coding: utf-8 -*-
"""Agent-vs-harness credit: the origin model's shell seams.

  * ctx.emit refuses every event type a scorer, the loop or the aggregation reads, and provenance keys, and still
    appends a workspace's own event types
  * the origin tag is validated (a harness origin needs a requester from the list; an agent origin has none)
  * trajectories that carry the legacy origin "recovery" read as harness through the one predicate

The tests fail if emit accepts any event type or provenance key, or if the origin module is missing.
"""
import os
import sys

import pytest

_RUNNER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RUNNER not in sys.path:
    sys.path.insert(0, _RUNNER)

import middleware_host as MH                  # noqa: E402


def _ctx(traj):
    return MH.MiddlewareContext(env=None, task={}, trajectory=traj, step=3, max_steps=10, used_actions=0,
                                judge_factory=lambda: None, errors=[])


@pytest.mark.parametrize("event_type", [
    "tool_call", "final_answer", "answer_attempt", "write_file", "agent_error", "tool_budget_exhausted",
    "repair_budget_exhausted", "circuit_breaker", "reconciliation", "reconcile_resolved", "reconcile_enter",
    "deliverable_budget_warning", "harness_decision", "harness_escalation", "harness_acquire_dispatch",
    "harness_infrastructure_error", "my_budget_note", "truncated_note", "escalation_hint"])
def test_emit_refuses_event_types_the_scorers_and_the_loop_read(event_type):
    traj = []
    ctx = _ctx(traj)
    with pytest.raises(ValueError):
        ctx.emit({"event_type": event_type, "tool": "click", "args": {"ref": "x"}})
    assert traj == []
    ctx.emit({"event_type": "obligation_nudge", "status": "ok"})        # a workspace's own type still lands
    assert traj == [{"event_type": "obligation_nudge", "status": "ok", "step": 3}]


@pytest.mark.parametrize("key", ["origin", "requester", "action_origin", "action_requester", "final_answer_author"])
def test_emit_refuses_provenance_keys(key):
    traj = []
    with pytest.raises(ValueError):
        _ctx(traj).emit({"event_type": "obligation_nudge", key: "agent"})
    assert traj == []


def test_emit_refuses_a_reserved_type_but_appends_every_workspace_type_h0_emits():
    traj = []
    ctx = _ctx(traj)
    with pytest.raises(ValueError):
        ctx.emit({"event_type": "final_answer", "thought": "a stub the scorer would read"})
    for et in ("obligation_nudge", "grounding_gap", "recovery_v3", "adaptive_acquisition", "adaptive_revise",
               "read_acquire_dispatch", "agent_reentry", "precommit_audit", "precommit_terminal", "write_not_landed",
               "write_misroute", "collection_nudge", "attachment_block"):
        ctx.emit({"event_type": et, "status": "ok"})
    assert [e["event_type"] for e in traj][:2] == ["obligation_nudge", "grounding_gap"] and len(traj) == 13


def test_origin_tag_is_validated():
    from harness import origin as O
    assert O.tag({}, "agent") == {"origin": "agent"}
    assert O.tag({}, "harness", "kernel_acquire", stage="after_action", capability="c") == {
        "origin": "harness", "requester": "kernel_acquire", "requester_stage": "after_action",
        "requester_capability": "c"}
    for bad in (("recovery", None), ("harness", None), ("harness", "someone"), ("agent", "next_read")):
        with pytest.raises(ValueError):
            O.tag({}, *bad)
    assert set(O.REQUESTERS) == {"recovery", "next_read", "kernel_acquire", "on_answer_read", "content_guard",
                                 "forced_deliverable_write", "candidate_merge", "withhold"}


def test_frozen_recovery_origin_reads_as_harness():
    from harness import origin as O
    frozen = {"event_type": "tool_call", "origin": "recovery", "audience": "both"}
    assert O.is_harness_event(frozen) and O.requester_of(frozen) == "legacy_recovery_origin"
    assert not O.is_harness_event({"event_type": "tool_call", "origin": "agent", "audience": "agent"})
    assert O.kernel_kwargs({"tool": "x"}) == {}
    assert O.kernel_kwargs({"tool": "x", "_harness_acquire": True, "_requester": "next_read"}) == \
        {"origin": "harness", "requester": "next_read"}
