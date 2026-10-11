"""ACQUIRE, REVISE and DEFER_FINAL, pinned as CONTRACTS rather than as whatever the hooks happen to do.

The failure mode guarded against: a decision that has a name at a hook and no behaviour there, which in an
artifact is indistinguishable from a capability that never fired.

  ACQUIRE      has one actuator, dispatched at all 3 hooks.
  REVISE       is a real repair turn at all 3 hooks, including after_action (not only text spliced into an observation).
  DEFER_FINAL  has its own branch at before_final, separate from REVISE's.
"""
import io
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RUN_PY = os.path.join(REPO, "runner", "run.py")

from harness import actuation as A          # noqa: E402


def _run_src():
    return io.open(RUN_PY, encoding="utf-8").read()


# ---------------------------------------------------------------- one actuator, every hook
def test_every_hook_dispatches_acquire_through_the_one_actuator():
    """after_action needs its own ACQUIRE branch; without it a decision reaches "effective" and produces no tool
    call."""
    src = _run_src()
    for hookvar, stage in (("_hb", "before_action"), ("_hpost", "after_action"), ("_hf", "before_final")):
        assert '_act.plan(%s, "%s"' % (hookvar, stage) in src, \
            "%s has no ACQUIRE actuator" % stage
    # and no code hand-builds the forced action
    assert '"_harness_acquire": True' not in src, \
        "a hand-built forced action bypasses the shared admission gate and telemetry"


def test_revise_is_a_repair_turn_at_every_hook():
    src = _run_src()
    # every repair-type result goes through the one accounting, _repair_result
    helper = src[src.index("    def _repair_result("):]
    helper = helper[:helper.index("        return within")]
    assert "_repair_turns += 1" in helper and "_in_repair[0] = True" in helper and "record_revision(" in helper
    for stage, var in (("before_action", "_hb"), ("after_action", "_hpost"), ("before_final", "_hf")):
        assert '_repair_result("%s", "kernel:" + %s.type' % (stage, var) in src, \
            "%s REVISE is not accounted as a repair turn" % stage
    # after_action must actually claim a turn, not only splice text into the observation
    aa = src[src.index('_hpost.type in ("REVISE", "BLOCK")'):]
    aa = aa[:aa.index('if _hpost.type == "ACQUIRE"')]
    assert '_repair_result("after_action"' in aa
    assert src.count("record_revision(") == helper.count("record_revision("), "a REVISE recorded outside the one accounting"


def test_defer_final_is_actuated_at_before_final_and_nowhere_else():
    src = _run_src()
    assert len(re.findall(r'record_defer\(', src)) == 1
    assert 'record_defer(_harness.ledger, "before_final"' in src
    assert '_hf.type == "DEFER_FINAL"' in src, "DEFER_FINAL needs its own branch, not REVISE's"
    assert '"REVISE", "BLOCK", "DEFER_FINAL"' not in src, \
        "sharing REVISE's branch is what left DEFER_FINAL with no behaviour of its own"


# ---------------------------------------------------------------- telemetry: five rungs, not one bit
class _Ledger(object):
    def __init__(self):
        self.acquire_count = 0
        self.validated_evidence_version = 0


class _Dec(object):
    def __init__(self, extra, capability="cap"):
        self.extra = extra
        self.capability = capability


def test_acquire_records_where_the_chain_stopped():
    led = _Ledger()
    ok = A.plan(_Dec({"next_action": {"tool": "fhir_search", "read_only": True}}), "after_action", 1, ledger=led)
    assert ok.dispatchable and ok.forced_action["_acquire_id"]
    A.record_execution(led, ok.acquire_id, executed=True, succeeded=True, evidence_added=False)

    bad = A.plan(_Dec({"next_action": {"tool": "write_file", "read_only": False}}), "before_action", 2, ledger=led)
    assert not bad.dispatchable and bad.failure_reason == A.NOT_READ_ONLY
    assert bad.events and bad.events[0]["event_type"] == "harness_acquire_refused"

    s = A.episode_summary(led)
    assert (s["acquire_requested"], s["acquire_attempted"], s["acquire_executed"]) == (2, 1, 1)
    assert s["acquire_succeeded"] == 1 and s["acquire_evidence_added"] == 0
    assert s["failure_reasons"] == {A.NOT_READ_ONLY: 1, "no_new_evidence": 1}


def test_effective_is_not_executed():
    """A plan that was never dispatched must not read as an execution."""
    led = _Ledger()
    A.plan(_Dec({}), "after_action", 1, ledger=led)          # no next_action at all
    s = A.episode_summary(led)
    assert s["acquire_requested"] == 1 and s["acquire_executed"] == 0
    assert s["failure_reasons"] == {A.NO_NEXT_ACTION: 1}


def test_both_by_stage_breakdowns_survive():
    led = _Ledger()
    p = A.plan(_Dec({"next_action": {"tool": "t", "read_only": True}}), "before_final", 1, ledger=led)
    A.record_execution(led, p.acquire_id, executed=True, succeeded=True, evidence_added=True)
    A.record_revision(led, "after_action", 2, repair_turn_started=True, repair_turn=1)
    A.record_defer(led, "before_final", 3, repair_turn_started=True, repair_turn=2, control_only=True)
    s = A.episode_summary(led)
    assert s["by_stage_acquire"]["before_final"]["executed"] == 1
    assert s["by_stage_revise_defer"]["after_action"]["REVISE"] == 1
    assert s["revise_repair_turns"] == 1 and s["defer_final_gated"] == 1


class _TelemetryLedger(object):
    def __init__(self):
        self.acquire_telemetry = []
        self.actuation_telemetry = []


def test_one_definition_of_executed_per_decision_kind():
    led = _TelemetryLedger()
    assert A.n_intervention_executed(led) == 0
    # ACQUIRE: requested and attempted are EARLIER rungs and touch nothing
    led.acquire_telemetry.append({"requested": True, "attempted": True, "executed": False})
    assert A.n_intervention_executed(led) == 0
    led.acquire_telemetry.append({"requested": True, "attempted": True, "executed": True})
    assert A.n_intervention_executed(led) == 1
    # REVISE: delivery WITHOUT a repair turn is a string in an observation
    A.record_revision(led, "after_action", 1, delivered=True, repair_turn_started=False)
    assert A.n_intervention_executed(led) == 1
    A.record_revision(led, "after_action", 2, delivered=True, repair_turn_started=True)
    assert A.n_intervention_executed(led) == 2
    # DEFER_FINAL: on budget exhaustion the answer is DELIVERED, so nothing was deferred
    A.record_defer(led, "before_final", 3, deferred=True, repair_turn_started=False, exhausted=True)
    assert A.n_intervention_executed(led) == 2
    A.record_defer(led, "before_final", 4, deferred=True, repair_turn_started=True)
    assert A.n_intervention_executed(led) == 3
    assert A.episode_summary(led)["n_intervention_executed"] == 3


def test_the_breakdown_by_capability_is_sortable_when_an_owner_is_none():
    """A read no capability raised (capability None) next to a capability's REVISE: the breakdown keys the None owner as
    "null" (what JSON writes for a None key), so sorting it works and the JSON is what a None key gives. Without this,
    sorting the breakdown raises TypeError (None against a string)."""
    import json
    led = _TelemetryLedger()
    led.acquire_telemetry.append({"requested": True, "attempted": True, "executed": True, "capability": None})
    A.record_revision(led, "before_action", 1, capability="scope_evidence", delivered=True, repair_turn_started=True)
    by = A.episode_summary(led)["n_intervention_executed_by_capability"]
    assert by == {"null": 1, "scope_evidence": 1}
    assert json.dumps(by, sort_keys=True) == '{"null": 1, "scope_evidence": 1}'
    assert json.dumps({None: 1}) == json.dumps({"null": 1})


def test_the_breakdown_by_capability_sums_to_the_total_and_filters_by_owner():
    led = _TelemetryLedger()
    for i, cap in enumerate(("obligation_lifecycle", "scope_evidence", "obligation_lifecycle")):
        A.record_revision(led, "before_action", i, capability=cap, delivered=True, repair_turn_started=True)
    led.acquire_telemetry.append({"requested": True, "attempted": True, "executed": True,
                                  "capability": "fhir_empty_targeted_search_broad_reacquire"})
    s = A.episode_summary(led)
    by = s["n_intervention_executed_by_capability"]
    assert by == {"obligation_lifecycle": 2, "scope_evidence": 1, "fhir_empty_targeted_search_broad_reacquire": 1}
    assert sum(by.values()) == s["n_intervention_executed"] == 4
    assert A.n_intervention_executed(led, capability="scope_evidence") == 1
    assert A.n_intervention_executed(led, capability="not_a_capability") == 0
