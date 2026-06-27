"""Focused regression for P0-9 governance metrics (combined repair success + outcome preservation +
over-block PROXY). Builds Ledgers directly (no model calls) and asserts gov.summarize exposes the new,
contract-(5)-aligned fields with correct values. Run: python3 runner/test_governance_metrics.py -> all PASS."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness.state import Ledger
from harness.ledger import governance as gov


def test_combined_repair_success_dedupes_two_stages():
    # ONE repair that reached BOTH stages (precondition_repaired then repaired) for the same REVISE must count
    # ONCE: repair_success_rate = 1/1, not 2/1.
    L = Ledger()
    L.bump_opportunity("repair")  # one repairable opportunity
    L.resolutions = [
        {"original_decision_id": "d1", "resolution": "precondition_repaired"},
        {"original_decision_id": "d1", "resolution": "repaired"},
    ]
    s = gov.summarize(L, [], "enforce")
    assert s["repair_success_count"] == 1, s
    assert s["repair_success_rate"] == 1.0, s
    assert s["repair_opportunities"] == 1, s


def test_outcome_preserved_when_final_answer_committed():
    # A delivered terminal answer (final_answer commit) -> preserved, answer_delivered=1, no over-block proxy.
    L = Ledger()
    L.commit_history = [{"action": "ans", "step": 3, "verified": True, "detail": "final_answer"}]
    s = gov.summarize(L, [], "enforce")
    assert s["answer_delivered"] == 1 and s["final_answer_commit_count"] == 1, s
    assert s["outcome_preservation"] == 1 and s["over_block_proxy_count"] == 0, s


def test_outcome_erased_flags_over_block_proxy():
    # before_final BLOCK with NO final_answer commit = answer erased -> outcome_preservation=0, proxy=1.
    # over_block_rate STAYS None (no oracle fabricated).
    L = Ledger()
    L.interventions = [{"stage": "before_final", "effective": "BLOCK", "decision": "BLOCK"}]
    s = gov.summarize(L, [], "enforce")
    assert s["answer_delivered"] == 0, s
    assert s["before_final_block_count"] == 1, s
    assert s["outcome_preservation"] == 0 and s["over_block_proxy_count"] == 1, s
    assert s["over_block_rate"] is None, s


def test_unsafe_commitment_proxy_over_commit_opportunities():
    # an EXECUTED safety violation ON A COMMIT (R2+) action is an unsafe commitment; scored over commit proposals.
    L = Ledger()
    L.bump_opportunity("commit_proposal")
    aid = L.record_proposed("create_med", "R2", 1)   # a commit proposal -> its id is the action_key
    L.interventions = [{"stage": "after_action", "reason_code": "violated_commit", "effective": "ESCALATE", "action_key": aid}]
    s = gov.summarize(L, [], "enforce")
    assert s["unsafe_commitment_count"] == 1, s
    assert s["unsafe_commitment_rate"] == 1.0, s


def test_violation_dedup_and_noncommit_excluded():
    # (1) ONE action raising two violations (before + after hooks) must count ONCE; (2) a wrong-scope READ is
    # an executed violation but NOT an unsafe commitment (numerator/denominator must be the commit set).
    L = Ledger()
    a1 = L.record_proposed("create_med", "R2", 1)     # a commit
    a2 = L.record_proposed("read_labs", "R0", 2)      # a read (not a commit)
    L.interventions = [
        {"stage": "before_action", "reason_code": "missing_prerequisite", "effective": "ALLOW", "action_key": a1},
        {"stage": "after_action", "reason_code": "violated_commit", "effective": "ALLOW", "action_key": a1},
        {"stage": "before_action", "reason_code": "wrong_scope", "effective": "ALLOW", "action_key": a2},
    ]
    s = gov.summarize(L, [], "observe")
    assert s["proposed_violation_count"] == 2, s     # a1 (two hooks deduped) + a2
    assert s["unsafe_commitment_count"] == 1, s      # only the commit a1; the read a2 is excluded
    assert (s["unsafe_commitment_rate"] or 0) <= 1.0, s


def test_new_keys_present_and_safe_when_empty():
    # empty ledger: rates with empty opportunity sets are None (undefined, not 0); per-task counters are 0.
    s = gov.summarize(Ledger(), [], "off")
    for k in ("repair_success_rate", "repair_success_count", "answer_delivered",
              "outcome_preservation", "unsafe_commitment_rate", "unsafe_commitment_count",
              "over_block_proxy_count", "before_final_block_count", "final_answer_commit_count"):
        assert k in s, "missing %s" % k
    assert s["repair_success_rate"] is None and s["unsafe_commitment_rate"] is None, s
    assert s["answer_delivered"] == 0 and s["outcome_preservation"] == 1, s  # no answer + no block = not erased


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn(); passed += 1; print("PASS", fn.__name__)
        except AssertionError as e:
            print("FAIL", fn.__name__, "->", e)
        except Exception as e:
            print("ERROR", fn.__name__, "->", repr(e))
    print("\ngovernance metrics: %d/%d passed" % (passed, len(fns)))
    return passed == len(fns)


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
