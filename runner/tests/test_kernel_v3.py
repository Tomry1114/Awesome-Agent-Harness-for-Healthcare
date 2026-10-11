"""Unit tests for Bounded Clinical Recovery v3 kernel (standalone; sys.exit(0) on pass, non-zero on fail).

Run: python3 runner/tests/test_kernel_v3.py

Uses STUB workflow/benchmark objects only - no model calls, deterministic, oracle-blind.
Asserts:
  (b) field classification: operational whitelist vs default-deny semantic
  (d) no commitment                 -> DECLINED_NO_COMMITMENT (not FAILED)
  (f) metrics buckets map correctly (correctly_blocked != failed)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import bindings as B
from workspace import H0, Workspace

_RC = Workspace(H0["gui"])
M = _RC.module("recovery/metrics.py")
RecoveryKernel = _RC.module("recovery/kernel.py").RecoveryKernel
WorkflowRegistry = _RC.module("recovery/registry.py").WorkflowRegistry
_C = _RC.module("recovery/contracts.py")
CommittedGoal, EpisodeResult = _C.CommittedGoal, _C.EpisodeResult
BLOCKED_NEEDS_DECISION, DECLINED_NO_COMMITMENT, VERIFIED, FAILED, NOT_APPLICABLE, UNKNOWN = (
    _C.BLOCKED_NEEDS_DECISION, _C.DECLINED_NO_COMMITMENT, _C.VERIFIED, _C.FAILED, _C.NOT_APPLICABLE, _C.UNKNOWN)


# -------------------------------------------------------------------------------------------------
# STUBS
# -------------------------------------------------------------------------------------------------
class StubWorkflow(object):
    def __init__(self, matches=True):
        self._matches = matches

    def match_goal(self, goal, ctx):
        return self._matches


class StubBenchmark(object):
    def __init__(self, commitments):
        self._commitments = commitments

    def context(self, task):
        return {"schema": {"operational_fields": ["episode_id", "idempotency_key"]}}

    def resolve_commitments(self, root, trajectory, goal, judge, ctx):
        return list(self._commitments)

    def should_trigger(self, lifecycle_event):
        return True

    def state_path(self, logical_name):
        return "state.%s" % logical_name


def _run(commitments, workflow):
    reg = WorkflowRegistry([workflow]) if workflow is not None else WorkflowRegistry([])
    bench = StubBenchmark(commitments)
    k = RecoveryKernel()
    return k.run_all_episodes(bench, reg, None, driver=None,
                              task={"task_id": "t1"}, trajectory=[], goal="g", judge=None)[0]


FAILS = []


def check(name, cond, extra=""):
    if cond:
        print("PASS %s" % name)
    else:
        print("FAIL %s %s" % (name, extra))
        FAILS.append(name)


# -------------------------------------------------------------------------------------------------
# (b) field classification: an operational whitelist entry vs a default-deny semantic field
# -------------------------------------------------------------------------------------------------
def test_b_field_classification():
    schema = {"operational_fields": ["episode_id"]}
    check("b_classify_op", B.classify_field("episode_id", schema) == B.OPERATIONAL)
    check("b_classify_sem", B.classify_field("drug", schema) == B.SEMANTIC)


# -------------------------------------------------------------------------------------------------
# (d) no commitment -> DECLINED_NO_COMMITMENT (not FAILED)
# -------------------------------------------------------------------------------------------------
def test_d_no_commitment():
    res = _run([], StubWorkflow())
    check("d_state", res.state == DECLINED_NO_COMMITMENT, res.state)
    check("d_not_failed", res.state != FAILED)
    check("d_bucket", M.classify(res) == M.CORRECTLY_BLOCKED, M.classify(res))
    check("d_not_eligible", M.is_eligible(res) is False, M.is_eligible(res))


# -------------------------------------------------------------------------------------------------
# (f) metrics buckets: correctly_blocked != failed; NOT_APPLICABLE routing
# -------------------------------------------------------------------------------------------------
def test_f_metrics_buckets():
    blocked = EpisodeResult(state=BLOCKED_NEEDS_DECISION)
    failed = EpisodeResult(state=FAILED)
    verified = EpisodeResult(state=VERIFIED)
    unknown = EpisodeResult(state=UNKNOWN)
    check("f_blocked", M.classify(blocked) == M.CORRECTLY_BLOCKED, M.classify(blocked))
    check("f_failed", M.classify(failed) == M.FAILED_RECOVERY, M.classify(failed))
    check("f_distinct", M.classify(blocked) != M.classify(failed))
    check("f_verified", M.classify(verified) == M.VERIFIED_RECOVERY)
    check("f_unknown", M.classify(unknown) == M.UNKNOWN_RECOVERY)
    check("f_na_blocked", M.classify(EpisodeResult(state=NOT_APPLICABLE)) == M.CORRECTLY_BLOCKED)
    # tally
    counts = M.tally([blocked, failed, verified, unknown])
    check("f_tally", counts[M.CORRECTLY_BLOCKED] == 1 and counts[M.FAILED_RECOVERY] == 1
          and counts[M.VERIFIED_RECOVERY] == 1 and counts[M.UNKNOWN_RECOVERY] == 1, counts)


# -------------------------------------------------------------------------------------------------
# bonus: no matching workflow -> NOT_APPLICABLE (a correct refusal, not FAILED)
# -------------------------------------------------------------------------------------------------
def test_g_not_applicable():
    goal = CommittedGoal(goal_id="g4", committed_fields={})
    res = _run([goal], StubWorkflow(matches=False))
    check("g_na", res.state == NOT_APPLICABLE, res.state)
    check("g_na_not_failed", res.state != FAILED)
    check("g_na_bucket", M.classify(res) == M.CORRECTLY_BLOCKED)


def main():
    for fn in [test_b_field_classification, test_d_no_commitment, test_f_metrics_buckets,
               test_g_not_applicable]:
        fn()
    if FAILS:
        print("\n%d CHECK(S) FAILED: %s" % (len(FAILS), FAILS))
        sys.exit(1)
    print("\nALL CHECKS PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
