"""Unit tests for Bounded Clinical Recovery v3 kernel (standalone; sys.exit(0) on pass, non-zero on fail).

Run: python3 runner/harness/recovery/test_kernel_v3.py

Uses STUB substrate/workflow/benchmark objects only - no model calls, deterministic, oracle-blind.
Asserts:
  (a) unbound semantic arg          -> BLOCKED_NEEDS_DECISION
  (b) system_metadata binds an operational field but is REJECTED for a semantic field
  (c) >1 irreversible_commit w/o transaction_contract -> compile error / FAILED
  (d) no commitment                 -> DECLINED_NO_COMMITMENT (not FAILED)
  (e) happy 4-step plan read->probe->irreversible_commit->verify -> VERIFIED
  (f) metrics buckets map correctly (correctly_blocked != failed)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from harness.recovery import (
    contracts as C, bindings as B, metrics as M,
    RecoveryKernel, WorkflowRegistry,
    CommittedGoal, RecoveryStep, Plan, Outcome, AffordanceBinding, EpisodeResult,
    validate_plan, PlanCompileError,
    BLOCKED_NEEDS_DECISION, DECLINED_NO_COMMITMENT, VERIFIED, FAILED, NOT_APPLICABLE, UNKNOWN,
)


# -------------------------------------------------------------------------------------------------
# STUBS
# -------------------------------------------------------------------------------------------------
class StubSubstrate(object):
    """Environment mechanics stub. Configurable per-kind Outcome + affordance behavior."""

    def __init__(self, affordance="ok", commit_class=C.RESULT_OK, probe_class=C.RESULT_OK,
                 created_id="Resource/1"):
        self.affordance = affordance        # "ok" | a BLOCKED_* string
        self.commit_class = commit_class
        self.probe_class = probe_class
        self.created_id = created_id
        self.calls = []

    def resolve_affordance(self, target_spec, observation):
        if self.affordance == "ok":
            return AffordanceBinding(target_spec=target_spec, ref="ref/%s" % target_spec, observation_hash="h")
        return self.affordance             # a BLOCKED_* terminal string

    def execute_primitive(self, kind, action, auth):
        self.calls.append((kind, auth.tier if auth is not None else None))
        if kind == C.IRREVERSIBLE_COMMIT:
            return Outcome(status=self.commit_class, created_id=self.created_id,
                           state_view={"committed": True})
        if kind in (C.READ, C.NAVIGATE, C.ACQUIRE, C.VERIFY):
            return Outcome(status=C.RESULT_OK, result={"read": kind}, state_view={"seen_%s" % kind: True})
        return Outcome(status=C.RESULT_OK, state_view={})

    def read_state(self, paths):
        return {"post": "read_back"}

    def classify_result(self, outcome):
        kind = None
        # probe reads are ordinary reads here unless configured otherwise
        st = getattr(outcome, "status", outcome)
        return st


class StubWorkflow(object):
    def __init__(self, plan, required=None, matches=True, verdict=True):
        self._plan = plan
        self._required = required or []
        self._matches = matches
        self._verdict = verdict

    def match_goal(self, goal, ctx):
        return self._matches

    def required_bindings(self, goal, ctx):
        return list(self._required)

    def compile_plan(self, goal, ctx):
        return self._plan

    def verify_effect(self, goal, state_view):
        return self._verdict


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


def _run(commitments, workflow, substrate=None):
    substrate = substrate or StubSubstrate()
    reg = WorkflowRegistry([workflow]) if workflow is not None else WorkflowRegistry([])
    bench = StubBenchmark(commitments)
    k = RecoveryKernel()
    return k.run_episode(bench, reg, substrate, driver=None,
                         task={"task_id": "t1"}, trajectory=[], goal="g", judge=None)


FAILS = []


def check(name, cond, extra=""):
    if cond:
        print("PASS %s" % name)
    else:
        print("FAIL %s %s" % (name, extra))
        FAILS.append(name)


# -------------------------------------------------------------------------------------------------
# (a) unbound semantic arg -> BLOCKED_NEEDS_DECISION
# -------------------------------------------------------------------------------------------------
def test_a_unbound_semantic_blocks():
    goal = CommittedGoal(goal_id="g1", committed_fields={})   # nothing committed
    plan = Plan(steps=[RecoveryStep(kind=C.READ, arg_specs=["drug"])])  # 'drug' is semantic, unbound
    wf = StubWorkflow(plan)
    res = _run([goal], wf)
    check("a_state", res.state == BLOCKED_NEEDS_DECISION, res.state)
    check("a_arg", res.blocked_argument == "drug", res.blocked_argument)
    check("a_path", res.path == C.PATH_DECISION, res.path)


# -------------------------------------------------------------------------------------------------
# (b) system_metadata binds an operational field but is rejected for a semantic field
# -------------------------------------------------------------------------------------------------
def test_b_operational_vs_semantic_source():
    schema = {"operational_fields": ["episode_id"]}
    # operational field -> binds from system_metadata
    ob = B.resolve_argument("episode_id", {B.SYSTEM_METADATA: {"episode_id": "e1"}}, schema=schema)
    check("b_op_bound", ob is not None and ob.source == B.SYSTEM_METADATA and ob.value == "e1",
          None if ob is None else (ob.source, ob.value))
    # SAME source rejected for a semantic field
    sb = B.resolve_argument("drug", {B.SYSTEM_METADATA: {"drug": "aspirin"}}, schema=schema)
    check("b_sem_rejected", sb is None, sb)
    # but a semantic source binds the semantic field
    sb2 = B.resolve_argument("drug", {B.AGENT_COMMITMENT: {"drug": "aspirin"}}, schema=schema)
    check("b_sem_bound", sb2 is not None and sb2.source == B.AGENT_COMMITMENT, sb2)
    # classification
    check("b_classify_op", B.classify_field("episode_id", schema) == B.OPERATIONAL)
    check("b_classify_sem", B.classify_field("drug", schema) == B.SEMANTIC)


# -------------------------------------------------------------------------------------------------
# (c) >1 irreversible_commit without transaction_contract -> compile error / FAILED
# -------------------------------------------------------------------------------------------------
def test_c_multi_commit():
    plan = Plan(steps=[
        RecoveryStep(kind=C.IRREVERSIBLE_COMMIT),
        RecoveryStep(kind=C.IRREVERSIBLE_COMMIT),
    ], transaction_contract=None)
    # direct compile-time raise
    raised = False
    try:
        validate_plan(plan)
    except PlanCompileError:
        raised = True
    check("c_raises", raised, "validate_plan did not raise")
    # kernel-level -> FAILED
    goal = CommittedGoal(goal_id="g2", committed_fields={})
    res = _run([goal], StubWorkflow(plan))
    check("c_failed", res.state == FAILED, res.state)
    check("c_bucket", M.classify(res) == M.FAILED_RECOVERY, M.classify(res))
    # with a transaction_contract it compiles (no raise)
    plan2 = Plan(steps=plan.steps, transaction_contract={"commits": ["a", "b"], "compensation": []})
    ok = True
    try:
        validate_plan(plan2)
    except PlanCompileError:
        ok = False
    check("c_contract_ok", ok, "transaction_contract still raised")


# -------------------------------------------------------------------------------------------------
# (d) no commitment -> DECLINED_NO_COMMITMENT (not FAILED)
# -------------------------------------------------------------------------------------------------
def test_d_no_commitment():
    plan = Plan(steps=[RecoveryStep(kind=C.READ)])
    res = _run([], StubWorkflow(plan))
    check("d_state", res.state == DECLINED_NO_COMMITMENT, res.state)
    check("d_not_failed", res.state != FAILED)
    check("d_bucket", M.classify(res) == M.CORRECTLY_BLOCKED, M.classify(res))
    check("d_not_eligible", M.is_eligible(res) is False, M.is_eligible(res))


# -------------------------------------------------------------------------------------------------
# (e) happy 4-step plan -> VERIFIED
# -------------------------------------------------------------------------------------------------
def test_e_happy_path():
    goal = CommittedGoal(goal_id="g3", goal_type="create_order",
                         committed_fields={"drug": "aspirin", "dose": "81mg", "patient": "Patient/1"})
    steps = [
        RecoveryStep(kind=C.READ, name="read_govern", arg_specs=["patient"]),
        RecoveryStep(kind=C.READ, name="existing_effect_probe", probe=True, arg_specs=["drug"]),
        RecoveryStep(kind=C.IRREVERSIBLE_COMMIT, name="create",
                     affordance_target="submit", arg_specs=["drug", "dose", "patient"],
                     manifest={"server_persisted": True}),
        RecoveryStep(kind=C.VERIFY, name="read_back"),
    ]
    plan = Plan(steps=steps, expected_postcondition={"paths": ["state.order"]})
    sub = StubSubstrate(commit_class=C.RESULT_OK)
    res = _run([goal], StubWorkflow(plan, verdict=True), substrate=sub)
    check("e_state", res.state == VERIFIED, (res.state, res.reason))
    check("e_completed", res.completed_steps == [0, 1, 2, 3], res.completed_steps)
    check("e_created", res.created_ids == ["Resource/1"], res.created_ids)
    check("e_auth", res.auth_status == C.AUTH_TIER_IRREVERSIBLE, res.auth_status)
    check("e_bucket", M.classify(res) == M.VERIFIED_RECOVERY, M.classify(res))
    check("e_engaged", M.is_engaged(res) is True)
    # exactly one commit executed at irreversible tier
    commit_calls = [c for c in sub.calls if c[0] == C.IRREVERSIBLE_COMMIT]
    check("e_one_commit", len(commit_calls) == 1 and commit_calls[0][1] == C.AUTH_TIER_IRREVERSIBLE,
          sub.calls)


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
    res = _run([goal], StubWorkflow(Plan(steps=[RecoveryStep(kind=C.READ)]), matches=False))
    check("g_na", res.state == NOT_APPLICABLE, res.state)
    check("g_na_not_failed", res.state != FAILED)
    check("g_na_bucket", M.classify(res) == M.CORRECTLY_BLOCKED)


# -------------------------------------------------------------------------------------------------
# bonus: affordance that cannot be located -> BLOCKED_UNRESOLVED_AFFORDANCE
# -------------------------------------------------------------------------------------------------
def test_h_affordance_block():
    goal = CommittedGoal(goal_id="g5", committed_fields={"drug": "x"})
    plan = Plan(steps=[RecoveryStep(kind=C.STAGED_WRITE, affordance_target="ghost", arg_specs=["drug"],
                                    manifest={"server_persisted": False})])
    sub = StubSubstrate(affordance=C.BLOCKED_UNRESOLVED_AFFORDANCE)
    res = _run([goal], StubWorkflow(plan, verdict=True), substrate=sub)
    check("h_aff", res.state == C.BLOCKED_UNRESOLVED_AFFORDANCE, res.state)
    check("h_aff_bucket", M.classify(res) == M.CORRECTLY_BLOCKED)


def main():
    for fn in [test_a_unbound_semantic_blocks, test_b_operational_vs_semantic_source,
               test_c_multi_commit, test_d_no_commitment, test_e_happy_path,
               test_f_metrics_buckets, test_g_not_applicable, test_h_affordance_block]:
        fn()
    total = 0
    # count checks via FAILS + printed PASS lines is implicit; report explicitly
    if FAILS:
        print("\n%d CHECK(S) FAILED: %s" % (len(FAILS), FAILS))
        sys.exit(1)
    print("\nALL CHECKS PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
