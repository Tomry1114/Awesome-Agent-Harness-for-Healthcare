"""Bounded Clinical Recovery v3 - structured-record (PB) stack unit tests.

Standalone: prints PASS/FAIL per check; sys.exit(0) iff every check passes, non-zero otherwise.
Run: python3 runner/harness/recovery/test_pb_v3.py

Uses a STUB in-memory record substrate (no network / no HAPI) + stub judges (no model calls). Oracle-blind:
the committed order comes only from the agent deliverable via the injected judge.

Asserts:
  (a) a FIRM committed order            -> a 4-step Plan with correctly-typed bindings.
  (b) a HEDGED order                    -> resolve_commitments returns [] (no goal, DECLINED downstream).
  (c) an existing-effect PRESENT        -> kernel returns ALREADY_REALIZED (no second create).
  (d) happy path through the REAL kernel-> VERIFIED + the record is created.
  (e) missing subject                   -> DECLINED/BLOCKED (never FAILED).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from harness.recovery import (
    contracts as C, metrics as M,
    RecoveryKernel, WorkflowRegistry,
    VERIFIED, FAILED, ALREADY_REALIZED, BLOCKED_NEEDS_DECISION, DECLINED_NO_COMMITMENT,
    IRREVERSIBLE_COMMIT,
)
from harness.recovery.substrate.fhir import FhirSubstrateAdapter
from harness.recovery.workflows.create_order import CreateOrderWorkflow
from harness.recovery.benchmark.pb import PbBenchmarkAdapter
from harness.recovery.benchmark import pb_register


# ------------------------------------------------------------------------------------------------
# STUB in-memory record substrate backend (search / create). No I/O.
# ------------------------------------------------------------------------------------------------
class InMemoryRecordBackend(object):
    def __init__(self, existing=None, fail_search=False, fail_create=False):
        self.store = [dict(r) for r in (existing or [])]
        self.fail_search = fail_search
        self.fail_create = fail_create
        self._n = 0
        self.creates = 0
        self.searches = 0

    def search(self, resource_type, subject):
        self.searches += 1
        if self.fail_search:
            return {"status": "failed", "error": "stub_search_failure"}
        entries = []
        for r in self.store:
            if resource_type is not None and r.get("resourceType") != resource_type:
                continue
            subj = ((r.get("subject") or {}).get("reference")
                    if isinstance(r.get("subject"), dict) else r.get("subject"))
            if subject is not None and subj != subject:
                continue
            entries.append({"resource": r})
        return {"entries": entries, "status": "ok"}

    def create(self, resource):
        self.creates += 1
        if self.fail_create:
            return {"status": "failed", "error": "stub_create_failure"}
        self._n += 1
        rid = "res-%d" % self._n
        r = dict(resource)
        r["id"] = rid
        self.store.append(r)
        return {"id": rid, "created": True, "status": "ok", "resource": r}


# ------------------------------------------------------------------------------------------------
# STUB judges (deterministic; no model calls). extract_committed_orders passes them a prompt string.
# ------------------------------------------------------------------------------------------------
def firm_judge(_prompt):
    # one firm, unconditional imaging order
    return ('{"orders": [{"text": "Order pelvic ultrasound", "category": "imaging", '
            '"conditional": false}]}')


def hedged_judge(_prompt):
    # the extractor drops conditional:true items -> [] (a hedged recommendation, not a commitment)
    return ('{"orders": [{"text": "Consider pelvic ultrasound if symptoms persist", '
            '"category": "imaging", "conditional": true}]}')


def _task(with_subject=True, with_prac=True):
    text = "The current date and time is 2026-07-01T09:00:00Z."
    if with_prac:
        text += " Practitioner ID: prac-77."
    ctx = {"text": text}
    if with_subject:
        ctx["patient_ref"] = "pt-123"
    return {"task_id": "t-pb", "goal": "Evaluate and manage the patient.",
            "context": ctx, "environment": {"type": "fhir"},
            "deliverable": "Assessment complete. Order pelvic ultrasound now."}


FAILS = []


def check(name, cond, extra=""):
    if cond:
        print("PASS %s" % name)
    else:
        print("FAIL %s :: %s" % (name, extra))
        FAILS.append(name)


# ================================================================================================
# (a) firm committed order -> 4-step plan with correctly-typed bindings
# ================================================================================================
def test_a_firm_order_four_step_plan():
    bench = PbBenchmarkAdapter()
    task = _task()
    ctx = bench.context(task)
    goals = bench.resolve_commitments(task, [], task["goal"], firm_judge, ctx)
    check("a_one_goal", len(goals) == 1, "n=%d" % len(goals))
    g = goals[0]
    check("a_goal_type", g.goal_type == "create_order", g.goal_type)
    check("a_code_text_agent", g.committed_fields.get("code_text") == "Order pelvic ultrasound",
          g.committed_fields)

    wf = CreateOrderWorkflow()
    check("a_matches", wf.match_goal(g, ctx) is True)
    plan = wf.compile_plan(g, ctx)
    kinds = [s.kind for s in plan.steps]
    check("a_four_steps", len(plan.steps) == 4, kinds)
    check("a_kinds", kinds == [C.READ, C.READ, C.IRREVERSIBLE_COMMIT, C.VERIFY], kinds)
    check("a_one_commit", sum(1 for k in kinds if k == IRREVERSIBLE_COMMIT) == 1, kinds)
    check("a_probe", plan.steps[1].probe is True)
    # plan compiles clean under the <=1 irreversible_commit invariant
    C.validate_plan(plan)
    # required bindings are correctly TYPED: code_text/subject/requester SEMANTIC; the rest OPERATIONAL.
    from harness.recovery import bindings as B
    schema = ctx["schema"]
    req = wf.required_bindings(g, ctx)
    check("a_sem_code", B.classify_field("code_text", schema) == B.SEMANTIC)
    check("a_sem_subject", B.classify_field("subject", schema) == B.SEMANTIC)
    check("a_op_authored", B.classify_field("authoredOn", schema) == B.OPERATIONAL)
    check("a_op_idem", B.classify_field("idempotency_key", schema) == B.OPERATIONAL)
    check("a_req_has", ("code_text" in req and "subject" in req and "authoredOn" in req), req)
    # the create step carries a real built resource (reusing effect_completion.build_order_resource)
    create_step = plan.steps[2]
    res = (create_step.action or {}).get("resource")
    check("a_resource_built", isinstance(res, dict) and res.get("resourceType") == "ServiceRequest"
          and res.get("code", {}).get("text") == "Order pelvic ultrasound", res)
    check("a_resource_subject", isinstance(res, dict)
          and res.get("subject", {}).get("reference") == "Patient/pt-123", res)


# ================================================================================================
# (b) hedged order -> resolve_commitments returns [] (no goal)
# ================================================================================================
def test_b_hedged_no_goal():
    bench = PbBenchmarkAdapter()
    task = _task()
    ctx = bench.context(task)
    goals = bench.resolve_commitments(task, [], task["goal"], hedged_judge, ctx)
    check("b_empty", goals == [], goals)
    # end-to-end: no commitment -> DECLINED_NO_COMMITMENT (a correctly-blocked, non-failure outcome)
    backend = InMemoryRecordBackend()
    stack = pb_register.build_stack(backend=backend)
    k = RecoveryKernel()
    res = k.run_episode(stack.benchmark_adapter, stack.workflow_registry, stack.substrate_adapter,
                        driver=None, task=task, trajectory=[], goal=task["goal"], judge=hedged_judge)
    check("b_declined", res.state == DECLINED_NO_COMMITMENT, res.state)
    check("b_bucket", M.classify(res) == M.CORRECTLY_BLOCKED, M.classify(res))
    check("b_no_create", backend.creates == 0, backend.creates)


# ================================================================================================
# (c) existing-effect PRESENT -> ALREADY_REALIZED (no second create)
# ================================================================================================
def test_c_already_realized():
    # the effect is ALREADY in the store for this subject
    existing = [{"resourceType": "ServiceRequest", "status": "active", "intent": "order",
                 "subject": {"reference": "Patient/pt-123"},
                 "code": {"text": "Order pelvic ultrasound"}, "id": "pre-1"}]
    backend = InMemoryRecordBackend(existing=existing)
    stack = pb_register.build_stack(backend=backend)
    task = _task()
    k = RecoveryKernel()
    res = k.run_episode(stack.benchmark_adapter, stack.workflow_registry, stack.substrate_adapter,
                        driver=None, task=task, trajectory=[], goal=task["goal"], judge=firm_judge)
    check("c_already", res.state == ALREADY_REALIZED, res.state)
    check("c_no_create", backend.creates == 0, backend.creates)
    check("c_bucket", M.classify(res) == M.VERIFIED_RECOVERY, M.classify(res))


# ================================================================================================
# (d) happy path through the REAL kernel -> VERIFIED + record created
# ================================================================================================
def test_d_happy_path_verified():
    backend = InMemoryRecordBackend()             # empty store
    stack = pb_register.build_stack(backend=backend)
    task = _task()
    k = RecoveryKernel()
    res = k.run_episode(stack.benchmark_adapter, stack.workflow_registry, stack.substrate_adapter,
                        driver=None, task=task, trajectory=[], goal=task["goal"], judge=firm_judge)
    # THE stub-substrate happy-path VERIFIED assertion:
    assert res.state == VERIFIED and res.created_ids == ["res-1"] and backend.creates == 1, \
        "happy-path expected VERIFIED + created res-1, got state=%s created_ids=%s creates=%d" % (
            res.state, res.created_ids, backend.creates)
    check("d_verified", res.state == VERIFIED, res.state)
    check("d_created", res.created_ids == ["res-1"], res.created_ids)
    check("d_one_create", backend.creates == 1, backend.creates)
    check("d_bucket", M.classify(res) == M.VERIFIED_RECOVERY, M.classify(res))
    check("d_engaged", M.is_engaged(res) is True)
    # the record actually landed in the store with the agent's order text
    landed = [r for r in backend.store if r.get("code", {}).get("text") == "Order pelvic ultrasound"]
    check("d_record_landed", len(landed) == 1 and landed[0]["id"] == "res-1", landed)


# ================================================================================================
# (e) missing subject -> DECLINED/BLOCKED, never FAILED
# ================================================================================================
def test_e_missing_subject_blocks():
    backend = InMemoryRecordBackend()
    stack = pb_register.build_stack(backend=backend)
    task = _task(with_subject=False)              # no patient_ref -> no subject resolvable
    k = RecoveryKernel()
    res = k.run_episode(stack.benchmark_adapter, stack.workflow_registry, stack.substrate_adapter,
                        driver=None, task=task, trajectory=[], goal=task["goal"], judge=firm_judge)
    check("e_not_failed", res.state != FAILED, res.state)
    check("e_blocked", res.state in (BLOCKED_NEEDS_DECISION, DECLINED_NO_COMMITMENT), res.state)
    check("e_blocked_arg", (res.blocked_argument in (None, "subject")), res.blocked_argument)
    check("e_no_create", backend.creates == 0, backend.creates)
    check("e_bucket", M.classify(res) == M.CORRECTLY_BLOCKED, M.classify(res))


def main():
    test_a_firm_order_four_step_plan()
    test_b_hedged_no_goal()
    test_c_already_realized()
    test_d_happy_path_verified()
    test_e_missing_subject_blocks()
    print("\n%d checks, %d failures" % (
        # count is informational
        0, len(FAILS)))
    if FAILS:
        print("FAILED: %s" % FAILS)
        sys.exit(1)
    print("ALL GREEN")
    sys.exit(0)


if __name__ == "__main__":
    main()
