"""Conformance for the Clinical Process Harness kernel (P0). Run: python3 runner/test_harness.py
Expects all PASS. No model calls; deterministic."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness as H
from harness import decision as D
from harness.kernel import HarnessKernel
from harness.compiler import build_contract, CompilerInputs, LeakError
from harness.capabilities.scope_evidence import ScopeEvidenceBinding
from harness.capabilities.obligation_lifecycle import ObligationLifecycle
from harness.capabilities.verify_commit import VerifyAndCommit
from harness.ledger import governance as gov

# a synthetic, oracle-free policy: subject=patient, one commit point needing one obligation.
POLICY = {
    "subject": {"type": "patient", "id_context_keys": ["patient_ref"]},
    "subject_arg_keys": ["patient"],
    "read_actions": ["fhir_search"],
    "commit_actions": ["create_medication_request"],
    "evidence_obligations": [{"id": "check_allergies",
                              "satisfied_by": {"tool": "fhir_search", "resource_type": "AllergyIntolerance"}}],
    "workflow_obligations": [{"id": "med_review", "requires": ["check_allergies"]}],
    "commit_points": [{"action": "create_medication_request", "risk": "R2",
                       "requires": ["med_review"], "postcondition": "read_back"}],
    "final_risk": "R2",
}
TASK = {"task_id": "t1", "goal": "order a safe med", "context": {"patient_ref": "Patient/123"},
        "environment": {"type": "fhir"}, "source_benchmark": "PhysicianBench"}


def _kernel(mode):
    contract = build_contract(TASK, env_type="fhir", policy=POLICY)
    caps = [ScopeEvidenceBinding(), ObligationLifecycle(), VerifyAndCommit()]
    from harness.risk import classify_risk
    return HarnessKernel(contract, caps, mode=mode, policy=POLICY, env_type="fhir",
                         risk_of=lambda a: classify_risk(a, contract, POLICY))


def test_decision_priority():
    c = D.combine([D.HarnessDecision(D.ALLOW), D.HarnessDecision(D.REVISE), D.HarnessDecision(D.BLOCK)])
    assert c.type == D.BLOCK, c.type
    c2 = D.combine([D.HarnessDecision(D.BLOCK), D.HarnessDecision(D.ESCALATE)])
    assert c2.type == D.ESCALATE, c2.type
    assert D.combine([]).type == D.ALLOW


def test_contract_compiled_subject_and_obligations():
    contract = build_contract(TASK, env_type="fhir", policy=POLICY)
    assert contract.subject == {"type": "patient", "id": "Patient/123"}, contract.subject
    assert "check_allergies" in contract.obligation_ids()
    assert contract.commit_point_for("create_medication_request")["risk"] == "R2"


def test_scope_block_enforce_vs_observe():
    # action targets a FOREIGN patient -> Module A
    foreign = {"type": "tool", "tool": "fhir_search", "args": {"patient": "Patient/456"}}
    # enforce -> BLOCK
    k = _kernel("enforce")
    eff = k.before_action(foreign, env_state=None, step=0)
    assert eff.type == D.BLOCK, eff.type
    assert eff.feedback and "subject_scope_mismatch" in str(eff.feedback)
    # observe -> effective ALLOW but the would-be BLOCK is recorded
    k2 = _kernel("observe")
    eff2 = k2.before_action(foreign, env_state=None, step=0)
    assert eff2.type == D.ALLOW, eff2.type
    assert any(iv["decision"] == "BLOCK" for iv in k2.ledger.interventions)


def test_in_scope_action_allowed():
    k = _kernel("enforce")
    ok = {"type": "tool", "tool": "fhir_search", "args": {"patient": "Patient/123", "resource_type": "AllergyIntolerance"}}
    assert k.before_action(ok, step=0).type == D.ALLOW


def test_commit_requires_obligation_then_satisfied():
    k = _kernel("enforce")
    commit = {"type": "tool", "tool": "create_medication_request", "args": {"patient": "Patient/123"}}
    # before doing the allergy check -> REVISE with missing obligation
    eff = k.before_action(commit, step=0)
    assert eff.type == D.REVISE, eff.type
    assert "check_allergies" in str(eff.feedback) or "med_review" in str(eff.feedback)
    # do the allergy check -> obligation satisfied
    allergy = {"type": "tool", "tool": "fhir_search", "args": {"patient": "Patient/123", "resource_type": "AllergyIntolerance"}}
    k.before_action(allergy, step=1)
    k.after_action(allergy, "AllergyIntolerance: penicillin", before_state={"a": 0}, after_state={"a": 1}, step=1)
    assert k.ledger.obligation_state("check_allergies") == "SATISFIED"
    assert k.ledger.obligation_state("med_review") == "SATISFIED"
    # now the commit passes the prerequisite gate
    assert k.before_action(commit, step=2).type == D.ALLOW


def test_final_answer_commit_gate():
    # a contract whose FINAL commit point requires an unmet obligation -> before_final REVISEs
    pol = {"subject": {"type": "medical_image", "id_context_keys": []},
           "evidence_obligations": [{"id": "img_evidence", "satisfied_by": {"tool": "ImageDescription"}}],
           "workflow_obligations": [{"id": "grounded", "requires": ["img_evidence"]}],
           "commit_points": [{"action": "final", "risk": "R2", "requires": ["grounded"]}],
           "final_risk": "R2"}
    task = {"task_id": "m0", "goal": "read the scan", "context": {}, "environment": {"type": "tool_sandbox"},
            "source_benchmark": "MedCTA"}
    contract = build_contract(task, env_type="tool_sandbox", policy=pol)
    from harness.risk import classify_risk
    k = HarnessKernel(contract, [ObligationLifecycle(), VerifyAndCommit()], mode="enforce", policy=pol,
                      env_type="tool_sandbox", risk_of=lambda a: classify_risk(a, contract, pol))
    # answer WITHOUT having produced image evidence -> REVISE
    eff = k.before_final("the mass is benign", step=0)
    assert eff.type == D.REVISE, eff.type
    assert "grounded" in str(eff.feedback) or "img_evidence" in str(eff.feedback)
    # produce image evidence -> grounded becomes satisfiable -> final allowed
    img = {"type": "tool", "tool": "ImageDescription", "args": {}}
    k.after_action(img, "a 3cm mass in the RUL", before_state=None, after_state=None, step=1)
    assert k.ledger.obligation_state("img_evidence") == "SATISFIED"
    assert k.before_final("the mass is benign", step=2).type == D.ALLOW


def test_assist_downgrades_block_to_revise():
    foreign = {"type": "tool", "tool": "fhir_search", "args": {"patient": "Patient/999"}}
    eff = _kernel("assist").before_action(foreign, step=0)
    assert eff.type == D.REVISE, eff.type        # assist never hard-blocks


def test_leak_firewall():
    # forbidden key in observed -> LeakError
    try:
        build_contract(TASK, env_type="fhir", policy=POLICY, observed=[{"gold_answer": "x"}])
        assert False, "leak not caught"
    except LeakError:
        pass
    # forbidden key in policy -> LeakError
    try:
        build_contract(TASK, env_type="fhir", policy=dict(POLICY, reference_trajectory=[1, 2]))
        assert False, "leak not caught"
    except LeakError:
        pass
    # a task carrying checkpoints/outcome is fine: those fields are whitelisted OUT, never seen
    dirty = dict(TASK, checkpoints=[{"id": "cp1"}], success=True, reference=[1])
    inp = CompilerInputs(dirty, env_type="fhir", policy=POLICY)
    assert not hasattr(inp, "checkpoints") and inp.goal == TASK["goal"]


def test_off_mode_returns_none():
    assert H.build_kernel(TASK, bench="PhysicianBench", env_type="fhir", mode="off") is None


def test_budget_caps_interventions():
    k = _kernel("enforce"); k.budget["max_interventions_per_task"] = 2
    foreign = {"type": "tool", "tool": "fhir_search", "args": {"patient": "Patient/456"}}
    effs = [k.before_action(foreign, step=i) for i in range(5)]
    n_block = sum(1 for e in effs if e.type == D.BLOCK)
    assert n_block == 2, n_block               # only 2 effective blocks; rest budget-exhausted -> ALLOW


def test_governance_summary_counts_wrong_scope():
    k = _kernel("enforce")
    k.before_action({"type": "tool", "tool": "fhir_search", "args": {"patient": "Patient/456"}}, step=0)
    s = gov.summarize(k.ledger, [], mode="enforce")
    assert s["wrong_scope_action_rate"] > 0, s


def test_capability_error_does_not_crash():
    class Boom(ScopeEvidenceBinding):
        def before_action(self, action, ctx):
            raise RuntimeError("boom")
    from harness.risk import classify_risk
    contract = build_contract(TASK, env_type="fhir", policy=POLICY)
    k = HarnessKernel(contract, [Boom()], mode="enforce", policy=POLICY, env_type="fhir",
                      risk_of=lambda a: classify_risk(a, contract, POLICY))
    assert k.before_action({"type": "tool", "tool": "fhir_search", "args": {}}, step=0).type == D.ALLOW


def test_p1_physicianbench_real_pack():
    """P1: the REAL physicianbench.yaml on real-shaped PB actions (no synthetic inline policy).
    Exercises wrong-patient BLOCK, medication-safety prerequisite REVISE -> satisfy -> ALLOW, and
    pattern-based risk (fhir_medication_request_create = R2; a search = R0)."""
    from harness.engines.policy import load_policy
    from harness.risk import classify_risk
    pol = load_policy(bench="PhysicianBench", env_type="fhir")
    assert pol.get("_pack_name") == "physicianbench", pol.get("_pack_name")
    task = {"task_id": "pb", "goal": "manage aberrant UDS",
            "context": {"patient_ref": "MRN6025656705"}, "environment": {"type": "fhir"},
            "source_benchmark": "PhysicianBench"}
    contract = build_contract(task, env_type="fhir", policy=pol)
    assert contract.subject == {"type": "patient", "id": "MRN6025656705"}, contract.subject
    k = HarnessKernel(contract, [ScopeEvidenceBinding(), ObligationLifecycle(), VerifyAndCommit()],
                      mode="enforce", policy=pol, env_type="fhir",
                      risk_of=lambda a: classify_risk(a, contract, pol))
    own = {"type": "tool", "tool": "fhir_observation_search_labs", "args": {"patient": "MRN6025656705"}}
    wrong = {"type": "tool", "tool": "fhir_observation_search_labs", "args": {"patient": "MRN9999999999"}}
    create = {"type": "tool", "tool": "fhir_medication_request_create",
              "args": {"patient": "MRN6025656705", "medication": "amoxicillin"}}
    assert k.before_action(wrong, step=0).type == D.BLOCK, "wrong patient must BLOCK"
    assert k.before_action(own, step=1).type == D.ALLOW, "own patient read ok"
    eff = k.before_action(create, step=2)
    assert eff.type == D.REVISE and "check_allergies" in str(eff.feedback), eff.feedback
    # satisfy the safety review with the real resource-specific tools
    allergy = {"type": "tool", "tool": "fhir_allergy_intolerance_search_active", "args": {"patient": "MRN6025656705"}}
    meds = {"type": "tool", "tool": "fhir_medication_request_search_orders", "args": {"patient": "MRN6025656705"}}
    k.after_action(allergy, "AllergyIntolerance: penicillin", {"x": 0}, {"x": 1}, step=3)
    k.after_action(meds, "MedicationRequest: lisinopril", {"x": 1}, {"x": 2}, step=4)
    assert k.ledger.obligation_state("check_allergies") == "SATISFIED"
    assert k.ledger.obligation_state("check_current_medications") == "SATISFIED"
    assert k.ledger.obligation_state("medication_safety_review") == "SATISFIED"
    assert k.before_action(create, step=5).type == D.ALLOW, "prerequisites met -> commit allowed"
    assert classify_risk(create, contract, pol) == "R2"
    assert classify_risk(own, contract, pol) == "R0"


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
    print("\nharness conformance: %d/%d passed" % (passed, len(fns)))
    return passed == len(fns)


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
