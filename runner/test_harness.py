"""Conformance for the Clinical Process Harness (semantic/substrate architecture).
Run: python3 runner/test_harness.py — expects all PASS. No model calls; deterministic."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness as H
from harness import decision as D
from harness.kernel import HarnessKernel
from harness.compiler import build_contract, CompilerInputs, LeakError
from harness.semantics import canonicalize
from harness.risk import classify_risk
from harness.capabilities.scope_evidence import ScopeEvidenceBinding
from harness.capabilities.obligation_lifecycle import ObligationLifecycle
from harness.capabilities.verify_commit import VerifyAndCommit
from harness.ledger import governance as gov

# a synthetic, oracle-free SUBSTRATE policy (record system): tool names live ONLY in manifest.actions.
POLICY = {
    "manifest": {
        "subject": {"type": "patient", "id_context_keys": ["patient_ref"], "from_args": ["patient"]},
        "actions": [
            {"match": {"tool_pattern": "allergy"}, "semantic_type": "read", "effect": "none",
             "resource": "AllergyIntolerance", "produces_evidence": {"source_class": "record"}},
            {"match": {"tool": "fhir_search"}, "semantic_type": "read", "effect": "none",
             "produces_evidence": {"source_class": "record", "resource_from_args": ["resource_type"]}},
            {"match": {"tool": "create_medication_request"}, "semantic_type": "create",
             "effect": "irreversible", "resource": "MedicationRequest"},
        ],
    },
    "evidence_obligations": [{"id": "check_allergies",
                             "satisfied_by": {"source_class": "record", "resource": "AllergyIntolerance"}}],
    "workflow_obligations": [{"id": "med_review", "requires": ["check_allergies"]}],
    "commit_points": [{"match": {"semantic_type": "create", "resource": "MedicationRequest"}, "risk": "R2",
                       "requires": ["med_review"], "postcondition": {"type": "state_transition"}}],
}
TASK = {"task_id": "t1", "goal": "order a safe med", "context": {"patient_ref": "Patient/123"},
        "environment": {"type": "fhir"}}


def _kernel(mode, policy=POLICY, task=TASK, env="fhir"):
    contract = build_contract(task, env_type=env, policy=policy)
    caps = [ScopeEvidenceBinding(), ObligationLifecycle(), VerifyAndCommit()]
    return HarnessKernel(contract, caps, mode=mode, policy=policy, env_type=env)


def _sem(action, policy=POLICY):
    return canonicalize(action, policy.get("manifest"))


def test_decision_priority():
    c = D.combine([D.HarnessDecision(D.ALLOW), D.HarnessDecision(D.REVISE), D.HarnessDecision(D.BLOCK)])
    assert c.type == D.BLOCK, c.type
    assert D.combine([D.HarnessDecision(D.BLOCK), D.HarnessDecision(D.ESCALATE)]).type == D.ESCALATE
    assert D.combine([]).type == D.ALLOW


def test_contract_compiled_subject_and_obligations():
    contract = build_contract(TASK, env_type="fhir", policy=POLICY)
    assert contract.subject == {"type": "patient", "id": "Patient/123"}, contract.subject
    assert "check_allergies" in contract.obligation_ids()
    create = {"type": "tool", "tool": "create_medication_request", "args": {"patient": "Patient/123"}}
    assert contract.commit_point_for(_sem(create))["risk"] == "R2"


def test_canonicalize_effect_and_risk():
    read = _sem({"type": "tool", "tool": "fhir_search", "args": {}})
    create = _sem({"type": "tool", "tool": "create_medication_request", "args": {}})
    assert read.semantic_type == "read" and read.effect == "none"
    assert create.semantic_type == "create" and create.effect == "irreversible"
    contract = build_contract(TASK, env_type="fhir", policy=POLICY)
    assert classify_risk(read, contract) == "R0"
    assert classify_risk(create, contract) == "R2"


def test_scope_block_enforce_vs_observe():
    foreign = {"type": "tool", "tool": "fhir_search", "args": {"patient": "Patient/456"}}
    k = _kernel("enforce")
    eff = k.before_action(foreign, step=0)
    assert eff.type == D.BLOCK, eff.type
    assert eff.feedback and "subject_scope_mismatch" in str(eff.feedback)
    k2 = _kernel("observe")
    assert k2.before_action(foreign, step=0).type == D.ALLOW
    assert any(iv["decision"] == "BLOCK" for iv in k2.ledger.interventions)


def test_in_scope_action_allowed():
    k = _kernel("enforce")
    ok = {"type": "tool", "tool": "fhir_search", "args": {"patient": "Patient/123", "resource_type": "AllergyIntolerance"}}
    assert k.before_action(ok, step=0).type == D.ALLOW


def test_commit_requires_obligation_then_satisfied():
    k = _kernel("enforce")
    commit = {"type": "tool", "tool": "create_medication_request", "args": {"patient": "Patient/123"}}
    assert k.before_action(commit, step=0).type == D.REVISE
    allergy = {"type": "tool", "tool": "fhir_search", "args": {"patient": "Patient/123", "resource_type": "AllergyIntolerance"}}
    k.before_action(allergy, step=1)
    k.after_action(allergy, "penicillin allergy", {"a": 0}, {"a": 1}, step=1)
    assert k.ledger.obligation_state("check_allergies") == "SATISFIED"
    assert k.ledger.obligation_state("med_review") == "SATISFIED"
    assert k.before_action(commit, step=2).type == D.ALLOW


def test_assist_downgrades_block_to_revise():
    foreign = {"type": "tool", "tool": "fhir_search", "args": {"patient": "Patient/999"}}
    assert _kernel("assist").before_action(foreign, step=0).type == D.REVISE


def test_leak_firewall():
    try:
        build_contract(TASK, env_type="fhir", policy=POLICY, observed=[{"gold_answer": "x"}]); assert False
    except LeakError:
        pass
    try:
        build_contract(TASK, env_type="fhir", policy=dict(POLICY, reference_trajectory=[1])); assert False
    except LeakError:
        pass
    dirty = dict(TASK, checkpoints=[{"id": "cp1"}], success=True, reference=[1])
    inp = CompilerInputs(dirty, env_type="fhir", policy=POLICY)
    assert not hasattr(inp, "checkpoints") and inp.goal == TASK["goal"]


def test_off_mode_returns_none():
    assert H.build_kernel(TASK, env_type="fhir", mode="off") is None


def test_budget_escalates_not_allows():
    k = _kernel("enforce"); k.budget["max_interventions_per_task"] = 2
    foreign = {"type": "tool", "tool": "fhir_search", "args": {"patient": "Patient/456"}}
    effs = [k.before_action(foreign, step=i) for i in range(5)]
    assert sum(1 for e in effs if e.type == D.BLOCK) == 2
    assert all(e.type != D.ALLOW for e in effs), "exhausted budget must NEVER silently ALLOW a BLOCK"
    assert effs[-1].type == D.ESCALATE, "over-budget -> ESCALATE (resource limit, not safety override)"


def test_opportunity_denominators():
    k = _kernel("enforce")
    k.before_action({"type": "tool", "tool": "fhir_search", "args": {"patient": "Patient/456"}}, step=0)
    s = gov.summarize(k.ledger, [], mode="enforce")
    assert s["wrong_scope_action_rate"] == 1.0 and s["wrong_scope_opportunities"] == 1, s
    # an action with no subject -> no subject-bearing opportunity (denominator stays honest)
    k2 = _kernel("enforce")
    s2 = gov.summarize(k2.ledger, [], mode="enforce")
    assert s2["wrong_scope_action_rate"] is None, s2


def test_failed_result_does_not_satisfy_obligation():
    """A failed/empty tool result is ATTEMPTED evidence, not VALIDATED -> it cannot satisfy an obligation."""
    k = _kernel("enforce")
    allergy = {"type": "tool", "tool": "fhir_search", "args": {"patient": "Patient/123", "resource_type": "AllergyIntolerance"}}
    k.after_action(allergy, "{error: HTTP 500}", {"a": 0}, {"a": 0}, step=1, result_ok=False)
    assert k.ledger.obligation_state("check_allergies") != "SATISFIED", "error result must not satisfy"
    k.after_action(allergy, "penicillin allergy", {"a": 0}, {"a": 1}, step=2, result_ok=True)
    assert k.ledger.obligation_state("check_allergies") == "SATISFIED"


def test_unverifiable_commit_not_recorded_verified():
    """A commit whose state is unobservable -> verification None (UNKNOWN), NOT verified=True."""
    k = _kernel("enforce")
    allergy = {"type": "tool", "tool": "fhir_search", "args": {"patient": "Patient/123", "resource_type": "AllergyIntolerance"}}
    k.after_action(allergy, "penicillin", {"a": 0}, {"a": 1}, step=1, result_ok=True)   # satisfy med_review
    create = {"type": "tool", "tool": "create_medication_request", "args": {"patient": "Patient/123"}}
    assert k.before_action(create, step=2).type == D.ALLOW
    k.after_action(create, "created", None, None, step=2, result_ok=True)   # unobservable state
    assert k.ledger.commit_history[-1]["verified"] is None, k.ledger.commit_history[-1]


def test_capability_error_escalates_in_enforce():
    """Fail-closed: a crashing capability must NOT silently become ALLOW. Under enforce it ESCALATEs
    (and is recorded as a capability error); under observe it records but stays effective-ALLOW."""
    class Boom(ScopeEvidenceBinding):
        def before_action(self, action, ctx):
            raise RuntimeError("boom")
    contract = build_contract(TASK, env_type="fhir", policy=POLICY)
    k = HarnessKernel(contract, [Boom()], mode="enforce", policy=POLICY, env_type="fhir")
    assert k.before_action({"type": "tool", "tool": "fhir_search", "args": {}}, step=0).type == D.ESCALATE
    assert k._capability_errors and k.audit()["status"] == "degraded"
    k2 = HarnessKernel(contract, [Boom()], mode="observe", policy=POLICY, env_type="fhir")
    assert k2.before_action({"type": "tool", "tool": "fhir_search", "args": {}}, step=0).type == D.ALLOW
    assert k2._capability_errors and k2.audit()["status"] == "degraded"


# ---- real SUBSTRATE packs (P1/P2/P3): same harness core, only adapter + pack differ ----------------

def test_p1_structured_record_pack():
    pol = H.load_policy(env_type="fhir")
    assert pol.get("_substrate") == "structured_record"
    task = {"task_id": "pb", "goal": "manage UDS", "context": {"patient_ref": "MRN6025656705"},
            "environment": {"type": "fhir"}}
    contract = build_contract(task, env_type="fhir", policy=pol)
    assert contract.subject == {"type": "patient", "id": "MRN6025656705"}, contract.subject
    k = HarnessKernel(contract, [ScopeEvidenceBinding(), ObligationLifecycle(), VerifyAndCommit()],
                      mode="enforce", policy=pol, env_type="fhir")
    wrong = {"type": "tool", "tool": "fhir_observation_search_labs", "args": {"patient": "MRN9999999999"}}
    own = {"type": "tool", "tool": "fhir_observation_search_labs", "args": {"patient": "MRN6025656705"}}
    create = {"type": "tool", "tool": "fhir_medication_request_create", "args": {"patient": "MRN6025656705"}}
    assert k.before_action(wrong, step=0).type == D.BLOCK
    assert k.before_action(own, step=1).type == D.ALLOW
    assert k.before_action(create, step=2).type == D.REVISE
    allergy = {"type": "tool", "tool": "fhir_allergy_intolerance_search_active", "args": {"patient": "MRN6025656705"}}
    meds = {"type": "tool", "tool": "fhir_medication_request_search_orders", "args": {"patient": "MRN6025656705"}}
    k.after_action(allergy, "penicillin", {"x": 0}, {"x": 1}, step=3)
    k.after_action(meds, "lisinopril", {"x": 1}, {"x": 2}, step=4)
    assert k.ledger.obligation_state("medication_safety_review") == "SATISFIED"
    assert k.before_action(create, step=5).type == D.ALLOW
    assert classify_risk(canonicalize(create, pol["manifest"]), contract) == "R2"


def test_p2_interactive_gui_pack():
    pol = H.load_policy(env_type="gui")
    assert pol.get("_substrate") == "interactive_gui"
    task = {"task_id": "hab", "goal": "Open denial DEN-001 for Martinez. Triage it.",
            "context": {"text": "Open denial DEN-001."}, "environment": {"type": "gui"}}
    contract = build_contract(task, env_type="gui", policy=pol)
    assert contract.subject == {"type": "admin_case", "id": "DEN-001"}, contract.subject
    k = HarnessKernel(contract, [ScopeEvidenceBinding(), ObligationLifecycle(), VerifyAndCommit()],
                      mode="enforce", policy=pol, env_type="gui")
    click = {"type": "tool", "tool": "click", "args": {}}
    eff = k.after_action(click, "ok", {"x": 0}, {"x": 1}, step=0, canonical_observation={"case_identity": "DEN-999"})
    assert eff.type == D.REVISE and "DEN-999" in str(eff.feedback), eff.feedback
    eff2 = k.after_action(click, "ok", {"x": 1}, {"x": 2}, step=1, canonical_observation={"case_identity": "DEN-001"})
    assert eff2.type == D.ALLOW
    submit = {"type": "tool", "tool": "submit", "args": {}}
    assert classify_risk(canonicalize(submit, pol["manifest"]), contract) == "R2"
    eff3 = k.after_action(submit, "submitted", {"status": "Draft"}, {"status": "Draft"}, step=2,
                          canonical_observation={"case_identity": "DEN-001"})
    assert eff3.type == D.REVISE


def test_p3_perceptual_tool_pack():
    pol = H.load_policy(env_type="tool_sandbox")
    assert pol.get("_substrate") == "perceptual"
    task = {"task_id": "m", "goal": "finding in the chest CT?", "context": {}, "environment": {"type": "tool_sandbox"}}
    contract = build_contract(task, env_type="tool_sandbox", policy=pol)
    k = HarnessKernel(contract, [ScopeEvidenceBinding(), ObligationLifecycle(), VerifyAndCommit()],
                      mode="enforce", policy=pol, env_type="tool_sandbox")
    assert k.before_final("a 3 cm mass", step=0).type == D.REVISE      # ungrounded
    region = {"type": "tool", "tool": "RegionAttributeDescription", "args": {}}
    k.after_action(region, "spiculated RUL nodule", None, None, step=1)
    assert k.ledger.obligation_state("image_derived_evidence") == "SATISFIED"
    # grounded, but NO judge configured -> claim-support is UNVERIFIABLE -> enforce ESCALATEs (fail-closed),
    # it does NOT pass just because grounding evidence exists. (With a judge -> see the semantic test.)
    assert k.before_final("a 3 cm mass", step=2).type == D.ESCALATE
    # external (search) evidence does NOT count as image grounding
    k2 = HarnessKernel(contract, [ScopeEvidenceBinding(), ObligationLifecycle(), VerifyAndCommit()],
                       mode="enforce", policy=pol, env_type="tool_sandbox")
    k2.after_action({"type": "tool", "tool": "GoogleSearch", "args": {}}, "web text", None, None, step=1)
    assert k2.ledger.obligation_state("image_derived_evidence") != "SATISFIED"
    assert k2.before_final("a 3 cm mass", step=2).type == D.REVISE


def test_p3_semantic_claim_support_judge():
    pol = H.load_policy(env_type="tool_sandbox")
    task = {"task_id": "m", "goal": "finding?", "context": {}, "environment": {"type": "tool_sandbox"}}
    contract = build_contract(task, env_type="tool_sandbox", policy=pol)

    def mk(judge_fn):
        k = HarnessKernel(contract, [ScopeEvidenceBinding(), ObligationLifecycle(), VerifyAndCommit()],
                          mode="enforce", policy=pol, env_type="tool_sandbox", judge_fn=judge_fn,
                          budget={"max_semantic_checks": 5})
        k.after_action({"type": "tool", "tool": "ImageDescription", "args": {}},
                       "a 3cm spiculated RUL nodule", None, None, step=0)
        return k

    assert mk(lambda p: '{"supported": true, "confidence": 0.9, "reason": "ok"}').before_final("RUL nodule", step=1).type == D.ALLOW
    assert mk(lambda p: '{"supported": false, "confidence": 0.9, "reason": "LUL vs RUL"}').before_final("LUL effusion", step=1).type == D.REVISE
    assert mk(lambda p: '{"supported": false, "confidence": 0.2, "reason": "unclear"}').before_final("maybe", step=1).type == D.ESCALATE
    nj = mk(None)   # judge unavailable: a contract that REQUIRES claim-support cannot pass in enforce
    assert nj.before_final("RUL nodule", step=1).type == D.ESCALATE
    assert any(r["rule_id"] == "semantic_claim_support" for r in nj.ledger.unresolved_risks)


def test_tool_renaming_invariance():
    """GENERALITY: rename every tool to an opaque id in the manifest; the SAME gates fire. The harness
    depends on capability SEMANTICS (manifest mapping), not on any tool name -> a new dataset that uses
    different tool names works by writing its manifest, with zero harness change."""
    pol = {
        "manifest": {
            "subject": {"type": "patient", "id_context_keys": ["patient_ref"], "from_args": ["patient"]},
            "actions": [
                {"match": {"tool": "xq7"}, "semantic_type": "read", "effect": "none",
                 "resource": "AllergyIntolerance", "produces_evidence": {"source_class": "record"}},
                {"match": {"tool": "zz9"}, "semantic_type": "create", "effect": "irreversible",
                 "resource": "MedicationRequest"},
            ],
        },
        "evidence_obligations": [{"id": "allergies",
                                 "satisfied_by": {"source_class": "record", "resource": "AllergyIntolerance"}}],
        "commit_points": [{"match": {"semantic_type": "create", "resource": "MedicationRequest"},
                           "risk": "R2", "requires": ["allergies"], "postcondition": {"type": "state_transition"}}],
    }
    task = {"task_id": "t", "goal": "g", "context": {"patient_ref": "Patient/123"}, "environment": {"type": "fhir"}}
    contract = build_contract(task, env_type="fhir", policy=pol)
    k = HarnessKernel(contract, [ScopeEvidenceBinding(), ObligationLifecycle(), VerifyAndCommit()],
                      mode="enforce", policy=pol, env_type="fhir")
    create = {"type": "tool", "tool": "zz9", "args": {"patient": "Patient/123"}}
    assert k.before_action(create, step=0).type == D.REVISE                 # missing prerequisite
    k.after_action({"type": "tool", "tool": "xq7", "args": {"patient": "Patient/123"}}, "ok", {"a": 0}, {"a": 1}, step=1)
    assert k.ledger.obligation_state("allergies") == "SATISFIED"            # satisfied by evidence class
    assert k.before_action(create, step=2).type == D.ALLOW
    assert k.before_action({"type": "tool", "tool": "xq7", "args": {"patient": "Patient/999"}}, step=3).type == D.BLOCK


def test_kernel_has_no_benchmark_names():
    """GENERALITY GUARD: the harness core must not know which benchmark it runs — it governs by SUBSTRATE.
    No benchmark proper-name may appear anywhere under runner/harness/. A new dataset works by writing an
    adapter manifest + substrate pack, with ZERO harness change."""
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "harness")
    forbidden = ("MedCTA", "PhysicianBench", "HealthAdminBench")
    hits = []
    for dp, _dn, fns in os.walk(root):
        if "__pycache__" in dp:
            continue
        for fn in fns:
            if fn.endswith(".py"):
                txt = open(os.path.join(dp, fn)).read()
                for name in forbidden:
                    if name in txt:
                        hits.append("%s:%s" % (os.path.relpath(os.path.join(dp, fn), root), name))
    assert not hits, "benchmark name leaked into harness core: %s" % hits


def test_three_layer_policy_composition():
    """The effective policy is composed from ADAPTER (manifest) + SUBSTRATE (generic) + CLINICAL modules.
    No layer is a benchmark; env_type picks a DEFAULT adapter that can be overridden."""
    pol = H.load_policy(env_type="fhir")
    assert pol["_adapter"] == "hapi_fhir" and pol["_substrate"] == "structured_record"
    assert pol["_clinical_modules"] == ["medication_safety"]
    assert pol["manifest"]["actions"], "manifest came from the adapter"
    # clinical obligations are present and clinical commit point is FIRST (matched before the substrate's
    # generic irreversible-write rule).
    assert any(o["id"] == "check_allergies" for o in pol["evidence_obligations"])
    assert pol["commit_points"][0]["match"].get("resource") == "MedicationRequest"
    assert any(cp["match"] == {"effect": "irreversible"} for cp in pol["commit_points"]), "substrate invariant present"


def test_clinical_module_scoped_to_its_resource():
    """GENERALITY: a NON-medication irreversible write (e.g. ServiceRequest) gets only the SUBSTRATE's
    generic write-verification — the medication-safety review does NOT fire. Clinical rules are scoped to
    the resources they name, not to the whole substrate."""
    pol = H.load_policy(env_type="fhir")
    task = {"task_id": "t", "goal": "place an order", "context": {"patient_ref": "Patient/1"},
            "environment": {"type": "fhir"}}
    contract = build_contract(task, env_type="fhir", policy=pol)
    k = HarnessKernel(contract, [ScopeEvidenceBinding(), ObligationLifecycle(), VerifyAndCommit()],
                      mode="enforce", policy=pol, env_type="fhir")
    # a service-request create maps to create/irreversible with NO MedicationRequest resource -> it matches
    # the substrate's generic {effect: irreversible} commit, NOT the medication clinical rule -> ALLOW
    # (no allergy review required), just a state-change postcondition.
    svc = {"type": "tool", "tool": "fhir_service_request_create", "args": {"patient": "Patient/1"}}
    assert k.before_action(svc, step=0).type == D.ALLOW, "non-medication write must not require med review"
    # contrast: a medication create DOES require the review
    med = {"type": "tool", "tool": "fhir_medication_request_create", "args": {"patient": "Patient/1"}}
    assert k.before_action(med, step=1).type == D.REVISE


def test_required_binding_rejects_unspecified_subject():
    """SAFETY: under `required` subject binding (FHIR), a commit that names NO subject must not be assumed
    to operate on the active subject -> REVISE (specify the subject)."""
    pol = H.load_policy(env_type="fhir")
    assert (pol["manifest"]["subject"].get("binding")) == "required"
    task = {"task_id": "t", "goal": "order", "context": {"patient_ref": "Patient/1"}, "environment": {"type": "fhir"}}
    contract = build_contract(task, env_type="fhir", policy=pol)
    k = HarnessKernel(contract, [ScopeEvidenceBinding(), ObligationLifecycle(), VerifyAndCommit()],
                      mode="enforce", policy=pol, env_type="fhir")
    nopt = {"type": "tool", "tool": "fhir_medication_request_create", "args": {}}   # no patient named
    assert k.before_action(nopt, step=0).type == D.REVISE
    assert any(f["reason_code"] == "subject_unspecified" for f in k.ledger.findings)
    # a no-subject READ does not satisfy an obligation either (evidence is not VALIDATED)
    rd = {"type": "tool", "tool": "fhir_allergy_intolerance_search_active", "args": {}}
    k.after_action(rd, "data", {"a": 0}, {"a": 1}, step=1, result_ok=True)
    assert k.ledger.obligation_state("check_allergies") != "SATISFIED"


def test_policy_loader_fail_closed():
    """A missing/typo'd policy layer must NOT silently vanish: it is recorded in _errors, and assist/enforce
    REFUSE to build (observe builds degraded)."""
    assert not H.load_policy(env_type="fhir").get("_errors")        # the real policy composes cleanly
    os.environ["MH_HARNESS_ADAPTER"] = "does_not_exist"
    try:
        assert H.load_policy(env_type="fhir").get("_errors"), "missing adapter must be an error"
        task = {"task_id": "t", "goal": "g", "context": {}, "environment": {"type": "fhir"}}
        try:
            H.build_kernel(task, env_type="fhir", mode="enforce"); assert False, "enforce must raise"
        except H.PolicyError:
            pass
        assert H.build_kernel(task, env_type="fhir", mode="observe") is not None   # observe builds (degraded)
    finally:
        os.environ.pop("MH_HARNESS_ADAPTER", None)


def test_findings_keep_lower_priority():
    """A hook whose winner is wrong_scope (BLOCK) must still record its missing_prerequisite finding."""
    pol = H.load_policy(env_type="fhir")
    task = {"task_id": "t", "goal": "order", "context": {"patient_ref": "Patient/1"}, "environment": {"type": "fhir"}}
    contract = build_contract(task, env_type="fhir", policy=pol)
    k = HarnessKernel(contract, [ScopeEvidenceBinding(), ObligationLifecycle(), VerifyAndCommit()],
                      mode="enforce", policy=pol, env_type="fhir")
    # a create on the WRONG patient AND missing the safety review: BLOCK wins, but both are recorded
    bad = {"type": "tool", "tool": "fhir_medication_request_create", "args": {"patient": "Patient/999"}}
    eff = k.before_action(bad, step=0)
    assert eff.type == D.BLOCK
    rcs = {f["reason_code"] for f in k.ledger.findings}
    assert "wrong_scope" in rcs and "missing_prerequisite" in rcs, rcs


def test_commit_point_merge_composes_modules():
    """A MedicationRequest create is constrained by BOTH the clinical module AND the substrate's generic
    write invariant -> the merged commit point unions their requires and keeps all postconditions."""
    pol = H.load_policy(env_type="fhir")
    task = {"task_id": "t", "goal": "order", "context": {"patient_ref": "Patient/1"}, "environment": {"type": "fhir"}}
    contract = build_contract(task, env_type="fhir", policy=pol)
    create = canonicalize({"type": "tool", "tool": "fhir_medication_request_create", "args": {"patient": "Patient/1"}},
                          pol["manifest"])
    cp = contract.commit_point_for(create)
    assert cp["match"]["composed_from"] == 2, cp           # medication_safety + substrate generic
    assert "medication_safety_review" in cp["requires"]
    assert len(cp["postconditions"]) >= 1


def test_final_answer_not_a_commit_in_record_substrate():
    """A plain 'task done' in an EHR is a terminal_response, NOT an irreversible commit (no commit point,
    no commit recorded). Only an adapter that declares the final irreversible (perceptual) makes it one."""
    pol = H.load_policy(env_type="fhir")
    task = {"task_id": "t", "goal": "review labs", "context": {"patient_ref": "Patient/1"}, "environment": {"type": "fhir"}}
    contract = build_contract(task, env_type="fhir", policy=pol)
    k = HarnessKernel(contract, [ScopeEvidenceBinding(), ObligationLifecycle(), VerifyAndCommit()],
                      mode="enforce", policy=pol, env_type="fhir")
    assert k.before_final("done, labs reviewed", step=0).type == D.ALLOW
    assert k.ledger.commit_history == [], k.ledger.commit_history


def test_typed_subject_identity():
    from harness.capabilities.scope_evidence import _same_subject
    assert _same_subject("Patient/123", "Patient/123")
    assert _same_subject("Patient/123", "123")            # one side untyped -> compare id only
    assert not _same_subject("Patient/123", "Encounter/123")   # same id, different TYPE -> not the same


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
