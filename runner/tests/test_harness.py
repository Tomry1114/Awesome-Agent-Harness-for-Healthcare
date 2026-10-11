"""Conformance for the Clinical Process Harness (semantic/substrate architecture).
Run: python3 runner/tests/test_harness.py — expects all PASS. No model calls; deterministic."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import harness as H
import workspace as W   # the default workspaces (W.H0) these tests compose: a run hands its workspace to every reader
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


def _kernel(policy=POLICY, task=TASK, env="fhir"):
    contract = build_contract(task, env_type=env, policy=policy)
    caps = [ScopeEvidenceBinding(), ObligationLifecycle(), VerifyAndCommit()]
    return HarnessKernel(contract, caps, policy=policy, env_type=env)


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


def test_scope_block():
    foreign = {"type": "tool", "tool": "fhir_search", "args": {"patient": "Patient/456"}}
    k = _kernel()
    eff = k.before_action(foreign, step=0)
    assert eff.type == D.BLOCK, eff.type
    assert eff.feedback and "subject_scope_mismatch" in str(eff.feedback)
    assert any(iv["decision"] == "BLOCK" for iv in k.ledger.interventions)


def test_in_scope_action_allowed():
    k = _kernel()
    ok = {"type": "tool", "tool": "fhir_search", "args": {"patient": "Patient/123", "resource_type": "AllergyIntolerance"}}
    assert k.before_action(ok, step=0).type == D.ALLOW


def test_commit_requires_obligation_then_satisfied():
    k = _kernel()
    commit = {"type": "tool", "tool": "create_medication_request", "args": {"patient": "Patient/123"}}
    assert k.before_action(commit, step=0).type == D.REVISE
    allergy = {"type": "tool", "tool": "fhir_search", "args": {"patient": "Patient/123", "resource_type": "AllergyIntolerance"}}
    k.before_action(allergy, step=1)
    k.after_action(allergy, "penicillin allergy", {"a": 0}, {"a": 1}, step=1, result_ok=True)
    assert k.ledger.obligation_state("check_allergies") == "SATISFIED"
    assert k.ledger.obligation_state("med_review") == "SATISFIED"
    assert k.before_action(commit, step=2).type == D.ALLOW


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


def test_budget_never_changes_a_safety_verdict_and_stops_advice():
    """The intervention budget bounds resources, not safety. A BLOCK stays a BLOCK past it (never a silent
    ALLOW, and not an ESCALATE, which after_action applies by ending the episode); an advisory REVISE past it is
    not issued (effective ALLOW, noted) and the episode goes on."""
    k = _kernel(); k.budget["max_interventions_per_task"] = 2
    foreign = {"type": "tool", "tool": "fhir_search", "args": {"patient": "Patient/456"}}
    effs = [k.before_action(foreign, step=i) for i in range(5)]
    assert [e.type for e in effs] == [D.BLOCK] * 5, "exhausted budget must NEVER change a BLOCK"
    rev = D.HarnessDecision(D.REVISE, capability="c", reason_code="advice", rule_id="r")
    eff = k._decide_effective(rev, "after_action")
    assert (eff.type, eff.note) == (D.ALLOW, "budget_exhausted_interventions")
    assert eff.events[0]["effective"] == D.ALLOW and eff.events[0]["note"] == "budget_exhausted_interventions"


def test_opportunity_denominators():
    k = _kernel()
    k.before_action({"type": "tool", "tool": "fhir_search", "args": {"patient": "Patient/456"}}, step=0)
    s = gov.summarize(k.ledger, [], mode="enforce")
    assert s["wrong_scope_action_rate"] == 1.0 and s["wrong_scope_opportunities"] == 1, s
    # an action with no subject -> no subject-bearing opportunity (denominator stays honest)
    k2 = _kernel()
    s2 = gov.summarize(k2.ledger, [], mode="enforce")
    assert s2["wrong_scope_action_rate"] is None, s2


def test_failed_result_does_not_satisfy_obligation():
    """A failed/empty tool result is ATTEMPTED evidence, not VALIDATED -> it cannot satisfy an obligation."""
    k = _kernel()
    allergy = {"type": "tool", "tool": "fhir_search", "args": {"patient": "Patient/123", "resource_type": "AllergyIntolerance"}}
    k.after_action(allergy, "{error: HTTP 500}", {"a": 0}, {"a": 0}, step=1, result_ok=False)
    assert k.ledger.obligation_state("check_allergies") != "SATISFIED", "error result must not satisfy"
    k.after_action(allergy, "penicillin allergy", {"a": 0}, {"a": 1}, step=2, result_ok=True)
    assert k.ledger.obligation_state("check_allergies") == "SATISFIED"


def test_unverifiable_commit_not_recorded_verified():
    """A commit whose state is unobservable -> verification None (UNKNOWN), NOT verified=True."""
    k = _kernel()
    allergy = {"type": "tool", "tool": "fhir_search", "args": {"patient": "Patient/123", "resource_type": "AllergyIntolerance"}}
    k.after_action(allergy, "penicillin", {"a": 0}, {"a": 1}, step=1, result_ok=True)   # satisfy med_review
    create = {"type": "tool", "tool": "create_medication_request", "args": {"patient": "Patient/123"}}
    assert k.before_action(create, step=2).type == D.ALLOW
    k.after_action(create, "created", None, None, step=2, result_ok=True)   # unobservable state
    assert k.ledger.commit_history[-1]["verified"] is None, k.ledger.commit_history[-1]


def test_capability_error_escalates_in_enforce():
    """Fail-closed: a crashing capability must NOT silently become ALLOW. It ESCALATEs (and is recorded as a
    capability error)."""
    class Boom(ScopeEvidenceBinding):
        def before_action(self, action, ctx):
            raise RuntimeError("boom")
    contract = build_contract(TASK, env_type="fhir", policy=POLICY)
    k = HarnessKernel(contract, [Boom()], policy=POLICY, env_type="fhir")
    assert k.before_action({"type": "tool", "tool": "fhir_search", "args": {}}, step=0).type == D.ESCALATE
    assert k._capability_errors and k.audit()["status"] == "degraded"


# ---- real SUBSTRATE packs: same harness core, only adapter + pack differ ----------------

def test_p1_structured_record_pack():
    pol = H.load_policy(env_type="fhir", workspace=W.H0["fhir"])
    assert pol.get("_substrate") == "structured_record"
    task = {"task_id": "pb", "goal": "manage UDS", "context": {"patient_ref": "MRN6025656705"},
            "environment": {"type": "fhir"}}
    contract = build_contract(task, env_type="fhir", policy=pol)
    assert contract.subject == {"type": "patient", "id": "MRN6025656705"}, contract.subject
    k = HarnessKernel(contract, [ScopeEvidenceBinding(), ObligationLifecycle(), VerifyAndCommit()],
                      policy=pol, env_type="fhir")
    wrong = {"type": "tool", "tool": "fhir_observation_search_labs", "args": {"patient": "MRN9999999999"}}
    own = {"type": "tool", "tool": "fhir_observation_search_labs", "args": {"patient": "MRN6025656705"}}
    create = {"type": "tool", "tool": "fhir_medication_request_create", "args": {"patient": "MRN6025656705"}}
    assert k.before_action(wrong, step=0).type == D.BLOCK
    assert k.before_action(own, step=1).type == D.ALLOW
    assert k.before_action(create, step=2).type == D.REVISE
    allergy = {"type": "tool", "tool": "fhir_allergy_intolerance_search_active", "args": {"patient": "MRN6025656705"}}
    meds = {"type": "tool", "tool": "fhir_medication_request_search_orders", "args": {"patient": "MRN6025656705"}}
    k.after_action(allergy, "penicillin", {"x": 0}, {"x": 1}, step=3, result_ok=True)
    k.after_action(meds, "lisinopril", {"x": 1}, {"x": 2}, step=4, result_ok=True)
    assert k.ledger.obligation_state("medication_safety_review") == "SATISFIED"
    assert k.before_action(create, step=5).type == D.ALLOW
    assert classify_risk(canonicalize(create, pol["manifest"]), contract) == "R2"


def test_p2_interactive_gui_pack():
    pol = H.load_policy(env_type="gui", workspace=W.H0["gui"])
    assert pol.get("_substrate") == "interactive_gui"
    task = {"task_id": "hab", "goal": "Open denial DEN-001 for Martinez. Triage it.",
            "context": {"text": "Open denial DEN-001."}, "environment": {"type": "gui"}}
    contract = build_contract(task, env_type="gui", policy=pol)
    assert contract.subject == {"type": "admin_case", "id": "DEN-001"}, contract.subject
    k = HarnessKernel(contract, [ScopeEvidenceBinding(), ObligationLifecycle(), VerifyAndCommit()],
                      policy=pol, env_type="gui")
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
    pol = H.load_policy(env_type="tool_sandbox", workspace=W.H0["tool_sandbox"])
    assert pol.get("_substrate") == "perceptual"
    task = {"task_id": "m", "goal": "finding in the chest CT?", "context": {}, "environment": {"type": "tool_sandbox"}}
    contract = build_contract(task, env_type="tool_sandbox", policy=pol)
    k = HarnessKernel(contract, [ScopeEvidenceBinding(), ObligationLifecycle(), VerifyAndCommit()],
                      policy=pol, env_type="tool_sandbox")
    assert k.before_final("a 3 cm mass", step=0).type == D.REVISE      # ungrounded
    region = {"type": "tool", "tool": "RegionAttributeDescription", "args": {}}
    k.after_action(region, "spiculated RUL nodule", None, None, step=1, result_ok=True)
    assert k.ledger.obligation_state("image_derived_evidence") == "SATISFIED"
    # grounded, but NO judge configured -> claim-support is UNVERIFIABLE -> enforce ESCALATEs (fail-closed),
    # it does NOT pass just because grounding evidence exists. (With a judge -> see the semantic test.)
    assert k.before_final("a 3 cm mass", step=2).type == D.ESCALATE
    # external (search) evidence does NOT count as image grounding
    k2 = HarnessKernel(contract, [ScopeEvidenceBinding(), ObligationLifecycle(), VerifyAndCommit()],
                       policy=pol, env_type="tool_sandbox")
    k2.after_action({"type": "tool", "tool": "GoogleSearch", "args": {}}, "web text", None, None, step=1)
    assert k2.ledger.obligation_state("image_derived_evidence") != "SATISFIED"
    assert k2.before_final("a 3 cm mass", step=2).type == D.REVISE


AUDIT_CLEAN = '{"addresses_task": true, "hard_violations": [], "repairable_gaps": [], "confidence": 0.9}'
AUDIT_HARD = ('{"addresses_task": true, "hard_violations": [{"type": "contradiction", "claim": "LUL effusion", '
              '"evidence_ids": ["E1"], "reason": "LUL vs RUL"}], "repairable_gaps": [], "confidence": 0.9}')


def test_p3_final_answer_adequacy_audit():
    """verify_commit's before_final on a claim-support postcondition is the adequacy audit (repair mode full): a
    LOCALIZED hard defect is must-resolve, a defect without cited evidence is not, and a missing judge ESCALATEs.
    A claim-support-shaped judge reply is not read by the audit."""
    pol = H.load_policy(env_type="tool_sandbox", workspace=W.H0["tool_sandbox"])
    task = {"task_id": "m", "goal": "finding?", "context": {}, "environment": {"type": "tool_sandbox"}}
    contract = build_contract(task, env_type="tool_sandbox", policy=pol)

    def mk(judge_fn):
        k = HarnessKernel(contract, [ScopeEvidenceBinding(), ObligationLifecycle(), VerifyAndCommit()],
                          policy=pol, env_type="tool_sandbox", judge_fn=judge_fn,
                          budget={"max_semantic_checks": 5})
        k.after_action({"type": "tool", "tool": "ImageDescription", "args": {}},
                       "a 3cm spiculated RUL nodule", None, None, step=0, result_ok=True)
        return k

    kc = mk(lambda p: AUDIT_CLEAN)
    assert kc.before_final("RUL nodule", step=1).type == D.ALLOW and kc.ctx.verification is True
    # a claim-support verdict carries no audit keys -> no defect, nothing verified
    ks = mk(lambda p: '{"supported": false, "confidence": 0.9, "reason": "LUL vs RUL"}')
    assert ks.before_final("LUL effusion", step=1).type == D.ALLOW and ks.ctx.verification is None
    # LOCALIZED hard defect (names the claim + cites evidence) -> must-resolve hard REVISE
    _dl = mk(lambda p: AUDIT_HARD).before_final("LUL effusion", step=1)
    assert _dl.type == D.REVISE and _dl.raw.reason_code == "evidence_contradiction", _dl.raw.to_dict()
    assert _dl.raw.extra.get("must_resolve") is True and "side_effecting" in _dl.raw.extra, _dl.raw.extra
    # hard defect citing no evidence -> not must-resolve, not escalated
    _nl = mk(lambda p: AUDIT_HARD.replace(', "evidence_ids": ["E1"]', "")).before_final("LUL effusion", step=1)
    assert _nl.type == D.ALLOW, _nl.raw.to_dict() if _nl.raw else _nl.type
    nj = mk(None)   # judge unavailable: a contract that REQUIRES claim-support cannot pass in enforce
    assert nj.before_final("RUL nodule", step=1).type == D.ESCALATE
    assert any(r["rule_id"] == "semantic_claim_support" for r in nj.ledger.unresolved_risks)


def test_p3_semantic_relation_tristate():
    """The judge sees the QUESTION/GOAL + PUBLIC CONTEXT; the relation tri-state separates a CONTRADICTION from
    INSUFFICIENT under-coverage. verify_claim_support is the engine the MedCTA answer-candidate strategy calls;
    verify_commit does not call it."""
    from harness.engines.semantic import verify_claim_support, CONTRADICTED, INSUFFICIENT
    cap = {}
    def jins(p):
        cap["prompt"] = p
        return '{"relation": "insufficient", "confidence": 0.8, "reason": "under-covers"}'
    v = verify_claim_support("finding in chest CT?", "ctx-text", "a 3 cm mass",
                             [{"type": "image", "value": "spiculated nodule"}], judge_fn=jins)
    assert v.relation == INSUFFICIENT and v.supported is None, v.to_dict()
    # the judge prompt actually contains the PUBLIC QUESTION/GOAL (oracle-safe, never gold).
    assert "QUESTION / GOAL" in cap["prompt"] and "finding in chest CT?" in cap["prompt"], cap["prompt"]
    # zero SELECTED evidence = under-coverage (INSUFFICIENT), NOT a 1.0-confidence contradiction.
    v0 = verify_claim_support("g", "c", "ans", [], judge_fn=jins)
    assert v0.relation == INSUFFICIENT and v0.supported is None, v0.to_dict()
    # a {"supported": ...} judge reply also parses (relation derived).
    vb = verify_claim_support("g", "c", "ans", [{"type": "image", "value": "x"}],
                              judge_fn=lambda p: '{"supported": false, "confidence": 0.9, "reason": "r"}')
    assert vb.relation == CONTRADICTED and vb.supported is False, vb.to_dict()


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
                      policy=pol, env_type="fhir")
    create = {"type": "tool", "tool": "zz9", "args": {"patient": "Patient/123"}}
    assert k.before_action(create, step=0).type == D.REVISE                 # missing prerequisite
    k.after_action({"type": "tool", "tool": "xq7", "args": {"patient": "Patient/123"}}, "ok", {"a": 0}, {"a": 1}, step=1, result_ok=True)
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
    pol = H.load_policy(env_type="fhir", workspace=W.H0["fhir"])
    assert pol["_adapter"] == "hapi_fhir" and pol["_substrate"] == "structured_record"
    assert pol["_clinical_modules"] == ["medication_safety", "chart_review"]
    assert pol["manifest"]["actions"], "manifest came from the adapter"
    # clinical obligations are present and clinical commit point is FIRST (matched before the substrate's
    # generic irreversible-write rule).
    assert any(o["id"] == "check_current_medications" for o in pol["evidence_obligations"])
    assert pol["commit_points"][0]["match"].get("resource") == "MedicationRequest"
    assert any(cp["match"] == {"effect": "irreversible"} for cp in pol["commit_points"]), "substrate invariant present"


def test_clinical_module_scoped_to_its_resource():
    """GENERALITY: a NON-medication irreversible write (e.g. ServiceRequest) gets only the SUBSTRATE's
    generic write-verification — the medication-safety review does NOT fire. Clinical rules are scoped to
    the resources they name, not to the whole substrate."""
    pol = H.load_policy(env_type="fhir", workspace=W.H0["fhir"])
    task = {"task_id": "t", "goal": "place an order", "context": {"patient_ref": "Patient/1"},
            "environment": {"type": "fhir"}}
    contract = build_contract(task, env_type="fhir", policy=pol)
    k = HarnessKernel(contract, [ScopeEvidenceBinding(), ObligationLifecycle(), VerifyAndCommit()],
                      policy=pol, env_type="fhir")
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
    pol = H.load_policy(env_type="fhir", workspace=W.H0["fhir"])
    assert (pol["manifest"]["subject"].get("binding")) == "required"
    task = {"task_id": "t", "goal": "order", "context": {"patient_ref": "Patient/1"}, "environment": {"type": "fhir"}}
    contract = build_contract(task, env_type="fhir", policy=pol)
    k = HarnessKernel(contract, [ScopeEvidenceBinding(), ObligationLifecycle(), VerifyAndCommit()],
                      policy=pol, env_type="fhir")
    nopt = {"type": "tool", "tool": "fhir_medication_request_create", "args": {}}   # no patient named
    assert k.before_action(nopt, step=0).type == D.REVISE
    assert any(f["reason_code"] == "subject_unspecified" for f in k.ledger.findings)
    # a no-subject READ does not satisfy an obligation either (evidence is not VALIDATED)
    rd = {"type": "tool", "tool": "fhir_allergy_intolerance_search_active", "args": {}}
    k.after_action(rd, "data", {"a": 0}, {"a": 1}, step=1, result_ok=True)
    assert k.ledger.obligation_state("check_allergies") != "SATISFIED"


def test_policy_loader_fail_closed():
    """A missing/typo'd policy layer must NOT silently vanish: it is recorded in _errors, and the kernel
    REFUSES to build."""
    assert not H.load_policy(env_type="fhir", workspace=W.H0["fhir"]).get("_errors")        # the real policy composes cleanly
    assert H.load_policy(adapter="does_not_exist", env_type="fhir", workspace=W.H0["fhir"]).get("_errors"), "missing adapter must be an error"
    task = {"task_id": "t", "goal": "g", "context": {}, "environment": {"type": "fhir"}}
    try:
        H.build_kernel(task, env_type="fhir", adapter="does_not_exist", workspace=W.H0["fhir"]); assert False, "enforce must raise"
    except H.PolicyError:
        pass


def test_findings_keep_lower_priority():
    """A hook whose winner is wrong_scope (BLOCK) must still record its missing_prerequisite finding."""
    pol = H.load_policy(env_type="fhir", workspace=W.H0["fhir"])
    task = {"task_id": "t", "goal": "order", "context": {"patient_ref": "Patient/1"}, "environment": {"type": "fhir"}}
    contract = build_contract(task, env_type="fhir", policy=pol)
    k = HarnessKernel(contract, [ScopeEvidenceBinding(), ObligationLifecycle(), VerifyAndCommit()],
                      policy=pol, env_type="fhir")
    # a create on the WRONG patient AND missing the safety review: BLOCK wins, but both are recorded
    bad = {"type": "tool", "tool": "fhir_medication_request_create", "args": {"patient": "Patient/999"}}
    eff = k.before_action(bad, step=0)
    assert eff.type == D.BLOCK
    rcs = {f["reason_code"] for f in k.ledger.findings}
    assert "wrong_scope" in rcs and "missing_prerequisite" in rcs, rcs


def test_commit_point_merge_composes_modules():
    """A MedicationRequest create is constrained by BOTH the clinical module AND the substrate's generic
    write invariant -> the merged commit point unions their requires and keeps all postconditions."""
    pol = H.load_policy(env_type="fhir", workspace=W.H0["fhir"])
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
    pol = H.load_policy(env_type="fhir", workspace=W.H0["fhir"])
    task = {"task_id": "t", "goal": "review labs", "context": {"patient_ref": "Patient/1"}, "environment": {"type": "fhir"}}
    contract = build_contract(task, env_type="fhir", policy=pol)
    k = HarnessKernel(contract, [ScopeEvidenceBinding(), ObligationLifecycle(), VerifyAndCommit()],
                      policy=pol, env_type="fhir")
    assert k.before_final("done, labs reviewed", step=0).type == D.ALLOW
    assert k.ledger.commit_history == [], k.ledger.commit_history


def test_typed_subject_identity():
    from harness.capabilities.scope_evidence import _same_subject
    assert _same_subject("Patient/123", "Patient/123")
    assert _same_subject("Patient/123", "123")            # one side untyped -> compare id only
    assert not _same_subject("Patient/123", "Encounter/123")   # same id, different TYPE -> not the same


def test_gui_prospective_commit_guard():
    """A GUI submit while the page shows the WRONG case is blocked BEFORE it executes (prospective), not
    only caught post-hoc. Non-commit navigation is allowed."""
    pol = H.load_policy(env_type="gui", workspace=W.H0["gui"])
    task = {"task_id": "hab", "goal": "Triage DEN-001.", "context": {"text": "DEN-001"}, "environment": {"type": "gui"}}
    contract = build_contract(task, env_type="gui", policy=pol)
    k = HarnessKernel(contract, [ScopeEvidenceBinding(), ObligationLifecycle(), VerifyAndCommit()],
                      policy=pol, env_type="gui")
    # land on the WRONG case (records the displayed subject)
    k.after_action({"type": "tool", "tool": "click", "args": {}}, "ok", {"x": 0}, {"x": 1}, step=0,
                   canonical_observation={"case_identity": "DEN-999"})
    eff = k.before_action({"type": "tool", "tool": "submit", "args": {}}, step=1)   # commit on wrong page
    assert eff.type == D.BLOCK and "DEN-999" in str(eff.feedback), eff.feedback
    # navigate to the right case -> submit now allowed prospectively
    k.after_action({"type": "tool", "tool": "click", "args": {}}, "ok", {"x": 1}, {"x": 2}, step=2,
                   canonical_observation={"case_identity": "DEN-001"})
    assert k.before_action({"type": "tool", "tool": "submit", "args": {}}, step=3).type == D.ALLOW


def test_repair_chain_records_repaired():
    """A missing-prereq REVISE on a commit, then obligations satisfied + the commit accepted, is a causal
    `repaired` resolution -> repair_success_rate = repaired / repairable-opportunities."""
    pol = H.load_policy(env_type="fhir", workspace=W.H0["fhir"])
    task = {"task_id": "t", "goal": "order", "context": {"patient_ref": "Patient/1"}, "environment": {"type": "fhir"}}
    contract = build_contract(task, env_type="fhir", policy=pol)
    k = HarnessKernel(contract, [ScopeEvidenceBinding(), ObligationLifecycle(), VerifyAndCommit()],
                      policy=pol, env_type="fhir")
    create = {"type": "tool", "tool": "fhir_medication_request_create", "args": {"patient": "Patient/1"}}
    assert k.before_action(create, step=0).type == D.REVISE          # opens a repair opportunity
    k.after_action({"type": "tool", "tool": "fhir_allergy_intolerance_search_active", "args": {"patient": "Patient/1"}},
                   "pen", {"a": 0}, {"a": 1}, step=1, result_ok=True)
    k.after_action({"type": "tool", "tool": "fhir_medication_request_search_orders", "args": {"patient": "Patient/1"}},
                   "lis", {"a": 1}, {"a": 2}, step=2, result_ok=True)
    assert k.before_action(create, step=3).type == D.ALLOW           # gate passes -> precondition_repaired
    assert any(r["resolution"] == "precondition_repaired" for r in k.ledger.resolutions)
    # actually EXECUTE the create with a real state change -> postcondition verified -> verified repair
    k.after_action(create, {"id": "rx-1"}, {"MedicationRequest": []}, {"MedicationRequest": [{"id": "rx-1"}]},
                   step=4, result_ok=True)
    assert k.ledger.commit_history[-1]["verified"] is True
    assert any(r["resolution"] == "repaired" for r in k.ledger.resolutions)
    s = gov.summarize(k.ledger, [], "enforce")
    assert s["verified_repair_rate"] == 1.0 and s["repair_opportunities"] == 1, s


def test_commit_verification_violated_on_no_state_change():
    """Counter-case: a commit that produces NO observable state change -> postcondition VIOLATED (verified
    False), recorded as a violated commit (not silently verified)."""
    pol = H.load_policy(env_type="fhir", workspace=W.H0["fhir"])
    task = {"task_id": "t", "goal": "order", "context": {"patient_ref": "Patient/1"}, "environment": {"type": "fhir"}}
    contract = build_contract(task, env_type="fhir", policy=pol)
    k = HarnessKernel(contract, [ScopeEvidenceBinding(), ObligationLifecycle(), VerifyAndCommit()],
                      policy=pol, env_type="fhir")
    create = {"type": "tool", "tool": "fhir_medication_request_create", "args": {"patient": "Patient/1"}}
    same = {"MedicationRequest": []}
    eff = k.after_action(create, "ok", same, same, step=0, result_ok=True)   # state unchanged
    assert eff.type == D.REVISE                       # violated postcondition
    assert k.ledger.commit_history[-1]["verified"] is False


def test_violation_prevented():
    """A would-be violation is PREVENTED (effective BLOCK), never executed."""
    pol = H.load_policy(env_type="fhir", workspace=W.H0["fhir"])
    task = {"task_id": "t", "goal": "g", "context": {"patient_ref": "Patient/1"}, "environment": {"type": "fhir"}}
    contract = build_contract(task, env_type="fhir", policy=pol)
    wrong = {"type": "tool", "tool": "fhir_observation_search_labs", "args": {"patient": "Patient/999"}}
    ke = HarnessKernel(contract, [ScopeEvidenceBinding(), ObligationLifecycle(), VerifyAndCommit()],
                       policy=pol, env_type="fhir")
    ke.before_action(wrong, step=0)
    se = gov.summarize(ke.ledger, [], "enforce")
    assert se["prevented_violation_count"] == 1 and se["executed_violation_count"] == 0, se


def test_unseen_adapter_zero_core_change():
    """ZERO-SHOT ADAPTER GENERALITY: a NEW EMR with entirely different tool names and a different patient-id
    field (mrn) runs the full wrong-patient / medication-safety / repair flow via load_policy(adapter=...),
    with ZERO change to the harness core, the substrate policy, or the clinical module. A new dataset is one
    new adapter file."""
    import workspace as W
    pol = H.load_policy(adapter="clinic_emr", workspace=W.H0["fhir"])   # the fhir workspace carries chart_review
    assert pol["_adapter"] == "clinic_emr" and pol["_substrate"] == "structured_record"
    assert pol["_clinical_modules"] == ["medication_safety", "chart_review"] and not pol.get("_errors")
    task = {"task_id": "u", "goal": "prescribe", "context": {"mrn": "M-7"},
            "environment": {"type": "fhir", "adapter": "clinic_emr"}}
    contract = build_contract(task, env_type="fhir", policy=pol)
    assert contract.subject == {"type": "patient", "id": "M-7"}, contract.subject
    k = HarnessKernel(contract, [ScopeEvidenceBinding(), ObligationLifecycle(), VerifyAndCommit()],
                      policy=pol, env_type="fhir")
    # wrong patient, brand-new tool name -> still BLOCK
    assert k.before_action({"type": "tool", "tool": "ehr_labs_fetch", "args": {"mrn": "M-999"}}, step=0).type == D.BLOCK
    assert k.before_action({"type": "tool", "tool": "ehr_labs_fetch", "args": {"mrn": "M-7"}}, step=1).type == D.ALLOW
    rx = {"type": "tool", "tool": "ehr_prescribe", "args": {"mrn": "M-7"}}
    assert k.before_action(rx, step=2).type == D.REVISE        # medication safety review still required
    k.after_action({"type": "tool", "tool": "ehr_allergy_review", "args": {"mrn": "M-7"}}, "pen", {"a": 0}, {"a": 1}, step=3, result_ok=True)
    k.after_action({"type": "tool", "tool": "ehr_med_list", "args": {"mrn": "M-7"}}, "lis", {"a": 1}, {"a": 2}, step=4, result_ok=True)
    assert k.ledger.obligation_state("medication_safety_review") == "SATISFIED"
    assert k.before_action(rx, step=5).type == D.ALLOW
    # EXECUTE the prescribe in the held-out EMR -> postcondition verified -> repaired (commit-effect
    # verification is portable too, not only the pre-commit gating)
    k.after_action(rx, {"id": "rx-1"}, {"MedicationRequest": []}, {"MedicationRequest": [{"id": "rx-1"}]},
                   step=6, result_ok=True)
    assert k.ledger.commit_history[-1]["verified"] is True
    assert any(r["resolution"] == "repaired" for r in k.ledger.resolutions)


def test_loader_rejects_invalid_enums():
    """A typo'd effect/semantic_type/binding/source_class is caught at load (would otherwise be accepted
    verbatim by the canonicalizer and silently mis-classify risk/scope)."""
    from harness.engines.policy import _validate_enums
    errs = []
    _validate_enums({"subject": {"binding": "requred"},
                     "actions": [{"match": {"tool": "x"}, "semantic_type": "craete", "effect": "irreversble",
                                  "produces_evidence": {"source_class": "recrod"}},
                                 {"match": {}}]}, errs)
    assert any("invalid_subject_binding" in e for e in errs)
    assert any("invalid_semantic_type" in e for e in errs)
    assert any("invalid_effect" in e for e in errs)
    assert any("invalid_source_class" in e for e in errs)
    assert any("action_rule_empty_match" in e for e in errs)


def test_uncovered_irreversible_escalates():
    """An irreversible action that NO commit point covers (no obligations/postcondition) fails closed."""
    pol = {"manifest": {"subject": {}, "actions": [{"match": {"tool": "danger"},
            "semantic_type": "create", "effect": "irreversible"}]},
           "evidence_obligations": [], "workflow_obligations": [], "commit_points": []}
    task = {"task_id": "t", "goal": "g", "context": {}, "environment": {"type": "fhir"}}
    contract = build_contract(task, env_type="fhir", policy=pol)
    k = HarnessKernel(contract, [ScopeEvidenceBinding(), ObligationLifecycle(), VerifyAndCommit()],
                      policy=pol, env_type="fhir")
    assert k.before_action({"type": "tool", "tool": "danger", "args": {}}, step=0).type == D.ESCALATE


def test_required_binding_rejects_subjectless_read():
    """required binding covers READS too: a search with no patient ('all patients') -> REVISE."""
    pol = H.load_policy(env_type="fhir", workspace=W.H0["fhir"])
    task = {"task_id": "t", "goal": "g", "context": {"patient_ref": "Patient/1"}, "environment": {"type": "fhir"}}
    contract = build_contract(task, env_type="fhir", policy=pol)
    k = HarnessKernel(contract, [ScopeEvidenceBinding(), ObligationLifecycle(), VerifyAndCommit()],
                      policy=pol, env_type="fhir")
    rd = {"type": "tool", "tool": "fhir_allergy_intolerance_search_active", "args": {}}
    assert k.before_action(rd, step=0).type == D.REVISE


def test_gui_subject_projected_from_raw_observation():
    """REAL-PATH: the displayed case is projected from the RAW env observation via the manifest's declared
    paths (full_state.fields.caseId), not a hand-fed canonical field; and it is STICKY (an empty/error
    observation does NOT erase the last known displayed subject)."""
    pol = H.load_policy(env_type="gui", workspace=W.H0["gui"])
    task = {"task_id": "hab", "goal": "Triage DEN-001.", "context": {"text": "DEN-001"}, "environment": {"type": "gui"}}
    contract = build_contract(task, env_type="gui", policy=pol)
    k = HarnessKernel(contract, [ScopeEvidenceBinding(), ObligationLifecycle(), VerifyAndCommit()],
                      policy=pol, env_type="gui")
    raw_wrong = {"full_state": {"fields": {"caseId": "DEN-999"}}}
    k.after_action({"type": "tool", "tool": "snapshot", "args": {}}, raw_wrong, {"x": 0}, {"x": 1}, step=0,
                   raw_observation=raw_wrong)
    # an empty observation must NOT clear the known displayed subject
    k.after_action({"type": "tool", "tool": "snapshot", "args": {}}, {"ok": True}, {"x": 1}, {"x": 2}, step=1,
                   raw_observation={"ok": True})
    eff = k.before_action({"type": "tool", "tool": "submit", "args": {}}, step=2)
    assert eff.type == D.BLOCK and "DEN-999" in str(eff.feedback), eff.feedback


def test_failed_commit_not_verified():
    """A commit whose TOOL CALL failed (result_ok=False) is NOT verified, even if the state hash happened
    to change — the commit did not land."""
    k = _kernel()
    allergy = {"type": "tool", "tool": "fhir_search", "args": {"patient": "Patient/123", "resource_type": "AllergyIntolerance"}}
    k.after_action(allergy, "penicillin", {"a": 0}, {"a": 1}, step=1, result_ok=True)
    create = {"type": "tool", "tool": "create_medication_request", "args": {"patient": "Patient/123"}}
    assert k.before_action(create, step=2).type == D.ALLOW
    eff = k.after_action(create, {"error": "HTTP 500"}, "old", "new", step=2, result_ok=False)  # tool failed
    assert eff.type == D.REVISE
    assert k.ledger.commit_history[-1]["verified"] is False, k.ledger.commit_history[-1]


def test_empty_result_not_validated():
    """A successful-but-EMPTY read ({} / {"ok": true}) carries no evidence -> ATTEMPTED, cannot satisfy."""
    k = _kernel()
    allergy = {"type": "tool", "tool": "fhir_search", "args": {"patient": "Patient/123", "resource_type": "AllergyIntolerance"}}
    k.after_action(allergy, {}, {"a": 0}, {"a": 1}, step=1, result_ok=True)
    assert k.ledger.obligation_state("check_allergies") != "SATISFIED"
    k.after_action(allergy, {"entries": [{"id": 1}]}, {"a": 1}, {"a": 2}, step=2, result_ok=True)   # real payload
    assert k.ledger.obligation_state("check_allergies") == "SATISFIED"


def test_final_answer_verified_repair():
    """A final-answer commit (perceptual) that was REVISE'd for missing grounding, then grounded + supported,
    becomes a VERIFIED repair (not only precondition_repaired)."""
    pol = H.load_policy(env_type="tool_sandbox", workspace=W.H0["tool_sandbox"])
    task = {"task_id": "m", "goal": "finding?", "context": {}, "environment": {"type": "tool_sandbox"}}
    contract = build_contract(task, env_type="tool_sandbox", policy=pol)
    judge = lambda p: AUDIT_CLEAN       # the adequacy audit addresses the task with no defect -> verified
    k = HarnessKernel(contract, [ScopeEvidenceBinding(), ObligationLifecycle(), VerifyAndCommit()],
                      policy=pol, env_type="tool_sandbox", judge_fn=judge,
                      budget={"max_semantic_checks": 5})
    assert k.before_final("RUL nodule", step=0).type == D.REVISE       # ungrounded -> opens repair
    k.after_action({"type": "tool", "tool": "ImageDescription", "args": {}}, "a 3cm RUL nodule",
                   None, None, step=1, result_ok=True)
    assert k.before_final("RUL nodule", step=2).type == D.ALLOW        # grounded + clean audit -> verified
    assert any(r["resolution"] == "repaired" for r in k.ledger.resolutions)
    assert gov.summarize(k.ledger, [], "enforce")["verified_repair_rate"] == 1.0


def test_fhir_nested_create_subject_and_lab_binding():
    """from_args supports dotted paths (nested FHIR create resource.subject.reference); the reference-range
    lookup is declared subject_binding: none (not patient-bound)."""
    pol = H.load_policy(env_type="fhir", workspace=W.H0["fhir"])
    create = {"type": "tool", "tool": "fhir_medication_request_create",
              "args": {"resource": {"resourceType": "MedicationRequest", "subject": {"reference": "Patient/1"}}}}
    assert canonicalize(create, pol["manifest"]).target_entity == "Patient/1"
    lab = canonicalize({"type": "tool", "tool": "get_lab_reference_range", "args": {}}, pol["manifest"])
    assert lab.subject_binding == "none"


def test_wrong_scope_rate_capped_at_one():
    """A single action examined in both before_action and after_action must not double-count: the rate
    numerator is UNIQUE actions and the denominator dedups opportunities per action -> rate <= 1."""
    pol = H.load_policy(env_type="gui", workspace=W.H0["gui"])
    task = {"task_id": "hab", "goal": "Triage DEN-001.", "context": {"text": "DEN-001"}, "environment": {"type": "gui"}}
    contract = build_contract(task, env_type="gui", policy=pol)
    k = HarnessKernel(contract, [ScopeEvidenceBinding(), ObligationLifecycle(), VerifyAndCommit()],
                      policy=pol, env_type="gui")
    raw = {"full_state": {"fields": {"caseId": "DEN-999"}}}
    k.after_action({"type": "tool", "tool": "click", "args": {}}, raw, {"x": 0}, {"x": 1}, step=0, raw_observation=raw)
    submit = {"type": "tool", "tool": "submit", "args": {}}
    k.before_action(submit, step=1)                                  # prospective wrong_scope (BLOCK)
    k.after_action(submit, raw, {"x": 1}, {"x": 2}, step=1, raw_observation=raw)   # retrospective wrong_scope
    s = gov.summarize(k.ledger, [], "enforce")
    assert s["wrong_scope_action_rate"] is not None and s["wrong_scope_action_rate"] <= 1.0, s


def test_admission_rejects_ambiguous_tool():
    """A tool that matches >1 action rule with DIFFERENT semantics is ambiguous (first-match would silently
    decide its risk) -> admission error; a tool matching one rule is fine."""
    from harness.engines.policy import admission_errors
    manifest = {"actions": [{"match": {"tool_pattern": "order"}, "semantic_type": "read", "effect": "none"},
                            {"match": {"tool": "med_order"}, "semantic_type": "create", "effect": "irreversible"}]}
    assert any("ambiguous_action_mapping" in e for e in admission_errors(manifest, ["med_order"]))
    assert not admission_errors(manifest, ["other_order"])


def test_evidence_not_truncated_for_grounding_judge():
    # The grounding judge must receive the FULL evidence text, not the 200-char audit preview — a truncated stub
    # makes the judge call well-grounded answers "unsupported" and REVISE/ESCALATE them. The ledger keeps `value_full`
    # for the judge; the audit (to_dict) keeps only the short `value` preview.
    pol = H.load_policy(env_type="tool_sandbox", workspace=W.H0["tool_sandbox"])
    task = {"task_id": "m", "goal": "finding?", "context": {}, "environment": {"type": "tool_sandbox"}}
    contract = build_contract(task, env_type="tool_sandbox", policy=pol)
    long_finding = ("CT abdomen. " + "portal vein shows an intraluminal filling defect; " * 14 + "TAIL_MARKER_PVT")
    assert len(long_finding) > 250 and long_finding.index("TAIL_MARKER_PVT") > 200  # marker lives past 200 chars
    seen = {}

    def echo_judge(prompt):
        seen["prompt"] = prompt
        return '{"supported": true, "confidence": 0.9, "reason": "ok"}'

    k = HarnessKernel(contract, [ScopeEvidenceBinding(), ObligationLifecycle(), VerifyAndCommit()],
                      policy=pol, env_type="tool_sandbox", judge_fn=echo_judge,
                      budget={"max_semantic_checks": 5})
    k.after_action({"type": "tool", "tool": "ImageDescription", "args": {}},
                   long_finding, None, None, step=0, result_ok=True)
    ev = k.ledger.evidence[-1]
    assert "TAIL_MARKER_PVT" in (ev.get("value_full") or ""), "full payload not retained for the judge"
    assert k.before_final("portal vein thrombosis", step=1).type == D.ALLOW
    assert "TAIL_MARKER_PVT" in seen.get("prompt", ""), "grounding judge saw a truncated evidence stub"
    aud_ev = k.ledger.to_dict()["evidence"][-1]
    assert "value_full" not in aud_ev and len(aud_ev.get("value", "")) <= 220  # audit stays compact


def test_revision_identity_resets_on_progress():
    """The per-action revision counter is keyed on the FULL action fingerprint
    (semantic_type, resource, target_entity, payload_hash, evidence_version, reason_code, capability).
    A TRULY identical repeated rejection accumulates toward the cap, past which it is not issued (effective
    ALLOW, noted max_revisions_exceeded); a revised payload OR new evidence (genuine progress) RESETS the
    counter so a progressing agent keeps receiving it."""
    rev = D.HarnessDecision(D.REVISE, capability="verify_commit", reason_code="ungrounded_claim", rule_id="r")
    man = POLICY.get("manifest")

    # (a) IDENTICAL final answer rejected over and over -> stuck loop -> not issued past the cap (=2).
    k = _kernel(); k.budget["max_revisions_per_action"] = 2
    k.ctx.sem = canonicalize({"type": "final", "answer": "a 3 cm mass"}, man)
    effs = [k._decide_effective(rev, "before_final") for _ in range(4)]
    assert [(e.type, e.note) for e in effs] == [(D.REVISE, None), (D.REVISE, None), (D.ALLOW, "max_revisions_exceeded"),
                                                (D.ALLOW, "max_revisions_exceeded")], effs

    # (b) a DIFFERENT answer every turn (still REVISE'd) = progress -> counter RESETS -> never escalates.
    k2 = _kernel(); k2.budget["max_revisions_per_action"] = 2
    outs = []
    for i in range(5):
        k2.ctx.sem = canonicalize({"type": "final", "answer": "draft v%d" % i}, man)
        outs.append(k2._decide_effective(rev, "before_final").type)
    assert all(o == D.REVISE for o in outs), outs

    # (c) SAME answer but NEW evidence each turn = progress (evidence_version bumps) -> RESETS -> no escalate.
    k3 = _kernel(); k3.budget["max_revisions_per_action"] = 2
    outs3 = []
    for i in range(5):
        k3.ctx.sem = canonicalize({"type": "final", "answer": "stuck"}, man)
        k3.ledger.add_evidence("record", {"i": i}, subject_id="Patient/123",
                               extra={"status": "VALIDATED", "scope_relation": "matched"})  # genuine new progress each turn
        outs3.append(k3._decide_effective(rev, "before_final").type)
    assert all(o == D.REVISE for o in outs3), outs3

    # (d) the identity bounds BEFORE_ACTION too: two DISTINCT tool args don't share a counter.
    k4 = _kernel(); k4.budget["max_revisions_per_action"] = 1
    k4.ctx.sem = canonicalize({"type": "tool", "tool": "fhir_search", "args": {"q": "A"}}, man)
    a1 = [k4._decide_effective(rev, "before_action").type for _ in range(2)]
    assert a1 == [D.REVISE, D.ALLOW], a1                         # same args -> accumulates -> not issued past it
    k4.ctx.sem = canonicalize({"type": "tool", "tool": "fhir_search", "args": {"q": "B"}}, man)
    assert k4._decide_effective(rev, "before_action").type == D.REVISE  # different args -> fresh counter


def test_flagged_final_recorded_as_delivered():
    # graceful degradation: run.py delivers a no-side-effect answer WITH a flag via kernel.record_flagged_final.
    # It must surface as a final-answer commit (verified=None) -> answer_delivered=1, outcome_preservation=1,
    # so the metric layer never reports a delivered answer as erased.
    pol = H.load_policy(env_type="tool_sandbox", workspace=W.H0["tool_sandbox"])
    task = {"task_id": "m", "goal": "finding?", "context": {}, "environment": {"type": "tool_sandbox"}}
    contract = build_contract(task, env_type="tool_sandbox", policy=pol)
    k = HarnessKernel(contract, [ScopeEvidenceBinding(), ObligationLifecycle(), VerifyAndCommit()],
                      policy=pol, env_type="tool_sandbox")
    k.record_flagged_final("a 3cm nodule", flag="unresolved_risk", step=2)
    fa = [c for c in k.ledger.commit_history if c.get("detail") == "final_answer"]
    assert len(fa) == 1 and fa[0].get("verified") is None and fa[0].get("verification_flag") == "unresolved_risk"
    m = gov.summarize(k.ledger, [], "enforce")
    assert m["answer_delivered"] == 1 and m["outcome_preservation"] == 1


def test_a_judge_sharing_the_agent_model_runs_and_is_recorded():
    # judge independence is RECORDED, never enforced. A judge that is the agent brain builds a kernel like any
    # other and the kernel says the run is self-judged.
    task = {"task_id": "m", "goal": "g", "context": {}, "environment": {"type": "tool_sandbox"}, "available_tools": []}
    _old = (os.environ.get("MH_HARNESS_JUDGE_MODEL"), os.environ.get("MH_API_MODEL"))
    try:
        os.environ["MH_API_MODEL"] = "collide-model"; os.environ["MH_HARNESS_JUDGE_MODEL"] = "collide-model"
        k = H.build_kernel(task, env_type="tool_sandbox", workspace=W.H0["tool_sandbox"])
        assert k.judge_independence == "shared_model_with_agent_or_tool"
    finally:
        for _k, _v in zip(("MH_HARNESS_JUDGE_MODEL", "MH_API_MODEL"), _old):
            os.environ.pop(_k, None) if _v is None else os.environ.__setitem__(_k, _v)


def test_a_judge_sharing_the_tool_backend_runs_and_is_recorded():
    # the same for the perception / tool backend, and a judge that shares neither is recorded independent.
    task = {"task_id": "m", "goal": "g", "context": {}, "environment": {"type": "tool_sandbox"}, "available_tools": []}
    _old = (os.environ.get("MH_HARNESS_JUDGE_MODEL"), os.environ.get("MH_API_MODEL"))
    try:
        os.environ["MH_API_MODEL"] = "brain"; os.environ["MH_HARNESS_JUDGE_MODEL"] = "vlm-x"
        k = H.build_kernel(task, env_type="tool_sandbox", tool_model="vlm-x", workspace=W.H0["tool_sandbox"])
        assert k.judge_independence == "shared_model_with_agent_or_tool"
        os.environ["MH_HARNESS_JUDGE_MODEL"] = "other-judge"
        k2 = H.build_kernel(task, env_type="tool_sandbox", tool_model="vlm-x", workspace=W.H0["tool_sandbox"])
        assert k2.judge_independence == "independent"
    finally:
        for _k, _v in zip(("MH_HARNESS_JUDGE_MODEL", "MH_API_MODEL"), _old):
            os.environ.pop(_k, None) if _v is None else os.environ.__setitem__(_k, _v)


def test_findings_use_canonical_action_key():
    contract = build_contract(TASK, env_type="fhir", policy=POLICY)
    k = HarnessKernel(contract, [ScopeEvidenceBinding(), ObligationLifecycle(), VerifyAndCommit()],
                      policy=POLICY, env_type="fhir")
    create = {"type": "tool", "tool": "create_medication_request", "args": {"patient": "Patient/123"}}
    k.before_action(create, {"a": 0}, step=1)
    assert k.ledger.findings, "expected a finding"
    assert all(str(f.get("action_key", "")).startswith("action-") for f in k.ledger.findings), \
        "findings must use the canonical action_key (action-N), not a separate act%d identity"


def _deliverable_cls():
    """PB's deliverable lifecycle lives in the fhir workspace's middleware."""
    from workspace import H0, Workspace
    return [c for c in Workspace(H0["fhir"]).middleware_classes() if c.__name__ == "DeliverableMiddleware"][0]


def test_pb_is_required_write_exact_path():
    d = _deliverable_cls()({"goal": "Write your note to output/assessment.txt now"})
    assert d.active
    assert d.is_required_write({"tool": "write_file", "args": {"file_path": "output/assessment.txt"}})
    assert not d.is_required_write({"tool": "write_file", "args": {"file_path": "output/wrong.txt"}})
    assert not d.is_required_write({"tool": "write_file", "args": {"path": "output/assessment.txt"}})   # undeclared
    assert not d.is_required_write({"tool": "fhir_read", "args": {}})


def test_validated_evidence_version_only_on_genuine_progress():
    # the kernel's per-action revision counter and authorizations key on validated_evidence_version -> a
    # FAILED/ATTEMPTED or FOREIGN read must NOT refresh it (otherwise a failing tool would buy unlimited revises).
    from harness.state import Ledger
    L = Ledger()
    L.add_evidence("x", "v", extra={"status": "ATTEMPTED"})
    assert L.evidence_version == 1 and L.validated_evidence_version == 0
    L.add_evidence("x", "v2", extra={"status": "VALIDATED", "scope_relation": "matched"})
    assert L.validated_evidence_version == 1
    L.add_evidence("x", "v3", extra={"status": "VALIDATED", "scope_relation": "foreign"})
    assert L.validated_evidence_version == 1   # foreign-subject evidence is not genuine progress here
    L.add_evidence("x", "v2", extra={"status": "VALIDATED", "scope_relation": "matched"})  # IDENTICAL to the earlier matched read
    assert L.validated_evidence_version == 1   # a repeated identical validated read is NOT new progress


def test_pre_commit_redundant_commit_blocked():
    # PRE-COMMIT CONTROL: an irreversible commit that already SUCCEEDED must be BLOCKED on re-execution
    # (prevents a redundant re-submit/re-create from corrupting the landed state).
    contract = build_contract(TASK, env_type="fhir", policy=POLICY)
    from harness.capabilities.verify_commit import commit_identity
    k = HarnessKernel(contract, [VerifyAndCommit()], policy=POLICY, env_type="fhir")
    create = {"type": "tool", "tool": "create_medication_request", "args": {"patient": "Patient/123", "med": "X"}}
    k.before_action(create, {"x": 0}, step=1)                       # first attempt -> populates ctx.sem
    k.ledger.completed_commits.add(commit_identity(k.ctx.sem, k.ledger))   # this EXACT intent succeeded
    eff = k.before_action(create, {"x": 0}, step=2)                # SAME intent re-attempted -> BLOCK
    assert eff.type == D.BLOCK and getattr(eff.raw, "reason_code", None) == "redundant_commit", (eff.type, getattr(eff.raw, "reason_code", None))
    # a DIFFERENT intent (different payload) of the SAME resource type is NOT blocked
    create2 = {"type": "tool", "tool": "create_medication_request", "args": {"patient": "Patient/123", "med": "Y"}}
    eff2 = k.before_action(create2, {"x": 0}, step=3)
    assert getattr(getattr(eff2, "raw", None), "reason_code", None) != "redundant_commit"


def test_unknown_commit_resolved_by_postcondition():
    # TRANSPORT timeout != commit failure: check the postcondition against the after-state FIRST. State
    # CHANGED -> the commit LANDED despite the timeout (ALLOW); state UNCHANGED -> it did not land (REVISE).
    contract = build_contract(TASK, env_type="fhir", policy=POLICY)
    create = {"type": "tool", "tool": "create_medication_request", "args": {"patient": "Patient/123"}}
    ka = HarnessKernel(contract, [VerifyAndCommit()], policy=POLICY, env_type="fhir")
    ea = ka.after_action(create, "TimeoutError", {"x": 0}, {"x": 1}, step=1, result_ok=False, result_status="unknown")
    assert ea.type == D.ALLOW and ka.ctx.verification is True, (ea.type, ka.ctx.verification)
    ku = HarnessKernel(contract, [VerifyAndCommit()], policy=POLICY, env_type="fhir")
    eu = ku.after_action(create, "TimeoutError", {"x": 0}, {"x": 0}, step=1, result_ok=False, result_status="unknown")
    assert eu.type == D.REVISE and ku.ctx.verification is False, (eu.type, ku.ctx.verification)
    kf = HarnessKernel(contract, [VerifyAndCommit()], policy=POLICY, env_type="fhir")
    ef = kf.after_action(create, "{error}", {"x": 0}, {"x": 0}, step=1, result_ok=False, result_status="failed")
    assert ef.type == D.REVISE and kf.ctx.verification is False, (ef.type, kf.ctx.verification)


def test_no_new_progress_steers_after_second_repeat():
    # loop-detection: a successful action that yields no new evidence, closes no obligation, and changes no
    # state is NOT progress; the SAME semantic action repeated that way -> REVISE(no_new_progress) on the 2nd.
    contract = build_contract(TASK, env_type="fhir", policy=POLICY)
    k = HarnessKernel(contract, [ObligationLifecycle()], policy=POLICY, env_type="fhir")
    read = {"type": "tool", "tool": "check_allergies", "args": {"patient": "Patient/123"}}
    st = {"x": 0}
    d1 = k.after_action(read, "noop", st, st, step=1, result_ok=True)   # 1st no-progress -> just record
    assert d1 is None or d1.type != D.REVISE, d1
    d2 = k.after_action(read, "noop", st, st, step=2, result_ok=True)   # 2nd no-progress -> steer
    assert d2 is not None and d2.type == D.REVISE and getattr(d2.raw, "reason_code", None) == "no_new_progress", \
        (getattr(d2, "type", None), getattr(getattr(d2, "raw", None), "reason_code", None))
    # and a progress action (state changed) resets -> no REVISE
    k2 = HarnessKernel(contract, [ObligationLifecycle()], policy=POLICY, env_type="fhir")
    k2.after_action(read, "noop", {"x": 0}, {"x": 1}, step=1, result_ok=True)   # state changed -> progress
    d = k2.after_action(read, "noop", {"x": 1}, {"x": 2}, step=2, result_ok=True)
    assert d is None or d.type != D.REVISE, d


def test_must_resolve_pending_resolution_lifecycle():
    # must-resolve: a LOCALIZED hard defect from the adequacy audit opens a pending_resolution the next answer must
    # close; repeats increment attempts; a clean audit clears it; non-VALIDATED evidence cannot establish
    # must-resolve (the commit's prerequisite is unmet, so obligation_lifecycle REVISEs first).
    pol = H.load_policy(env_type="tool_sandbox", workspace=W.H0["tool_sandbox"])
    task = {"task_id": "m", "goal": "finding?", "context": {}, "environment": {"type": "tool_sandbox"}}
    contract = build_contract(task, env_type="tool_sandbox", policy=pol)

    def mk(jf, ok=True):
        k = HarnessKernel(contract, [ScopeEvidenceBinding(), ObligationLifecycle(), VerifyAndCommit()],
                          policy=pol, env_type="tool_sandbox", judge_fn=jf,
                          budget={"max_semantic_checks": 9})
        k.after_action({"type": "tool", "tool": "ImageDescription", "args": {}},
                       "a 3cm spiculated RUL nodule", None, None, step=0, result_ok=ok)
        return k

    cj = lambda p: AUDIT_HARD
    k = mk(cj)
    d1 = k.before_final("LUL effusion", step=1)
    assert d1.type == D.REVISE and d1.raw.reason_code == "evidence_contradiction", d1.raw.to_dict()
    pr = k.ledger.pending_resolution
    assert pr and pr["violation_type"] == "audit_hard" and pr["critical_claim"] == "LUL effusion" \
        and pr["evidence_ids"] == ["E1"] and pr["attempts"] == 1, pr
    # agent re-answers but conflict persists -> attempts increments, still unresolved
    d2 = k.before_final("still LUL effusion", step=2)
    assert d2.type == D.REVISE and k.ledger.pending_resolution["attempts"] == 2, k.ledger.pending_resolution
    # agent truly resolves -> a clean audit clears the pending resolution
    k.ctx.judge_fn = lambda p: AUDIT_CLEAN
    d3 = k.before_final("RUL nodule", step=3)
    assert d3.type == D.ALLOW and k.ledger.pending_resolution is None, (d3.type, k.ledger.pending_resolution)
    # NON-VALIDATED evidence basis (failed tool -> ATTEMPTED) cannot establish must-resolve -> no pending
    kf = mk(cj, ok=False)
    df = kf.before_final("LUL effusion", step=1)
    assert kf.ledger.pending_resolution is None and df.raw.reason_code != "evidence_contradiction", \
        (df.raw.reason_code, kf.ledger.pending_resolution)


def test_selective_repair_foundation():
    # Layer-2 foundation: the adequacy auditor splits HARD vs REPAIRABLE; the A/B selector is conservative.
    from harness.engines.semantic import audit_answer
    from workspace import H0, Workspace   # the A/B selector lives in the answer-candidate strategy
    _ac = sys.modules[Workspace(H0["tool_sandbox"]).load("middleware/answer_candidate.py", "AnswerCandidate").__module__]
    compare_answer_candidates, adopt_revised = _ac.compare_answer_candidates, _ac.adopt_revised
    jf = lambda p: ('{"addresses_task": false, "hard_violations": [], "repairable_gaps": '
                    '[{"type": "missed_discriminator", "claim": "Phrygian cap", "evidence_ids": ["E1"], '
                    '"critique": "ignores the duplicated lumen"}], "confidence": 0.82}')
    a = audit_answer("variant?", "ctx", "Phrygian cap", [{"type": "image", "value": "duplicated lumen"}], judge_fn=jf)
    assert not a.hard_violations and len(a.repairable_gaps) == 1 and a.top_gap()["type"] == "missed_discriminator"
    # no judge -> no fabricated defects
    assert audit_answer("g", "c", "a", [], judge_fn=None).repairable_gaps == []
    # conservative adopt: only a clear, resolved, no-new-hard improvement replaces A
    mk = lambda **kw: {"preferred": "revised", "margin": 0.3, "critique_resolved": True,
                       "revised_new_hard_violation": False, **kw}
    assert adopt_revised(mk()) is True
    assert adopt_revised(mk(margin=0.1)) is False                 # below tau -> keep A
    assert adopt_revised(mk(critique_resolved=False)) is False    # critique unresolved -> keep A
    assert adopt_revised(mk(revised_new_hard_violation=True)) is False  # B adds a hard violation -> keep A
    assert adopt_revised(mk(preferred="uncertain")) is False      # uncertain -> keep A
    assert adopt_revised({}) is False                             # no comparison -> keep A


def test_selective_repair_trigger_auto_recedes():
    # AUTO-RECEDING amplification: a repairable gap engages candidate repair ONLY on a real deficit
    # (addresses_task False, or auditor confidence >= threshold). A competent answer with a low-confidence
    # minor gap is ALLOWED -> the amplification layer trips less as the model improves.
    # The trigger is repair policy: MedCTA's workspace kernel middleware answer_repair reads verify_commit's audit
    # (verify_commit does not raise the REVISE itself).
    from workspace import H0, Workspace
    AnswerRepair = Workspace(H0["tool_sandbox"]).load("middleware/answer_repair.py", "AnswerRepair")
    pol = H.load_policy(env_type="tool_sandbox", workspace=W.H0["tool_sandbox"])
    task = {"task_id": "m", "goal": "finding?", "context": {}, "environment": {"type": "tool_sandbox"}}
    contract = build_contract(task, env_type="tool_sandbox", policy=pol)

    def mk(jf):
        k = HarnessKernel(contract, [ScopeEvidenceBinding(), ObligationLifecycle(), VerifyAndCommit(), AnswerRepair()],
                          policy=pol, env_type="tool_sandbox", judge_fn=jf,
                          budget={"max_semantic_checks": 5})
        k.after_action({"type": "tool", "tool": "ImageDescription", "args": {}},
                       "a 3cm RUL nodule", None, None, step=0, result_ok=True)
        return k

    low = ('{"addresses_task": true, "hard_violations": [], "repairable_gaps": '
           '[{"type": "over_specific", "claim": "x", "evidence_ids": ["E1"], "critique": "minor"}], "confidence": 0.4}')
    hi = ('{"addresses_task": false, "hard_violations": [], "repairable_gaps": '
          '[{"type": "unanswered", "claim": "x", "evidence_ids": ["E1"], "critique": "no dx"}], "confidence": 0.4}')
    conf = ('{"addresses_task": true, "hard_violations": [], "repairable_gaps": '
            '[{"type": "over_specific", "claim": "x", "evidence_ids": ["E1"], "critique": "c"}], "confidence": 0.9}')
    assert mk(lambda p: low).before_final("ans", step=1).type == D.ALLOW          # below threshold -> recede
    d2 = mk(lambda p: hi).before_final("ans", step=1)
    assert d2.type == D.REVISE and d2.raw.reason_code == "process_gap"          # not addressed -> repair
    assert d2.raw.capability == "answer_repair"
    d3 = mk(lambda p: conf).before_final("ans", step=1)
    assert d3.type == D.REVISE and d3.raw.reason_code == "process_gap"          # high-conf gap -> repair
    assert d3.raw.capability == "answer_repair"


def test_active_readback_reconcile_write():
    # infrastructure: a write that CLAIMED success but did not persist on read-back -> confirmed False.
    import environments as _E
    f = _E.PbNativeEnv.__new__(_E.PbNativeEnv)
    f._get = lambda path: {"id": "abc", "resourceType": "MedicationRequest"} if "abc" in path else {"error": "not found"}
    assert f.reconcile_write("fhir_create", {}, {"id": "abc", "resourceType": "MedicationRequest"})["confirmed"] is True
    assert f.reconcile_write("fhir_create", {}, {"id": "zzz", "resourceType": "MedicationRequest"})["confirmed"] is False
    assert f.reconcile_write("fhir_search", {}, {"entries": []})["confirmed"] is None     # read -> no-op
    # base env cannot reconcile -> None (safe no-op)
    assert _E.EnvironmentAdapter.__new__(_E.EnvironmentAdapter).reconcile_write("x", {}, {})["confirmed"] is None


def test_active_readback_portal_submit():
    # portal: a submit whose click left the persisted case state unchanged -> NOT confirmed.
    import environments as _E
    g = _E.GuiEnvReal.__new__(_E.GuiEnvReal)
    g.page = object()                       # non-None so reconcile engages
    g._read_state = lambda: None
    g.full_state = {"emr": {"case": "DEN-1", "disposition": "appeals"}}
    assert g.reconcile_write("submit", {}, {"ok": True, "state_changed": True})["confirmed"] is True
    assert g.reconcile_write("submit", {}, {"ok": True, "state_changed": False})["confirmed"] is False
    g.full_state = {}
    assert g.reconcile_write("submit", {}, {"ok": True})["confirmed"] is False        # nothing persisted
    assert g.reconcile_write("click", {}, {"ok": True})["confirmed"] is None          # not a commit


def test_unknown_unobservable_commit_reconciles_not_terminates():
    # an UNKNOWN commit whose effect is UNOBSERVABLE -> RECONCILE (restricted read-back recovery), NOT a
    # terminal ESCALATE. (Observable cases are resolved by the postcondition-first check in the other test.)
    contract = build_contract(TASK, env_type="fhir", policy=POLICY)
    create = {"type": "tool", "tool": "create_medication_request", "args": {"patient": "Patient/123"}}
    k = HarnessKernel(contract, [VerifyAndCommit()], policy=POLICY, env_type="fhir")
    e = k.after_action(create, "TimeoutError", None, None, step=1, result_ok=False, result_status="unknown")
    assert e.type == D.RECONCILE, (e.type, k.ctx.verification)
    assert k.ctx.verification is None


def test_goal_spec_compilation_oracle_blind():
    # the PUBLIC goal compiles into a structured, checkable goal_spec -- no judge -> None (fail-safe).
    from harness.engines.semantic import compile_goal_spec, audit_answer
    jf = lambda p: ('{"requested_operation": "submit a denial appeal", "required_effects": '
                    '["case status appealed"], "required_fields": ["appeal_reason"], "forbidden_effects": [], '
                    '"success_observables": ["disposition recorded"]}')
    gs = compile_goal_spec("Appeal the denial for DEN-1", "payer Aetna", judge_fn=jf)
    assert gs["requested_operation"] == "submit a denial appeal" and gs["required_fields"] == ["appeal_reason"]
    assert compile_goal_spec("g", "c", judge_fn=None) is None        # fail-safe
    # the auditor accepts goal_spec and still parses a verdict (goal-aware adequacy)
    aj = lambda p: ('{"addresses_task": false, "hard_violations": [], "repairable_gaps": [{"type": '
                    '"unanswered", "claim": "x", "evidence_ids": ["E1"], "critique": "missing appeal_reason"}], '
                    '"confidence": 0.9}')
    au = audit_answer("appeal?", "ctx", "done", [{"type": "e", "value": "x"}], judge_fn=aj, goal_spec=gs)
    assert au.top_gap() is not None and "goal_spec" not in str(au.to_dict())   # spec is in the prompt, not the verdict


def test_goal_alignment_before_commit():
    # BEFORE a commit, check the proposed write against the public goal_spec; missing required field
    # -> REVISE(goal_misalignment) so the agent completes it before committing.
    from workspace import H0, Workspace
    GoalAlignment = Workspace(H0["fhir"]).load("middleware/goal_alignment.py", "GoalAlignment")
    contract = build_contract(TASK, env_type="fhir", policy=POLICY)
    contract.meta = dict(contract.meta or {})
    contract.meta["goal_spec"] = {"requested_operation": "order X", "required_fields": ["dose"],
                                  "required_effects": [], "forbidden_effects": [], "success_observables": []}
    create = {"type": "tool", "tool": "create_medication_request", "args": {"patient": "Patient/123"}}
    kf = HarnessKernel(contract, [GoalAlignment(), VerifyAndCommit()], policy=POLICY,
                       env_type="fhir", judge_fn=lambda p: '{"aligned": false, "findings": [{"target_type": "resource_path", "target_path": "dose", "defect_type": "missing", "repair_operation": "ADD", "required_change": "Add the dose to the medication order.", "protected_paths": ["patient"]}]}',  # Scoped Repair structured finding
                       budget={"max_semantic_checks": 5})
    eff = kf.before_action(create, {"x": 0}, step=1)
    # a create/update is an INTERMEDIATE build step -> a completeness finding is recorded as
    # advisory, NOT a blocking REVISE (blocking intermediate creates derails the agent). Only a
    # terminal `submit` is gated. So a create yields ALLOW + an advisory record.
    assert eff.type == D.ALLOW, eff.type
    assert kf.ledger.advisories, "the completeness finding should be recorded as advisory"
    ka = HarnessKernel(contract, [GoalAlignment(), VerifyAndCommit()], policy=POLICY,
                       env_type="fhir", judge_fn=lambda p: '{"aligned": true, "missing": []}',
                       budget={"max_semantic_checks": 5})
    eff2 = ka.before_action(create, {"x": 0}, step=1)
    assert getattr(getattr(eff2, "raw", None), "reason_code", None) != "goal_misalignment"


def test_freshness_novelty_repeat_query_does_not_fire():
    import tempfile, os as _os
    ws = tempfile.mkdtemp()
    _os.makedirs(_os.path.join(ws, "output"), exist_ok=True)
    open(_os.path.join(ws, "assessment.txt"), "w").write("note reflecting the TSH lab")
    class _Env:  # minimal env: workspace + no state-summary staleness
        workspace = ws
    d = _deliverable_cls()({"goal": "Write your note to output/assessment.txt now"})
    import middleware_host as MH
    ctx = lambda: MH.MiddlewareContext(env=_Env(), task={"goal": "g"}, trajectory=[], step=4, max_steps=100,
                                       used_actions=0, judge_factory=lambda: None, errors=[])
    assert d.active and not d._missing(ctx())
    lab = {"resourceType": "Observation", "id": "obs-1",
           "meta": {"versionId": "1"}, "effectiveDateTime": "2026-01-01", "code": {"text": "TSH"}}
    def _sr(entries, step):
        return {"step": step, "event_type": "tool_call", "tool": "fhir_search",
                "result": {"entries": entries, "resourceType": "Bundle"}}
    # (1) retrieve lab (pre) -> write -> RE-QUERY THE SAME lab (post): NOT stale, no nudge
    traj = [_sr([lab], 1),
            {"step": 2, "event_type": "tool_call", "tool": "write_file",
             "args": {"path": "output/assessment.txt"}},
            _sr([lab], 3)]
    assert d.deliverable_freshness_nudge(ctx(), traj, None) is None
    # (2) a NEW lab (different id) retrieved post-write -> stale, nudge fires
    d.stale_nudges = 0
    newlab = {"resourceType": "Observation", "id": "obs-2",
              "meta": {"versionId": "1"}, "effectiveDateTime": "2026-01-02", "code": {"text": "prolactin"}}
    traj2 = traj[:2] + [_sr([lab, newlab], 3)]
    res = d.deliverable_freshness_nudge(ctx(), traj2, None)
    assert res is not None and "new patient data" in res[1]
    # (3) an UPDATED version of the same id (versionId bump) post-write -> stale, fires
    d.stale_nudges = 0
    amended = dict(lab, meta={"versionId": "2"})
    traj3 = traj[:2] + [_sr([amended], 3)]
    assert d.deliverable_freshness_nudge(ctx(), traj3, None) is not None


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



def test_chart_review_portable_to_held_out_emr():
    """ZERO-SHOT GENERALITY for the DATA-review module: chart_review (the SAME goal-conditional clinical
    module) is satisfiable on the held-out EMR too. Querying the EMR's OWN read tools (ehr_demographics/
    problems/labs_fetch) binds Patient/Condition/Observation record evidence that resolves the review
    obligations and OPENS the deliverable gate -- showing `produces_evidence` is a per-substrate ADAPTER
    fact, not a FHIR special-case, so the review obligation cannot loop unsatisfied on this substrate."""
    import workspace as W
    pol = H.load_policy(adapter="clinic_emr", workspace=W.H0["fhir"])   # the fhir workspace carries chart_review
    assert "chart_review" in pol["_clinical_modules"]
    task = {"task_id": "cr",
            "goal": "Review the patient chart: demographics, problem list, and recent labs, then write a summary note.",
            "context": {"mrn": "M-7"}, "environment": {"type": "fhir", "adapter": "clinic_emr"}}
    contract = build_contract(task, env_type="fhir", policy=pol)
    k = HarnessKernel(contract, [ScopeEvidenceBinding(), ObligationLifecycle(), VerifyAndCommit()],
                      policy=pol, env_type="fhir")
    note = {"type": "tool", "tool": "scratch_note", "args": {"mrn": "M-7", "text": "summary"}}
    # deliverable BEFORE any chart review -> REVISE (chart_reviewed unmet)
    assert k.before_action(note, step=0).type == D.REVISE
    # query the EMR's own read tools -> Patient / Condition / Observation record evidence (subject-bound,
    # standard {entries: [...]} collection envelope)
    k.after_action({"type": "tool", "tool": "ehr_demographics_fetch", "args": {"mrn": "M-7"}},
                   {"entries": [{"name": "Jane Doe", "dob": "1950-01-01"}]}, {"a": 0}, {"a": 1}, step=1, result_ok=True)
    k.after_action({"type": "tool", "tool": "ehr_problems_fetch", "args": {"mrn": "M-7"}},
                   {"entries": [{"code": "gout"}, {"code": "CKD"}]}, {"a": 1}, {"a": 2}, step=2, result_ok=True)
    k.after_action({"type": "tool", "tool": "ehr_labs_fetch", "args": {"mrn": "M-7"}},
                   {"entries": [{"creatinine": 1.4}]}, {"a": 2}, {"a": 3}, step=3, result_ok=True)
    assert k.ledger.obligation_state("check_demographics") == "SATISFIED"
    assert k.ledger.obligation_state("check_problem_list") == "SATISFIED"
    assert k.ledger.obligation_state("check_labs") == "SATISFIED"
    assert k.ledger.obligation_state("chart_reviewed") == "SATISFIED"
    # deliverable AFTER review -> gate opens (no infinite REVISE, no starved deliverable)
    assert k.before_action(note, step=4).type == D.ALLOW


def test_chart_review_absent_category_still_resolves_on_emr():
    """A patient who genuinely LACKS a category: the agent queries labs, the EMR returns an EMPTY collection
    -> CHECKED_ABSENT -> the review obligation STILL resolves ('checked = looked, present or absent'), so the
    agent is not trapped in an unsatisfiable loop. (On the held-out substrate; relies on the adapter's default
    absence_when_empty result-semantics.)"""
    import workspace as W
    pol = H.load_policy(adapter="clinic_emr", workspace=W.H0["fhir"])
    task = {"task_id": "cr2", "goal": "Review recent labs and write a note.",
            "context": {"mrn": "M-7"}, "environment": {"type": "fhir", "adapter": "clinic_emr"}}
    contract = build_contract(task, env_type="fhir", policy=pol)
    k = HarnessKernel(contract, [ScopeEvidenceBinding(), ObligationLifecycle(), VerifyAndCommit()],
                      policy=pol, env_type="fhir")
    # confirmed-empty read (entries: []), still subject-bound to M-7
    k.after_action({"type": "tool", "tool": "ehr_labs_fetch", "args": {"mrn": "M-7"}},
                   {"entries": []}, {"a": 0}, {"a": 1}, step=1, result_ok=True)
    assert k.ledger.obligation_state("check_labs") == "SATISFIED"
