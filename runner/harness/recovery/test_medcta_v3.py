"""Unit tests for the Bounded Clinical Recovery v3 EVIDENCE-ACQUISITION + AGENT-REENTRY path (perceptual).

Standalone: sys.exit(0) on pass, non-zero on fail. No model calls, no GPU, oracle-blind.
Run: python3 runner/harness/recovery/test_medcta_v3.py

Uses a STUB perception env (FakeEnv, no VLM) behind the REAL PerceptualSubstrateAdapter, a STUB judge, and a
STUB 're-entry' callable that returns a canned agent-regenerated answer B. Asserts:
  (a) answered-without-looking            -> an ACQUIRE-only plan (no commit / no staged_write) -> VERIFIED
  (b) region not derivable from question  -> BLOCKED_AMBIGUOUS_TARGET
  (c) acquired IMAGE evidence supports B   -> ACCEPTED (adopt B)
  (d) B not supported by the delta         -> KEPT_ORIGINAL
  (e) claim grounded only in GoogleSearch web text -> REJECTED as image grounding -> KEPT_ORIGINAL
  (f) region read fell back (localization.resolved False) -> BLOCKED_MISSING_EVIDENCE
Plus: the whole path emits NO irreversible_commit and NO created ids (read-only), and requires agent re-entry.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from harness.recovery import contracts as C, metrics as M, RecoveryKernel
from harness.recovery.substrate.perceptual import PerceptualSubstrateAdapter
from harness.recovery.workflows.evidence_acquisition import EvidenceAcquisitionWorkflow, AgentReentrySignal
from harness.recovery.benchmark.medcta import MedctaBenchmarkAdapter
from harness.recovery.benchmark.medcta_register import build_registry, build_stack
from harness.recovery.acceptance import evaluate as accept_evaluate, AcceptanceResult, _is_image_grounded


# ------------------------------------------------------------------------------------------------- FIXTURES
QUESTION = "Based on the CT image, what type of venous thrombosis is present?"
ANSWER_A = "The portal vein shows a bland thrombus (acute)."          # answered WITHOUT looking
ANSWER_B = "The portal vein shows a neoplastic tumor thrombus (malignant invasion)."

TOOLS = [
    {"name": "OCR", "signature": "(image)->text"},
    {"name": "ImageDescription", "signature": "(image)->text"},
    {"name": "RegionAttributeDescription", "signature": "(image,region)->text"},
    {"name": "GoogleSearch", "signature": "(query)->text"},
    {"name": "Calculator", "signature": "(expr)->number"},
]

RESOLVED_TRUE = {"text": "hypodense filling defect expanding the portal vein",
                 "localization": {"requested": "porta hepatis", "mode": "semantic", "resolved": True}}
RESOLVED_FALSE = {"text": "axial CT of the abdomen",
                  "localization": {"requested": "porta hepatis", "mode": "none", "resolved": False}}


def make_task():
    return {"task_id": "MCTA-t", "context": {"text": QUESTION, "images": [{"asset_id": "img-0"}]},
            "available_tools": TOOLS, "environment": {"type": "tool_sandbox"}}


class FakeEnv(object):
    """Stub perception env (no VLM). Returns canned tool outputs so the REAL substrate exercises its own
    localization/empty mechanics without a GPU."""

    def __init__(self, region_output):
        self.region_output = region_output
        self.calls = []

    def available_tools(self):
        return [t["name"] for t in TOOLS]

    def call_tool(self, name, args):
        self.calls.append((name, args))
        if name == "RegionAttributeDescription":
            return {"tool": name, "args": args, "output": self.region_output}
        if name == "ImageDescription":
            return {"tool": name, "args": args, "output": {"text": "axial CT of the abdomen"}}
        if name == "OCR":
            return {"tool": name, "args": args, "output": "[no text]"}
        if name == "GoogleSearch":
            return {"tool": name, "args": args, "output": "portal vein thrombosis per literature"}
        return {"tool": name, "args": args, "output": ""}


class FakeBenchJudge(object):
    """Stub judge for the benchmark adapter (oracle-blind claim decomposition + discriminator). region=None
    models a question from which no perceptual target can be derived."""

    def __init__(self, region="porta hepatis"):
        self.region = region

    def __call__(self, prompt):
        if "Decompose the agent FINAL ANSWER into atomic claims" in prompt:
            return json.dumps({"claims": [{"text": "filling defect in the porta hepatis",
                                           "claim_type": "perceptual", "region": self.region,
                                           "modality": "CT", "attribute": "filling defect"}]})
        if "most discriminating OBSERVABLE feature" in prompt or "TWO most plausible interpretations" in prompt:
            return json.dumps({"hypotheses": ["bland thrombus", "tumor thrombus"],
                               "region": self.region, "attribute": "filling defect"})
        return "{}"


class FakeAcceptJudge(object):
    """Stub judge for the acceptance gate. supported=True models new IMAGE evidence that directly supports the
    A->B core flip (>=0.8 confidence); supported=False models an unsupported flip."""

    def __init__(self, supported=True):
        self.supported = supported

    def __call__(self, prompt):
        p = prompt
        if "Extract the CORE decision" in p:
            if "neoplastic" in p:                       # signature of B
                return json.dumps({"requested_operation": None, "primary_conclusion": "tumor thrombus",
                                   "polarity": "present", "target": "portal vein",
                                   "severity_or_priority": "urgent", "recommended_action": None})
            return json.dumps({"requested_operation": None, "primary_conclusion": "bland thrombus",
                               "polarity": "present", "target": "portal vein",
                               "severity_or_priority": "routine", "recommended_action": None})
        if "Two DECISION SIGNATURES" in p:
            return json.dumps({"changed": True, "changed_slots": ["primary_conclusion"],
                               "uncertain": False, "reason": "primary conclusion changed"})
        if "NEWLY ACQUIRED" in p:
            return json.dumps({"supported": self.supported, "changed_slots": ["primary_conclusion"],
                               "supporting_evidence_ids": (["obs-0"] if self.supported else []),
                               "confidence": (0.9 if self.supported else 0.1), "reason": "delta"})
        if "comparing TWO candidate FINAL ANSWERS" in p:
            return json.dumps({"preferred": "revised", "margin": 0.6, "critique_resolved": True,
                               "revised_new_hard_violation": False, "reason": "clearly better"})
        if "REMOVED or CONTRADICTED substantive content" in p:
            return json.dumps({"preserved": True, "removed": ""})
        return "{}"


IMG_EVIDENCE = [{"evidence_id": "obs-0", "type": "RegionAttributeDescription",
                 "value_full": "hypodense filling defect expanding the portal vein",
                 "source_channel": "radiology_image", "source_instance_id": "image:primary",
                 "extractor": "image_vlm", "region": "porta hepatis"}]
WEB_EVIDENCE = [{"evidence_id": "ev-web", "type": "GoogleSearch",
                 "value_full": "portal vein thrombosis is commonly bland per literature",
                 "source_channel": "external_web", "source_instance_id": "web:domain:example.org",
                 "extractor": "web_search"}]
GOAL_SPEC = {"goal": QUESTION, "public_context": QUESTION}


# ------------------------------------------------------------------------------------------------- HARNESS
FAILS = []


def check(name, cond):
    print(("PASS" if cond else "FAIL"), name)
    if not cond:
        FAILS.append(name)


def run_episode(env, bench_judge_region="porta hepatis"):
    """Wire a real stack (stub env + stub judge) and run one acquire episode; return (result, stack, ctx, goal)."""
    task = make_task()
    trajectory = {"final_answer": ANSWER_A, "observations": []}      # answered WITHOUT looking
    substrate = PerceptualSubstrateAdapter(env)
    bench = MedctaBenchmarkAdapter(judge_fn=FakeBenchJudge(region=bench_judge_region))
    registry = build_registry()
    kernel = RecoveryKernel()
    ctx = kernel._build_ctx(bench, task, driver=env)
    commitments = bench.resolve_commitments(task, trajectory, None, None, ctx)
    goal = commitments[0] if commitments else None
    res = kernel.run_episode(bench, registry, substrate, env, task, trajectory, None, None)
    return res, substrate, ctx, goal, registry


# ================================================================================================= (a)
def test_a_acquire_plan_no_commit():
    env = FakeEnv(RESOLVED_TRUE)
    res, substrate, ctx, goal, registry = run_episode(env)
    check("(a) a commitment was resolved (answered-without-looking is an evidence gap)", goal is not None)

    wf = registry.all()[0]
    plan = wf.compile_plan(goal, ctx)
    kinds = [s.kind for s in plan.steps]
    check("(a) plan emits at least one ACQUIRE step", any(k == C.ACQUIRE for k in kinds))
    check("(a) plan emits NO irreversible_commit", C.IRREVERSIBLE_COMMIT not in kinds)
    check("(a) plan emits NO staged_write", C.STAGED_WRITE not in kinds)
    check("(a) plan emits ONLY acquire steps", all(k == C.ACQUIRE for k in kinds))
    check("(a) plan has no transaction_contract (no commit)", plan.transaction_contract is None)
    check("(a) workflow declares emits_commit == False", wf.emits_commit is False)

    check("(a) episode VERIFIED (evidence acquired)", res.state == C.VERIFIED)
    check("(a) NO created ids (read-only path, no mutation)", not res.created_ids)
    check("(a) metrics bucket = verified_recovery", M.classify(res) == M.VERIFIED_RECOVERY)


# ================================================================================================= (b)
def test_b_region_not_derivable_ambiguous():
    env = FakeEnv(RESOLVED_TRUE)
    res, substrate, ctx, goal, registry = run_episode(env, bench_judge_region=None)
    check("(b) commitment still resolved (evidence gap exists)", goal is not None)
    check("(b) target region is null (not derivable from the question)",
          (goal.committed_fields or {}).get("region") is None)
    check("(b) episode -> BLOCKED_AMBIGUOUS_TARGET", res.state == C.BLOCKED_AMBIGUOUS_TARGET)
    check("(b) blocked is CORRECTLY_BLOCKED (not a failure)", M.classify(res) == M.CORRECTLY_BLOCKED)
    check("(b) no tool was ever called", env.calls == [])


# ================================================================================================= (f)
def test_f_localization_fallback_missing_evidence():
    env = FakeEnv(RESOLVED_FALSE)                                    # region read fell back to whole image
    res, substrate, ctx, goal, registry = run_episode(env)
    check("(f) localization.resolved False -> BLOCKED_MISSING_EVIDENCE",
          res.state == C.BLOCKED_MISSING_EVIDENCE)
    check("(f) blocked argument is the operational evidence gate",
          res.blocked_argument == "evidence_acquired")
    check("(f) correctly_blocked bucket", M.classify(res) == M.CORRECTLY_BLOCKED)


# ================================================================================================= (c)
def test_c_evidence_supports_b_accept():
    # run the acquire episode (image evidence obtained), then re-enter the agent and gate B.
    env = FakeEnv(RESOLVED_TRUE)
    res, substrate, ctx, goal, registry = run_episode(env)
    wf = registry.all()[0]
    state_view = substrate.read_state(["evidence_ledger"])
    signal = wf.agent_reentry(goal, state_view, ctx)
    check("(c) agent-reentry signal requires the ROOT AGENT to regenerate B", signal.required is True)
    check("(c) signal carries acquired IMAGE evidence", len(signal.acquired_evidence) >= 1)
    check("(c) acquired evidence is image-grounded", _is_image_grounded(signal.acquired_evidence[0]))

    def reentry_fn(sig):                                             # STUB root-agent re-invocation -> canned B
        return ANSWER_B
    candidate_b = reentry_fn(signal)

    acc = accept_evaluate(ANSWER_A, candidate_b, GOAL_SPEC,
                          all_evidence=signal.acquired_evidence, new_evidence=signal.acquired_evidence,
                          judge_fn=FakeAcceptJudge(supported=True))
    check("(c) harness did NOT author B (B came from the re-entry callable)", candidate_b == ANSWER_B)
    check("(c) acceptance = ACCEPTED (adopt B)", acc.state == C.ACCEPTED and acc.adopted)
    check("(c) accepted maps to verified_recovery", M.classify(acc) == M.VERIFIED_RECOVERY)


# ================================================================================================= (d)
def test_d_b_not_supported_keep_original():
    env = FakeEnv(RESOLVED_TRUE)
    res, substrate, ctx, goal, registry = run_episode(env)
    wf = registry.all()[0]
    signal = wf.agent_reentry(goal, substrate.read_state(["evidence_ledger"]), ctx)
    acc = accept_evaluate(ANSWER_A, ANSWER_B, GOAL_SPEC,
                          all_evidence=signal.acquired_evidence, new_evidence=signal.acquired_evidence,
                          judge_fn=FakeAcceptJudge(supported=False))
    check("(d) acceptance = KEPT_ORIGINAL", acc.state == C.KEPT_ORIGINAL and not acc.adopted)
    check("(d) reason cites unsupported delta", "not_supported_by_delta" in acc.reason)
    check("(d) kept_original maps to correctly_blocked", M.classify(acc) == M.CORRECTLY_BLOCKED)


# ================================================================================================= (e)
def test_e_web_only_grounding_rejected():
    # B flips the core decision but the ONLY new evidence is GoogleSearch web text -> not image grounding.
    acc = accept_evaluate(ANSWER_A, ANSWER_B, GOAL_SPEC,
                          all_evidence=WEB_EVIDENCE, new_evidence=WEB_EVIDENCE,
                          judge_fn=FakeAcceptJudge(supported=True))     # even a POSITIVE judge cannot rescue it
    check("(e) web evidence is NOT image grounding", not _is_image_grounded(WEB_EVIDENCE[0]))
    check("(e) web-only new evidence stripped before delta check", acc.image_evidence == [])
    check("(e) acceptance = KEPT_ORIGINAL", acc.state == C.KEPT_ORIGINAL)
    check("(e) reason records web-only rejection", "web_only_evidence_rejected" in acc.reason)


# ================================================================================================= extras
def test_declined_when_no_gap():
    # agent already LOOKED (its trajectory covers the region) -> no evidence gap -> DECLINED_NO_COMMITMENT.
    task = make_task()
    covered = {"observation_id": "o0", "tool_capability": "RegionAttributeDescription",
               "region": "porta hepatis", "modality": "CT",
               "attributes_observed": ["filling defect"], "result_status": "valid",
               "content": "filling defect present"}
    trajectory = {"final_answer": ANSWER_A, "observations": [covered]}
    bench = MedctaBenchmarkAdapter(judge_fn=FakeBenchJudge())
    ctx = bench.context(task)
    commitments = bench.resolve_commitments(task, trajectory, None, None, ctx)
    check("(g) already-looked -> no commitment", commitments == [])

    kernel = RecoveryKernel()
    res = kernel.run_episode(bench, build_registry(), PerceptualSubstrateAdapter(FakeEnv(RESOLVED_TRUE)),
                             None, task, trajectory, None, None)
    check("(g) kernel reports DECLINED_NO_COMMITMENT (not a failure)",
          res.state == C.DECLINED_NO_COMMITMENT)
    check("(g) declined is not failed", M.classify(res) != M.FAILED_RECOVERY)


def test_substrate_read_only_and_affordance_blocks():
    sub = PerceptualSubstrateAdapter(FakeEnv(RESOLVED_TRUE))
    # a mutation kind is refused (this substrate performs no writes)
    out = sub.execute_primitive(C.IRREVERSIBLE_COMMIT, {"action": {}}, auth=object())
    check("(h) substrate refuses irreversible_commit (read-only)", out.status == C.RESULT_FAILED)
    # null region -> ambiguous target
    obs = {"available_tools": TOOLS}
    check("(h) null region -> BLOCKED_AMBIGUOUS_TARGET",
          sub.resolve_affordance({"region": None}, obs) == C.BLOCKED_AMBIGUOUS_TARGET)
    # a region with NO perception tool available -> unresolved affordance
    check("(h) no perception tool -> BLOCKED_UNRESOLVED_AFFORDANCE",
          sub.resolve_affordance({"region": "porta hepatis"},
                                 {"available_tools": [{"name": "Calculator", "signature": "(expr)->number"}]})
          == C.BLOCKED_UNRESOLVED_AFFORDANCE)
    # localization.resolved False classifies as UNKNOWN (evidence not actually obtained)
    from harness.recovery.contracts import Outcome
    check("(h) resolved-False result classifies UNKNOWN",
          sub.classify_result(Outcome(status=C.RESULT_OK, result=RESOLVED_FALSE)) == C.RESULT_UNKNOWN)


def test_stack_builds():
    stack = build_stack(FakeEnv(RESOLVED_TRUE), judge_fn=FakeBenchJudge())
    check("(i) build_stack wires substrate", stack.substrate_adapter is not None)
    check("(i) build_stack wires a workflow registry", len(stack.workflow_registry) == 1)
    check("(i) build_stack wires the benchmark adapter", stack.benchmark_adapter is not None)


def main():
    test_a_acquire_plan_no_commit()
    test_b_region_not_derivable_ambiguous()
    test_f_localization_fallback_missing_evidence()
    test_c_evidence_supports_b_accept()
    test_d_b_not_supported_keep_original()
    test_e_web_only_grounding_rejected()
    test_declined_when_no_gap()
    test_substrate_read_only_and_affordance_blocks()
    test_stack_builds()
    print("-" * 60)
    if FAILS:
        print("FAILED %d check(s): %s" % (len(FAILS), FAILS))
        sys.exit(1)
    print("ALL CHECKS PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
