"""MedCTA selector CLOSED-LOOP test, judge- and VLM-independent. Forces the semantic
helpers to the 'genuinely supported' verdicts a correct E1+E2 case would produce (E2 addresses the
discriminator; E1+E2 jointly support B; E2 alone is a discriminating increment; comparator prefers B;
reverify ALLOW) and asserts evaluate_candidate ADOPTS B. If it KEEPS A here, the selector control flow --
not the VLM, not the budget -- is the blocker."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from workspace import H0, Workspace
# the adoption gates are the MedCTA workspace's answer-candidate middleware; patch that module's helpers
S = sys.modules[Workspace(H0["tool_sandbox"]).load("middleware/answer_candidate.py", "AnswerCandidate").__module__]

fails = []

# force each gate's judge-backed helper to the correct verdict for a genuinely-supported B.
S.extract_decision_signature = lambda gs, ans, judge_fn=None: {"core": ("portal+smv" if ("mesenteric" in ans.lower() or "smv" in ans.lower()) else "portal"), "slots": {}}
S.compare_decision_signatures = lambda a, b, judge_fn=None: {"changed": a != b, "changed_slots": ["diagnosis"], "uncertain": False, "reason": "core flip"}
S.verify_change_against_delta = lambda goal, sa, sb, ne, judge_fn=None: {"supported": True, "changed_slots": ["diagnosis"], "supporting_evidence_ids": ["E2"], "confidence": 0.9}
try:
    from harness.engines.semantic import SemanticVerdict
    S.verify_claim_support = lambda goal, pub, ans, ev, judge_fn=None: SemanticVerdict(True, 0.9, "E1+E2 support B", relation="supports")
except Exception:
    class _V:
        def __init__(s): s.supported = True
    S.verify_claim_support = lambda goal, pub, ans, ev, judge_fn=None: _V()
S.compare_answer_candidates = lambda goal, pub, a, b, crit, ev, judge_fn=None: {"preferred": "revised", "margin": 0.5, "critique_resolved": True, "revised_new_hard_violation": False, "reason": "B strictly better"}
S.preserves_non_target_content = lambda a, b, crit, judge_fn=None: True

class _AllowRV:
    type = "ALLOW"; events = []

intervention = {"type": "ACQUIRE", "root_answer": "Portal vein thrombosis.",
                "critique": "confirm whether the SMV is involved", "baseline_ev_ids": set()}
B = "Portal vein thrombosis with superior mesenteric vein thrombosis."
goal_spec = {"goal": "name the venous thrombosis present", "public_context": ""}
E1 = {"evidence_id": "E1", "status": "VALIDATED", "scope_relation": "own", "value_full": "portal venous filling defect"}
E2 = {"evidence_id": "E2", "status": "VALIDATED", "scope_relation": "own", "value_full": "SMV also contains a filling defect"}

dec, reason = S.evaluate_candidate(intervention, B, goal_spec, [E1, E2], [E2],
                                   judge_fn=(lambda p: "{}"), reverify_fn=(lambda ans: _AllowRV()))
if dec != "ADOPT_B":
    fails.append("SELECTOR CONTROL-FLOW BUG: KEEP_A despite two-stage support + comparator revised(margin .5) "
                 "+ reverify ALLOW -> reason=%s" % reason)

# NEGATIVE control: if the delta contribution is FALSE (E2 irrelevant), it must KEEP_A even with total support.
S.verify_change_against_delta = lambda goal, sa, sb, ne, judge_fn=None: {"supported": False, "changed_slots": [], "supporting_evidence_ids": [], "confidence": 0.0}
dec2, reason2 = S.evaluate_candidate(intervention, B, goal_spec, [E1, E2], [E2],
                                     judge_fn=(lambda p: "{}"), reverify_fn=(lambda ans: _AllowRV()))
if dec2 != "KEEP_A":
    fails.append("SAFETY: an irrelevant delta must KEEP_A, got %s (%s)" % (dec2, reason2))

if fails:
    print("FAIL:")
    for x in fails:
        print("  -", x)
    sys.exit(1)
print("PASS: selector ADOPTS a genuinely-supported B (control flow correct); rejects an irrelevant delta.")
sys.exit(0)
