"""Module C — Risk-Adaptive Verify-and-Commit. Operates on ctx.risk (from the declared effect) + the
commit point's POSTCONDITION predicate (generic evaluator). No tool names / dataset checks.

  R3 (declared unjudgeable)         -> before_action ESCALATE
  R2 commit (effect=irreversible)   -> after_action verifies the postcondition predicate
  final answer (a commit)           -> before_final semantic claim<->evidence support (if asked)
"""
from ..capability import Capability
from .. import decision as D
from ..risk import at_least, R2, R3
from ..predicates import evaluate as eval_predicate


class VerifyAndCommit(Capability):
    name = "verify_commit"

    def before_action(self, action, ctx):
        if ctx.risk == R3:
            return self._decide(D.ESCALATE, rule_id="unjudgeable_high_risk", deterministic=True,
                                reason="action is high-risk and cannot be reliably adjudicated",
                                feedback="This action is high-risk and cannot be auto-verified; escalating.")
        return None

    def after_action(self, action, result, before_state, after_state, ctx):
        if not at_least(ctx.risk or R2, R2):
            return None
        cp = ctx.contract.commit_point_for(ctx.sem) if (ctx.contract and ctx.sem) else None
        post = (cp or {}).get("postcondition")
        # GENERIC predicate evaluation: True verified, False violated, None unverifiable (no guess).
        verdict = eval_predicate(post, before_state, after_state, ctx.sem)
        if verdict is False:
            rid = post.get("type") if isinstance(post, dict) else (post or "post_commit_no_state_change")
            return self._decide(
                D.REVISE, rule_id=rid, deterministic=True,
                reason="commit postcondition not satisfied (observable state unchanged / inconsistent)",
                feedback="The commit did not produce the expected state change — re-check and retry.")
        return None

    def before_final(self, answer, ctx):
        cp = ctx.contract.commit_point_for(ctx.sem) if (ctx.contract and ctx.sem) else None
        post = (cp or {}).get("postcondition")
        ptype = post.get("type") if isinstance(post, dict) else post
        if not ptype or "support" not in str(ptype):
            return None
        if not ctx.spend_semantic():
            ctx.ledger.add_unresolved_risk("semantic_claim_support",
                                           "claim<->evidence not verified (no judge / budget spent)")
            return None
        from ..engines.semantic import verify_claim_support
        v = verify_claim_support(answer, list(ctx.ledger.evidence), judge_fn=ctx.judge_fn)
        if v.supported is True:
            return None
        if v.supported is False and (v.confidence or 0) >= 0.5:
            return self._decide(
                D.REVISE, rule_id=ptype, deterministic=False, extra={"semantic": v.to_dict()},
                reason="final answer not supported by the gathered evidence: %s" % v.reason,
                feedback="Your answer is not supported by the evidence you gathered (%s) — re-examine "
                         "before answering." % v.reason)
        return self._decide(
            D.ESCALATE, rule_id="semantic_low_confidence", deterministic=False,
            extra={"semantic": v.to_dict()},
            reason="claim<->evidence support is low-confidence/unknown: %s" % v.reason)
