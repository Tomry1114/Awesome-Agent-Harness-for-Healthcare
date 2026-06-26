"""Module C — Risk-Adaptive Verify-and-Commit (the key module: the shared gap is LOW Verification).

Risk tiers gate the control strategy:
  R0 read           -> allow + record
  R1 reversible     -> prospective (pre-) check (subject + prerequisites; A/B supply those)
  R2 commit         -> prospective check AND retrospective (post-) verification
  R3 unjudgeable    -> ESCALATE

P0 retrospective check is GENERIC + deterministic: after an R2 commit, re-read the environment and
require that state actually changed (an API "success" that left the world unchanged is not success).
Per-dataset post-conditions (read-back of the created resource / case status != Draft / claim-evidence
linking for the final answer) are layered in P1–P3. The final answer is itself a commit point.
"""
from ..capability import Capability
from .. import decision as D
from ..risk import classify_risk, at_least, R2, R3
from ..engines.deterministic import state_changed


class VerifyAndCommit(Capability):
    name = "verify_commit"

    def before_action(self, action, ctx):
        risk = (ctx.risk_of(action) if ctx.risk_of else classify_risk(action, ctx.contract, ctx.policy))
        if risk == R3:
            return self._decide(D.ESCALATE, rule_id="unjudgeable_high_risk", deterministic=True,
                                reason="action is high-risk and cannot be reliably adjudicated",
                                feedback="This action is high-risk and cannot be auto-verified; escalating.")
        return None

    def after_action(self, action, result, before_state, after_state, ctx):
        risk = (ctx.risk_of(action) if ctx.risk_of else classify_risk(action, ctx.contract, ctx.policy))
        if not at_least(risk, R2):
            return None
        cp = ctx.contract.commit_point_for(_name(action)) if ctx.contract else None
        post = (cp or {}).get("postcondition")
        # DETERMINISTIC retrospective check (no keyword guessing): a commit whose effect on the
        # environment is OBSERVABLE and DID NOT change the state is not a real commit. When the state is
        # not observable (state_changed is None), we do NOT guess -> no false REVISE. The structured,
        # per-dataset postcondition verifier (read-back of the created resource / case status) is layered
        # in P1–P3 via `post`; this generic check is the deterministic floor.
        if state_changed(before_state, after_state) is False:
            return self._decide(
                D.REVISE, rule_id=post or "post_commit_no_state_change", deterministic=True,
                reason="commit left the (observable) environment state unchanged",
                feedback="The commit did not change the environment state — re-check and retry.")
        return None

    def before_final(self, answer, ctx):
        # P3 SEMANTIC commit check: when the contract's final commit point asks for claim<->evidence
        # support, verify the answer is SUPPORTED by the gathered (image-derived) evidence using the
        # INDEPENDENT injected judge. Fail-safe: no judge / budget spent -> do NOT block (record that the
        # claim was not verified). supported -> ALLOW; unsupported(high conf) -> REVISE; low-conf/unknown
        # -> ESCALATE (cannot reliably adjudicate).
        cp = ctx.contract.commit_point_for("final") if ctx.contract else None
        post = (cp or {}).get("postcondition")
        if not post or "support" not in str(post):
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
                D.REVISE, rule_id=post, deterministic=False, extra={"semantic": v.to_dict()},
                reason="final answer not supported by image-derived evidence: %s" % v.reason,
                feedback="Your answer is not supported by the image evidence you gathered (%s) — "
                         "re-examine the image before answering." % v.reason)
        return self._decide(
            D.ESCALATE, rule_id="semantic_low_confidence", deterministic=False,
            extra={"semantic": v.to_dict()},
            reason="claim<->evidence support is low-confidence/unknown: %s" % v.reason)


def _name(action):
    if not isinstance(action, dict):
        return ""
    if action.get("type") == "final":
        return "final"
    return action.get("tool") or action.get("action") or ""


