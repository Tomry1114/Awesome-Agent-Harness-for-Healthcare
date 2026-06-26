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
        # final answer = commit. P0: ALLOW (claim<->evidence linking is Module-C / P3 work).
        # Keep the hook so P3 can require key claims be backed by subject-scoped evidence.
        return None


def _name(action):
    if not isinstance(action, dict):
        return ""
    if action.get("type") == "final":
        return "final"
    return action.get("tool") or action.get("action") or ""


