"""Module — Goal Alignment (P1.3). BEFORE an operational commit (create/update/submit), check the PROPOSED
write against the compiled public goal_spec + the current draft state: does it satisfy the requested
operation, required fields, and required effects? If not, REVISE with the specific missing requirement so the
agent completes the write correctly BEFORE committing. This verifies the commit satisfies the TASK GOAL, not
just that a state change happened. Oracle-blind (goal_spec is compiled from the PUBLIC goal only). Active only
when MH_REPAIR is soft/select/full (ablatable amplification layer)."""
import os
from ..capability import Capability
from .. import decision as D
from ..risk import at_least, R2


class GoalAlignment(Capability):
    # LAYER (see HARNESS_DESIGN.md): AMPLIFICATION -- goal-aware pre-commit verification (provisional, MH_REPAIR-gated)
    name = "goal_alignment"

    def before_action(self, action, ctx):
        if os.environ.get("MH_REPAIR", "hard") not in ("soft", "select", "full"):
            return None                                            # ablatable: off unless repair is enabled
        if not at_least(ctx.risk or R2, R2):
            return None                                            # only operational commits
        if not (ctx.sem and getattr(ctx.sem, "semantic_type", None) in ("create", "update", "submit")):
            return None
        gs = (ctx.contract.meta or {}).get("goal_spec") if (ctx.contract and ctx.contract.meta) else None
        if not gs or not ctx.judge_fn or not ctx.spend_semantic():
            return None
        from ..engines.semantic import verify_goal_alignment
        r = verify_goal_alignment(gs, ctx.current_state, getattr(ctx.sem, "raw", action), ctx.judge_fn)
        if r.get("aligned") is False and r.get("missing"):
            _m = "; ".join(str(x) for x in r["missing"])
            return self._decide(
                D.REVISE, rule_id="goal_alignment", reason_code="goal_misalignment", deterministic=False,
                missing_obligations=list(r.get("missing") or []),
                reason="proposed commit does not yet satisfy the task goal: %s" % _m,
                feedback="Before you commit, your draft does NOT yet satisfy the task requirement(s): %s. "
                         "Complete or correct them in the form/payload, THEN commit." % _m)
        return None
