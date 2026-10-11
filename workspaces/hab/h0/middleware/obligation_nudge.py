# -*- coding: utf-8 -*-
"""Obligation nudge: do not accept a premature final while the task's completion obligations are unmet.

At the agent's final, if the task's public obligations are not yet observably realized and enough budget remains,
the final is turned back with a CONTINUE nudge. Oracle-blind: whether they are met is the shell's obligation model
(ctx.obligations(): the task's completion conjunction, composed from the public goal and the agent's own decisions, over
the persisted portal state), never a gold value. Bounded, so
it can never loop forever.

Host before_final (runner/middleware_host.py). The shell applies the Nudge: the final is refused and the agent gets a
turn; the repair budget stays locked and no repair turn is counted.
"""
from workspace_api import Nudge

MAX_NUDGES = 3        # per task
MIN_REMAINING = 6     # actions of budget that must be left for continuing to be worth it

MESSAGE = ("You are finalizing, but the task's required workflow is NOT yet complete and you still have "
           "%d action(s) of budget left. Do NOT stop here. Re-read the current page and the task goal, "
           "then continue toward the unmet objective: reach the required surface, complete the required "
           "fields/actions, submit, and record the result. Take the next concrete action now.")


class ObligationNudge(object):
    def __init__(self, task):
        self.nudges = 0

    def before_final(self, ctx, answer):
        if ctx.budget_spent_at_turn_start:
            return None
        if self.nudges >= MAX_NUDGES:
            return None
        remaining = ctx.max_steps - ctx.live_used_actions   # agent + recovery actions now, recovery included
        if remaining < MIN_REMAINING:
            return None
        try:
            met, decisions, _conjuncts = ctx.obligations()
        except Exception:
            return None                                     # never block finishing on a checker error
        if met:
            return None
        self.nudges += 1
        ctx.emit({"step": ctx.step, "event_type": "obligation_nudge", "unmet": True, "decisions": list(decisions or ()),
                  "attempt": self.nudges, "status": "ok"})
        return Nudge("obligation", MESSAGE % remaining, "obligation_unmet", "CONTINUE")
