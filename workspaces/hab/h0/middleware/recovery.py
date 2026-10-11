# -*- coding: utf-8 -*-
"""Recovery: closed-loop GUI completion at the agent's final, and once more after a forced stop.

At every agent final the strategy (workspaces/hab/h0/recovery/, Bounded Clinical Recovery) tries to complete the
effect the agent committed to -- navigate, fill what the agent's own values or the persisted records bind, acquire the
evidence reads the agent skipped, submit where the agent began the submission -- and verifies it against the task's
completion conjunction. When the gap is one it must not author (a field needing new content, a decision needing
evidence) it hands the final back to the agent naming the gap (re-entry). Oracle-blind: the agent's own trajectory,
the public task and the persisted portal state, never gold.

Host middleware (runner/middleware_host.py), first in the HAB manifest:
    before_final(ctx, answer)  run one recovery invocation; then maybe re-entry -> Revision(stage="reentry")
    after_episode(ctx)         after tool_budget_exhausted, run one invocation in the shell's post-stop allowance

What is the shell's and not decided here: whether an invocation may start (ctx.begin_harness_actions: a kernel, the
per-task total cap), every action (ctx.request_action / ctx.read: ceilings, irreversible-commit checks, write provenance,
the kernel with origin=recovery, execution, recording, the content guard), the post-stop allowance and its size
(ctx.post_stop_allowance), and what a Revision does to the loop. What is decided here: the strategy, the blocked-episode
streak stop (MAX_BLOCKED_RECOVERY), re-entry detection, text and rounds (MAX_REENTRY_ROUNDS), and taking the post-stop
allowance at all.
"""
import workspace_api as _W
from workspace_api import Revision

_entry = _W.load_relative(__file__, "../recovery/entry.py")

MAX_BLOCKED_RECOVERY = 2      # consecutive BLOCKED recovery invocations before recovery stops re-engaging
MAX_REENTRY_ROUNDS = 2        # root-agent re-entry rounds on an evidence/authoring gap

AUTHORING_MESSAGE = ("Your submission cannot be finalized yet: the field '%s' has no grounded source in what you have "
                     "observed. Open the relevant records (the referral / clinical note / coverage) and re-evaluate '%s' "
                     "from them, then complete and submit the form. Do NOT reuse a prior value unless the records "
                     "support it.")
EVIDENCE_MESSAGE = ("Your submission cannot be finalized yet: a required decision is not grounded in the available "
                    "evidence. Open the relevant records (denial details / remittance / clinical note) and re-derive "
                    "your decision and note from them before submitting.")


def reentry_request(summary):
    """The first episode of an invocation summary that asks for the root agent: {kind, field} or None.
    needs_new_content -> authoring (the field after the marker); re_entry_requested / decision_readiness /
    BLOCKED_NEEDS_DECISION -> evidence (the episode's blocked_argument, or '')."""
    for ep in ((summary or {}).get("episodes") or []):
        reason = str(ep.get("reason") or "")
        state = str(ep.get("episode_state") or "")
        if "needs_new_content" in reason:
            return {"kind": "authoring", "field": reason.split("needs_new_content:", 1)[-1].strip(": ")}
        if "re_entry_requested" in reason or "decision_readiness" in reason or state == "BLOCKED_NEEDS_DECISION":
            return {"kind": "evidence", "field": (ep.get("blocked_argument") or "")}
    return None


class Recovery(object):
    def __init__(self, task):
        self.blocked_streak = 0          # consecutive invocations whose last episode was BLOCKED (across invocations)
        self.reentry_rounds = 0

    def before_final(self, ctx, answer):
        summary = self._invoke(ctx, answer, ctx.begin_harness_actions)
        # RE-ENTRY: recovery found a gap it must NOT author -> hand control back to the root agent with the precise gap
        # (field / evidence to open), never a value. Bounded; never once the budget was spent at the turn's start.
        if ctx.budget_spent_at_turn_start or self.reentry_rounds >= MAX_REENTRY_ROUNDS:
            return None
        req = reentry_request(summary)
        if req is None:
            return None
        self.reentry_rounds += 1
        if req["kind"] == "authoring" and req.get("field"):
            msg = AUTHORING_MESSAGE % (req["field"], req["field"])
        else:
            msg = EVIDENCE_MESSAGE
        ctx.emit({"step": ctx.step, "event_type": "agent_reentry", "round": self.reentry_rounds, "kind": req["kind"],
                  "field": req.get("field"), "status": "ok"})
        return Revision("reentry", msg, "agent_reentry")

    def after_episode(self, ctx):
        # FORCE-STOP: the agent spent its budget without a final (a weak agent ignores the give-final feedback), so no
        # before_final ran. The forced stop is the agent-done boundary: one invocation, as at a clean final, in the
        # shell's post-stop allowance.
        self._invoke(ctx, "", ctx.post_stop_allowance)

    def _invoke(self, ctx, root_content, open_invocation):
        """One recovery invocation, or None. Recovery runs at every final, so re-answer loops can re-engage it;
        the shell caps its actions, and here K consecutive BLOCKED invocations stop it re-engaging."""
        try:
            step = open_invocation()
            if step is None or self.blocked_streak >= MAX_BLOCKED_RECOVERY:
                return None
            summary = _entry.run_recovery_v3(ctx.task, root_content, "before_final", ctx.trajectory,
                                             driver=_entry.HarnessDriver(ctx, step),
                                             goal=(ctx.contract_meta or {}).get("goal"), step=step)
            last = [e for e in ctx.trajectory if e.get("event_type") == "recovery_v3"]
            if last and "BLOCKED" in str(last[-1].get("episode_state") or last[-1].get("reason") or ""):
                self.blocked_streak += 1
            else:
                self.blocked_streak = 0
            return summary
        except Exception as exc:
            ctx.error("recovery_v3: %r" % (exc,))
        return None
