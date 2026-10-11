# -*- coding: utf-8 -*-
"""HealthAdminBench H0 middleware: the pre-commit audit (host before_tool).

A terminal commit -- a `submit`, or a click the shell's classifier (adapters/hab/gui_control_model.is_terminal_action)
resolves to a commit control on the live page -- does not run while a required control on that page is still empty
(HTML required / aria-required, or an asterisk in the label). The first MAX_PRECOMMIT refusals name the fields to
fill (precommit_audit); every later one, without a bound, tells the agent not to submit an incomplete form and to give
its best final answer (precommit_terminal). Both unlock the repair budget (Revision). Harness reads are not audited.
Public form state only, never a value.
"""
from workspace_api import is_control_complete, is_terminal_action
from workspace_api import Revision

MAX_PRECOMMIT = 3


def terminal_commit(ctx, action):
    """(is the action a terminal commit, the live controls it was classified against: ctx.controls(), the shell's
    read-only control model of the current page)."""
    if action.get("tool") not in ("submit", "click"):
        return False, None
    try:
        controls = ctx.controls()
    except Exception:
        controls = []
    try:
        return bool(is_terminal_action(action.get("tool"), (action.get("args") or {}).get("ref"), controls)), controls
    except Exception:
        return False, controls


class PrecommitAudit(object):
    def __init__(self, task):
        self.rounds = 0          # precommit_audit refusals so far; precommit_terminal is not counted

    def before_tool(self, ctx, action):
        if action.get("_harness_acquire"):
            return None
        terminal, controls = terminal_commit(ctx, action)
        if not terminal:
            return None
        empty = [(c.get("label") or c.get("name") or c.get("ref")) for c in controls
                 if (c.get("required") or "*" in str(c.get("label") or "")) and not is_control_complete(c)]
        if not empty:
            return None
        fields = ", ".join(str(x)[:40] for x in empty[:6])
        if self.rounds < MAX_PRECOMMIT:
            self.rounds += 1
            ctx.emit({"step": ctx.step, "event_type": "precommit_audit", "round": self.rounds,
                      "blocked_tool": action.get("tool"), "required_empty": empty[:8], "status": "ok"})
            return Revision("precommit",
                            "Pre-commit check blocked this submission: these REQUIRED fields are still empty: %s. "
                            "Fill each one from the available records (patient demographics / referral / coverage "
                            "/ clinical note) before submitting. Do NOT resubmit until they are complete." % fields,
                            "precommit_block")
        ctx.emit({"step": ctx.step, "event_type": "precommit_terminal",
                  "blocked_tool": action.get("tool"), "required_empty": empty[:8], "status": "ok"})
        return Revision("precommit_terminal",
                        "Required fields (%s) are still empty after repeated attempts. Do NOT submit an "
                        "incomplete form; give your best final answer from the evidence gathered." % fields,
                        "precommit_terminal")
