# -*- coding: utf-8 -*-
"""The episode's reconcile state: one unconfirmed commit, the agent's or the harness's.

A commit the kernel's after_action answers RECONCILE (its effect is unknown: it timed out, or its read-back is ambiguous)
puts the episode in restricted recovery until the commit is read back. Agent commits and harness commits share this one
state, so an unconfirmed harness write also restricts the agent's writes, and the harness action channel does not write
while an agent commit is unconfirmed:

    entered by    the kernel's after_action RECONCILE on an agent action (run.py) or on a harness write (the harness
                  action channel, runner/harness/action_service.py)
    restricts     every write of either origin: an agent write is refused with reconcile_block (one repair turn); a
                  harness write, and a content-guard restore, is refused as unconfirmed_commit_pending
    resolved by   the agent's reads: env.reconcile_write on the pending commit; confirmed -> cleared,
                  not landed -> cleared with RECONCILE feedback, unknown -> one read of the budget spent; an exhausted
                  budget ends the episode (harness_escalation reconcile_budget_exhausted)

A harness read does not resolve it (restricted mode counts the agent's reads).
"""


class ReconcileState(object):
    __slots__ = ("pending",)

    def __init__(self):
        self.pending = None      # {action, res, budget, origin, requester, step}

    @property
    def active(self):
        return self.pending is not None

    @property
    def origin(self):
        return (self.pending or {}).get("origin")

    def enter(self, action, res, origin, requester=None, step=None, budget=3):
        self.pending = {"action": {"tool": (action or {}).get("tool"), "args": (action or {}).get("args") or {}},
                        "res": res, "budget": budget, "origin": origin, "requester": requester, "step": step}

    def clear(self):
        self.pending = None
