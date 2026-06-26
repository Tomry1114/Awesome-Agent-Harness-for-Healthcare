"""Capability base class — Modules A / B / C all implement this 3-hook interface.

The kernel calls each registered capability at three stages and combines their decisions by precedence:
  before_action(action, ctx)  -> HarnessDecision | None     (prospective gate)
  after_action(action, result, before_state, after_state, ctx) -> HarnessDecision | None  (retrospective)
  before_final(answer, ctx)   -> HarnessDecision | None     (final answer = a commit point)

`ctx` is a HarnessContext (ledger + contract + risk + policy + mode). A capability mutates the LEDGER
(records evidence/obligations) but returns only a decision; it never executes the action itself.
Returning None == no opinion (== ALLOW for that capability).
"""
from .decision import HarnessDecision, ALLOW


class Capability:
    name = "capability"

    def on_contract(self, ctx):
        """Hook once after the contract is compiled (e.g. declare obligations into the ledger)."""
        return None

    def before_action(self, action, ctx):
        return None

    def after_action(self, action, result, before_state, after_state, ctx):
        return None

    def before_final(self, answer, ctx):
        return None

    # helpers for subclasses
    def _allow(self, **kw):
        return HarnessDecision(ALLOW, capability=self.name, **kw)

    def _decide(self, type, **kw):
        return HarnessDecision(type, capability=self.name, **kw)


class HarnessContext:
    """What every capability + the kernel share for one task."""
    __slots__ = ("ledger", "contract", "policy", "mode", "step", "env_type", "risk_of", "observation")

    def __init__(self, ledger, contract, policy, mode, env_type=None, risk_of=None):
        self.ledger = ledger
        self.contract = contract
        self.policy = policy or {}
        self.mode = mode
        self.step = 0
        self.env_type = env_type
        self.risk_of = risk_of      # callable(action) -> "R0".."R3"
        self.observation = None     # canonical_observation of the most recent action (after_action only)
