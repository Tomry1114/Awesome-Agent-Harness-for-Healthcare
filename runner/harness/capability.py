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
    __slots__ = ("ledger", "contract", "policy", "mode", "step", "env_type", "risk_of", "observation",
                 "judge_fn", "judge_model", "semantic_remaining", "manifest", "sem", "risk",
                 "result_ok", "result_status", "verification", "last_observation", "observed_subject", "displayed_subject",
                 "action_key", "evidence_version_before", "current_state")

    def __init__(self, ledger, contract, policy, mode, env_type=None, risk_of=None,
                 judge_fn=None, judge_model=None, semantic_budget=0, manifest=None):
        self.ledger = ledger
        self.contract = contract
        self.policy = policy or {}
        self.mode = mode
        self.step = 0
        self.env_type = env_type
        self.risk_of = risk_of      # legacy; the kernel now sets ctx.sem + ctx.risk per action
        self.observation = None     # canonical_observation of the most recent action (after_action only)
        self.manifest = manifest or {}   # substrate manifest (adapter layer); tool->semantic mapping
        self.sem = None             # SemanticAction of the current action (set by the kernel)
        self.risk = None            # risk tier of the current action (set by the kernel)
        self.result_ok = None       # did the current action's tool result succeed? (set by the kernel)
        self.result_status = None   # tri-state ok|failed|unknown (timeout/ambiguous) -> unknown commit = ESCALATE
        self.current_state = None   # STRUCTURED env snapshot at before_action (pre-commit form/resource state)
        self.verification = None     # tri-state commit verification: True/False/None (set by verify_commit)
        self.last_observation = None  # the most recent canonical_observation
        self.action_key = None       # canonical id of the current action (dedup before/after interventions)
        self.observed_subject = None   # subject displayed by THIS action's observation (manifest-projected)
        self.displayed_subject = None  # last-known displayed subject, carried into before_action (sticky)
        self.judge_fn = judge_fn    # injected judge: callable(prompt:str) -> str|None (INDEPENDENT model)
        self.judge_model = judge_model
        self.semantic_remaining = int(semantic_budget or 0)

    def spend_semantic(self):
        """Consume one semantic-check from the budget. Returns True if a check is allowed."""
        if self.judge_fn is None or self.semantic_remaining <= 0:
            return False
        self.semantic_remaining -= 1
        return True
