"""Harness decision types — the single, small decision vocabulary the kernel emits.

Precedence (low->high): ALLOW < RECONCILE < REVISE < ACQUIRE < DEFER_FINAL < BLOCK < ESCALATE.
DEFER_FINAL is SOFT: the final is not accepted yet and the agent continues; it is never terminal.
A decision is produced by a capability (Module A/B/C) at a stage (before_action / after_action /
before_final). The kernel combines all capability decisions for a stage by precedence and then applies
the run MODE (observe / assist / enforce) to get the EFFECTIVE decision.
"""

ALLOW = "ALLOW"
RECONCILE = "RECONCILE"   # UNKNOWN commit -> restricted read-back recovery (not terminal)
ACQUIRE = "ACQUIRE"       # harness acquires read-only evidence, feeds it back, agent re-reasons
REVISE = "REVISE"
DEFER_FINAL = "DEFER_FINAL"   # soft gate on the FINAL only: the final is not accepted yet and the agent
                              # CONTINUES. NOT a BLOCK (forbids no action) and NOT terminal -- the kernel
                              # alone decides ESCALATE/terminate on a canonical invariant.
BLOCK = "BLOCK"
ESCALATE = "ESCALATE"

# higher number = higher precedence. DEFER_FINAL sits below BLOCK -- a soft "final not yet", never terminal.
_PRIORITY = {ALLOW: 0, RECONCILE: 1, REVISE: 2, ACQUIRE: 3, DEFER_FINAL: 4, BLOCK: 5, ESCALATE: 6}
DECISIONS = (ALLOW, RECONCILE, ACQUIRE, REVISE, DEFER_FINAL, BLOCK, ESCALATE)

# The decision.extra key a capability puts the agent's text in: what the agent is told about this decision, in the
# raising capability's own words (the shell reads it with middleware_host.decision_text). It is not part of the
# verdict: the agent's turn records it, and the decision's records (to_dict, the kernel's harness_decision event)
# leave it out.
AGENT_TEXT = "agent_text"


class HarnessDecision:
    """One capability's verdict on a proposed action / final answer. `feedback` is the structured,
    leak-safe message handed back to the agent (it never contains gold answers or reference traces)."""

    __slots__ = ("type", "stage", "capability", "rule_id", "reason_code", "reason", "missing_obligations",
                 "suggested_capabilities", "avoid_capabilities", "feedback", "deterministic", "risk", "extra")

    def __init__(self, type=ALLOW, stage=None, capability=None, rule_id=None, reason=None,
                 missing_obligations=None, suggested_capabilities=None, feedback=None,
                 deterministic=True, risk=None, extra=None, reason_code=None, avoid_capabilities=None):
        if type not in _PRIORITY:
            raise ValueError("unknown decision %r" % (type,))
        self.type = type
        self.stage = stage
        self.capability = capability
        self.rule_id = rule_id
        self.reason_code = reason_code   # STRUCTURED category for metrics (rule_id is for audit only)
        self.reason = reason
        self.missing_obligations = list(missing_obligations or [])
        self.suggested_capabilities = list(suggested_capabilities or [])
        self.avoid_capabilities = list(avoid_capabilities or [])   # capabilities to STOP repeating (loop-steer)
        self.feedback = feedback
        self.deterministic = bool(deterministic)
        self.risk = risk
        self.extra = dict(extra or {})

    @property
    def priority(self):
        return _PRIORITY[self.type]

    def to_dict(self):
        d = {"decision": self.type, "stage": self.stage, "capability": self.capability,
             "rule_id": self.rule_id, "reason_code": self.reason_code, "reason": self.reason,
             "missing_obligations": self.missing_obligations,
             "suggested_capabilities": self.suggested_capabilities,
             "deterministic": self.deterministic, "risk": self.risk}
        if self.feedback is not None:
            d["feedback"] = self.feedback
        extra = {k: v for k, v in self.extra.items() if k != AGENT_TEXT}
        if extra:
            d["extra"] = extra
        return d


def allow(stage=None, capability=None, **kw):
    return HarnessDecision(ALLOW, stage=stage, capability=capability, **kw)


# ---- CONFLICT, PER HOOK -------------------------------------------------------------------------
# What is genuinely exclusive at each hook, stated as the hook's own semantics rather than as one global
# severity rank over all action types.
#
# A global max would assert that the action CLASS decides who matters, so an ACQUIRE always beats a REVISE.
# A missing-lab ACQUIRE and an inconsistent-reasoning REVISE can both be right, and the correct order may be
# ACQUIRE -> new observation -> REVISE rather than ACQUIRE instead of REVISE.
#
# WHAT IS ACTUALLY EXCLUSIVE. Control flow, and only control flow:
#
#   before_action   whether the proposed action RUNS.  BLOCK/ESCALATE veto it; ACQUIRE defers it to run
#                   a read first; REVISE hands the turn back. At most one of those can hold.
#   after_action    the action has already run. ACQUIRE inserts a read before the agent's next turn.
#   before_final    whether the final answer is ACCEPTED. DEFER_FINAL says not yet.
#
# So resolve() returns the one control-flow decision AND the others, instead of dropping them. The others
# get no effect: they are not delivered and their actuators do not run.
_CONTROL_FLOW = {
    "before_action": (ESCALATE, BLOCK, ACQUIRE, DEFER_FINAL, REVISE, RECONCILE),
    "after_action":  (ESCALATE, BLOCK, ACQUIRE, REVISE, RECONCILE),
    "before_final":  (ESCALATE, BLOCK, DEFER_FINAL, ACQUIRE, REVISE, RECONCILE),
}


class Resolution(object):
    """The hook's outcome: one decision that shapes control flow, and everything else that still applies.

    `also` holds the decisions that were RAISED and then outranked. They are not applied and not delivered,
    and nothing runs their actuator: control flow is exclusive, and here delivery is too.
    """

    __slots__ = ("winner", "also", "stage")

    def __init__(self, winner, also, stage=None):
        self.winner, self.also, self.stage = winner, list(also), stage

    def __iter__(self):                      # so `winner, also = resolve(...)` reads naturally
        return iter((self.winner, self.also))


def resolve(decisions, stage=None):
    """-> Resolution. Which decision shapes control flow at THIS hook, and which others still apply.

    Order within a hook is the hook's own semantics (see _CONTROL_FLOW); ties keep the FIRST capability
    that raised that level, so capability registration order is the tie-break.
    """
    live = [d for d in (decisions or []) if d is not None and d.type != ALLOW]
    order = _CONTROL_FLOW.get(stage) or tuple(sorted(_PRIORITY, key=_PRIORITY.get, reverse=True))
    winner = None
    for kind in order:
        for d in live:
            if d.type == kind:
                winner = d
                break
        if winner is not None:
            break
    if winner is None:
        return Resolution(HarnessDecision(ALLOW, stage=stage), [], stage)
    return Resolution(winner, [d for d in live if d is not winner], stage)


def combine(decisions, stage=None):
    """The control-flow decision alone: the one-value form of resolve() for callers that only
    steer on the winner; anything that reports what a hook DID must read resolve().also as well, or it is
    reporting a max and calling it the outcome."""
    return resolve(decisions, stage=stage).winner
