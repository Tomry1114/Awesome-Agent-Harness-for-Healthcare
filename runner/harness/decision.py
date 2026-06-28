"""Harness decision types — the single, small decision vocabulary the kernel emits.

Exactly four decisions, with a fixed precedence:  ESCALATE > BLOCK > REVISE > ALLOW.
A decision is produced by a capability (Module A/B/C) at a stage (before_action / after_action /
before_final). The kernel combines all capability decisions for a stage by precedence and then applies
the run MODE (observe / assist / enforce) to get the EFFECTIVE decision.
"""

ALLOW = "ALLOW"
REVISE = "REVISE"
BLOCK = "BLOCK"
ESCALATE = "ESCALATE"

# higher number = higher precedence
_PRIORITY = {ALLOW: 0, REVISE: 1, BLOCK: 2, ESCALATE: 3}
DECISIONS = (ALLOW, REVISE, BLOCK, ESCALATE)


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
        if self.extra:
            d["extra"] = self.extra
        return d


def allow(stage=None, capability=None, **kw):
    return HarnessDecision(ALLOW, stage=stage, capability=capability, **kw)


def combine(decisions, stage=None):
    """Pick the highest-precedence decision; ALLOW when the list is empty / all ALLOW. Ties keep the
    FIRST capability that raised that level (stable: capability order = registration order)."""
    winner = None
    for d in decisions:
        if d is None:
            continue
        if winner is None or d.priority > winner.priority:
            winner = d
    if winner is None:
        winner = HarnessDecision(ALLOW, stage=stage)
    return winner
