"""Semantic engine — judge-backed checks (claim<->evidence support, low-confidence escalation).

P0: a stub that returns UNKNOWN/low-confidence so callers fail SAFE (escalate / don't claim a pass)
rather than fabricate a verdict. Wiring it to the harness judge (independent of agent + tool backend,
budgeted by max_semantic_checks) is P3 work. Keeping the seam here means no capability hard-codes a
model call.
"""

UNKNOWN = "UNKNOWN"


class SemanticVerdict:
    __slots__ = ("supported", "confidence", "reason")

    def __init__(self, supported=None, confidence=0.0, reason=None):
        self.supported = supported       # True / False / None(unknown)
        self.confidence = confidence
        self.reason = reason


def claim_supported_by_evidence(claim, evidence, judge=None):
    """Does `claim` follow from subject-scoped `evidence`? P0: unknown (fail-safe). P3: judge-backed."""
    if judge is None:
        return SemanticVerdict(supported=None, confidence=0.0, reason="semantic_judge_unavailable")
    # P3: call the independent harness judge here, count against max_semantic_checks.
    return SemanticVerdict(supported=None, confidence=0.0, reason="not_implemented")
