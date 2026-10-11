"""Gap-to-recovery routing (Selective Capability Amplification).

The 7 capabilities are DETECTORS: each emits GapProposal objects describing WHY a recovery is warranted. A
single rule-first Router maps the highest-priority concrete gap to exactly ONE recovery primitive. Core is
oracle-blind and substrate-agnostic: it reasons over GoalContract obligations / commit requirements and the
EvidenceLedger's validated units, never over benchmark names or gold trajectories. Adapters supply the
read/observe/mutation affordances; core only sees abstract gap -> recovery -> affordance.

Recovery primitives (only three can RAISE outcome):
  ACQUIRE  -- obtain new read-only evidence that influences the decision (MISSING_CONTEXT / UNOBSERVED_FEATURE)
  COMPLETE -- finish a deterministic, verifiable missing effect (INCOMPLETE_EFFECT)
  CONSULT  -- evidence complete but interpretation uncertain -> structured specialist (INTERPRETATION)
Plus the non-amplifying safe routes: RECONCILE_READ_ONLY (UNRESOLVED_POSTCONDITION) and ALLOW (no gap).
"""
from dataclasses import dataclass, field

# gap types (priority order = the Router's rule order; earliest = most reliable F->T, latest = riskiest)
MISSING_CONTEXT = "MISSING_CONTEXT"
INCOMPLETE_EFFECT = "INCOMPLETE_EFFECT"
UNRESOLVED_POSTCONDITION = "UNRESOLVED_POSTCONDITION"
UNOBSERVED_FEATURE = "UNOBSERVED_FEATURE"
INTERPRETATION = "INTERPRETATION"
NO_ACTIONABLE_GAP = "NO_ACTIONABLE_GAP"

# recovery primitives
ACQUIRE = "ACQUIRE"
COMPLETE = "COMPLETE"
RECONCILE_READ_ONLY = "RECONCILE_READ_ONLY"
CONSULT = "CONSULT"
ALLOW = "ALLOW"

_GAP_PRIORITY = (MISSING_CONTEXT, INCOMPLETE_EFFECT, UNRESOLVED_POSTCONDITION, UNOBSERVED_FEATURE, INTERPRETATION)
_GAP_TO_RECOVERY = {
    MISSING_CONTEXT: ACQUIRE,
    INCOMPLETE_EFFECT: COMPLETE,
    UNRESOLVED_POSTCONDITION: RECONCILE_READ_ONLY,
    UNOBSERVED_FEATURE: ACQUIRE,
    INTERPRETATION: CONSULT,
}


@dataclass
class GapProposal:
    gap_type: str
    missing_unit: str = None
    proposed_recovery: str = None
    affordance: dict = None                 # adapter-resolved read/observe/mutation affordance for the recovery
    expected_progress_token: str = None     # what NEW progress this recovery should produce (evidence id / effect)
    trace_evidence: list = field(default_factory=list)
    confidence: float = 0.0
    risk: str = None

    def __post_init__(self):
        if self.proposed_recovery is None:
            self.proposed_recovery = _GAP_TO_RECOVERY.get(self.gap_type, ALLOW)


def route(gaps):
    """Rule-first: pick the highest-priority CONCRETE gap and return its single recovery proposal. Deterministic
    gaps (context/effect/postcondition) are tried before the riskier perceptual/interpretation gaps. No
    actionable gap -> a NO_ACTIONABLE_GAP/ALLOW proposal."""
    by_type = {}
    for g in (gaps or []):
        by_type.setdefault(g.gap_type, g)
    for gt in _GAP_PRIORITY:
        if gt in by_type:
            return by_type[gt]
    return GapProposal(NO_ACTIONABLE_GAP, proposed_recovery=ALLOW)



