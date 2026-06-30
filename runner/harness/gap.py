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


def detect_missing_context(required_units, validated_units):
    """The required-context gap: obligation ids a commit REQUIRES minus the ledger's VALIDATED units.
    Substrate-agnostic -- `required_units` come from the GoalContract commit point, `validated_units` from the
    EvidenceLedger. Returns the missing unit ids (order preserved)."""
    vset = set(validated_units or [])
    return [u for u in (required_units or []) if u not in vset]


def missing_context_proposal(missing_units, affordance_for=None, admit=True):
    """Wrap the FIRST missing required-context unit as a MISSING_CONTEXT -> ACQUIRE GapProposal. `affordance_for`
    maps a unit id -> a read-only affordance dict (adapter); no affordance -> no proposal (cannot acquire it)."""
    if not admit or not missing_units:
        return None
    unit = missing_units[0]
    aff = affordance_for(unit) if affordance_for else None
    if not aff:
        return None                                        # no read affordance -> not recoverable here
    return GapProposal(MISSING_CONTEXT, missing_unit=unit, affordance=aff,
                       expected_progress_token="validated:%s" % unit, confidence=1.0, risk=aff.get("risk"))
