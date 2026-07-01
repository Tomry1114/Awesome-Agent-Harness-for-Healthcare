"""Evidence-state semantics + acquisition dedup (general recovery core).

Substrate-agnostic. Separates two things the old `_has_payload -> VALIDATED` conflated:
  (1) is the OBSERVATION resolved?  -> the obligation was checked (PRESENT or ABSENT both resolve it)
  (2) did it yield DECISION-CHANGING evidence?  -> only PRESENT can support a core change.
An ABSENT result ("checked, none found") SATISFIES an obligation but does NOT justify flipping a core decision.
Core reasons over EvidenceState; adapters map raw tool results -> state via a declared result-semantics spec.
No benchmark names, no resource literals here.
"""
from dataclasses import dataclass, field

# --- evidence state ---
PRESENT = "PRESENT"     # query resolved, decision-relevant content found
ABSENT = "ABSENT"       # query resolved, confirmed no content (obligation checked, not decision-changing)
UNKNOWN = "UNKNOWN"     # scope/status/result uncertain (timeout, ambiguous, partial)
FAILED = "FAILED"       # tool call failed
_RESOLVED = (PRESENT, ABSENT)          # the obligation was actually checked
_RETRYABLE = (UNKNOWN, FAILED)         # may be retried (bounded)


def classify_evidence_state(result, semantics=None):
    """Map a raw tool result -> EvidenceState using an ADAPTER-declared result-semantics spec (not core guesses).
    semantics: {collection_paths:[...], count_paths:[...], absence_when_empty: bool}. No spec -> best-effort:
    error -> FAILED; a collection present & empty -> ABSENT; non-empty -> PRESENT; otherwise UNKNOWN."""
    if not isinstance(result, dict):
        return UNKNOWN if result is not None else FAILED
    if result.get("error"):
        return FAILED
    sem = semantics or {}
    # explicit count
    for cp in (sem.get("count_paths") or []):
        v = result.get(cp)
        if isinstance(v, (int, float)):
            return PRESENT if v > 0 else (ABSENT if sem.get("absence_when_empty", True) else UNKNOWN)
    # collection paths
    cols = sem.get("collection_paths") or [k for k in ("entries", "results", "items", "data") if k in result]
    for cp in cols:
        v = result.get(cp)
        if isinstance(v, list):
            return PRESENT if len(v) > 0 else (ABSENT if sem.get("absence_when_empty", True) else UNKNOWN)
    if result.get("timeout") or result.get("status") == "unknown":
        return UNKNOWN
    # a written/created/updated payload is PRESENT progress
    if any(result.get(k) for k in ("written", "created", "updated", "value", "output", "observation")):
        return PRESENT
    return UNKNOWN


def is_resolved(state):
    return state in _RESOLVED

def is_decision_changing(state):
    return state == PRESENT

def is_retryable(state):
    return state in _RETRYABLE


@dataclass(frozen=True)
class AcquisitionKey:
    target_entity: str
    evidence_unit: str
    normalized_query: str = ""
    environment_version: int = 0

    def tuple(self):
        return (self.target_entity, self.evidence_unit, self.normalized_query, self.environment_version)


@dataclass
class EvidenceRequest:
    obligation_id: str
    target_entity: str                      # the CURRENT task entity every acquisition must bind to
    evidence_unit: str                      # abstract unit (resource / region / field-group)
    affordance: dict = None                 # adapter-compiled {tool, args, ...}
    query: dict = field(default_factory=dict)
    expected_result_semantics: dict = None


class AcquisitionLog:
    """Dedup + bounded-retry ledger for acquisitions. Same (entity, unit, query, env-version) that already
    RESOLVED (PRESENT/ABSENT) is not re-run; UNKNOWN/FAILED may retry once."""
    def __init__(self):
        self._states = {}    # key.tuple() -> (state, attempts)

    def state_of(self, key):
        return self._states.get(key.tuple(), (None, 0))[0]

    def should_acquire(self, key, max_retry=1):
        st, att = self._states.get(key.tuple(), (None, 0))
        if st is None:
            return True
        if is_resolved(st):
            return False                     # already checked -> never repeat
        return att <= max_retry              # UNKNOWN/FAILED -> bounded retry

    def record(self, key, state):
        _, att = self._states.get(key.tuple(), (None, 0))
        self._states[key.tuple()] = (state, att + 1)
