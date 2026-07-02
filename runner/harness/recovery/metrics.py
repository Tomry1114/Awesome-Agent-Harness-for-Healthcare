"""Bounded Clinical Recovery v3 - metrics taxonomy.

Per task, an EpisodeResult falls into exactly one OUTCOME bucket, plus two orthogonal counters
(eligible / engaged) used for reporting the recoverable population honestly.

  eligible          - a gap-classified recoverable episode was POSSIBLE (a committed goal existed).
  engaged           - a plan/acquire actually RAN (at least one step executed, or a commit dispatched).
  verified_recovery - VERIFIED / ACCEPTED / ALREADY_REALIZED (the effect is realized).
  correctly_blocked - any BLOCKED_* / DECLINED_* / NOT_APPLICABLE / KEPT_ORIGINAL: a CORRECT refusal.
  failed_recovery   - FAILED.
  unknown_recovery  - UNKNOWN.

DECLINED_NO_COMMITMENT / NOT_APPLICABLE / BLOCKED_* are NEVER counted as failures.
"""
from .contracts import (
    VERIFIED, ACCEPTED, ALREADY_REALIZED, KEPT_ORIGINAL,
    FAILED, UNKNOWN, DECLINED_NO_COMMITMENT, NOT_APPLICABLE,
    BLOCKED_STATES, DECLINED_STATES,
)

# Outcome buckets
ELIGIBLE = "eligible"
ENGAGED = "engaged"
VERIFIED_RECOVERY = "verified_recovery"
CORRECTLY_BLOCKED = "correctly_blocked"
FAILED_RECOVERY = "failed_recovery"
UNKNOWN_RECOVERY = "unknown_recovery"

OUTCOME_BUCKETS = (VERIFIED_RECOVERY, CORRECTLY_BLOCKED, FAILED_RECOVERY, UNKNOWN_RECOVERY)
ALL_BUCKETS = (ELIGIBLE, ENGAGED) + OUTCOME_BUCKETS

_VERIFIED = frozenset({VERIFIED, ACCEPTED, ALREADY_REALIZED})
# A block, a declined, or a non-regression "kept the original answer" are all CORRECT refusals.
_BLOCKED = set(BLOCKED_STATES) | set(DECLINED_STATES) | {KEPT_ORIGINAL}


def _state_of(result_or_state):
    return getattr(result_or_state, "state", result_or_state)


def classify(result_or_state):
    """Map an EpisodeResult (or a bare state string) to one of OUTCOME_BUCKETS.

    correctly_blocked is its OWN bucket and is NEVER failed_recovery."""
    st = _state_of(result_or_state)
    if st in _VERIFIED:
        return VERIFIED_RECOVERY
    if st in _BLOCKED:
        return CORRECTLY_BLOCKED
    if st == FAILED:
        return FAILED_RECOVERY
    if st == UNKNOWN:
        return UNKNOWN_RECOVERY
    # Non-terminal / progress states are not final outcomes; report as unknown for safety.
    return UNKNOWN_RECOVERY


def is_eligible(result):
    """A committed goal existed (recovery was possible). DECLINED_NO_COMMITMENT is the only ineligible
    outcome; NOT_APPLICABLE still counts as eligible (a goal existed but no workflow matched)."""
    return _state_of(result) != DECLINED_NO_COMMITMENT


def is_engaged(result):
    """A plan/acquire actually ran (>=1 completed step) or a mutation was dispatched (created_ids)."""
    completed = getattr(result, "completed_steps", None) or []
    created = getattr(result, "created_ids", None) or []
    return bool(completed) or bool(created)


def tally(results):
    """Aggregate a list of EpisodeResults into the full taxonomy for per-dataset reporting."""
    counts = {b: 0 for b in ALL_BUCKETS}
    for r in (results or []):
        if is_eligible(r):
            counts[ELIGIBLE] += 1
        if is_engaged(r):
            counts[ENGAGED] += 1
        counts[classify(r)] += 1
    return counts
