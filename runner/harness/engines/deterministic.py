"""Deterministic engine — small, oracle-free predicate helpers shared by capabilities.

Holds NO task-specific gold. Just utilities for the deterministic (non-judge) checks: id equality,
state-change detection, simple field presence. Anything needing a model lives in semantic.py.
"""


def same_subject(a, b):
    return _norm(a) == _norm(b) and _norm(a) != ""


def state_changed(before, after):
    if before is None or after is None:
        return None                 # unknown -> let the caller decide (never fabricate)
    try:
        return before != after
    except Exception:
        return None


def _norm(x):
    return str(x or "").strip().lower().split("/")[-1]
