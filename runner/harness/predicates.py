"""Postcondition predicates — a tiny, generic DSL evaluated over canonical state. No per-dataset verifier.

A commit point declares a postcondition as a typed predicate; the same evaluator serves EHR resource
creation, GUI form submission, file writes, DB updates. Predicates that need STRUCTURED state (field
values) are unverifiable when the substrate only exposes an opaque state token -> returns None
(unknown), never a fabricated pass.

  state_transition   the (observable) state changed                         -> needs only before/after token
  object_exists      a target object is present in after_state              -> needs structured state
  field_equals       after_state.<path> == expected                         -> needs structured state
  no_unexpected_side_effect  only the intended target changed               -> needs structured state
"""


def evaluate(predicate, before_state, after_state, sem=None):
    """Returns True (verified), False (violated), or None (cannot verify with the available state). The
    core knows ONLY generic predicate types — NO domain aliases (no 'case_status_not_draft' etc.). A
    structured predicate over an opaque state token -> None (UNKNOWN), never auto-downgraded to a bare
    state change."""
    if isinstance(predicate, str):
        predicate = {"type": predicate}
    t = (predicate or {}).get("type")
    if t in ("state_transition", None):
        return _changed(before_state, after_state)
    if t in ("object_exists", "field_equals", "field_not_equals", "no_unexpected_side_effect",
             "target_consistency"):
        if not isinstance(after_state, dict):
            return None                       # opaque state token -> cannot verify structurally -> UNKNOWN
        return _structured(t, predicate, before_state, after_state)
    return None                               # unknown / domain-specific type -> not a core predicate


def _changed(before, after):
    if before is None or after is None:
        return None                           # state not observable -> unknown (never fabricate)
    try:
        return before != after
    except Exception:
        return None


def _structured(t, p, before, after):
    if t == "object_exists":
        return bool(_get(after, p.get("path")))
    if t == "field_equals":
        return _get(after, p.get("path")) == p.get("expected")
    if t == "field_not_equals":
        return _get(after, p.get("path")) != p.get("unexpected")
    if t == "target_consistency":
        return _get(after, p.get("path")) == p.get("expected")
    if t == "no_unexpected_side_effect":
        # only the declared target path differs between before/after
        keys = set((before or {}).keys()) | set((after or {}).keys())
        changed = [k for k in keys if (before or {}).get(k) != (after or {}).get(k)]
        allowed = set(p.get("allowed_changed", []))
        return all(k in allowed for k in changed)
    return None


def _get(state, path):
    if not path:
        return None
    cur = state
    for part in str(path).split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur
