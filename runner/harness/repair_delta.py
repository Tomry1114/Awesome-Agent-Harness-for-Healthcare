"""Repair delta validation — the load-bearing check that makes Scoped Repair NON-DEGRADING.

A repair is accepted ONLY when ALL hold:
  target_resolved            : the named defect is actually fixed
  protected_content_preserved: no protected content was overwritten / dropped / mutated
  no_new_conflict            : the patch introduced no contradiction (judge-optional; deterministic no-op)

This is what prevents HAB-12/15-style regressions: 'added a rationale' is NOT enough if the case-specific
clinical summary was replaced by "Reviewed all information". The validator compares the BASELINE projection
(captured when the finding was delivered) with the AFTER projection (the agent's repaired candidate), per
the finding's operation semantics. Reindex-tolerant for list targets (REMOVE shifts indices), so protected
items are matched by CONTENT membership, not position.
"""
from __future__ import annotations
import re
from dataclasses import dataclass

from .repair import RepairOperation


@dataclass
class RepairVerdict:
    accepted: bool
    reason: str            # target_resolved | target_not_resolved | repair_regression | repair_introduced_conflict
    detail: str = ""


def _tokens(s):
    return set(re.findall(r"[a-z0-9]+", str(s).lower()))


def _retained(before, after, thresh=0.6):
    """Is the SUBSTANCE of `before` still present in `after`? Empty before -> nothing to lose (True).
    Structured / numeric / bool -> must be EQUAL (PB dose/route must not drift). Text -> exact-substring OR
    >= thresh token overlap (so an APPEND keeps it, an OVERWRITE drops it)."""
    if before is None or (isinstance(before, str) and not before.strip()):
        return True
    if isinstance(before, bool) or isinstance(after, bool):
        return before == after
    if isinstance(before, (int, float)) or isinstance(after, (int, float)):
        return before == after
    if isinstance(before, (dict, list)) or isinstance(after, (dict, list)):
        return before == after
    bs = str(before).strip()
    if bs and bs in str(after):
        return True
    bt = _tokens(before)
    if not bt:
        return True
    return (len(bt & _tokens(after)) / len(bt)) >= thresh


def _present(v):
    return not (v is None or (isinstance(v, str) and not v.strip())
                or (isinstance(v, (list, dict)) and len(v) == 0))


def _leaf_count(o):
    """Number of non-empty scalar leaves in a structure -- a cheap measure of 'how much content exists'."""
    n = 0
    for v in _walk(o):
        if not isinstance(v, (dict, list)) and _present(v):
            n += 1
    return n


def _effect_landed(before_root, after_root, effect_paths):
    """Did the required EFFECT land at one of the finding's declared equivalent paths? (item 3) -- a value
    became newly present at an effect_path that was empty at baseline. Replaces the old global leaf-count
    heuristic (too loose: any unrelated write resolved it; appends were missed). Resolution is now SPECIFIC to
    the substrate's declared persistence path(s), not 'something somewhere changed'."""
    from .repair_surface import resolve
    for p in (effect_paths or []):
        if _present(resolve(after_root, p)) and not _present(resolve(before_root, p)):
            return True
    return False


def _walk(o):
    if isinstance(o, dict):
        yield o
        for v in o.values():
            yield from _walk(v)
    elif isinstance(o, list):
        yield o
        for v in o:
            yield from _walk(v)
    else:
        yield o


def _retained_anywhere(value, after_proj, thresh=0.6):
    """Membership fallback for list elements whose index shifted: is `value`'s substance present ANYWHERE in
    the after-root collection?"""
    if value is None:
        return True
    for v in _walk(after_proj.get("root")):
        if _retained(value, v, thresh) and _retained(v, value, thresh):
            return True
    return False


def validate_repair(finding, before_proj, after_proj, surface=None, effect_paths=None):
    """Pure verdict from two projections. `effect_paths` (item 3) are the substrate-declared equivalent
    persistence paths for an additive defect; resolution checks the effect landed at one of them, not the
    judge-guessed target alone."""
    op = finding.operation
    dt = finding.defect_type
    tb = before_proj.get("target")
    ta = after_proj.get("target")

    # ---- 1. target resolved (operation-aware) -----------------------------------------------------------
    if op in (RepairOperation.REMOVE,) or dt == "unsupported":
        # the SPECIFIC baseline claim value must be gone from the collection (reindex-tolerant)
        if op == RepairOperation.REMOVE and _retained_anywhere(tb, after_proj):
            return RepairVerdict(False, "target_not_resolved", "claim not removed")
    elif op == RepairOperation.VERIFY:
        if _present(ta) and ta == tb and not finding.evidence_refs:
            return RepairVerdict(False, "target_not_resolved", "claim unchanged / unverified")
    elif op in (RepairOperation.ADD,) or dt in ("missing", "insufficient_content"):
        if not _present(ta):
            # item 3: resolve by EFFECT at a substrate-DECLARED equivalent path (not a guessed target, not a
            # global "something changed"). Without a declared equivalence the only effect path is the target
            # itself -> stays unresolved (and the attempt cap stops the churn), which is the honest behavior.
            if not _effect_landed(before_proj.get("root"), after_proj.get("root"), effect_paths):
                return RepairVerdict(False, "target_not_resolved", "required effect not observed")
        elif dt == "insufficient_content" and ta == tb:
            return RepairVerdict(False, "target_not_resolved", "content unchanged")
    else:   # EDIT / REPLACE / conflicting / wrong_operation
        if not _present(ta) or ta == tb:
            return RepairVerdict(False, "target_not_resolved", "no change applied to target")

    # ---- 2. protected content preserved -----------------------------------------------------------------
    pb = before_proj.get("protected") or {}
    pa = after_proj.get("protected") or {}
    for path, bval in pb.items():
        if not _present(bval):
            continue                                   # nothing substantive to lose
        if _retained(bval, pa.get(path)):
            continue                                   # preserved in place (equal / appended-to)
        if "[" in path and _retained_anywhere(bval, after_proj):
            continue                                   # list element survived a reindex
        return RepairVerdict(False, "repair_regression", "protected content lost: %s" % path)

    # ---- 3. no new conflict (deterministic no-op; judge can extend) --------------------------------------
    return RepairVerdict(True, "target_resolved")
