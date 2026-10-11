"""Multi-value COLLECTION field handling for GUI recovery (diagnosis codes / CPT codes / documents).

A real portal collects a SET via type-one -> Add -> type-next -> Add, but generic single-value binding
fills a control once. This module provides the pure, benchmark-neutral primitives:
  - COLLECTION_CONCEPTS: which field concepts are sets;
  - collection_delta(committed, present): the ordered values still to add (committed MINUS already-present);
  - present_values(state_or_controls, concept): best-effort read of the values already added (chips).
Values come ONLY from agent commitment / read evidence / authoritative state -- the harness never invents a
code. Oracle-blind; no gold. NOTE: the committed-SET extraction must accumulate repeated same-concept
entries into a list instead of overwriting (see committed_collection), and the live type->Add->re-observe
loop + chip detection depend on how the portal renders added values.
"""
import re

COLLECTION_CONCEPTS = {"diagnosis", "cpt", "supporting_documents", "procedure_codes"}


def _norm(v):
    return re.sub(r"[\s\-\.]+", "", str(v or "")).lower()


def collection_delta(committed, present):
    """Ordered values still to add: each committed value (de-duplicated, order-preserving) whose normalized
    form is not already present. Never invents a value -- only what was committed/observed."""
    seen, pres = set(), {_norm(p) for p in (present or []) if str(p or "").strip()}
    out = []
    for v in (committed or []):
        n = _norm(v)
        if not n or n in pres or n in seen:
            continue
        seen.add(n)
        out.append(v)
    return out


def present_values(items, concept=None):
    """Best-effort read of the values already ADDED to a collection (chips / list entries). Accepts a list of
    chip strings, or a list of control dicts (uses their value / label). Generic; no gold."""
    out = []
    for it in (items or []):
        if isinstance(it, str):
            if it.strip():
                out.append(it.strip())
        elif isinstance(it, dict):
            v = str(it.get("value") or it.get("text") or it.get("label") or "").strip()
            if v:
                out.append(v)
    return out


# billing/clinical code patterns shared with grounding_audit (adapters/hab/common.py) so they can't drift
from adapters.hab.common import ICD10_RE as _ICD_RE, CPT_HCPCS_RE as _CPT_RE
_CONCEPT_PAT = {"diagnosis": _ICD_RE, "cpt": _CPT_RE, "procedure_codes": _CPT_RE}


# =====================================================================================================
# The collection EXECUTOR. Unlike a coverage nudge (which an agent can ignore), the executor AUTONOMOUSLY
# completes the agent's OWN committed set, so it can land the outcome without agent cooperation.
# SAFETY BOUNDARY: the values it types come STRICTLY from the agent's own trajectory (typed) or
# authoritative read evidence -- it never invents a code (that would be authoring content). It only re-adds
# a value the agent already produced but that did not land as a chip (an execution gap: type -> Add -> chip
# failed / the agent typed but never clicked Add).
# =====================================================================================================

def committed_collection(trajectory, concept):
    """The agent's OWN collection values for a concept, accumulated (NOT overwritten) across the trajectory,
    order-preserving + de-duplicated. A value qualifies if the agent typed it into a field whose key names the
    concept OR the typed value itself matches the concept's code pattern (ICD for diagnosis, CPT/HCPCS for
    procedure). Oracle-blind: strictly the agent's typed values, never gold, never invented."""
    pat = _CONCEPT_PAT.get(concept)
    key_hint = {"diagnosis": ("diagnos", "icd"), "cpt": ("cpt", "hcpcs", "procedure", "service"),
                "procedure_codes": ("cpt", "hcpcs", "procedure", "service")}.get(concept, ())
    seen, out = set(), []
    for e in (trajectory or []):
        # HAB agents write fields with "fill"; "type" is accepted as an alias and "select" for dropdowns.
        # Without "fill" this would return an empty set for agents that typed codes into a code field,
        # starving the whole type->Add executor downstream.
        if not isinstance(e, dict) or e.get("tool") not in ("fill", "type", "select"):
            continue
        args = e.get("args") or {}
        key = str(args.get("field") or args.get("ref") or args.get("target") or args.get("name") or "").lower()
        val = args.get("text") if args.get("text") is not None else args.get("value")
        val = str(val or "").strip()
        if not val:
            continue
        key_match = any(h in key for h in key_hint)
        # accept the typed value only if the field names the concept, OR (field unknown) the value is a code
        # of this concept -- so a free-text note that merely mentions a code is not harvested as a commitment.
        if key_match:
            cands = pat.findall(val) if pat else [val]
        elif pat and pat.fullmatch(val.replace(" ", "")):
            cands = [val]
        else:
            cands = []
        for c in cands:
            n = _norm(c)
            if n and n not in seen:
                seen.add(n)
                out.append(c.strip())
    return out


def next_collection_action(committed, present, input_ref, add_ref):
    """One ATOMIC step of the type->Add loop, or None when the set is already covered. Given the agent's
    committed values and the values already present (chips), returns the next value to add and the refs to
    drive it: {'value', 'input_ref', 'add_ref'}. The caller fills input_ref with value, clicks add_ref, then
    RE-OBSERVES and calls again (the chip for `value` should now be present, shrinking the delta). Never
    returns a value that is not in `committed` (no invented codes)."""
    missing = collection_delta(committed, present)
    if not missing or not input_ref:
        return None
    return {"value": missing[0], "input_ref": input_ref, "add_ref": add_ref, "remaining": len(missing)}
