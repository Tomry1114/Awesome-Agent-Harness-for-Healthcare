"""Slot-level commitments, deltas, and localized patches (general recovery core).

A commitment (a clinical plan, an answer, a form/action) decomposes into typed SLOTS. Recovery then works at
slot granularity instead of rewriting the whole artifact: find the AFFECTED slot, generate ONLY its patch,
deterministically merge, and promote per-slot. This avoids the "one local fix drowned by a whole-text
comparator -> uncertain -> keep A" failure. Substrate-agnostic: slot_types are declared by adapters/policy
(PB: diagnosis/medication/workup/follow_up; MedCTA: primary_conclusion/polarity/region/severity; HAB:
field/disposition/submit_state). No benchmark names here.
"""
from dataclasses import dataclass, field


@dataclass
class CommitmentSlot:
    slot_id: str
    slot_type: str = None
    target: str = None
    value: str = None
    evidence_ids: list = field(default_factory=list)
    required_by_goal: bool = False


@dataclass
class SlotDelta:
    slot_id: str
    change: str                 # added | removed | changed | unchanged
    old_value: str = None
    new_value: str = None
    required: bool = False


@dataclass
class PatchProposal:
    target_slot: str
    operation: str              # add | replace | remove | qualify
    new_value: str = None
    supporting_evidence_ids: list = field(default_factory=list)
    preserve_other_slots: bool = True


def _norm(v):
    return " ".join(str(v or "").split()).strip().lower()


def compute_slot_delta(slots_a, slots_b):
    """Per-slot diff A->B. slots_* : [CommitmentSlot]. Returns [SlotDelta] over the union of slot_ids."""
    a = {s.slot_id: s for s in (slots_a or [])}
    b = {s.slot_id: s for s in (slots_b or [])}
    out = []
    for sid in list(a.keys()) + [k for k in b.keys() if k not in a]:
        sa, sb = a.get(sid), b.get(sid)
        req = bool((sa and sa.required_by_goal) or (sb and sb.required_by_goal))
        if sa and not sb:
            out.append(SlotDelta(sid, "removed", sa.value, None, req))
        elif sb and not sa:
            out.append(SlotDelta(sid, "added", None, sb.value, req))
        elif _norm(sa.value) != _norm(sb.value):
            out.append(SlotDelta(sid, "changed", sa.value, sb.value, req))
        else:
            out.append(SlotDelta(sid, "unchanged", sa.value, sb.value, req))
    return out


def missing_required_slots(required_slot_ids, present_slots):
    """Goal-completeness gap: required slot ids that are ABSENT/empty in the draft. present_slots:[CommitmentSlot]."""
    present = {s.slot_id for s in (present_slots or []) if _norm(s.value)}
    return [sid for sid in (required_slot_ids or []) if sid not in present]


def adopt_slot_patch(deltas, supported_slot_ids):
    """Slot-level promotion. Adopt iff EVERY changed/added slot is evidence-supported AND no required slot was
    removed/emptied. `supported_slot_ids`: slot_ids whose change is justified by the evidence/effect delta (or a
    goal-completeness fill). A whole-text comparator can no longer veto a locally-supported fix.
    Returns (adopt: bool, reason)."""
    supported = set(supported_slot_ids or [])
    changed = [d for d in deltas if d.change in ("changed", "added", "removed")]
    if not changed:
        return (False, "no_slot_change")
    for d in changed:
        if d.required and (d.change == "removed" or (d.change == "changed" and not _norm(d.new_value))):
            return (False, "required_slot_removed:%s" % d.slot_id)   # emptying a required slot == removal
        if d.slot_id not in supported:
            return (False, "unsupported_slot_change:%s" % d.slot_id)
    return (True, "all_changed_slots_supported")
