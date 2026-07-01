"""Plan-completeness clinical module (Phase 3) -- COMPLETE recovery for goal-required-but-missing slots.

A deliverable often FAILS not because a claim is wrong but because a component the PUBLIC GOAL requires is
simply ABSENT (no follow-up, no workup, no monitoring plan). This module produces that missing component and
merges it in WITHOUT touching existing content -- the F->T path least dependent on a chance-positive record.

Pipeline (all oracle-blind: reads only the public goal + the agent's own draft; never gold/checkpoint):
  goal   -> extract_required_slots           (which sections a complete deliverable must contain)
  draft  -> parse_present_slots              (which required sections the draft already addresses)
  missing = missing_required_slots(...)      (goal-completeness gap)
  per missing slot -> generate_slot_content  (structured specialist; ONLY that slot)
  deterministic MERGE (append labeled sections; existing content preserved verbatim)
  slot-level PROMOTION via adopt_slot_patch  (adopt iff no required slot dropped/emptied and every ADDED
                                              slot is a goal-completeness fill -- a whole-text comparator can
                                              no longer veto a locally-correct completion)

Substrate-agnostic: the DELIVERABLE surface (which tool/arg carries the draft) is resolved by run.py/adapter;
this module operates purely on (content, goal, judge_fn). No benchmark names.
"""
from .slot import CommitmentSlot, compute_slot_delta, missing_required_slots, adopt_slot_patch
from .engines.semantic import extract_required_slots, parse_present_slots, generate_slot_content


def _slot_objs(present):
    return [CommitmentSlot(slot_id=s["slot_id"], value=s.get("value") or "<present>",
                           required_by_goal=bool(s.get("required_by_goal", True))) for s in (present or [])]


def compute_completeness_patch(content, goal, judge_fn):
    """Localized goal-completeness patch. Returns a dict:
      {applied, merged_content, required, present, missing, filled, reason, deltas_n}
    Fail-safe toward applied=False (unchanged content) on any missing input, empty requirement set, complete
    draft, empty generation, or a promotion veto. Never edits/removes existing content -- APPEND-only."""
    base = {"applied": False, "merged_content": content, "required": [], "present": [],
            "missing": [], "filled": [], "reason": None, "deltas_n": 0}
    if not str(content or "").strip() or judge_fn is None:
        base["reason"] = "no_content_or_judge"; return base

    required = extract_required_slots(goal, judge_fn)               # [{slot_id, description}]
    base["required"] = [r["slot_id"] for r in required]
    if not required:
        base["reason"] = "no_required_slots"; return base

    present = parse_present_slots(content, required, judge_fn)      # [dict slot] present
    base["present"] = [s["slot_id"] for s in present]
    missing = missing_required_slots([r["slot_id"] for r in required], _slot_objs(present))
    base["missing"] = list(missing)
    if not missing:
        base["reason"] = "complete"; return base

    additions, filled = [], []                                     # (slot_id, title, body)
    for sid in missing:
        desc = next((r["description"] for r in required if r["slot_id"] == sid), sid)
        gen = generate_slot_content(goal, content, {"slot_id": sid, "description": desc}, judge_fn)
        if not gen:
            continue
        additions.append((sid, gen["section_title"], gen["content"])); filled.append(sid)
    base["filled"] = list(filled)
    if not additions:
        base["reason"] = "no_slot_generated"; return base

    # deterministic merge -- APPEND labeled sections; existing draft preserved verbatim
    appended = "\n\n".join("## %s\n%s" % (title, body) for (_sid, title, body) in additions)
    merged = content.rstrip() + "\n\n" + appended + "\n"

    # slot-level promotion: A = present slots; B = present slots + newly filled slots (present ones UNCHANGED)
    slots_a = _slot_objs(present)
    slots_b = slots_a + [CommitmentSlot(slot_id=s, value=b, required_by_goal=True) for (s, _t, b) in additions]
    deltas = compute_slot_delta(slots_a, slots_b)
    base["deltas_n"] = len(deltas)
    adopt, why = adopt_slot_patch(deltas, supported_slot_ids=filled)  # goal-completeness fills justify the adds
    if not adopt:
        base["reason"] = "not_adopted:%s" % why; return base

    base["applied"] = True; base["merged_content"] = merged; base["reason"] = "filled_required_slots"
    return base
