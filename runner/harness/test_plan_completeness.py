"""Tests for the plan-completeness clinical module (Phase 3). Fake judge -> deterministic, no gateway."""
import json
import sys

sys.path.insert(0, "runner")

from harness.plan_completeness import compute_completeness_patch


def make_judge(required_slots, present_ids, gen_map, generic=False):
    """required_slots: [{slot_id,description}]; present_ids: set; gen_map: slot_id -> (title, body) or None."""
    def judge(prompt):
        if "list ONLY the sections" in prompt:            # extract_required_slots
            if generic:
                return json.dumps({"required_slots": []})
            return json.dumps({"required_slots": required_slots})
        if "REQUIRED-section checklist" in prompt:        # parse_present_slots
            present = [s for s in present_ids]
            absent = [r["slot_id"] for r in required_slots if r["slot_id"] not in present_ids]
            return json.dumps({"present": present, "absent": absent})
        if "clinical documentation specialist" in prompt:  # generate_slot_content
            for sid, tb in gen_map.items():
                if ("\n%s --" % sid) in prompt or ("%s --" % sid) in prompt:
                    if tb is None:
                        return json.dumps({"section_title": "", "content": ""})
                    return json.dumps({"section_title": tb[0], "content": tb[1]})
            return json.dumps({"section_title": "", "content": ""})
        return "{}"
    return judge


REQ = [{"slot_id": "assessment", "description": "diagnosis/assessment"},
       {"slot_id": "treatment", "description": "treatment plan"},
       {"slot_id": "follow_up", "description": "follow-up/monitoring plan"}]

DRAFT = "## Assessment\nAbnormal uterine bleeding, likely anovulatory.\n\n## Treatment\nStart combined OCP."


def test_missing_slot_filled():
    j = make_judge(REQ, {"assessment", "treatment"},
                   {"follow_up": ("Follow-up", "Reassess in 3 months; CBC if bleeding persists.")})
    r = compute_completeness_patch(DRAFT, "Provide a management plan with assessment, treatment, and follow-up.", j)
    assert r["applied"] is True, r
    assert r["missing"] == ["follow_up"], r
    assert r["filled"] == ["follow_up"], r
    assert "Reassess in 3 months" in r["merged_content"]
    assert "Start combined OCP" in r["merged_content"]           # existing content preserved verbatim
    assert r["merged_content"].startswith(DRAFT.rstrip()[:20])   # append-only, draft leads
    print("test_missing_slot_filled OK")


def test_complete_draft_noop():
    j = make_judge(REQ, {"assessment", "treatment", "follow_up"}, {})
    r = compute_completeness_patch(DRAFT, "goal", j)
    assert r["applied"] is False and r["reason"] == "complete", r
    assert r["merged_content"] == DRAFT
    print("test_complete_draft_noop OK")


def test_generic_goal_no_requirements():
    j = make_judge(REQ, set(), {}, generic=True)
    r = compute_completeness_patch(DRAFT, "Write something.", j)
    assert r["applied"] is False and r["reason"] == "no_required_slots", r
    print("test_generic_goal_no_requirements OK")


def test_no_judge_noop():
    r = compute_completeness_patch(DRAFT, "goal", None)
    assert r["applied"] is False and r["reason"] == "no_content_or_judge", r
    assert r["merged_content"] == DRAFT
    print("test_no_judge_noop OK")


def test_empty_generation_noop():
    j = make_judge(REQ, {"assessment", "treatment"}, {"follow_up": None})   # specialist returns empty
    r = compute_completeness_patch(DRAFT, "goal", j)
    assert r["applied"] is False and r["reason"] == "no_slot_generated", r
    assert r["merged_content"] == DRAFT
    print("test_empty_generation_noop OK")


def test_multiple_missing_all_filled():
    j = make_judge(REQ, {"assessment"},
                   {"treatment": ("Treatment", "Start combined OCP."),
                    "follow_up": ("Follow-up", "Reassess in 3 months.")})
    r = compute_completeness_patch("## Assessment\nAUB.", "goal", j)
    assert r["applied"] is True, r
    assert set(r["filled"]) == {"treatment", "follow_up"}, r
    assert "Start combined OCP" in r["merged_content"] and "Reassess in 3 months" in r["merged_content"]
    print("test_multiple_missing_all_filled OK")


def test_empty_content_noop():
    j = make_judge(REQ, set(), {})
    r = compute_completeness_patch("", "goal", j)
    assert r["applied"] is False and r["reason"] == "no_content_or_judge", r
    print("test_empty_content_noop OK")


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); n += 1
    print("\n%d/%d plan_completeness tests passed" % (n, n))
