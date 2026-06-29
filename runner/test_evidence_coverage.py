"""Acceptance tests for the evidence_coverage core: claim-conditioned observational coverage.

Encodes the review's tightenings: only PERCEPTUAL claims are strictly gated; 3 distinct defect/op cases;
deterministic-first (judge only at the margin); conservative when unsure; tools come from the affordance
registry, never invented; protected_paths preserve other grounded claims.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness.repair import RepairOperation
from harness.observation import Observation, Claim, coverage_findings, region_observed
from harness import affordance

OBS_RLL = Observation("obs-1", "inspect_region", subject="image-1", region="right-lower-lobe",
                      modality="CT", attributes_observed=("nodule", "margin"), result_status="valid")


def _by_path(findings):
    return {f.target_path: f for f in findings}


def test_unobserved_target_reacquire():
    claims = [Claim("c1", 0, "perceptual", region="left-upper-lobe", modality="CT", attribute="mass")]
    f = coverage_findings(claims, [OBS_RLL], "T")
    assert len(f) == 1 and f[0].defect_type == "unobserved_target"
    assert f[0].operation == RepairOperation.REACQUIRE_EVIDENCE


def test_fully_covered_silent():
    claims = [Claim("c1", 0, "perceptual", region="right-lower-lobe", modality="CT", attribute="nodule")]
    assert coverage_findings(claims, [OBS_RLL], "T") == []


def test_region_seen_attribute_unclear_judge_unsupported():
    claims = [Claim("c1", 0, "perceptual", region="right-lower-lobe", modality="CT", attribute="calcification")]
    # region observed, attribute 'calcification' not in attributes_observed -> judge consulted
    f = coverage_findings(claims, [OBS_RLL], "T", semantic_support={"c1": False})
    assert len(f) == 1 and f[0].defect_type == "unsupported_by_observation"
    assert f[0].operation == RepairOperation.VERIFY_OR_REMOVE


def test_region_seen_attribute_unclear_no_verdict_silent():
    # conservative: no judge verdict -> do NOT flag
    claims = [Claim("c1", 0, "perceptual", region="right-lower-lobe", modality="CT", attribute="calcification")]
    assert coverage_findings(claims, [OBS_RLL], "T") == []


def test_untraceable_claim_edit_or_remove():
    claims = [Claim("c1", 0, "perceptual", region=None, attribute="something")]
    f = coverage_findings(claims, [OBS_RLL], "T")
    assert len(f) == 1 and f[0].defect_type == "untraceable_claim"
    assert f[0].operation == RepairOperation.EDIT_OR_REMOVE


def test_background_and_recommendation_ignored():
    claims = [Claim("c1", 0, "background", text="malignancy usually has irregular margins"),
              Claim("c2", 1, "recommendation", text="recommend biopsy")]
    assert coverage_findings(claims, [OBS_RLL], "T") == []


def test_interpretive_ok_with_covered_premise():
    claims = [Claim("c1", 0, "perceptual", region="right-lower-lobe", modality="CT", attribute="nodule"),
              Claim("c2", 1, "interpretive", text="findings favor malignancy")]
    assert coverage_findings(claims, [OBS_RLL], "T") == []


def test_interpretive_flagged_without_any_covered_premise():
    claims = [Claim("c1", 0, "perceptual", region="left-lung", modality="CT", attribute="mass"),  # unobserved
              Claim("c2", 1, "interpretive", text="findings favor malignancy")]
    fs = _by_path(coverage_findings(claims, [OBS_RLL], "T"))
    assert "answer.claims[0]" in fs and fs["answer.claims[0]"].defect_type == "unobserved_target"
    assert "answer.claims[1]" in fs and fs["answer.claims[1]"].defect_type == "unsupported_by_observation"


def test_invalid_observation_does_not_cover():
    bad = Observation("obs-x", "inspect_region", region="right-lower-lobe", modality="CT",
                      attributes_observed=("nodule",), result_status="invalid")
    claims = [Claim("c1", 0, "perceptual", region="right-lower-lobe", modality="CT", attribute="nodule")]
    f = coverage_findings(claims, [bad], "T")
    assert len(f) == 1 and f[0].defect_type == "unobserved_target"


def test_protected_paths_preserve_other_covered_claims():
    claims = [Claim("c1", 0, "perceptual", region="right-lower-lobe", modality="CT", attribute="nodule"),  # covered
              Claim("c2", 1, "perceptual", region="left-lung", modality="CT", attribute="mass")]           # unobserved
    f = _by_path(coverage_findings(claims, [OBS_RLL], "T"))["answer.claims[1]"]
    assert "answer.claims[0]" in f.protected_paths   # the grounded sibling must be preserved


# ---- affordance registry: tools come from the manifest, never invented --------------------------------
MEDCTA_TOOLS = [{"name": "OCR", "signature": "(image)"},
                {"name": "ImageDescription", "signature": "(image)"},
                {"name": "RegionAttributeDescription", "signature": "(image, region, attribute)"},
                {"name": "GoogleSearch", "signature": "(query)"}]


def test_affordance_picks_real_perception_tools_excludes_search():
    tools = affordance.select_tools(MEDCTA_TOOLS, region="right-lower-lobe", modality="CT")
    assert "RegionAttributeDescription" in tools and "GoogleSearch" not in tools
    assert tools[0] == "RegionAttributeDescription"   # region/attribute tool ranked first


def test_affordance_empty_for_non_perception_substrate():
    form_tools = [{"name": "type", "signature": "(ref, text)"}, {"name": "submit", "signature": "()"}]
    assert affordance.select_tools(form_tools) == []


if __name__ == "__main__":
    import traceback
    fails = 0
    for nm, fn in sorted(globals().items()):
        if nm.startswith("test_") and callable(fn):
            try:
                fn(); print("PASS", nm)
            except Exception:
                fails += 1; print("FAIL", nm); traceback.print_exc()
    print("evidence_coverage:", "ALL PASS" if not fails else "%d FAILED" % fails)
    sys.exit(1 if fails else 0)
