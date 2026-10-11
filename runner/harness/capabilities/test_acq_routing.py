"""Test for audit->acquisition routing: an unanswered GLOBAL required element (e.g. the stain) with
region=null routes to a WHOLE-IMAGE acquisition on the attribute-aware tool instead of being dropped."""
import os, sys, types
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from workspace import H0, Workspace
AE = sys.modules[Workspace(H0["tool_sandbox"]).load("middleware/active_evidence.py", "ActiveEvidence").__module__]
import harness.engines.semantic as SEM


class _Ledger:
    def __init__(self):
        self.successful_acquisitions = 0
        self.acquire_count = 0
        self.observations = []
        self.acquisition_attempts = []


class _Contract:
    def __init__(self):
        self.meta = {"goal": "identify the tissue and determine the stain",
                     "available_tools": [{"name": "RegionAttributeDescription"}, {"name": "ImageDescription"},
                                         {"name": "OCR"}]}


class _Ctx:
    def __init__(self, req):
        self.judge_fn = (lambda p: "{}")
        self.final_is_commit = True
        self.contract = _Contract()
        self.ledger = _Ledger()
        self.workspace_state = {"pending_required_element": req}   # the audit's target, as answer_repair records it
    def has_judge(self):
        return True

    def semantic_check(self):
        return self.judge_fn


def test_global_required_element_routes_to_whole_image_acquisition():
    # the discriminator (correctly) returns NO region for the stain -- it is a whole-image property.
    SEM_orig = AE.elicit_discriminator if hasattr(AE, "elicit_discriminator") else None
    import harness.engines.semantic as _s
    _orig = _s.elicit_discriminator
    _s.elicit_discriminator = lambda answer, goal, observations=None, judge_fn=None, required_element=None: {
        "hypotheses": ["h1", "h2"], "region": None, "attribute": "overall stain / chromogen colour pattern"}
    try:
        cap = AE.ActiveEvidence() if hasattr(AE, "ActiveEvidence") else _find_cap()
        ctx = _Ctx("determine the stain")
        dec = cap.before_final("The tissue is cartilage/growth-plate.", ctx)
        assert dec is not None, "a global required element must still produce an ACQUIRE, not None"
        na = (getattr(dec, "extra", None) or {}).get("next_action") or {}
        # the ATTRIBUTE must actually reach the tool -- route to the attribute-aware region tool
        # with a whole-image region_query, NOT ImageDescription (which drops the attribute).
        assert "attribute" in str(na.get("tool", "")).lower() or "region" in str(na.get("tool", "")).lower(), \
            "global element must route to the ATTRIBUTE-AWARE tool (RegionAttr), got %r" % (na,)
        assert "stain" in str(na.get("args", {}).get("attribute", "")).lower(), \
            "the required attribute (stain) must be passed to the tool, got %r" % (na.get("args"),)
        assert "whole" in str(na.get("args", {}).get("region_query", "")).lower(), \
            "a global element must carry a whole-image region_query, got %r" % (na.get("args"),)
    finally:
        _s.elicit_discriminator = _orig


def _find_cap():
    for n in dir(AE):
        o = getattr(AE, n)
        if isinstance(o, type) and hasattr(o, "before_final"):
            return o()
    raise RuntimeError("ActiveEvidence capability class not found")


if __name__ == "__main__":
    test_global_required_element_routes_to_whole_image_acquisition()
    print("PASS: region-null global required element routes to whole-image acquisition.")
