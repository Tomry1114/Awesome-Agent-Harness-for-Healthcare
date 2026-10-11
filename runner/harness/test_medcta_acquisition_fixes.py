"""Counterexample tests for MedCTA evidence acquisition. Standalone: sys.exit(0) on pass, 1 on fail. No model,
no GPU. Each check asserts a specific behaviour with an input that a weaker implementation would get wrong,
covering reliability classification, information-gain diversification, and success counting."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.observation import (observation_reliability, RELIABILITY_UNCERTAIN, RELIABILITY_CONFIDENT,
                                  RELIABILITY_LOC_FAILED, RELIABILITY_UNOBSERVED)
from harness import affordance as A

fails = []
def check(cond, msg):
    if not cond:
        fails.append(msg)


# ===================================================================== RELIABILITY ==========================
check(observation_reliability("the SMV is not clearly visualized", "smv identity") == RELIABILITY_UNCERTAIN,
      "reliability: 'not clearly visualized' must be OBSERVED_UNCERTAIN")
check(observation_reliability("no pulmonary embolus is identified; the vessels are patent",
                              "pulmonary embolus") == RELIABILITY_CONFIDENT,
      "reliability: a CONFIDENT NEGATIVE finding must NOT be flagged uncertain")
check(observation_reliability("no definite fracture", "fracture") == RELIABILITY_CONFIDENT,
      "reliability: 'no definite fracture' is a confident negative, not uncertain")
check(observation_reliability("a vessel is present", "smv", localization_resolved=False) == RELIABILITY_LOC_FAILED,
      "reliability: structured localization_resolved=False must be LOCALIZATION_FAILED")
check(observation_reliability("the defect is in the main portal vein. the distal branch is not clearly seen.",
                              "portal vein") == RELIABILITY_CONFIDENT,
      "reliability: a target answered confidently must stay CONFIDENT despite an unrelated hedge elsewhere")
check(observation_reliability("the liver parenchyma is homogeneous", "superior mesenteric vein") == RELIABILITY_UNOBSERVED,
      "reliability: an attribute never addressed must be UNOBSERVED")
check(observation_reliability("the portal vein is patent and normal", "SMV identity") == RELIABILITY_UNOBSERVED,
      "reliability(abbrev): a portal-only output with target SMV must be UNOBSERVED (SMV not dropped as <4 chars)")
check(observation_reliability("the SMV is patent. a distal branch is not clearly visualized.", "SMV identity") == RELIABILITY_CONFIDENT,
      "reliability(abbrev): SMV answered confidently must stay CONFIDENT despite an unrelated hedge")


# ===================================================================== DIVERSIFICATION ======================
TOOLS = [{"name": "RegionAttributeDescription"}, {"name": "ImageDescription"}, {"name": "OCR"}]

# E1 tried -> E2 must differ (not the same tool+region+attribute)
p1 = [{"tool": "RegionAttributeDescription", "region": "porta hepatis", "attribute": "SMV identity"}]
t, a = A.select_diversified_tool(TOOLS, region="porta hepatis", attribute="SMV identity", prior_attempts=p1)
check(not (t == "RegionAttributeDescription" and a == "SMV identity"),
      "diversify: E2 must not repeat E1's exact tool+attribute")
check(t == "RegionAttributeDescription",
      "diversify(#5): an anatomical requirement must NOT route to OCR/ImageDescription")

# anatomical requirement must never pick OCR even as a 'different tool'
check(t != "OCR", "diversify(#5): anatomical need must not select OCR")

# contrastive E2 already tried -> E3 must NOT regenerate the same contrastive prompt -> router returns None
contrastive = A._contrastive_attribute("SMV identity")
p2 = p1 + [{"tool": "RegionAttributeDescription", "region": "porta hepatis", "attribute": contrastive}]
t3, a3 = A.select_diversified_tool(TOOLS, region="porta hepatis", attribute="SMV identity", prior_attempts=p2)
check(t3 is None, "diversify(#3): after tool+contrastive exhausted, router must return None (no repeat)")

# text/label requirement -> OCR is the right modality
tt, _ = A.select_diversified_tool(TOOLS, region="corner", attribute="the text label annotation", prior_attempts=[])
check(tt == "OCR", "diversify(#5): a text/label requirement must route to OCR")


# ===================================================================== SUCCESS COUNT ========================
# Uses the REAL add_evidence(type, value, subject_id, source_event, source_type, extra) signature. A repeated
# identical VALIDATED OUTPUT must NOT raise validated_evidence_version, EVEN when the request args differ.
from harness.state import Ledger
from harness.capabilities.scope_evidence import _full

def _add(led, output_text, args):
    # value_full is what the novelty signature hashes; _full extracts the OUTPUT only from the envelope.
    envelope = {"tool": "RegionAttributeDescription", "args": args, "mode": "semantic",
                "output": {"text": output_text}}
    led.add_evidence(type="perception", value=output_text[:80], subject_id="img1",
                     source_event="step-1", source_type="perception",
                     extra={"modality": "image", "resource": "perception", "value_full": _full(envelope),
                            "source_class": "perception", "status": "VALIDATED", "scope_relation": "own"})

led = Ledger()
v0 = led.validated_evidence_version
_add(led, "the SMV is patent", {"attribute": "SMV identity"})
v1 = led.validated_evidence_version
check(v1 > v0, "success(#1): a NEW validated output must raise validated_evidence_version")
# SAME output, DIFFERENT request args (a re-worded / contrastive attribute) -> NOT new evidence.
_add(led, "the SMV is patent", {"attribute": "SMV identity -- report the DISCRIMINATING features"})
v2 = led.validated_evidence_version
check(v2 == v1, "success(#1,P0-3): SAME output with DIFFERENT request args must NOT bump the version")
# a genuinely different output IS new evidence.
_add(led, "the SMV contains a filling defect", {"attribute": "SMV identity"})
v3 = led.validated_evidence_version
check(v3 > v2, "success(#1): a genuinely different output must bump the version")

# _full must not carry request args: identical output, different args -> identical value_full.
e1 = {"tool": "RegionAttributeDescription", "args": {"attribute": "A"}, "output": {"text": "same output"}}
e2 = {"tool": "RegionAttributeDescription", "args": {"attribute": "B totally different prompt"}, "output": {"text": "same output"}}
check(_full(e1) == _full(e2), "P0-3: _full must hash the OUTPUT only, not the request args")


if fails:
    print("FAIL (%d):" % len(fails))
    for x in fails:
        print("  -", x)
    sys.exit(1)
print("PASS: acquisition-fix counterexamples (reliability #6/#7, diversification #2/#3/#5, success #1)")
sys.exit(0)
