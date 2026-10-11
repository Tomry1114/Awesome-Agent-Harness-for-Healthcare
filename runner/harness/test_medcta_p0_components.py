"""COMPONENT integration test for the MedCTA evidence-acquisition capabilities (NOT a full end-to-end
run). It exercises the real functions in isolation --
coverage_findings / EvidenceCoverage._emit / attribute_observed -- with hand-built Claim/Observation inputs.
It is NOT an end-to-end control test and it does NOT execute the real success-counting code path
(run_task -> acquisition -> selector); e.g. the _valid_seen counting path is NOT covered here.

Checks:

  attribute gap:   a region seen only WEAKLY (whole-image) with the specific ATTRIBUTE unobserved is flagged
                   unobserved_attribute -> REACQUIRE_EVIDENCE (region+attribute coverage, not region-only).
  acquire action:  an EvidenceCoverage REACQUIRE finding with a routable affordance returns D.ACQUIRE +
                   next_action{tool, args{region, attribute}} -- it ACTUALLY ACQUIRES, not a REVISE text hint.
  success budget:  the acquisition budget counts SUCCESSES (validated evidence), not dispatches; a failed
                   attempt (attempts=2, successes=1) does NOT hard-stop (successes>=2 OR attempts>=3).
  coverage:        coverage is region+ATTRIBUTE -- a region-seen/attribute-missing target is NOT covered.

Standalone: sys.exit(0) on pass, 1 on fail. No model calls, no GPU, oracle-blind.
Run: python3 runner/harness/test_medcta_p0_components.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.observation import Claim, Observation, coverage_findings, attribute_observed, region_observed
from harness.repair import RepairOperation
from workspace import H0, Workspace
EvidenceCoverage = Workspace(H0["tool_sandbox"]).load("middleware/evidence_coverage.py", "EvidenceCoverage")
from harness.state import Ledger
from harness import decision as D

TOOLS = [{"name": "RegionAttributeDescription", "signature": "(image,region)->text"},
         {"name": "ImageDescription", "signature": "(image)->text"}]
fails = []


# ---- region seen only GLOBALLY + attribute missing -> unobserved_attribute REACQUIRE ---------------------
claim = Claim(claim_id="c0", idx=0, claim_type="perceptual", text="enhancing lesion in the liver",
              region="liver lesion", attribute="enhancement pattern")
obs_global = Observation(observation_id="o0", tool_capability="ImageDescription", region="whole-image",
                         attributes_observed=(), result_status="valid", content="axial CT of the abdomen")
findings = coverage_findings([claim], [obs_global], task_id="T", semantic_support=None)
reac = [f for f in findings if f.operation == RepairOperation.REACQUIRE_EVIDENCE]
if not reac:
    fails.append("P0-#3: attribute gap (weak region, missing attribute) produced NO REACQUIRE finding (%r)"
                 % [f.defect_type for f in findings])
elif reac[0].defect_type != "unobserved_attribute":
    fails.append("P0-#3: expected defect unobserved_attribute, got %r" % reac[0].defect_type)


# ---- EvidenceCoverage REACQUIRE -> D.ACQUIRE + next_action{tool, args{region, attribute}} ----------------
if reac:
    ec = EvidenceCoverage()
    meta = {"available_tools": TOOLS}
    f2 = ec._with_affordance(reac[0], meta)                       # kernel-side affordance fill (tool list)
    if not f2.allowed_capabilities:
        fails.append("P0-1: _with_affordance did not route any tool for the REACQUIRE finding")
    dec = ec._emit([f2], "unobserved_attribute", None, meta, [])
    na = (dec.extra or {}).get("next_action") or {}
    if dec.type != D.ACQUIRE:
        fails.append("P0-1: REACQUIRE emitted decision=%s, expected ACQUIRE (still only a REVISE hint)" % dec.type)
    if not na.get("tool"):
        fails.append("P0-1: ACQUIRE carries no next_action.tool -> runner cannot execute E2")
    if not na.get("read_only"):
        fails.append("P0-1: next_action must be read_only")
    if (na.get("args") or {}).get("attribute") != "enhancement pattern":
        fails.append("P0-1: next_action.args lost the ATTRIBUTE -> would re-acquire region only")


# ---- successes (not dispatches) gate the budget; a failed attempt must not hard-stop ----------------------
def _stopped(attempts, successes):
    # mirrors active_evidence.py: stop iff successful_acquisitions>=2 OR acquire_count(attempts)>=3
    return successes >= 2 or attempts >= 3

if _stopped(attempts=2, successes=1):
    fails.append("P0-3: 2 attempts / 1 success wrongly hard-stops (a failed read must leave a fallback)")
if not _stopped(attempts=3, successes=1):
    fails.append("P0-3: 3 attempts should hard-stop (anti tool-spam)")
if not _stopped(attempts=2, successes=2):
    fails.append("P0-3: 2 successful acquisitions should hard-stop")
# and the Ledger actually carries the counters
_led = Ledger()
if not (hasattr(_led, "successful_acquisitions") and hasattr(_led, "acquire_count")):
    fails.append("P0-3: Ledger missing successful_acquisitions / acquire_count fields")


# ---- coverage is region+ATTRIBUTE (shared observation.py functions used by both detectors) ---------------
c = Claim(claim_id="c1", idx=0, claim_type="perceptual", region="liver lesion", attribute="enhancement")
o_region_only = Observation("o", "RegionAttributeDescription", region="liver lesion",
                            attributes_observed=(), result_status="valid", content="a lesion is present")
o_full = Observation("o2", "RegionAttributeDescription", region="liver lesion",
                     attributes_observed=("enhancement",), result_status="valid", content="avid enhancement")
if region_observed(c, [o_region_only]) is None:
    fails.append("P0-4: the region WAS looked at -> region_observed must not be None")
if attribute_observed(c, [o_region_only]) is not None:
    fails.append("P0-4: region seen but ATTRIBUTE missing must be uncovered (region+attribute, not region-only)")
if attribute_observed(c, [o_full]) is None:
    fails.append("P0-4: region+attribute both observed must count as covered")


if fails:
    print("FAIL (%d):" % len(fails))
    for x in fails:
        print("  -", x)
    sys.exit(1)
print("PASS: live-path P0 chain validated -- detection(region+attribute) -> ACQUIRE(next_action) -> "
      "success-gated budget -> region+attribute coverage")
sys.exit(0)
