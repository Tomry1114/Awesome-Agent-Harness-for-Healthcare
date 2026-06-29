"""Part-11 acceptance tests for the Scoped Repair layer (substrate-agnostic core).

These encode the contract: a repair is accepted ONLY if the named defect is fixed AND protected content is
preserved -- identically for HAB form fields, PB FHIR paths, and MedCTA answer claims. Plus: vague,
non-localized findings are never emitted, and a delivered finding is not re-nagged until the agent acts.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness.repair import RepairFinding, RepairOperation, make_finding_id, parse_findings
from harness.repair_surface import surface_for
from harness.repair_delta import validate_repair
from harness.state import Ledger, FindingStatus


def _finding(target_path, op, defect, protected, ttype="field", rule="scoped_repair"):
    return RepairFinding(
        finding_id=make_finding_id("T", rule, ttype, target_path, defect),
        rule_id=rule, target_type=ttype, target_path=target_path, defect_type=defect,
        operation=op, required_change="x", protected_paths=tuple(protected))


# --------------------------------------------------------------------------------------------------------
# parse: localized findings only (no vague "write a triage note")
# --------------------------------------------------------------------------------------------------------
def test_parse_drops_vague_and_keeps_localized():
    raw = {"aligned": False, "findings": [
        {"defect_type": "missing", "required_change": ""},                       # no target -> drop
        {"target_path": "form.x", "defect_type": "missing"},                     # no change/op -> drop
        {"target_path": "form.disposition_rationale", "defect_type": "insufficient_content",
         "repair_operation": "ADD", "required_change": "Add medical-necessity evidence.",
         "protected_paths": ["form.clinical_summary"]},                          # complete -> keep
    ]}
    fs = parse_findings(raw, "T")
    assert len(fs) == 1
    assert fs[0].target_path == "form.disposition_rationale"
    assert fs[0].operation == RepairOperation.ADD
    assert parse_findings({"aligned": True, "findings": []}, "T") == []


# --------------------------------------------------------------------------------------------------------
# HAB
# --------------------------------------------------------------------------------------------------------
HAB_BEFORE = {"form": {"patient_name": "Dorothy Harris", "case_id": "CLM-2025-00016", "disposition": "deny",
                       "clinical_summary": "CO-50 denial for Dorothy Harris due to medical necessity; MRI not covered.",
                       "disposition_rationale": ""}}
HAB_F = _finding("form.disposition_rationale", RepairOperation.ADD, "insufficient_content",
                 ["form.patient_name", "form.case_id", "form.disposition", "form.clinical_summary"])


def test_hab_positive_repair_allows():
    surf = surface_for("gui")
    before = surf.project(HAB_BEFORE, None, HAB_F)
    after_state = {"form": dict(HAB_BEFORE["form"],
                                disposition_rationale="MRI medically necessary: documented acute neuro deficit per record.")}
    after = surf.project(after_state, None, HAB_F)
    v = validate_repair(HAB_F, before, after)
    assert v.accepted and v.reason == "target_resolved", v


def test_hab_regression_overwrites_summary_revise():
    surf = surface_for("gui")
    before = surf.project(HAB_BEFORE, None, HAB_F)
    # added a rationale BUT replaced the case-specific clinical summary with boilerplate (HAB-12/15 pattern)
    after_state = {"form": dict(HAB_BEFORE["form"],
                                disposition_rationale="Reviewed all information.",
                                clinical_summary="Reviewed all available denial information.")}
    after = surf.project(after_state, None, HAB_F)
    v = validate_repair(HAB_F, before, after)
    assert (not v.accepted) and v.reason == "repair_regression", v


# --------------------------------------------------------------------------------------------------------
# dedup / lifecycle
# --------------------------------------------------------------------------------------------------------
def test_dedup_not_reemitted_until_state_changes():
    led = Ledger()
    surf = surface_for("gui")
    proj = surf.project(HAB_BEFORE, None, HAB_F)
    assert led.repair_decision(HAB_F, proj)[0] == "new"
    led.open_finding(HAB_F, proj, step=1)
    led.mark_delivered(HAB_F.finding_id, proj, step=1)
    # same projection -> agent has not acted -> suppress (no re-nag)
    assert led.repair_decision(HAB_F, proj)[0] == "suppress"
    # changed projection -> agent acted -> revalidate
    changed = surf.project({"form": dict(HAB_BEFORE["form"], disposition_rationale="now filled")}, None, HAB_F)
    assert led.repair_decision(HAB_F, changed)[0] == "revalidate"
    # after RESOLVED -> always suppress
    led.resolve_finding(HAB_F.finding_id)
    assert led.repair_findings[HAB_F.finding_id].status == FindingStatus.RESOLVED
    assert led.repair_decision(HAB_F, changed)[0] == "suppress"


# --------------------------------------------------------------------------------------------------------
# PB (FHIR) — structured preservation must be exact
# --------------------------------------------------------------------------------------------------------
PB_BEFORE = {"MedicationRequest": {"subject": "Patient/7", "medication": "amoxicillin",
                                   "dosageInstruction": [{"doseAndRate": 500, "route": "oral", "timing": None}]}}
PB_F = _finding("MedicationRequest.dosageInstruction[0].timing", RepairOperation.ADD, "missing",
                ["MedicationRequest.subject", "MedicationRequest.medication",
                 "MedicationRequest.dosageInstruction[0].doseAndRate",
                 "MedicationRequest.dosageInstruction[0].route"], ttype="resource_path")


def test_pb_positive_add_frequency_allows():
    surf = surface_for("fhir")
    before = surf.project(PB_BEFORE, None, PB_F)
    after_state = {"MedicationRequest": {"subject": "Patient/7", "medication": "amoxicillin",
                                         "dosageInstruction": [{"doseAndRate": 500, "route": "oral",
                                                                "timing": "every 8 hours"}]}}
    after = surf.project(after_state, None, PB_F)
    v = validate_repair(PB_F, before, after)
    assert v.accepted, v


def test_pb_regression_dose_changed_revise():
    surf = surface_for("fhir")
    before = surf.project(PB_BEFORE, None, PB_F)
    after_state = {"MedicationRequest": {"subject": "Patient/7", "medication": "amoxicillin",
                                         "dosageInstruction": [{"doseAndRate": 250, "route": "oral",
                                                                "timing": "every 8 hours"}]}}
    after = surf.project(after_state, None, PB_F)
    v = validate_repair(PB_F, before, after)
    assert (not v.accepted) and v.reason == "repair_regression", v


# --------------------------------------------------------------------------------------------------------
# MedCTA — reindex-tolerant claim removal
# --------------------------------------------------------------------------------------------------------
MEDCTA_BEFORE = {"findings": [
    {"id": "f0", "text": "gallbladder wall thickening, supported by image"},
    {"id": "f1", "text": "pericholecystic fluid, no supporting image evidence"},
    {"id": "f2", "text": "distended gallbladder, supported by image"}]}
MEDCTA_F = _finding("answer.findings[1]", RepairOperation.REMOVE, "unsupported",
                    ["answer.findings[0]", "answer.findings[2]"], ttype="claim")


def test_medcta_positive_remove_unsupported_allows():
    surf = surface_for("tool_sandbox")
    before = surf.project(None, MEDCTA_BEFORE, MEDCTA_F)
    after_cand = {"findings": [MEDCTA_BEFORE["findings"][0], MEDCTA_BEFORE["findings"][2]]}
    after = surf.project(None, after_cand, MEDCTA_F)
    v = validate_repair(MEDCTA_F, before, after)
    assert v.accepted, v


def test_medcta_regression_drops_supported_revise():
    surf = surface_for("tool_sandbox")
    before = surf.project(None, MEDCTA_BEFORE, MEDCTA_F)
    # removed the unsupported claim BUT also deleted the two supported findings -> collapsed to boilerplate
    after_cand = {"findings": [{"id": "x", "text": "Reviewed image. No specific findings."}]}
    after = surf.project(None, after_cand, MEDCTA_F)
    v = validate_repair(MEDCTA_F, before, after)
    assert (not v.accepted) and v.reason == "repair_regression", v


if __name__ == "__main__":
    import traceback
    fails = 0
    for nm, fn in sorted(globals().items()):
        if nm.startswith("test_") and callable(fn):
            try:
                fn(); print("PASS", nm)
            except Exception:
                fails += 1; print("FAIL", nm); traceback.print_exc()
    print("scoped_repair:", "ALL PASS" if not fails else "%d FAILED" % fails)
    sys.exit(1 if fails else 0)
