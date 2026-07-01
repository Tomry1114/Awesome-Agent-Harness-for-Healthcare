"""Tests for effect_completion (Phase 4b, v2). Fake env -> exercises EvidenceState + fail-closed probes."""
import sys
sys.path.insert(0, "runner")
from harness.effect_completion import (context_refs, required_evidence_resource_types,
                                       resolve_read_evidence, inspect_existing_effect, build_order_resource,
                                       resource_type_for_category)
from harness.evidence_state import PRESENT, ABSENT, UNKNOWN, FAILED, is_resolved

R = []
def ck(n, c): R.append((n, bool(c))); print(("OK  " if c else "FAIL") + " " + n)


class FakeEnv:
    """Programmable fhir_search responses keyed by resourceType. value: dict result, 'raise', or 'error'."""
    def __init__(self, table): self.table = table; self.calls = []
    def call_tool(self, name, args):
        self.calls.append((name, args))
        v = self.table.get(args.get("resourceType"))
        if v == "raise": raise RuntimeError("boom")
        if v == "error": return {"error": "HTTP 500"}
        return v if v is not None else {"entries": []}


TASK = {"context": {"patient_ref": "MRN2970753705",
        "text": "The current date and time is 2022-02-06T23:17:00+00:00. (Practitioner ID: dr-teresa-hayes)."}}
POLICY = {"required_tool_before_action": ["fhir_search(AllergyIntolerance)", "fhir_search(MedicationRequest)"]}
r = context_refs(TASK)
SUBJ = r["subject"]

# context + build (unchanged core)
ck("subject_ref", SUBJ == "Patient/MRN2970753705")
ck("authoredOn", r["authoredOn"] == "2022-02-06T23:17:00+00:00")
ck("req_types", required_evidence_resource_types(POLICY) == ["AllergyIntolerance", "MedicationRequest"])
rt, res = build_order_resource({"text": "Order pelvic ultrasound", "category": "imaging"}, r)
ck("build_ok", rt == "ServiceRequest" and res["status"] == "active" and res["intent"] == "order"
   and "pelvic ultrasound" in res["code"]["text"].lower())

# resolve_read_evidence: PRESENT / ABSENT / FAILED classification (governance gate)
env_present = FakeEnv({"AllergyIntolerance": {"entries": [{"resource": {"id": "a1"}}]}})
ck("gov_present", resolve_read_evidence(env_present, "AllergyIntolerance", SUBJ)["state"] == PRESENT)
env_absent = FakeEnv({"AllergyIntolerance": {"entries": []}})
ck("gov_absent", resolve_read_evidence(env_absent, "AllergyIntolerance", SUBJ)["state"] == ABSENT)
ck("gov_absent_resolved", is_resolved(resolve_read_evidence(env_absent, "AllergyIntolerance", SUBJ)["state"]))
env_err = FakeEnv({"AllergyIntolerance": "error"})
ck("gov_failed_not_resolved", not is_resolved(resolve_read_evidence(env_err, "AllergyIntolerance", SUBJ)["state"]))
env_raise = FakeEnv({"AllergyIntolerance": "raise"})
ck("gov_raise_failed", resolve_read_evidence(env_raise, "AllergyIntolerance", SUBJ)["state"] == FAILED)

# inspect_existing_effect: FAIL-CLOSED. probe error -> UNKNOWN (NOT absent -> caller must not create)
ck("insp_error_unknown", inspect_existing_effect(FakeEnv({"ServiceRequest": "error"}), "ServiceRequest", SUBJ)["state"] == UNKNOWN)
ck("insp_raise_unknown", inspect_existing_effect(FakeEnv({"ServiceRequest": "raise"}), "ServiceRequest", SUBJ)["state"] == UNKNOWN)
ck("insp_empty_absent", inspect_existing_effect(FakeEnv({"ServiceRequest": {"entries": []}}), "ServiceRequest", SUBJ)["state"] == ABSENT)
_present = inspect_existing_effect(FakeEnv({"ServiceRequest": {"entries": [
    {"resource": {"id": "sr1", "code": {"text": "Pelvic ultrasound transvaginal"}}}]}}), "ServiceRequest", SUBJ)
ck("insp_present", _present["state"] == PRESENT and "Pelvic ultrasound transvaginal" in _present["texts"] and "sr1" in _present["matched_ids"])

n = sum(1 for _, c in R if c)
print("\n%d/%d effect_completion(v2) tests passed" % (n, len(R)))
assert n == len(R), "FAILURES"
