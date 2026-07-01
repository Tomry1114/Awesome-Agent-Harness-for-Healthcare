"""Tests for effect_completion pure helpers (Phase 4b). No env / no gateway."""
import sys
sys.path.insert(0, "runner")
from harness.effect_completion import (context_refs, required_evidence_resource_types,
                                       searched_resource_types, missing_required_evidence, build_order_resource)

R = []
def ck(n, c): R.append((n, bool(c))); print(("OK  " if c else "FAIL") + " " + n)

TASK = {"context": {"patient_ref": "MRN2970753705",
        "text": "The current date and time is 2022-02-06T23:17:00+00:00. You are an Obstetrician-Gynecologist "
                "(Practitioner ID: dr-teresa-hayes). Patient MRN2970753705."}}
POLICY = {"required_tool_before_action": ["fhir_search(AllergyIntolerance)", "fhir_search(MedicationRequest)"]}

r = context_refs(TASK)
ck("subject_ref", r["subject"] == "Patient/MRN2970753705")
ck("authoredOn", r["authoredOn"] == "2022-02-06T23:17:00+00:00")
ck("requester", r["requester"] == "Practitioner/dr-teresa-hayes")
ck("no_ref_empty", context_refs({"context": {}}) == {})

ck("req_ev_types", required_evidence_resource_types(POLICY) == ["AllergyIntolerance", "MedicationRequest"])

# AUB trajectory: MedicationRequest searched, AllergyIntolerance NOT -> Allergy is the missing governance read
traj = [{"event_type": "tool_call", "tool": "fhir_search", "args": {"resourceType": "MedicationRequest"}},
        {"event_type": "tool_call", "tool": "fhir_search", "args": {"resourceType": "Observation"}}]
ck("searched", searched_resource_types(traj) == {"MedicationRequest", "Observation"})
ck("missing_gov_allergy", missing_required_evidence(POLICY, traj) == ["AllergyIntolerance"])
ck("no_missing_when_all", missing_required_evidence(POLICY, traj + [
    {"event_type": "tool_call", "tool": "fhir_search", "args": {"resourceType": "AllergyIntolerance"}}]) == [])

# build the ServiceRequest the eval (validate_service_order) will accept
rt, res = build_order_resource({"text": "Order pelvic ultrasound with transvaginal approach", "category": "imaging"}, r)
ck("build_rt", rt == "ServiceRequest")
ck("build_status_intent", res["status"] == "active" and res["intent"] == "order")
ck("build_subject", res["subject"]["reference"] == "Patient/MRN2970753705")
ck("build_codetext", "pelvic ultrasound" in res["code"]["text"].lower())
ck("build_authored", res["authoredOn"] == "2022-02-06T23:17:00+00:00")
ck("build_requester", res["requester"]["reference"] == "Practitioner/dr-teresa-hayes")

# medication category -> MedicationRequest
rt2, res2 = build_order_resource({"text": "Start metformin 500mg", "category": "medication"}, r)
ck("build_med_rt", rt2 == "MedicationRequest")

# fail-safe: no subject -> no resource
ck("no_subject_no_res", build_order_resource({"text": "x", "category": "imaging"}, {}) == (None, None))
ck("blank_text_no_res", build_order_resource({"text": "  ", "category": "imaging"}, r) == (None, None))

n = sum(1 for _, c in R if c)
print("\n%d/%d effect_completion tests passed" % (n, len(R)))
assert n == len(R), "FAILURES"
