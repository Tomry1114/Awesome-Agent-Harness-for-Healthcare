"""AdapterCompiler tests (Commit A) -- the compiler emits the CORRECT per-resource query (the live-bug regression)."""
import sys
sys.path.insert(0, "runner")
from harness.adapter_compiler import compile_evidence_request, format_subject, result_semantics_for

MANIFEST = {
    "subject": {"type": "patient"},
    "evidence_affordances": [
        {"evidence_unit": "AllergyIntolerance", "tool": "fhir_search", "subject_arg": "patient",
         "subject_ref_style": "typed_ref", "static_args": {"resourceType": "AllergyIntolerance"},
         "result_semantics": {"collection_paths": ["entries"], "absence_when_empty": True}},
        {"evidence_unit": "ServiceRequest", "tool": "fhir_search", "subject_arg": "subject",
         "subject_ref_style": "typed_ref", "static_args": {"resourceType": "ServiceRequest"}},
        {"evidence_unit": "LabPanel", "tool": "labs_api", "subject_arg": "mrn", "subject_ref_style": "bare_id",
         "static_args": {}},
    ],
}
R = []
def ck(n, c): R.append((n, bool(c))); print(("OK  " if c else "FAIL") + " " + n)

# THE live-bug regression: AllergyIntolerance -> patient=Patient/<id> (NOT subject)
req = compile_evidence_request(MANIFEST, "AllergyIntolerance", "MRN2970753705", obligation_id="o1")
ck("allergy_tool", req.affordance["tool"] == "fhir_search")
ck("allergy_param_is_patient", "patient" in req.affordance["args"] and "subject" not in req.affordance["args"])
ck("allergy_typed_ref", req.affordance["args"]["patient"] == "Patient/MRN2970753705")
ck("allergy_resourcetype", req.affordance["args"]["resourceType"] == "AllergyIntolerance")
ck("allergy_obligation", req.obligation_id == "o1" and req.evidence_unit == "AllergyIntolerance")
ck("allergy_semantics", req.expected_result_semantics.get("absence_when_empty") is True)

# ServiceRequest -> subject=Patient/<id> (a different param, adapter-chosen)
sr = compile_evidence_request(MANIFEST, "ServiceRequest", "MRN2970753705")
ck("sr_param_is_subject", sr.affordance["args"].get("subject") == "Patient/MRN2970753705" and "patient" not in sr.affordance["args"])

# bare_id style + already-typed input
ck("bare_id_style", compile_evidence_request(MANIFEST, "LabPanel", "MRN123").affordance["args"]["mrn"] == "MRN123")
ck("typed_input_kept", format_subject(MANIFEST, "Patient/MRN9", "typed_ref") == "Patient/MRN9")
ck("bare_from_typed", format_subject(MANIFEST, "Patient/MRN9", "bare_id") == "MRN9")

# fail-safe: no affordance / no subject -> None (core never fabricates a query)
ck("no_affordance_none", compile_evidence_request(MANIFEST, "Unknown", "MRN1") is None)
ck("no_subject_none", compile_evidence_request(MANIFEST, "AllergyIntolerance", "") is None)

# result_semantics_for falls back to a generic collection spec
ck("semantics_declared", result_semantics_for(MANIFEST, "AllergyIntolerance").get("absence_when_empty") is True)
ck("semantics_default", result_semantics_for(MANIFEST, "Nope")["collection_paths"] == ["entries"])

n = sum(1 for _, c in R if c)
print("\n%d/%d adapter_compiler tests passed" % (n, len(R)))
assert n == len(R), "FAILURES"
