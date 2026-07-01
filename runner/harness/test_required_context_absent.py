"""Commit A acceptance (unit): ABSENT resolves the obligation; a resolved unit is NOT re-ACQUIRED; the ACQUIRE
uses the adapter-compiled affordance (patient param). Uses a minimal fake ctx/ledger."""
import sys, os
sys.path.insert(0, "runner")
os.environ["MH_REPAIR"] = "full"
from harness.capabilities.required_context import RequiredContext

MANIFEST = {
    "subject": {"type": "patient"},
    "evidence_affordances": [
        {"evidence_unit": "AllergyIntolerance", "tool": "fhir_search", "subject_arg": "patient",
         "subject_ref_style": "typed_ref", "static_args": {"resourceType": "AllergyIntolerance"},
         "result_semantics": {"collection_paths": ["entries"], "absence_when_empty": True}},
    ],
}

class Led:
    def __init__(self, evidence): self.evidence = evidence; self.acquire_count = 0
    def subject_id(self): return "MRN2970753705"

class Ctx:
    def __init__(self, led): self.ledger = led; self.manifest = MANIFEST

R = []
def ck(n, c): R.append((n, bool(c))); print(("OK  " if c else "FAIL") + " " + n)

rc = RequiredContext()
REQ = [("allergy_obl", "AllergyIntolerance")]

# 1) NO evidence yet -> ACQUIRE, and the affordance uses patient=Patient/<id> (adapter-compiled)
d1 = rc._missing_obligation_acquire(Ctx(Led([])), REQ)
ck("acquire_when_missing", d1 is not None and d1.type == "ACQUIRE")
na = (d1.raw.extra if (d1 and getattr(d1, "raw", None)) else {}).get("next_action") or (d1.extra or {}).get("next_action") if d1 else {}
# decision.extra path may differ; pull from the decision object robustly
na = None
if d1 is not None:
    ex = getattr(d1, "extra", None) or (getattr(d1, "raw", None) and getattr(d1.raw, "extra", None)) or {}
    na = ex.get("next_action")
ck("acquire_uses_patient_param", bool(na) and na.get("args", {}).get("patient") == "Patient/MRN2970753705"
   and "subject" not in na.get("args", {}))

# 2) ABSENT evidence for the unit (confirmed no allergies) -> obligation RESOLVED -> NO re-ACQUIRE
absent_ev = [{"resource": "AllergyIntolerance", "evidence_state": "ABSENT", "scope_relation": "matched", "subject_id": "MRN2970753705"}]
d2 = rc._missing_obligation_acquire(Ctx(Led(absent_ev)), REQ)
ck("absent_resolves_no_reacquire", d2 is None)

# 3) PRESENT evidence -> also resolved -> no ACQUIRE
present_ev = [{"resource": "AllergyIntolerance", "evidence_state": "PRESENT", "scope_relation": "matched", "subject_id": "MRN2970753705"}]
ck("present_resolves", rc._missing_obligation_acquire(Ctx(Led(present_ev)), REQ) is None)

# 4) FAILED evidence -> NOT resolved -> still ACQUIRE (the live-bug case: a failed read must not resolve)
failed_ev = [{"resource": "AllergyIntolerance", "evidence_state": "FAILED", "scope_relation": "matched", "subject_id": "MRN2970753705"}]
ck("failed_not_resolved_reacquire", rc._missing_obligation_acquire(Ctx(Led(failed_ev)), REQ) is not None)

# 5) foreign-subject ABSENT does NOT resolve
foreign_ev = [{"resource": "AllergyIntolerance", "evidence_state": "ABSENT", "scope_relation": "foreign"}]
ck("foreign_absent_not_resolved", rc._missing_obligation_acquire(Ctx(Led(foreign_ev)), REQ) is not None)

n = sum(1 for _, c in R if c)
print("\n%d/%d required_context ABSENT/dedup tests passed" % (n, len(R)))
assert n == len(R), "FAILURES"
