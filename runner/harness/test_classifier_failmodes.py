"""Classifier failure-mode tests (#5): a failed/uncertain read must NEVER be misread as ABSENT."""
import sys
sys.path.insert(0, "runner")
from harness.evidence_state import classify_evidence_state, PRESENT, ABSENT, UNKNOWN, FAILED

SEM = {"collection_paths": ["entries"], "absence_when_empty": True}
R = []
def ck(n, c): R.append((n, bool(c))); print(("OK  " if c else "FAIL") + " " + n)

# the dangerous combos the old classifier got wrong (empty collection + a failure/uncertain signal)
ck("empty+timeout->UNKNOWN",   classify_evidence_state({"entries": [], "timeout": True}, SEM) == UNKNOWN)
ck("empty+success_false->FAILED", classify_evidence_state({"entries": [], "success": False}, SEM) == FAILED)
ck("empty+ok_false->FAILED",   classify_evidence_state({"entries": [], "ok": False}, SEM) == FAILED)
ck("empty+status_error->FAILED", classify_evidence_state({"entries": [], "status": "error"}, SEM) == FAILED)
ck("empty+status_failed->FAILED", classify_evidence_state({"entries": [], "status": "failed"}, SEM) == FAILED)
ck("empty+partial->UNKNOWN",   classify_evidence_state({"entries": [], "partial": True}, SEM) == UNKNOWN)
ck("empty+status_unknown->UNKNOWN", classify_evidence_state({"entries": [], "status": "unknown"}, SEM) == UNKNOWN)

# genuine clean cases still classify correctly
ck("clean_empty->ABSENT",      classify_evidence_state({"entries": []}, SEM) == ABSENT)
ck("clean_nonempty->PRESENT",  classify_evidence_state({"entries": [{"x": 1}]}, SEM) == PRESENT)
ck("hard_error->FAILED",       classify_evidence_state({"error": "HTTP 500"}, SEM) == FAILED)
ck("status_ok_empty->ABSENT",  classify_evidence_state({"entries": [], "status": "ok"}, SEM) == ABSENT)

n = sum(1 for _, c in R if c)
print("\n%d/%d classifier failure-mode tests passed" % (n, len(R)))
assert n == len(R), "FAILURES"
