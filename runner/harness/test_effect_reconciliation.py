"""Tests for effect reconciliation (Phase 4 detector). Pure -- no judge, no gateway."""
import sys
sys.path.insert(0, "runner")
from harness.effect_reconciliation import is_realized, unrealized_commitments, _keywords

R = []
def ck(n, c): R.append((n, bool(c))); print(("OK  " if c else "FAIL") + " " + n)

# is_realized: conservative toward True
ck("realized_match", is_realized("Order pelvic ultrasound", ["Pelvic ultrasound transvaginal", "CBC"]) is True)
ck("unrealized_absent", is_realized("Order pelvic ultrasound", ["CBC", "TSH panel"]) is False)
ck("realized_empty_kw", is_realized("order the", []) is True)                 # nothing salient -> never create
ck("unrealized_no_state", is_realized("Order pelvic ultrasound", []) is False)

# the real AUB case: agent committed the order, state has ZERO orders -> unrealized
committed = [{"text": "Order pelvic ultrasound with transvaginal approach", "category": "imaging"}]
u = unrealized_commitments(committed, [])
ck("aub_unrealized_one", len(u) == 1 and u[0].category == "imaging")
ck("aub_keywords", "ultrasound" in u[0].keywords and "pelvic" in u[0].keywords)

# once the order exists in state -> nothing to complete
ck("aub_realized_after_create",
   unrealized_commitments(committed, ["ServiceRequest: pelvic ultrasound (transvaginal), active"]) == [])

# empty / degenerate
ck("empty_committed", unrealized_commitments([], ["x"]) == [])
ck("blank_text_skipped", unrealized_commitments([{"text": "  "}], []) == [])

# multiple, mixed realized/unrealized
multi = [{"text": "Order pelvic ultrasound", "category": "imaging"},
         {"text": "Start metformin 500mg", "category": "medication"}]
u2 = unrealized_commitments(multi, ["MedicationRequest: metformin 500 mg oral, active"])
ck("multi_one_unrealized", len(u2) == 1 and "ultrasound" in u2[0].keywords)

# stopword hygiene: 'order'/'study' don't create false matches
ck("stopword_no_false_match", is_realized("Order pelvic ultrasound study", ["order a study now"]) is False)

n = sum(1 for _, c in R if c)
print("\n%d/%d effect_reconciliation tests passed" % (n, len(R)))
assert n == len(R), "FAILURES"
