"""order_integrity shadow VERIFY -- catches an agent's self-inconsistent ServiceRequest orders
(wrong code for the named test / one code on two tests / supersede-without-revoke) WITHOUT authoring
any clinical decision or reading the checkpoint graders. Fixtures mirror order sets from PB adrenal-workup
trajectories of different agent models: one set with 3 real defects must be caught; two clean sets must NOT
false-positive. Run from any dir."""
import os, sys
RUNNER_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RUNNER_ROOT)
from harness.order_integrity import scan_orders

R = []
def ck(n, c):
    R.append((n, bool(c)))
    if not c: print("FAIL:", n)

def classes(f):
    return sorted(x["class"] for x in f)

# --- order set with 3 genuine defects --------------------------------------------------------------------
qwen = [
    {"code_code": "51698-3", "code_display": "Metanephrines free [Moles/volume] in Plasma", "status": "active"},
    {"code_code": "8625-6", "code_display": "Aldosterone [Mass/volume] in Serum or Plasma",
     "note_text": "Aldosterone with paired plasma renin activity for ARR screen", "status": "active"},   # renin is CONTEXT, must NOT flag
    {"code_code": "2191-5", "code_display": "Potassium [Moles/volume] in Serum or Plasma", "status": "active"},
    {"code_code": "2191-5", "code_display": "Dehydroepiandrosterone sulfate [Mass/volume] in Serum or Plasma",
     "note_text": "DHEA-S for androgen workup", "status": "active"},                                     # DUP_CODE with the potassium above
    {"code_code": "2823-3", "code_display": "Potassium [Moles/volume] in Serum or Plasma",
     "note_text": "Corrected order replaces erroneous order 212155", "status": "active"},                # SUPERSEDE_NOT_REVOKED (nothing revoked)
    {"code_code": "1751-7", "code_display": "Albumin [Mass/volume] in Serum or Plasma",
     "note_text": "Dexamethasone level to accompany 1mg overnight DST", "status": "active"},             # INTENT_CODE_MISMATCH
]
fq = scan_orders(qwen)
ck("qwen: DUP_CODE caught (2191-5 on two tests)", "DUP_CODE" in classes(fq))
ck("qwen: INTENT_CODE_MISMATCH caught (albumin coded, dexamethasone named)", "INTENT_CODE_MISMATCH" in classes(fq))
ck("qwen: SUPERSEDE_NOT_REVOKED caught", "SUPERSEDE_NOT_REVOKED" in classes(fq))
ck("qwen: the ARR context mention did NOT create a false mismatch", sum(1 for x in fq if x["class"] == "INTENT_CODE_MISMATCH") == 1)
ck("qwen: exactly the 3 real defects", len(fq) == 3)

# --- clean orders with CONTEXTUAL analyte mentions + a panel (must be 0) ------------------------------------
doubao = [
    {"code_code": "13983-4", "code_display": "DHEA-sulfate (DHEA-S), serum",
     "note_text": "Adjunctive test for subclinical Cushing (suppression suggests autonomous cortisol secretion).", "status": "active"},  # cortisol = context
    {"code_code": "29588-3", "code_display": "ACTH, plasma",
     "note_text": "ACTH to interpret alongside the dexamethasone suppression result", "status": "active"},   # dexamethasone = context, ACTH is named+coded
    {"code_code": "24321-7", "code_display": "Basic metabolic panel (BMP): Na, K, Cl, CO2, BUN, Cr, glucose, Ca",
     "note_text": "Check potassium specifically given hypertension.", "status": "active"},                  # panel contains potassium -> exempt
]
fd = scan_orders(doubao)
ck("doubao: clean, no false positives", fd == [])

# --- clean, distinct codes (must be 0) ----------------------------------------------------------------------
minimax = [
    {"code_code": "15074-8", "code_display": "Cortisol, serum post 1 mg dexamethasone suppression (overnight DST)", "status": "active"},
    {"code_code": "49708-9", "code_display": "Metanephrines, fractionated, plasma", "status": "active"},
    {"code_code": "3087-0", "code_display": "Aldosterone, plasma + Plasma Renin Activity (ARR)", "status": "active"},
    {"code_code": "2192-5", "code_display": "Dehydroepiandrosterone sulfate (DHEA-S), serum", "status": "active"},
]
fm = scan_orders(minimax)
ck("minimax: clean, no false positives", fm == [])

# --- content-free / oracle-blind guarantee: no clinical value or code is ever emitted -----------------
allf = fq + fd + fm
ck("advisory NEVER supplies a corrective code/value (only flags inconsistency)",
   all("suggested_code" not in x and "correct_code" not in x and "fix_value" not in x for x in allf))

ok = sum(1 for _, c in R if c)
print("\n%d/%d passed" % (ok, len(R)))
sys.exit(0 if ok == len(R) else 1)
