#!/usr/bin/env python3
"""Build ref_ranges.json (benchmark-fixed reference ranges, Tier 1/2) from docs/01.

range_type: interval | upper_threshold | lower_threshold | clinical_threshold | context_required
NOT for clinical decision-making — reproducible benchmark grading only. Pending clinician review.
"""
import json
SOURCE = "conventional adult reference ranges (benchmark-fixed v1; pending clinician review)"
rows = []
def add(loinc, disp, unit, accepted, rtype, direction, low, high, sex="any", note=""):
    rows.append({"loinc": loinc, "display": disp, "canonical_unit": unit, "accepted_units": accepted,
                 "range_type": rtype, "abnormal_direction": direction, "low": low, "high": high,
                 "sex": sex, "age_min": 18, "age_max": None, "clinical_context_required": rtype == "context_required",
                 "source": SOURCE, "benchmark_note": note or "Adult fixed benchmark range."})
def addMF(loinc, disp, unit, accepted, m, f, note=""):
    add(loinc, disp, unit, accepted, "interval", "both", m[0], m[1], "male", note)
    add(loinc, disp, unit, accepted, "interval", "both", f[0], f[1], "female", note)

# --- interval (both-sided) ---
add("2823-3", "Potassium", "mmol/L", ["mEq/L"], "interval", "both", 3.5, 5.0)
addMF("2160-0", "Creatinine", "mg/dL", [], (0.7, 1.3), (0.6, 1.1))
addMF("4544-3", "Hematocrit", "%", [], (41, 50), (36, 44))
add("3094-0", "Urea nitrogen (BUN)", "mg/dL", [], "interval", "both", 7, 20)
add("17861-6", "Calcium", "mg/dL", [], "interval", "both", 8.5, 10.2)
add("2951-2", "Sodium", "mmol/L", ["mEq/L"], "interval", "both", 135, 145)
addMF("718-7", "Hemoglobin", "g/dL", [], (13.5, 17.5), (12.0, 15.5))
add("2075-0", "Chloride", "mmol/L", ["mEq/L"], "interval", "both", 98, 107)
add("777-3", "Platelets", "K/uL", ["10*3/uL", "10\\u00b3/uL"], "interval", "both", 150, 450)
add("6690-2", "Leukocytes (WBC)", "K/uL", ["10*3/uL"], "interval", "both", 4.5, 11.0)
add("2028-9", "CO2 (Bicarbonate)", "mmol/L", ["mEq/L"], "interval", "both", 22, 29)
add("787-2", "MCV", "fL", [], "interval", "both", 80, 100)
add("785-6", "MCH", "pg", [], "interval", "both", 27, 33)
add("786-4", "MCHC", "g/dL", [], "interval", "both", 32, 36)
addMF("789-8", "Erythrocytes (RBC)", "MIL/uL", ["10*6/uL"], (4.7, 6.1), (4.2, 5.4))
add("788-0", "RDW", "%", [], "interval", "both", 11.5, 14.5)
add("33037-3", "Anion gap", "mmol/L", ["mEq/L"], "interval", "both", 8, 12)
add("1975-2", "Total bilirubin", "mg/dL", [], "interval", "both", 0.1, 1.2)
add("1742-6", "ALT", "U/L", [], "interval", "both", 7, 56)
add("1920-8", "AST", "U/L", [], "interval", "both", 10, 40)
add("1751-7", "Albumin", "g/dL", [], "interval", "both", 3.5, 5.0)
add("2885-2", "Total protein", "g/dL", [], "interval", "both", 6.0, 8.3)
add("6768-6", "Alkaline phosphatase", "U/L", [], "interval", "both", 44, 147)
add("10834-0", "Globulin", "g/dL", [], "interval", "both", 2.0, 3.5)
add("5905-5", "Monocytes [%]", "%", [], "interval", "both", 2, 8)
add("770-8", "Neutrophils [%]", "%", [], "interval", "both", 40, 70)
add("736-9", "Lymphocytes [%]", "%", [], "interval", "both", 20, 40)
add("714-6", "Eosinophils [%]", "%", [], "interval", "both", 1, 4)
add("706-2", "Basophils [%]", "%", [], "interval", "both", 0.5, 1)
add("2777-1", "Phosphorus", "mg/dL", [], "interval", "both", 2.5, 4.5)
add("3016-3", "TSH", "uIU/mL", ["mIU/L"], "interval", "both", 0.4, 4.0)
add("19123-9", "Magnesium", "mg/dL", [], "interval", "both", 1.7, 2.2)
add("2532-0", "LDH", "U/L", [], "interval", "both", 140, 280)
add("1968-7", "Direct bilirubin", "mg/dL", [], "interval", "both", 0.0, 0.3)
addMF("2276-4", "Ferritin", "ng/mL", ["ug/L"], (24, 300), (12, 150))

# --- threshold / risk-target / context ---
add("2345-7", "Glucose", "mg/dL", [], "clinical_threshold", "both", 70, 200, "any",
    "Serum/plasma glucose; fasting status NOT encoded -> conservative: <70 low, >=200 clinically high. Optional fasting interval 70-99 (declare if used).")
add("33914-3", "eGFR", "mL/min/1.73m2", ["mL/min/1.73 m2"], "lower_threshold", "low", 60, None, "any",
    ">=60 not CKD-flagged; >=90 normal. <60 = abnormal (risk stratification).")
add("2089-1", "LDL cholesterol", "mg/dL", [], "risk_target", "high", None, 100, "any", ">=100 non-ideal (treatment target, not a population normal range).")
add("2571-8", "Triglyceride", "mg/dL", [], "upper_threshold", "high", None, 150, "any", ">=150 high.")
add("4548-4", "Hemoglobin A1c", "%", [], "clinical_threshold", "high", None, 6.5, "any", ">=6.5 diabetes (diagnostic threshold; normal <5.7).")
addMF("2085-9", "HDL cholesterol", "mg/dL", [], (40, None), (50, None))  # placeholder; overwrite below
# fix HDL: lower_threshold (low is bad), sex-specific cutoff
rows = [r for r in rows if r["loinc"] != "2085-9"]
add("2085-9", "HDL cholesterol", "mg/dL", [], "lower_threshold", "low", 40, None, "male", "M <40 low (higher is better).")
add("2085-9", "HDL cholesterol", "mg/dL", [], "lower_threshold", "low", 50, None, "female", "F <50 low (higher is better).")
add("2093-3", "Cholesterol (total)", "mg/dL", [], "risk_target", "high", None, 200, "any", ">=200 non-ideal.")
add("6301-6", "INR", "", [], "context_required", "both", 0.8, 1.1, "any",
    "Assumes patient is NOT on anticoagulation; v1 SKIP unless anticoag status known.")

out = {"_meta": {"spec": "benchmark-fixed reference ranges v1", "source": SOURCE,
                 "disclaimer": "Used for reproducible benchmark grading, not for clinical decision-making.",
                 "tier": "Tier 1/2", "n_entries": len(rows)},
       "ranges": rows}
import sys
path = sys.argv[1] if len(sys.argv) > 1 else "ref_ranges.json"
json.dump(out, open(path, "w"), indent=1, ensure_ascii=False)
print(f"wrote {len(rows)} range entries ({len({r['loinc'] for r in rows})} distinct LOINC) -> {path}")
