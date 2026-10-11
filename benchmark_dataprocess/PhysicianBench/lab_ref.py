#!/usr/bin/env python3
"""Benchmark-fixed lab reference-range lookup + classification (enhancement #1).

Powers the augmented `get_lab_reference_range(loinc, sex, age, unit)` tool and the abnormal-lab
verifier. Unit handling is a HARD check: canonical/accepted-alias -> classify; else -> skip (unit_mismatch).
"""
import json, os

_PATH = os.path.join(os.path.dirname(__file__), "ref_ranges.json")

def load(path=_PATH):
    return json.load(open(path))["ranges"]

def _norm(u):
    return (u or "").replace(" ", "").lower()

def get_lab_reference_range(loinc, sex=None, age=None, unit=None, ranges=None):
    """Return the matching range entry (sex-specific preferred, else 'any'); None if LOINC unknown."""
    ranges = ranges if ranges is not None else load()
    cands = [r for r in ranges if r["loinc"] == loinc]
    if not cands:
        return None
    sx = (sex or "").lower()
    pick = next((r for r in cands if r["sex"] == sx), None) or next((r for r in cands if r["sex"] == "any"), None) or cands[0]
    return pick

def unit_ok(entry, unit):
    if not unit:
        return True  # no unit on observation -> assume canonical
    u = _norm(unit)
    if u == _norm(entry["canonical_unit"]):
        return True
    return any(u == _norm(a) for a in entry.get("accepted_units", []))

def classify(value, loinc, sex=None, unit=None, ranges=None):
    """-> {status, ...}. status in {normal, low, high, abnormal, skip, unknown_loinc}."""
    e = get_lab_reference_range(loinc, sex, None, unit, ranges)
    if e is None:
        return {"status": "unknown_loinc", "loinc": loinc}
    if not unit_ok(e, unit):
        return {"status": "skip", "reason": "unit_mismatch", "loinc": loinc,
                "got_unit": unit, "expected": e["canonical_unit"], "accepted": e.get("accepted_units", [])}
    if e["range_type"] == "context_required":
        return {"status": "skip", "reason": "clinical_context_required", "loinc": loinc, "note": e.get("benchmark_note")}
    lo, hi = e.get("low"), e.get("high")
    rt, direction = e["range_type"], e["abnormal_direction"]
    res = {"loinc": loinc, "display": e["display"], "range_type": rt, "low": lo, "high": hi, "unit": e["canonical_unit"]}
    below = lo is not None and value < lo
    above = hi is not None and value > hi
    if rt in ("interval",):
        res["status"] = "low" if below else ("high" if above else "normal")
    elif rt == "lower_threshold":   # low is abnormal (e.g. HDL, eGFR)
        res["status"] = "low" if below else "normal"
    elif rt in ("upper_threshold", "risk_target", "clinical_threshold"):
        # high (>=) is abnormal; clinical_threshold may also have a low cutoff (e.g. glucose)
        hi_abn = hi is not None and value >= hi
        lo_abn = lo is not None and value < lo and direction == "both"
        res["status"] = "high" if hi_abn else ("low" if lo_abn else "normal")
    else:
        res["status"] = "normal"
    return res

if __name__ == "__main__":
    import sys
    # quick self-test
    rs = load()
    print("loaded", len(rs), "entries")
    for v, code, sx, u in [(2.9, "2823-3", "any", "mmol/L"), (5.6, "2823-3", "any", "mEq/L"),
                            (260, "2345-7", "any", "mg/dL"), (45, "33914-3", "any", "mL/min/1.73m2"),
                            (130, "2089-1", "any", "mg/dL"), (6.9, "4548-4", "any", "%"),
                            (1.5, "2823-3", "any", "g/dL")]:
        print(v, code, "->", classify(v, code, sx, u))
