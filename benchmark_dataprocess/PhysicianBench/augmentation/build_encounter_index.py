#!/usr/bin/env python3
"""#3 (external index): reconstruct synthetic ENCOUNTERS from FHIR timestamps — patient-level EHR
-> encounter-level workflow. Does NOT inject FHIR; emits encounter_index.json. Encounter = one
calendar day with >=1 resource (configurable). Strengthens Lifecycle/Context for PhysicianBench.
"""
import os, json, argparse, urllib.request, urllib.parse, collections

DATE_FIELD = {  # resourceType -> candidate date fields (first present wins)
    "Observation": ["effectiveDateTime"], "MedicationRequest": ["authoredOn"],
    "Condition": ["onsetDateTime", "recordedDate"], "Procedure": ["performedDateTime"],
    "DocumentReference": ["date"],
}

def get(base, path):
    req = urllib.request.Request(base + path, headers={"Accept": "application/fhir+json"})
    return json.load(urllib.request.urlopen(req, timeout=60))

def day_of(res, rt):
    for f in DATE_FIELD[rt]:
        v = res.get(f)
        if v: return v[:10]
    pp = res.get("performedPeriod") or {}
    if pp.get("start"): return pp["start"][:10]
    return None

def build_for_patient(base, mrn):
    days = collections.defaultdict(lambda: collections.defaultdict(list))
    times = collections.defaultdict(list)
    for rt in DATE_FIELD:
        try:
            b = get(base, f"/{rt}?subject=Patient/{mrn}&_count=1000")
        except Exception:
            continue
        for e in b.get("entry", []):
            r = e.get("resource", {}); d = day_of(r, rt)
            if d and r.get("id"):
                days[d][rt].append(r["id"]); times[d].append(d)
    encs = []
    for i, d in enumerate(sorted(days), 1):
        encs.append({"patient_ref": mrn, "encounter_id": f"synthetic-enc-{mrn}-{i:03d}",
                     "time_window": {"start": d + "T00:00:00", "end": d + "T23:59:59"},
                     "linked_resources": {rt: days[d].get(rt, []) for rt in DATE_FIELD},
                     "provenance": "augmented"})
    return encs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fhir-base", default="http://localhost:38080/fhir")
    ap.add_argument("--tasks-unified", default=os.path.join(os.path.dirname(__file__), "..", "tasks_unified.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "encounter_index.json"))
    args = ap.parse_args()
    mrns = []
    for line in open(args.tasks_unified):
        t = json.loads(line); pr = (t.get("context") or {}).get("patient_ref")
        if pr and pr not in mrns: mrns.append(pr)
    if args.limit: mrns = mrns[:args.limit]
    all_enc = []
    for i, mrn in enumerate(mrns, 1):
        all_enc.extend(build_for_patient(args.fhir_base, mrn))
    out = {"_meta": {"provenance": "augmented", "rule": "encounter = calendar day with >=1 resource",
                     "n_patients": len(mrns), "n_encounters": len(all_enc), "injected_to_fhir": False},
           "encounters": all_enc}
    json.dump(out, open(args.out, "w"), indent=1, ensure_ascii=False)
    print(f"patients={len(mrns)} encounters={len(all_enc)} -> {args.out}")
    if all_enc:
        e = all_enc[0]; print("sample:", e["encounter_id"], e["time_window"]["start"],
              {k: len(v) for k, v in e["linked_resources"].items() if v})

if __name__ == "__main__":
    main()
