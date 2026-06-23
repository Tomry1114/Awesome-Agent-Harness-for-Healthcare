#!/usr/bin/env python3
"""Track-B retag: category=reasoning -> dimension=Verification (correctness layer).

B口径: Execution=完成度(action happened); Verification=正确性(output correct vs ground truth).
The 'reasoning' checkpoints judge whether the agent's conclusion is CORRECT against clinical/
admin/answer ground truth -> that is Verification, not Execution. Single consistent rule across
all three benchmarks. Backs up each file, records original_dimension for audit/reversibility.
"""
import json, os, collections, shutil

ROOT = "benchmark_dataprocess"
BENCHES = ["PhysicianBench", "HealthAdminBench", "MedCTA"]


def dist(path):
    c = collections.Counter()
    for l in open(path):
        for cp in (json.loads(l).get("checkpoints") or []):
            c[cp.get("dimension")] += 1
    return dict(sorted(c.items()))


for b in BENCHES:
    p = os.path.join(ROOT, b, "tasks_unified.jsonl")
    if not os.path.exists(p):
        print("SKIP (missing):", p); continue
    before = dist(p)
    bak = p + ".bak_btag"
    if not os.path.exists(bak):
        shutil.copy(p, bak)
    out, moved = [], 0
    for l in open(p):
        t = json.loads(l)
        for cp in (t.get("checkpoints") or []):
            if cp.get("category") == "reasoning" and cp.get("dimension") != "Verification":
                cp["original_dimension"] = cp.get("dimension")
                cp["original_subdimension"] = cp.get("subdimension")
                cp["dimension"] = "Verification"
                cp["subdimension"] = "result_verification"
                cp["retag_rule"] = "track_B: category=reasoning -> Verification"
                moved += 1
        out.append(json.dumps(t, ensure_ascii=False))
    open(p, "w").write("\n".join(out) + "\n")
    after = dist(p)
    print("== %s ==  moved %d reasoning cp -> Verification" % (b, moved))
    print("  before:", before)
    print("  after :", after)
