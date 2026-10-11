#!/usr/bin/env python
"""The FIXED, dataset-level train/val/test split over ALL 342 tasks (PB 100 + HAB 135 + MedCTA 107).
DETERMINISTIC + STRATIFIED (5:2:3 by exact per-stratum proportion, round-half-up favoring test/val). Roles:
  held_in/train -- tasks whose failures may be inspected when developing harness mechanisms.
  held_in/val   -- tasks for tuning and refinement decisions. NOT the final verdict.
  held_out/test (SEALED) -- the final evaluation ONLY; NEVER used in any tuning or refinement.
Stratified per bench so every group is represented in each split (no "only-easy" leakage):
  PB by clinical PHASE (Dx/Workup/Treatment/Medication) ; HAB by (category x difficulty) ; MedCTA by modality.
A SINGLETON stratum (size 1) goes to the SEALED test (can't 3-split; a rare group is covered at final evaluation, never tuned on).
Re-run to regenerate the .txt files identically (no RNG, no timestamps)."""
import json, collections, os
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
def load(f): return [json.loads(l) for l in open(f) if l.strip()]
def tid(r): return r.get("task_id") or r.get("id")
def pb_phase(r):
    sp = str(r.get("specialty") or "?"); return sp.split(",")[1].strip() if "," in sp else sp
BENCH = {
 "PhysicianBench": ("benchmark_dataprocess/PhysicianBench/tasks_unified.jsonl", pb_phase),
 "HealthAdminBench": ("benchmark_dataprocess/HealthAdminBench/tasks_unified.jsonl",
                      lambda r: (tid(r).split("-")[1] if len(tid(r).split("-")) > 1 else "?", str(r.get("difficulty")))),
 "MedCTA": ("benchmark_dataprocess/MedCTA/tasks_unified.jsonl", lambda r: str(r.get("anatomical_region") or "?")),
}
PATTERN = ["train", "train", "test", "train", "val", "train", "test", "train", "val", "test"]  # 5:2:3
def build():
    splits = {"train": [], "val": [], "test": []}
    for bench, (f, keyfn) in BENCH.items():
        strata = collections.defaultdict(list)
        for r in load(os.path.join(ROOT, f)): strata[keyfn(r)].append(tid(r))
        for k in sorted(strata, key=str):
            mem = sorted(strata[k]); N = len(mem)   # members sorted lexically by task id; each stratum is cut in that order
            if N == 1:                        # a singleton stratum -> the SEALED test (covered at final evaluation)
                splits["test"].append(mem[0]); continue
            n_test = int(0.3 * N + 0.5)       # 5:2:3 by exact per-stratum proportion; round-half-UP favors test/val
            n_val = int(0.2 * N + 0.5)        # so remainders do NOT pile into train
            n_train = N - n_val - n_test
            splits["train"] += mem[:n_train]
            splits["val"] += mem[n_train:n_train + n_val]
            splits["test"] += mem[n_train + n_val:]
    return splits
if __name__ == "__main__":
    sp = build()
    for s, name in (("train", "held_in_train"), ("val", "held_in_val"), ("test", "held_out_test")):
        open(os.path.join(HERE, name + ".txt"), "w").write("\n".join(sorted(sp[s])) + "\n")
    print("train=%d val=%d test=%d (held_in=%d held_out=%d)" %
          (len(sp["train"]), len(sp["val"]), len(sp["test"]), len(sp["train"]) + len(sp["val"]), len(sp["test"])))
