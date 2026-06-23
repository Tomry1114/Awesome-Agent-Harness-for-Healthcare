#!/usr/bin/env python3
"""Metric-dissociation analysis (the main thesis). On the now-7/7 clean data, show that a single
task_success/outcome metric CONFLATES orthogonal failure modes the harness dimensions separate.
Reuses aggregate_report._remap so per-task dimension scores reflect the current (track-B + rescored)
taxonomy. No API calls."""
import json, os, sys, glob, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate_report import _load, _remap, MODULES

BENCHES = [("results_pbC/gpt5", "PhysicianBench"), ("results_mctaC/gpt5", "MedCTA"), ("results_hab10/gpt5", "HealthAdminBench")]
PROCESS = ["Tooling", "Lifecycle", "Observability", "Verification", "Governance"]  # non-outcome dims


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return round(num / (dx * dy), 2) if dx and dy else None


def analyze():
    per_task = []  # (bench, tid, outcome, {dim:score})
    for d, bench in BENCHES:
        for r in _remap(_load(d), bench):
            ds = r.get("dimension_scores") or {}
            outcome = 1.0 if r.get("success") else 0.0
            per_task.append((bench, r.get("task_id"), outcome, ds))

    print("== per-task: outcome vs process dimensions (dissociation = success but a process dim < 0.5) ==")
    dissoc = 0
    for bench, tid, oc, ds in per_task:
        lows = [m for m in PROCESS if isinstance(ds.get(m), (int, float)) and ds[m] < 0.5]
        flag = ""
        if oc >= 1.0 and lows:
            dissoc += 1
            flag = "  <<< SUCCESS but fails: " + ",".join(lows)
        print("  %-14s %-26s outcome=%d%s" % (bench, (tid or "")[:26], int(oc), flag))
    print("\n  -> %d/%d 'successful' tasks still fail >=1 process/safety dimension" % (dissoc, len(per_task)))

    print("\n== outcome vs each dimension: correlation + mean(score | success) vs mean(score | fail) ==")
    print("  %-14s %6s  %12s  %12s" % ("dim", "corr", "mean|success", "mean|fail"))
    for m in MODULES:
        pairs = [(oc, ds[m]) for _, _, oc, ds in per_task if isinstance(ds.get(m), (int, float))]
        if len(pairs) < 3:
            continue
        ocs = [p[0] for p in pairs]
        sc = [p[1] for p in pairs]
        succ = [s for o, s in pairs if o >= 1.0]
        fail = [s for o, s in pairs if o < 1.0]
        corr = _pearson(ocs, sc)
        ms = round(sum(succ) / len(succ), 2) if succ else None
        mf = round(sum(fail) / len(fail), 2) if fail else None
        print("  %-14s %6s  %12s  %12s" % (m, corr, ms, mf))

    print("\n== interpretation ==")
    print("  Low |corr| and overlapping success/fail means => that dimension is ORTHOGONAL to outcome:")
    print("  the single success metric does not capture it. High corr => outcome already reflects it.")


if __name__ == "__main__":
    analyze()
