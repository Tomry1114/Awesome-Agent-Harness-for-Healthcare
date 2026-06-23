#!/usr/bin/env python3
"""Honest, methodology-corrected dissociation re-analysis (addresses the peer-review critique).

Fixes vs the old runner/dissociation.py:
  - PER-BENCHMARK STRATIFIED (never pool the 30 tasks -> avoids Simpson's-paradox confounding by
    benchmark identity, which drove the old negative correlations).
  - STRICT-ONLY: dimension scores come from score_eligible checkpoints (proxy excluded by construction
    in dimension_scores). Constant columns (no variance, e.g. HAB Verification all-0) are flagged and
    excluded from correlation (a constant cannot correlate).
  - Reports outcome VARIANCE per benchmark; if outcome is constant (e.g. PB 0/10) it states plainly that
    NO within-benchmark correlation is possible.
  - Flags non-independent-judge dimensions (same-family judge) as not-primary-evidence.
  - Prefers TASK-LEVEL dissociation CASES (success-but-low-process) over a fragile n=10 correlation.

Usage: python runner/strict_dissociation.py [pb_dir medcta_dir hab_dir]
"""
import json, os, sys, glob, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate_report import _load, _remap, MODULES

DEFAULT = [("results_pbC/gpt5", "PhysicianBench"),
           ("results_mctaD/gpt5", "MedCTA"),      # fresh default-config run
           ("results_hab10/gpt5", "HealthAdminBench")]
PROCESS = ["Tooling", "Lifecycle", "Observability", "Verification", "Governance"]


def _pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return None  # constant column -> undefined
    return round(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (dx * dy), 2)


def _nonindep(results):
    q = collections.Counter()
    for r in results:
        for x in (r.get("qualification") or []):
            q[x] += 1
    return q.get("non_independent_judge", 0)


def analyze(d, bench):
    results = _remap(_load(d), bench)
    n = len(results)
    if not n:
        print("== %s: no data at %s ==" % (bench, d)); return
    rows = [(r.get("task_id"), 1.0 if r.get("success") else 0.0, r.get("dimension_scores") or {}) for r in results]
    outcomes = [o for _, o, _ in rows]
    nind = _nonindep(results)
    print("\n================  %s  (n=%d, strict-only)  ================" % (bench, n))
    print("outcome: %d/%d success%s" % (int(sum(outcomes)), n,
          "   [non_independent_judge on %d/%d tasks -> judge-derived dims are NOT primary evidence]" % (nind, n) if nind else ""))

    if len(set(outcomes)) == 1:
        print("  ⚠ outcome has NO variance (all %g) -> NO within-benchmark correlation is possible." % outcomes[0])
    else:
        print("  %-14s %6s %12s %12s %s" % ("dim", "corr", "mean|succ", "mean|fail", "note"))
        for m in MODULES:
            pairs = [(o, ds[m]) for _, o, ds in rows if isinstance(ds.get(m), (int, float))]
            if len(pairs) < 3:
                continue
            vals = [v for _, v in pairs]
            if len(set(round(v, 3) for v in vals)) == 1:
                print("  %-14s %6s %12s %12s constant -> excluded" % (m, "—", "—", "—")); continue
            oc = [o for o, _ in pairs]
            succ = [v for o, v in pairs if o >= 1.0]; fail = [v for o, v in pairs if o < 1.0]
            print("  %-14s %6s %12s %12s %s" % (
                m, _pearson(oc, vals),
                round(sum(succ) / len(succ), 2) if succ else "—",
                round(sum(fail) / len(fail), 2) if fail else "—",
                "judge=non-indep" if (nind and m in ("Verification", "Governance")) else ""))

    # task-level dissociation cases (more defensible than n=10 correlation)
    cases = []
    for tid, o, ds in rows:
        if o >= 1.0:
            lows = [m for m in PROCESS if isinstance(ds.get(m), (int, float)) and ds[m] < 0.5]
            if lows:
                cases.append((tid, lows))
    if cases:
        print("  task-level cases (SUCCESS but a strict process dim < 0.5):")
        for tid, lows in cases:
            print("    - %-26s fails: %s" % (tid, ",".join(lows)))
    else:
        print("  task-level cases: none (no success task fails a strict process dim)")


if __name__ == "__main__":
    benches = DEFAULT
    print("METHODOLOGY: per-benchmark stratified, strict-only, constant/non-independent flagged, NO 30-task pooling.")
    for d, b in benches:
        analyze(d, b)
    print("\nHONEST READ: with ~10 tasks/benchmark and (PB) zero outcome variance, correlations are not")
    print("statistically established; task-level dissociation CASES are the defensible evidence. Full")
    print("n>=100 x multi-seed x multi-model needed before any correlation claim.")
