#!/usr/bin/env python3
"""Read-only: re-map existing run bundles to the CURRENT tasks_unified dimension tags (track-B retag)
and recompute dimension means WITHOUT re-running models. Joins result.json checkpoints to the current
task checkpoint definitions by checkpoint id, then re-aggregates strict (score_eligible!=False) means.
"""
import json, os, sys, glob, collections

MODULES = ["Execution", "Tooling", "Context", "Lifecycle", "Observability", "Verification", "Governance"]
ROOT = "benchmark_dataprocess"


def id_to_dim(bench):
    m = {}
    for l in open(os.path.join(ROOT, bench, "tasks_unified.jsonl")):
        for cp in (json.loads(l).get("checkpoints") or []):
            m[cp.get("id")] = (cp.get("dimension"), cp.get("subdimension"), cp.get("weight", 1.0))
    return m


def recompute(agent_dir, bench):
    idmap = id_to_dim(bench)
    passw = collections.defaultdict(float)
    totw = collections.defaultdict(float)
    for rp in sorted(glob.glob(os.path.join(agent_dir, "*", "result.json"))):
        r = json.load(open(rp))
        for c in (r.get("checkpoints") or []):
            if c.get("score_eligible", False) is not True:   # fail-CLOSED: only explicitly-eligible cps count
                continue
            dim = idmap.get(c.get("id"), (c.get("dimension"), None, 1.0))[0]
            w = idmap.get(c.get("id"), (None, None, 1.0))[2]
            st = c.get("checkpoint_status")
            if st in ("passed", "failed"):
                totw[dim] += w
                if st == "passed":
                    passw[dim] += w
    return {m: (round(passw[m] / totw[m], 3) if totw[m] else None) for m in MODULES}


if __name__ == "__main__":
    for agent_dir, bench in [("results_pbC/gpt5", "PhysicianBench"), ("results_mctaC/gpt5", "MedCTA")]:
        print("==", bench, "(remapped to track-B tags) ==")
        d = recompute(agent_dir, bench)
        for m in MODULES:
            print("  %-14s %s" % (m, d[m] if d[m] is not None else "— 无题"))
