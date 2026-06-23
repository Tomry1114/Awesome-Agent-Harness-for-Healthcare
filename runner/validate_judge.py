#!/usr/bin/env python3
"""Automatable parts of tool_use_quality judge validation (human inter-rater is Rui's part):
  (a) repeat-stability: re-judge each task and compare to the stored score (mean |Δ|, % within 0.2);
  (b) bias check: per-task correlation of tool_execution_hygiene (all-calls-succeeded proxy) vs
      tool_use_quality — if quality just tracks hygiene, the judge adds nothing / is success-biased.
No bundles are mutated (2nd score kept in memory only)."""
import json, os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tool_use_judge import _gateway, _parse, _evidence, _SYS, SUBS
from proxy_verifiers import proxy_dimensions


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return round(num / (dx * dy), 2) if dx and dy else None


def run(agent_dir):
    stored, repeat, hyg = [], [], []
    for rp in sorted(glob.glob(os.path.join(agent_dir, "*", "result.json"))):
        bdir = os.path.dirname(rp)
        r = json.load(open(rp))
        s = next((c.get("score") for c in (r.get("checkpoints") or []) if c.get("id") == "cp_tool_use_quality"), None)
        if s is None:
            continue
        v = _parse(_gateway(_SYS, _evidence(bdir)))
        if not v:
            continue
        s2 = sum(float(v.get(k, 0)) for k in SUBS) / (len(SUBS) * 2.0)
        stored.append(s); repeat.append(round(s2, 3))
        tp = os.path.join(bdir, "trajectory.jsonl")
        evs = [json.loads(l) for l in open(tp)] if os.path.exists(tp) else []
        h = proxy_dimensions(evs).get("tool_execution_hygiene", {}).get("score")
        hyg.append(h if h is not None else 0.0)
        print("  %-30s stored=%.2f repeat=%.2f hygiene=%.2f" % (os.path.basename(bdir), s, s2, hyg[-1]))
    n = len(stored)
    if n:
        diffs = [abs(a - b) for a, b in zip(stored, repeat)]
        within = sum(1 for d in diffs if d <= 0.2)
        print("\n(a) repeat-stability: mean|Δ|=%.3f, within0.2=%d/%d, corr=%s" % (
            sum(diffs) / n, within, n, _pearson(stored, repeat)))
        print("(b) bias check: corr(hygiene, quality)=%s  (low/neg => judge NOT just rewarding all-success)" % (
            _pearson(hyg, stored)))


if __name__ == "__main__":
    run(sys.argv[1])
