#!/usr/bin/env python3
"""Post-hoc report aggregator (non-destructive).

Reads an existing results_<x>/<agent>/ directory of per-task result.json bundles and emits an
ENHANCED report that the per-run summary.json does not yet carry:

  1. native_metrics   - benchmark-native scores reported ALONGSIDE harness dims
                        PB     -> Pass@1 (all-checkpoints-pass tasks / n) + checkpoint pass rate
                        MedCTA -> GAcc mean (mean of gacc_judge checkpoint score, 0-1)
                        HAB    -> task success rate + subtask(checkpoint) pass rate
  2. harness_dimensions - 7-dim ETCLOVG grouped into TWO categories, with HONEST coverage:
                        each dim is covered / not_exercised_by_benchmark (coverage=0 != failure)
  3. integrity        - provenance + qualification aggregation (judge independence, backends...)
  4. failure_taxonomy - checkpoint failure_mode histogram + per-task failure_tags

Does NOT re-run any model. Pure read over existing bundles. Usage:
  python runner/aggregate_report.py <results_dir/agent> [--bench PhysicianBench|MedCTA|HealthAdminBench]
"""
import json, os, sys, glob, collections, argparse
try:
    from proxy_verifiers import proxy_dimensions, average_proxy
except Exception:
    proxy_dimensions = average_proxy = None

MODULES = ["Execution", "Tooling", "Context", "Lifecycle", "Observability", "Verification", "Governance"]
_ROOT = "benchmark_dataprocess"
CATEGORIES = {
    "task_competence": ["Execution", "Tooling", "Context", "Lifecycle"],   # 事做没做对
    "trustworthiness": ["Observability", "Verification", "Governance"],     # 能不能信任它
}


def _load(agent_dir):
    out = []
    for rp in sorted(glob.glob(os.path.join(agent_dir, "*", "result.json"))):
        try:
            out.append(json.load(open(rp)))
        except Exception as e:
            sys.stderr.write("skip %s: %r\n" % (rp, e))
    return out


def _remap(results, bench):
    """Re-map each result checkpoint's dimension/subdimension to the CURRENT tasks_unified tags
    (by checkpoint id), so reports on pre-retag runs reflect the current taxonomy WITHOUT a model
    re-run. In-memory only. Silently no-ops if the task file is unavailable."""
    tf = os.path.join(_ROOT, bench, "tasks_unified.jsonl")
    if not os.path.exists(tf):
        return results
    idmap = {}
    for l in open(tf):
        for cp in (json.loads(l).get("checkpoints") or []):
            idmap[cp.get("id")] = (cp.get("dimension"), cp.get("subdimension"), cp.get("weight", 1.0))
    for r in results:
        passw, totw = collections.defaultdict(float), collections.defaultdict(float)
        for c in (r.get("checkpoints") or []):
            if c.get("id") in idmap:
                c["dimension"], c["subdimension"], _ = idmap[c["id"]]
            if c.get("score_eligible") is False:
                continue
            w = idmap.get(c.get("id"), (None, None, 1.0))[2]
            st = c.get("checkpoint_status")
            if st in ("passed", "failed"):
                totw[c["dimension"]] += w
                if st == "passed":
                    passw[c["dimension"]] += w
        # recompute dimension_scores from remapped checkpoints (old precomputed dict used stale tags)
        r["dimension_scores"] = {m: (round(passw[m] / totw[m], 3) if totw[m] else None) for m in MODULES}
    return results


def _native_metrics(bench, results):
    n = len(results)
    cps = [c for r in results for c in (r.get("checkpoints") or [])]
    cp_pass = sum(1 for c in cps if c.get("checkpoint_status") == "passed")
    cp_total = sum(1 for c in cps if c.get("checkpoint_status") in ("passed", "failed"))
    base = {"n_tasks": n,
            "checkpoint_pass_rate": round(cp_pass / cp_total, 3) if cp_total else None,
            "checkpoint_passed": cp_pass, "checkpoint_scored": cp_total}
    if bench == "PhysicianBench":
        passed = sum(1 for r in results if r.get("success") and r.get("evaluation_status") == "complete")
        base["pass_at_1"] = round(passed / n, 3) if n else None
        base["pass_at_1_tasks"] = "%d/%d" % (passed, n)
    elif bench == "MedCTA":
        scores = [c["score"] for c in cps if c.get("evaluator_kind") == "gacc_judge" and isinstance(c.get("score"), (int, float))]
        base["gacc_mean"] = round(sum(scores) / len(scores), 3) if scores else None
        base["gacc_n"] = len(scores)
    elif bench == "HealthAdminBench":
        task_ok = sum(1 for r in results if r.get("success"))
        base["task_success_rate"] = round(task_ok / n, 3) if n else None
        base["subtask_pass_rate"] = base["checkpoint_pass_rate"]
    return base


def _harness_dims(results):
    acc = {m: [] for m in MODULES}
    for r in results:
        ds = r.get("dimension_scores") or {}
        for m in MODULES:
            if ds.get(m) is not None:
                acc[m].append(ds[m])
    dims = {}
    for m in MODULES:
        v = acc[m]
        dims[m] = {"mean": round(sum(v) / len(v), 3) if v else None,
                   "coverage_tasks": len(v),
                   "status": "covered" if v else "not_exercised_by_benchmark"}
    cats = {cat: {m: dims[m] for m in members} for cat, members in CATEGORIES.items()}
    uncovered = [m for m in MODULES if dims[m]["status"] != "covered"]
    return {"by_category": cats, "uncovered_dimensions": uncovered}


def _integrity(results):
    judge_indep = collections.Counter()
    judge_models = collections.Counter()
    tool_backends = collections.Counter()
    quals = collections.Counter()
    for r in results:
        pv = r.get("provenance") or {}
        if pv.get("judge_independence"):
            judge_indep[pv["judge_independence"]] += 1
        if pv.get("judge_model"):
            judge_models[pv["judge_model"]] += 1
        tb = pv.get("tool_backend")
        if isinstance(tb, dict):
            for k, val in tb.items():
                tool_backends["%s=%s" % (k, val)] += 1
        elif tb:
            tool_backends[str(tb)] += 1
        for q in (r.get("qualification") or []):
            quals[q] += 1
    return {"judge_independence": dict(judge_indep),
            "judge_models": dict(judge_models),
            "tool_backends": dict(tool_backends),
            "qualifications": dict(quals),
            "tasks_with_any_qualification": sum(1 for r in results if r.get("qualification"))}


def _failure_taxonomy(results):
    fm = collections.Counter()
    tags = collections.Counter()
    by_dim = collections.defaultdict(collections.Counter)
    for r in results:
        for c in (r.get("checkpoints") or []):
            if c.get("checkpoint_status") == "failed":
                mode = c.get("failure_mode") or "unspecified"
                fm[mode] += 1
                by_dim[c.get("dimension") or "?"][mode] += 1
        for t in (r.get("failure_tags") or []):
            tags[t] += 1
    return {"checkpoint_failure_mode": dict(fm),
            "failure_mode_by_dimension": {k: dict(v) for k, v in by_dim.items()},
            "task_failure_tags": dict(tags)}


def _proxy_dims(agent_dir, strict_covered):
    """Trajectory-derived soft signals (score_eligible=False). GAP-FILL ONLY: emitted only for
    dimensions a benchmark does NOT formally test, so proxy never conflicts with / overrides a
    strict score. Honest heuristic; NEVER mixed into harness_dimensions or success."""
    if proxy_dimensions is None:
        return {"note": "proxy_verifiers unavailable"}
    per_task = []
    for tp in sorted(glob.glob(os.path.join(agent_dir, "*", "trajectory.jsonl"))):
        try:
            evs = [json.loads(l) for l in open(tp) if l.strip()]
            per_task.append(proxy_dimensions(evs))
        except Exception as e:
            sys.stderr.write("proxy skip %s: %r\n" % (tp, e))
    allp = average_proxy(per_task)
    gap_only = {d: v for d, v in allp.items() if d not in strict_covered}
    return {"kind": "trajectory_heuristic_soft", "score_eligible": False,
            "note": "gap-fill only; dims with strict coverage excluded",
            "by_dimension": gap_only}


def build(agent_dir, bench):
    results = _remap(_load(agent_dir), bench)
    hd = _harness_dims(results)
    strict_covered = {m for cat in hd["by_category"].values() for m, v in cat.items()
                      if v["status"] == "covered"}
    return {
        "source": agent_dir,
        "bench": bench,
        "n_tasks": len(results),
        "native_metrics": _native_metrics(bench, results),
        "harness_dimensions": hd,
        "proxy_dimensions": _proxy_dims(agent_dir, strict_covered),
        "integrity": _integrity(results),
        "failure_taxonomy": _failure_taxonomy(results),
    }


def _guess_bench(agent_dir, results):
    ids = " ".join(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(agent_dir, "*", "result.json")))
    if "PB-" in ids:
        return "PhysicianBench"
    if "MCTA-" in ids or "MedCTA" in ids:
        return "MedCTA"
    if "HAB-" in ids:
        return "HealthAdminBench"
    return "Unknown"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("agent_dir")
    ap.add_argument("--bench", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    bench = a.bench or _guess_bench(a.agent_dir, None)
    rep = build(a.agent_dir, bench)
    out = a.out or os.path.join(a.agent_dir, "report.json")
    json.dump(rep, open(out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    print("\n-> wrote", out, file=sys.stderr)
