#!/usr/bin/env python3
"""Batch runner: filter tasks, run with stub agent, write a per-task RESULT BUNDLE + summary.json.

Bundle layout (per task, Harness-Bench style):
  results/<agent>/<task_id>/{task.json, trajectory.jsonl, result.json, workspace/output/, summary in summary.json}

Filters (B): --source-benchmark, --has-dimension, --has-subdimension, --governance-only.
Usage: python3 runner/run_batch.py --bench PhysicianBench --governance-only --limit 26 --out results/
"""
import os, sys, json, argparse, collections, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import run as R

MODULES = ["Execution", "Tooling", "Context", "Lifecycle", "Observability", "Verification", "Governance"]

def select_tasks(bench, args):
    p = os.path.join(ROOT, "benchmark_dataprocess", bench, "tasks_unified.jsonl")
    out = []
    for line in open(p):
        t = json.loads(line)
        dims = {c.get("dimension") for c in t.get("checkpoints", [])}
        subs = {c.get("subdimension") for c in t.get("checkpoints", [])}
        if args.has_dimension and args.has_dimension not in dims: continue
        if args.has_subdimension and args.has_subdimension not in subs: continue
        if args.governance_only and "Governance" not in dims: continue
        out.append(t["task_id"])
    return out[:args.limit]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--agent", default="stub")
    ap.add_argument("--fhir-base", default=None)
    ap.add_argument("--reset-mode", default="none", choices=["none", "restore_pristine", "per_task"])
    ap.add_argument("--source-benchmark", default=None)  # reserved (single-bench batch for now)
    ap.add_argument("--has-dimension", default=None)
    ap.add_argument("--has-subdimension", default=None)
    ap.add_argument("--governance-only", action="store_true")
    ap.add_argument("--out", default="results/")
    args = ap.parse_args()

    base_out = os.path.join(ROOT, args.out) if not os.path.isabs(args.out) else args.out
    agent_out = os.path.join(base_out, args.agent)
    os.makedirs(agent_out, exist_ok=True)
    if args.reset_mode == "restore_pristine":
        R.reset_fhir("restore_pristine")

    tids = select_tasks(args.bench, args)
    status_hist = collections.Counter(); tag_hist = collections.Counter()
    buckets = collections.Counter()  # complete_success/partial_success/failed/error/task_error
    dim_acc = {m: [] for m in MODULES}; proxy_dim_acc = {m: [] for m in MODULES}; schema_ok = 0; rows = []
    proxy_cp_total = 0
    tasks_by_id = {json.loads(l)["task_id"]: json.loads(l)
                   for l in open(os.path.join(ROOT, "benchmark_dataprocess", args.bench, "tasks_unified.jsonl"))}

    for tid in tids:
        if args.reset_mode == "per_task": R.reset_fhir("per_task")
        try:
            res = R.run_task(args.bench, tid, args.agent, args.fhir_base)
        except Exception as e:
            buckets["task_error"] += 1; rows.append({"task": tid, "error": repr(e)}); continue
        # --- per-task bundle (A) ---
        bdir = os.path.join(agent_out, tid); os.makedirs(os.path.join(bdir, "workspace"), exist_ok=True)
        json.dump(tasks_by_id.get(tid, {}), open(os.path.join(bdir, "task.json"), "w"), indent=1, ensure_ascii=False)
        with open(os.path.join(bdir, "trajectory.jsonl"), "w") as f:
            for ev in res.get("_trajectory", []): f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        clean = {k: v for k, v in res.items() if not k.startswith("_")}
        json.dump(clean, open(os.path.join(bdir, "result.json"), "w"), indent=1, ensure_ascii=False)
        # #13: full checkpoint raw results (detail/note/judge_backend) for debugging
        vdir = os.path.join(bdir, "verifier_logs"); os.makedirs(vdir, exist_ok=True)
        json.dump(res.get("_checkpoints_full", []), open(os.path.join(vdir, "checkpoints_full.json"), "w"),
                  indent=1, ensure_ascii=False, default=str)
        jd = res.get("_job_dir")
        if jd and os.path.isdir(os.path.join(jd, "workspace", "output")):
            shutil.copytree(os.path.join(jd, "workspace", "output"),
                            os.path.join(bdir, "workspace", "output"), dirs_exist_ok=True)
        # --- aggregate ---
        if res.get("_schema") == "OK": schema_ok += 1
        for c in res["checkpoints"]: status_hist[c["checkpoint_status"]] += 1
        for t in res["failure_tags"]: tag_hist[t] += 1
        for m in MODULES:
            if res["dimension_scores"].get(m) is not None: dim_acc[m].append(res["dimension_scores"][m])
            if res.get("proxy_dimension_scores", {}).get(m) is not None: proxy_dim_acc[m].append(res["proxy_dimension_scores"][m])
        proxy_cp_total += res.get("proxy_evaluated_checkpoints", 0)
        if res["success"]:
            buckets["complete_success" if res["evaluation_status"] == "complete" else "partial_success"] += 1
        elif res["evaluation_status"] == "error" or any(c["checkpoint_status"] == "error" for c in res["checkpoints"]):
            buckets["error"] += 1
        elif res["evaluation_status"] in ("proxy_only", "not_evaluated"):
            buckets[res["evaluation_status"]] += 1
        else:
            buckets["failed"] += 1
        rows.append({"task": tid, "success": res["success"], "evaluation_status": res["evaluation_status"],
                     "dimension_scores": res["dimension_scores"]})

    summary = {"bench": args.bench, "agent": args.agent, "n_tasks": len(tids),
               "filters": {"has_dimension": args.has_dimension, "has_subdimension": args.has_subdimension,
                           "governance_only": args.governance_only},
               "schema_valid": f"{schema_ok}/{len(tids)}",
               "success_buckets": dict(buckets),
               "dimension_means": {m: (round(sum(v) / len(v), 3) if v else None) for m, v in dim_acc.items()},
               "dimension_coverage_tasks": {m: len(v) for m, v in dim_acc.items()},
               "proxy_dimension_means": {m: (round(sum(v) / len(v), 3) if v else None) for m, v in proxy_dim_acc.items()},
               "proxy_evaluated_checkpoints": proxy_cp_total,
               "checkpoint_status_histogram": dict(status_hist),
               "failure_tag_counts": dict(tag_hist), "tasks": rows}
    json.dump(summary, open(os.path.join(agent_out, "summary.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: summary[k] for k in ("bench", "n_tasks", "schema_valid", "success_buckets",
                      "dimension_means", "checkpoint_status_histogram", "failure_tag_counts")},
                     indent=1, ensure_ascii=False))
    print("bundles + summary.json ->", agent_out)

if __name__ == "__main__":
    main()
