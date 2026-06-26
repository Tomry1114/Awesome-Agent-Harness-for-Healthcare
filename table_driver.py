import json, os, sys
sys.path.insert(0, "runner")
import aggregate_report as A

HEAD = A._current_git_head()
BUNDLES = [
    ("res2_m55_pb", "PhysicianBench", "gpt-5.5"),
    ("res2_m54_pb", "PhysicianBench", "gpt-5.4-mini"),
    ("res2_m55_hab", "HealthAdminBench", "gpt-5.5"),
    ("res2_m54_hab", "HealthAdminBench", "gpt-5.4-mini"),
    ("res2_m55_mcta", "MedCTA", "gpt-5.5"),
    ("res2_m54_mcta", "MedCTA", "gpt-5.4-mini"),
]
DIMS = ["Execution", "Tooling", "Context", "Lifecycle", "Observability", "Verification", "Governance"]

def fmt(x):
    return "NA" if x is None else ("%.3f" % x if isinstance(x, float) else str(x))

# ---- prove zero model calls during build(): arm tripwires (mirror of conformance item f) ----
import gateway as _gw
_orig = _gw.chat
tripped = []
_gw.chat = lambda *a, **k: tripped.append("gateway.chat") or {"ok": False, "content": None}
import urllib.request as _ur
_ou = _ur.urlopen
_ur.urlopen = lambda *a, **k: tripped.append("urlopen")

reps = {}
try:
    for d, bench, model in BUNDLES:
        reps[(d, bench, model)] = A.build(os.path.join(d, "gpt5"), bench)
finally:
    _gw.chat = _orig
    _ur.urlopen = _ou
print("AGGREGATE_ZERO_MODEL_CALLS:", tripped == [], "tripped=%r" % tripped)
print("HEAD:", HEAD)

# ---- disk==report NUMBER agreement (the core check, separate from the dirty-worktree gate) ----
print("\n--- disk number agreement & provenance per bundle ---")
for (d, bench, model), rep in reps.items():
    gc = rep["governance_consistency"]
    aud = rep["governance_audit"]
    shas = sorted({a.get("code_sha") for a in aud.values() if a.get("code_sha")})
    code_ok = bool(shas) and shas == [HEAD]
    num_match = (gc["report_harness_governance"] == gc["disk_reportable_mean"])
    print("%-16s %-16s code_sha==HEAD(all tasks)=%s | report_gov=%s disk_mean=%s number_match=%s | artifact=%s dirty=%s"
          % (d, model, code_ok, fmt(gc["report_harness_governance"]), fmt(gc["disk_reportable_mean"]),
             num_match, gc["artifact_status"], gc["dirty_worktree_any"]))

# ---- HAB in-scope count (subject-scope established) ----
print("\n--- HAB Governance subject-scope (established/in-scope per task) ---")
import scoring as _sc, substrate as _S
for d, bench, model in BUNDLES:
    if bench != "HealthAdminBench":
        continue
    ad = os.path.join(d, "gpt5")
    import glob
    inscope = 0; total = 0; details = []
    for rp in sorted(glob.glob(os.path.join(ad, "*", "result.json"))):
        bdir = os.path.dirname(rp); tid = os.path.basename(bdir)
        tpath = os.path.join(bdir, "task.json")
        task = json.load(open(tpath)) if os.path.exists(tpath) else {"source_benchmark": bench}
        traj = os.path.join(bdir, "trajectory.jsonl")
        evs = [json.loads(l) for l in open(traj) if l.strip()] if os.path.exists(traj) else []
        dp = _S.dimension_policy(task, _S.get_plugin(bench))
        r = _sc.governance_subject_scope(evs, dp, task)
        total += 1
        est = r["scope_boundary"].get("established_assigned")
        if est: inscope += 1
        details.append((tid, est, r["score"], r["violated"]))
    print("%s %s: established_assigned %d/%d" % (d, model, inscope, total))
    for tid, est, sc, v in details:
        print("    %-28s established=%s score=%s violated=%s" % (tid, est, sc, v))

# ---- FINAL TABLE rows ----
print("\n=== FINAL TABLE DATA ===")
rows = []
for (d, bench, model), rep in reps.items():
    out = rep["outcome"]
    diag = (out.get("outcome_diagnostics") or {})
    hs = rep["harness_dimensions"]
    aud = rep["governance_audit"]
    row = {"bench": bench, "model": model,
           "outcome_score": out.get("score"), "outcome_metric": out.get("metric"),
           "checkpoint_pass_rate": diag.get("checkpoint_pass_rate"),
           "dims": {}}
    for m in DIMS:
        c = hs[m]
        row["dims"][m] = {"score": c.get("score"),
                          "numeric_cov": c.get("numeric_coverage"),
                          "reportable_cov": c.get("reportable_coverage"),
                          "tier": c.get("evidence_tier"),
                          "formal": c.get("formal_analysis_eligible")}
    rows.append(row)
json.dump(rows, open("/tmp/table_rows.json", "w"), indent=1)
print("ROWS_DUMPED", len(rows))
