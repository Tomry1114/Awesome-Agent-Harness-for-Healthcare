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
print("HEAD=%s" % HEAD)
allrep = {}
for d, bench, model in BUNDLES:
    rep = A.build(os.path.join(d, "gpt5"), bench)
    allrep[(d, bench, model)] = rep
    gc = rep["governance_consistency"]
    cs = rep["coverage_summary"]
    # per-task code_sha == HEAD check + artifact_status
    aud = rep["governance_audit"]
    shas = sorted({a.get("code_sha") for a in aud.values() if a.get("code_sha")})
    all_head = bool(shas) and shas == [HEAD]
    out = rep["outcome"]
    print("\n===== %s | %s | %s =====" % (d, bench, model))
    print("artifact_status=%s code_sha_matches_head=%s disk_equals_report=%s metadata_agrees=%s"
          % (gc["artifact_status"], gc["code_sha_matches_head"], gc["disk_equals_report"], gc["metadata_agrees"]))
    print("disk_code_shas=%s  (==HEAD? %s)  n_disk_reportable=%s" % (gc["disk_code_shas"], all_head, gc["n_disk_reportable"]))
    print("dirty_worktree_any=%s" % gc["dirty_worktree_any"])
    print("strict_dimensions=%s  formal_coverage=%s" % (cs["strict_dimensions"], cs["formal_coverage"]))
    print("numeric_coverage=%s reportable_coverage=%s" % (cs["numeric_coverage"], cs["reportable_coverage"]))
    # n_scored uniform across 7 dims
    hs = rep["harness_dimensions"]
    ns = {m: hs[m]["n_scored"] for m in DIMS}
    print("n_scored_per_dim=%s  qualified=%s" % (ns, rep["qualified_profile"]["n_qualified"]))
    print("outcome.score=%s metric=%s checkpoint_pass_rate=%s"
          % (out.get("score"), out.get("metric"),
             (out.get("outcome_diagnostics") or {}).get("checkpoint_pass_rate")))
    # eval errors
    ee = rep.get("evaluator_errors_by_dimension") or []
    print("evaluator_errors=%d" % len(ee))

# dump full reports for table building
json.dump({("%s|%s|%s" % k): v for k, v in allrep.items()},
          open("/tmp/agg_all.json", "w"))
print("\nDUMPED")
