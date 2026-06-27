"""P4 report: MedCTA across harness modes — Native Outcome + 7 dims + the §12 harness metrics."""
import json, glob, os
MODES = ["off", "observe", "assist", "enforce"]
MODS = ["Execution", "Tooling", "Context", "Lifecycle", "Observability", "Verification", "Governance"]
HMET = ["wrong_scope_action_rate", "missing_prerequisite_rate",
        "verified_commit_rate", "violated_commit_rate", "unknown_verification_rate", "unverified_commit_rate",
        "executed_violation_rate", "post_commit_failure_count",
        "precondition_repair_rate", "verified_repair_rate", "n_interventions", "escalation_rate"]


def load_report(mode):
    p = "res4_mcta_%s/gpt5/report.json" % mode
    return json.load(open(p)) if os.path.exists(p) else None


def harness_metrics(mode):
    """ELIGIBLE-OPPORTUNITY mean of per-task harness.metrics: each metric is averaged ONLY over the tasks
    where it is DEFINED (not None), divided by its OWN count — never by the total task count. A metric that
    is None for half the tasks (its opportunity set was empty there) must not be diluted toward 0."""
    acc, cnt = {}, {}
    n = 0
    for f in glob.glob("res4_mcta_%s/gpt5/*/result.json" % mode):
        h = (json.load(open(f)).get("harness") or {}).get("metrics") or {}
        if not h:
            continue
        n += 1
        for k in HMET:
            v = h.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                acc[k] = acc.get(k, 0) + v
                cnt[k] = cnt.get(k, 0) + 1
    means = {k: round(acc[k] / cnt[k], 3) for k in acc}       # per-metric eligible mean
    cov = {k: "%d/%d" % (cnt.get(k, 0), n) for k in HMET}     # coverage (how many tasks defined it)
    return means, n, cov


def dim(r, m):
    return ((r.get("harness_dimensions") or {}).get(m) or {}).get("score")


print("P4  MedCTA  n=10  gpt-5.5 agent · gpt-5.4 harness judge (independent) · modes")
print("=" * 78)
reps = {m: load_report(m) for m in MODES}
hm = {m: harness_metrics(m) for m in MODES}
print("%-26s %-9s %-9s %-9s %-9s" % ("metric", *MODES))
print("-" * 78)
o = lambda m: (reps[m] or {}).get("outcome", {})
print("%-26s %-9s %-9s %-9s %-9s" % ("OUTCOME (GAcc>=0.5)",
      *[o(m).get("score") for m in MODES]))
print("%-26s %-9s %-9s %-9s %-9s" % ("  eval coverage",
      *[o(m).get("native_evaluation_coverage") for m in MODES]))
print("-" * 78)
for d in MODS:
    print("%-26s %-9s %-9s %-9s %-9s" % (d, *[dim(reps[m], d) if reps[m] else None for m in MODES]))
print("-" * 78)
print("HARNESS metrics (eligible-opportunity mean  [coverage = tasks where defined]):")
for k in HMET:
    print("  %-24s %s" % (k, " ".join("%s=%s[%s]" % (m, hm[m][0].get(k), hm[m][2].get(k)) for m in MODES)))
print("-" * 78)
print("interventions tasks-with-harness: " + " ".join("%s=%s" % (m, hm[m][1]) for m in MODES))
print("\nReading: off=Base (no harness). observe records what the harness WOULD do (effective ALLOW).")
print("assist=feedback only. enforce=full gates. Each metric is averaged over its OWN eligible tasks")
print("(coverage shown in []); over_block_rate is None (needs a held-out legality oracle).")
