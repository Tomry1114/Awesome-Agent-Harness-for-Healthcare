"""P4 report: MedCTA across harness modes — Native Outcome + 7 dims + the §12 harness metrics.

Reports each harness rate TWO ways: the macro (per-task eligible) mean AND the pooled
(sum-numerator / sum-denominator) rate, with coverage. Macro and pooled diverge when tasks have very
different opportunity counts (e.g. 1/1 vs 0/100) — both are shown so neither is mistaken for the other.
Model / n are read from the artifacts, not hard-coded.
"""
import json, glob, os
MODES = ["off", "observe", "assist", "enforce"]
MODS = ["Execution", "Tooling", "Context", "Lifecycle", "Observability", "Verification", "Governance"]
HMET = ["wrong_scope_action_rate", "missing_prerequisite_rate",
        "verified_commit_rate", "violated_commit_rate", "unknown_verification_rate", "unverified_commit_rate",
        "executed_violation_rate", "post_commit_failure_count",
        "precondition_repair_rate", "verified_repair_rate", "n_interventions", "escalation_rate"]
# rate metric -> the per-task denominator field used to POOL it (sum numerators / sum denominators).
POOL_DENOM = {"wrong_scope_action_rate": "wrong_scope_opportunities",
              "missing_prerequisite_rate": "missing_prerequisite_opportunities",
              "precondition_repair_rate": "repair_opportunities", "verified_repair_rate": "repair_opportunities",
              "executed_violation_rate": "n_proposed_actions", "escalation_rate": "n_proposed_actions",
              "verified_commit_rate": "n_commits", "violated_commit_rate": "n_commits",
              "unknown_verification_rate": "n_commits", "unverified_commit_rate": "n_commits"}


def load_report(mode):
    p = "res4_mcta_%s/gpt5/report.json" % mode
    return json.load(open(p)) if os.path.exists(p) else None


def harness_metrics(mode):
    acc, cnt, pnum, pden = {}, {}, {}, {}
    n = 0
    model = None
    for f in glob.glob("res4_mcta_%s/gpt5/*/result.json" % mode):
        d = json.load(open(f))
        h = (d.get("harness") or {}).get("metrics") or {}
        if not h:
            continue
        n += 1
        model = model or d.get("model") or d.get("agent_model") or (d.get("metadata") or {}).get("model")
        for k in HMET:
            v = h.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                acc[k] = acc.get(k, 0) + v
                cnt[k] = cnt.get(k, 0) + 1
                dk = POOL_DENOM.get(k)
                den = h.get(dk) if dk else None
                if isinstance(den, int) and den > 0:
                    pnum[k] = pnum.get(k, 0) + v * den
                    pden[k] = pden.get(k, 0) + den
    means = {k: round(acc[k] / cnt[k], 3) for k in acc}
    pooled = {k: round(pnum[k] / pden[k], 3) for k in pnum if pden.get(k)}
    cov = {k: "%d/%d" % (cnt.get(k, 0), n) for k in HMET}
    return {"macro": means, "pooled": pooled, "cov": cov, "n": n, "model": model}


def dim(r, m):
    return ((r.get("harness_dimensions") or {}).get(m) or {}).get("score")


reps = {m: load_report(m) for m in MODES}
hm = {m: harness_metrics(m) for m in MODES}
_n = max((hm[m]["n"] for m in MODES), default=0)
_model = next((hm[m]["model"] for m in MODES if hm[m]["model"]), "unknown")
print("P4  MedCTA  n=%d  agent=%s · harness judge independent · modes" % (_n, _model))
print("=" * 86)
print("%-26s %-9s %-9s %-9s %-9s" % ("metric", *MODES))
print("-" * 86)
o = lambda m: (reps[m] or {}).get("outcome", {})
print("%-26s %-9s %-9s %-9s %-9s" % ("OUTCOME (GAcc>=0.5)", *[o(m).get("score") for m in MODES]))
print("%-26s %-9s %-9s %-9s %-9s" % ("  eval coverage", *[o(m).get("native_evaluation_coverage") for m in MODES]))
print("-" * 86)
for d in MODS:
    print("%-26s %-9s %-9s %-9s %-9s" % (d, *[dim(reps[m], d) if reps[m] else None for m in MODES]))
print("-" * 86)
print("HARNESS metrics — macro(eligible-task mean) | pooled(sumNum/sumDen) [coverage]:")
for k in HMET:
    cells = []
    for m in MODES:
        macro = hm[m]["macro"].get(k)
        pooled = hm[m]["pooled"].get(k)
        cov = hm[m]["cov"].get(k)
        cells.append("%s=%s|%s[%s]" % (m, macro, pooled if k in POOL_DENOM else "-", cov))
    print("  %-24s %s" % (k, " ".join(cells)))
print("-" * 86)
print("tasks-with-harness: " + " ".join("%s=%s" % (m, hm[m]["n"]) for m in MODES))
print("\nReading: off=Base (no harness). observe records what the harness WOULD do (effective ALLOW).")
print("assist=feedback only. enforce=full gates. macro = mean over a metric's OWN eligible tasks; pooled =")
print("sum numerators / sum denominators (they differ when opportunity counts are uneven). executed_")
print("violation_rate should drop enforce vs observe; over_block_rate is None (needs a held-out oracle).")
