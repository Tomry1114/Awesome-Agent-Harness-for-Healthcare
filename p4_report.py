"""P4 report: MedCTA across harness modes — Native Outcome + 7 dims + the §12 harness metrics.

Only ELIGIBLE runs enter the comparison (evaluated, schema-valid, harness active, no runtime errors), the
four modes must share the SAME task set (else the bundle is flagged invalid), each rate is reported macro
(per-task eligible mean) AND pooled (sum integer numerators / sum denominators — NOT rate×denom), and
agent/judge models + judge independence are read from the artifacts (never asserted).
"""
import json, glob, os
MODES = ["off", "observe", "assist", "enforce"]
MODS = ["Execution", "Tooling", "Context", "Lifecycle", "Observability", "Verification", "Governance"]
HMET = ["wrong_scope_action_rate", "missing_prerequisite_rate",
        "verified_commit_rate", "violated_commit_rate", "unknown_verification_rate", "unverified_commit_rate",
        "executed_violation_rate", "post_commit_failure_count",
        "precondition_repair_rate", "verified_repair_rate", "n_interventions", "escalation_rate"]
# rate -> (integer numerator field, denominator field) for POOLED = sum(num)/sum(den).
POOL = {"wrong_scope_action_rate": ("wrong_scope_count", "wrong_scope_opportunities"),
        "missing_prerequisite_rate": ("missing_prerequisite_count", "missing_prerequisite_opportunities"),
        "verified_commit_rate": ("verified_commit_count", "n_commits"),
        "violated_commit_rate": ("violated_commit_count", "n_commits"),
        "unknown_verification_rate": ("unknown_verification_count", "n_commits"),
        "executed_violation_rate": ("executed_violation_count", "n_proposed_actions"),
        "precondition_repair_rate": ("precondition_repair_count", "repair_opportunities"),
        "verified_repair_rate": ("verified_repair_count", "repair_opportunities")}


def load_report(mode):
    p = "res4_mcta_%s/gpt5/report.json" % mode
    return json.load(open(p)) if os.path.exists(p) else None


def eligible(d, mode):
    """A run enters the comparison only if it actually evaluated, is schema-valid, and (for a harness mode)
    the harness was active with no runtime errors. Returns (ok, exclusion_reason)."""
    if d.get("evaluation_status") not in ("complete", "completed", "evaluated"):
        return False, "not_evaluated"
    sv = d.get("schema_validation")
    if isinstance(sv, dict) and sv.get("valid") is False:
        return False, "schema_invalid"
    if mode != "off":
        h = d.get("harness") or {}
        if h.get("status") != "active":
            return False, "harness_%s" % (h.get("status") or "missing")
        if h.get("runtime_errors"):
            return False, "runtime_error"
        if h.get("requested_mode") and h.get("effective_mode") and h["requested_mode"] != h["effective_mode"]:
            return False, "mode_mismatch"
    return True, None


def harness_metrics(mode):
    macc, mcnt, pnum, pden = {}, {}, {}, {}
    n = 0
    model = judge = None
    task_ids = set()
    excluded = {}
    for f in glob.glob("res4_mcta_%s/gpt5/*/result.json" % mode):
        d = json.load(open(f))
        tid = d.get("task_id") or os.path.basename(os.path.dirname(f))
        ok, why = eligible(d, mode)
        if not ok:
            excluded[why] = excluded.get(why, 0) + 1
            continue
        n += 1; task_ids.add(tid)
        prov = d.get("provenance") or {}
        model = model or d.get("model") or d.get("agent_model") or prov.get("agent_model")
        h = d.get("harness") or {}
        judge = judge or (h.get("audit") or {}).get("runtime_judge_model")
        hm = h.get("metrics") or {}
        for k in HMET:
            v = hm.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                macc[k] = macc.get(k, 0) + v; mcnt[k] = mcnt.get(k, 0) + 1
        for k, (nf, df) in POOL.items():
            num, den = hm.get(nf), hm.get(df)
            if isinstance(num, int) and isinstance(den, int) and den > 0:
                pnum[k] = pnum.get(k, 0) + num; pden[k] = pden.get(k, 0) + den
        vc, uc, nc = hm.get("violated_commit_count"), hm.get("unknown_verification_count"), hm.get("n_commits")
        if all(isinstance(x, int) for x in (vc, uc, nc)) and nc > 0:
            pnum["unverified_commit_rate"] = pnum.get("unverified_commit_rate", 0) + vc + uc
            pden["unverified_commit_rate"] = pden.get("unverified_commit_rate", 0) + nc
    return {"macro": {k: round(macc[k] / mcnt[k], 3) for k in macc},
            "pooled": {k: round(pnum[k] / pden[k], 3) for k in pnum if pden.get(k)},
            "cov": {k: "%d/%d" % (mcnt.get(k, 0), n) for k in HMET},
            "n": n, "model": model, "judge": judge, "task_ids": task_ids, "excluded": excluded}


def dim(r, m):
    return ((r.get("harness_dimensions") or {}).get(m) or {}).get("score")


reps = {m: load_report(m) for m in MODES}
hm = {m: harness_metrics(m) for m in MODES}
_model = next((hm[m]["model"] for m in MODES if hm[m]["model"]), "unknown")
_judge = next((hm[m]["judge"] for m in MODES if hm[m]["judge"]), None)
_indep = ("independent" if (_judge and _model and _judge != _model)
          else ("non_independent" if (_judge and _model and _judge == _model) else "unknown"))
_sets = {m: hm[m]["task_ids"] for m in MODES if hm[m]["n"]}
_paired = len({frozenset(s) for s in _sets.values()}) <= 1 if _sets else False

print("P4  MedCTA  agent=%s · harness_judge=%s (%s) · modes" % (_model, _judge or "none", _indep))
print("eligible n per mode: " + " ".join("%s=%d" % (m, hm[m]["n"]) for m in MODES)
      + "   PAIRED_TASK_SET=%s" % ("yes" if _paired else "NO -> INVALID_BUNDLE"))
for m in MODES:
    if hm[m]["excluded"]:
        print("  excluded[%s]: %s" % (m, hm[m]["excluded"]))
if not _paired:
    print("  WARNING: modes do not share an identical task set — cross-mode comparison is NOT valid.")
print("=" * 92)
print("%-26s %-9s %-9s %-9s %-9s" % ("metric", *MODES))
print("-" * 92)
o = lambda m: (reps[m] or {}).get("outcome", {})
print("%-26s %-9s %-9s %-9s %-9s" % ("OUTCOME (GAcc>=0.5)", *[o(m).get("score") for m in MODES]))
print("-" * 92)
for d in MODS:
    print("%-26s %-9s %-9s %-9s %-9s" % (d, *[dim(reps[m], d) if reps[m] else None for m in MODES]))
print("-" * 92)
print("HARNESS metrics — macro(eligible mean) | pooled(sumNum/sumDen) [coverage]:")
for k in HMET:
    cells = []
    for m in MODES:
        cells.append("%s=%s|%s[%s]" % (m, hm[m]["macro"].get(k),
                                       hm[m]["pooled"].get(k) if k in POOL or k == "unverified_commit_rate" else "-",
                                       hm[m]["cov"].get(k)))
    print("  %-24s %s" % (k, " ".join(cells)))
print("-" * 92)
print("Reading: pooled = sum integer numerators / sum denominators (exact, not rate×denom). executed_")
print("violation_rate should drop enforce vs observe. A non-paired/degraded bundle is flagged, not averaged.")
