"""Harness vs no-harness comparison across 3 datasets (off vs enforce). Reports native OUTCOME, the FULL
native_metrics set, all 7 process dimensions, and the harness intervention metrics (from per-task
result.json harness.metrics). Pure read of artifacts. MCTA enforce reads res5b (evidence-truncation fix)."""
import json, glob, os
# (stem_off, stem_enforce) — MCTA enforce uses the fixed re-run res5b.
DS = [("PhysicianBench", "res5_pb_off", "res5_pb_enforce"),
      ("MedCTA", "res5_mcta_off", "res5b_mcta_enforce"),
      ("HealthAdminBench", "res5_hab_off", "res5_hab_enforce")]
MODS = ["Execution", "Tooling", "Context", "Lifecycle", "Observability", "Verification", "Governance"]
HM = ["n_proposed_actions", "n_commits", "n_interventions", "n_findings", "escalation_rate",
      "wrong_scope_action_rate", "missing_prerequisite_rate", "unverified_commit_rate",
      "verified_commit_rate", "violated_commit_rate", "unknown_verification_rate",
      "executed_violation_rate", "post_commit_failure_count", "over_block_rate"]


def rep(d): return json.load(open(d + "/gpt5/report.json")) if os.path.exists(d + "/gpt5/report.json") else None
def dim(r, m): return ((r.get("harness_dimensions") or {}).get(m) or {}).get("score") if r else None
def outcome(r): return (r.get("outcome") or {}).get("score") if r else None
def native(r): return (r.get("native_metrics") or {}) if r else {}


def harness_agg(stem):
    acc, cnt, n, model, judge = {}, {}, 0, None, None
    for f in glob.glob("%s/gpt5/*/result.json" % stem):
        d = json.load(open(f)); h = d.get("harness") or {}; m = h.get("metrics") or {}
        prov = d.get("provenance") or {}
        model = model or prov.get("agent_model") or d.get("agent_model")
        if not m:
            continue
        n += 1
        judge = judge or (h.get("audit") or {}).get("runtime_judge_model")
        for k, v in m.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                acc[k] = acc.get(k, 0) + v; cnt[k] = cnt.get(k, 0) + 1
    agg = {k: (round(acc[k] / cnt[k], 3) if "rate" in k else acc[k]) for k in acc}
    return agg, n, model, judge


def f(v):
    if v is None: return "  -  "
    return ("%.3f" % v) if isinstance(v, float) else str(v)


def delta(a, b):
    return ("%+.3f" % (b - a)) if (isinstance(a, (int, float)) and isinstance(b, (int, float))) else "-"


for bench, soff, senf in DS:
    ro, re_ = rep(soff), rep(senf)
    ha, hn, hmodel, hjudge = harness_agg(senf)
    _, on, omodel, _ = harness_agg(soff)
    print("=" * 80)
    print("%-18s agent=%s | harness_judge=%s | bundles off=%d enf=%d (harness-active=%d)"
          % (bench, hmodel or omodel or "?", hjudge or "none",
             len(glob.glob(soff + "/gpt5/*/result.json")), len(glob.glob(senf + "/gpt5/*/result.json")), hn))
    print("-" * 80)
    print("  %-28s %9s %9s  %s" % ("", "OFF", "ENFORCE", "Δ"))
    print("  %-28s %9s %9s  %s" % ("NATIVE OUTCOME", f(outcome(ro)), f(outcome(re_)), delta(outcome(ro), outcome(re_))))
    print("  -- native_metrics (full) --")
    no, ne = native(ro), native(re_)
    for k in sorted(set(no) | set(ne)):
        print("  %-28s %9s %9s  %s" % (k, f(no.get(k)), f(ne.get(k)), delta(no.get(k), ne.get(k))))
    print("  -- 7 process dimensions --")
    for m in MODS:
        print("  %-28s %9s %9s  %s" % (m, f(dim(ro, m)), f(dim(re_, m)), delta(dim(ro, m), dim(re_, m))))
    print("  -- harness intervention (enforce) --")
    for k in HM:
        if k in ha:
            print("  %-28s %20s" % (k, f(ha[k])))
print("=" * 80)
