"""Harness vs no-harness comparison across 3 datasets (off vs enforce). Reports native OUTCOME, the FULL
native_metrics set, all 7 process dimensions, and the harness intervention metrics (from per-task
result.json harness.metrics). Pure read of artifacts. MCTA enforce reads res5b (evidence-truncation fix)."""
import json, glob, os
# (stem_off, stem_enforce) — MCTA enforce uses the fixed re-run res5b.
import sys as _sys
_P = _sys.argv[1] if len(_sys.argv) > 1 else "res6"   # run prefix (argv) -> never read a stale hardcoded dir
DS = [("PhysicianBench", "%s_pb_off" % _P, "%s_pb_enforce" % _P),
      ("MedCTA", "%s_mcta_off" % _P, "%s_mcta_enforce" % _P),
      ("HealthAdminBench", "%s_hab_off" % _P, "%s_hab_enforce" % _P)]
MODS = ["Execution", "Tooling", "Context", "Lifecycle", "Observability", "Verification", "Governance"]
HM = ["n_proposed_actions", "n_commits", "n_interventions", "n_findings", "escalation_rate",
      "wrong_scope_action_rate", "missing_prerequisite_rate", "unverified_commit_rate",
      "verified_commit_rate", "violated_commit_rate", "unknown_verification_rate",
      "executed_violation_rate", "post_commit_failure_count", "over_block_rate",
      "repair_success_rate", "unsafe_commitment_rate", "outcome_preservation", "answer_delivered",
      "over_block_proxy_count"]


def rep(d): return json.load(open(d + "/gpt5/report.json")) if os.path.exists(d + "/gpt5/report.json") else None
def dim(r, m): return ((r.get("harness_dimensions") or {}).get(m) or {}).get("score") if r else None
def outcome(r): return (r.get("outcome") or {}).get("score") if r else None
def native(r): return (r.get("native_metrics") or {}) if r else {}


def harness_agg(stem):
    acc, cnt, n, model, judge, ojudge, oj_ind = {}, {}, 0, None, None, None, None
    for f in glob.glob("%s/gpt5/*/result.json" % stem):
        d = json.load(open(f)); h = d.get("harness") or {}; m = h.get("metrics") or {}
        prov = d.get("provenance") or {}
        model = model or prov.get("agent_model") or d.get("agent_model")
        _oj = (prov.get("judges") or {}).get("outcome") or {}
        ojudge = ojudge or _oj.get("model"); oj_ind = oj_ind or _oj.get("independence")
        if not m:
            continue
        n += 1
        judge = judge or (h.get("audit") or {}).get("runtime_judge_model")
        for k, v in m.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                acc[k] = acc.get(k, 0) + v; cnt[k] = cnt.get(k, 0) + 1
    agg = {k: (round(acc[k] / cnt[k], 3) if "rate" in k else acc[k]) for k in acc}
    return agg, n, model, judge, ojudge, oj_ind


def f(v):
    if v is None: return "  -  "
    return ("%.3f" % v) if isinstance(v, float) else str(v)


def delta(a, b):
    return ("%+.3f" % (b - a)) if (isinstance(a, (int, float)) and isinstance(b, (int, float))) else "-"


for bench, soff, senf in DS:
    ro, re_ = rep(soff), rep(senf)
    ha, hn, hmodel, hjudge, h_oj, h_oj_ind = harness_agg(senf)
    _, on, omodel, _, o_oj, o_oj_ind = harness_agg(soff)
    print("=" * 80)
    _ojm = h_oj or o_oj or "none"
    _ojflag = ("  [!! OUTCOME JUDGE DID NOT RUN -> native outcome invalid]" if _ojm == "offline_whitelist_proxy"
               else ("  [!! non-independent outcome judge]"
                     if (o_oj_ind == "shared_model_with_agent_or_tool" or h_oj_ind == "shared_model_with_agent_or_tool")
                     else ""))
    print("%-18s agent=%s | harness_judge=%s | outcome_judge=%s (indep=%s)%s | bundles off=%d enf=%d (harness-active=%d)"
          % (bench, hmodel or omodel or "?", hjudge or "none", _ojm, h_oj_ind or o_oj_ind or "n/a", _ojflag,
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
