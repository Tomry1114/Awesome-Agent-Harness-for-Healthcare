#!/usr/bin/env python3
"""ONE-SHOT scoring entry point: post-hoc Clinical Conduct Boundaries rescore + report aggregation in a single command.

It runs the two post-processing steps that must happen IN ORDER:
    1) rescore_judges.py <dir> --bench B   (post-hoc judge -> writes result.rescored.json[Clinical Conduct Boundaries])
    2) aggregate_report.py <dir> --bench B  (PURE-READ -> report.json with the 7-dim harness_dimensions)
Running them out of order, or skipping step 1, silently yields a 6/7 report with Clinical Conduct Boundaries=None.
This wrapper always runs both in order and LOUDLY reports if the full 7/7 was not achieved, and why.

    python runner/scorers/score.py <agent_dir> [--bench PhysicianBench|MedCTA|HealthAdminBench]
                                       [--judge-model MODEL] [--no-rescore] [--out PATH]

Design invariant: aggregate_report.build stays PURE-READ (no model call; conformance-enforced).
The judge/model call lives ONLY in rescore_judges.rescore, invoked here FIRST. This module is the
orchestrator that is allowed to call both.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                 # runner/scorers/ (sibling scoring modules)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # runner/ (substrate, harness, gateway)
import rescore_judges
import aggregate_report

MODULES = ["Care Action Fulfillment", "Clinical Resource Competence", "Patient–Evidence Alignment", "Clinical Workflow Continuity", "Clinical Auditability", "Closed-Loop Verification", "Clinical Conduct Boundaries"]


def score(agent_dir, bench=None, judge_model=None, do_rescore=True, out=None):
    """Run the full scoring pipeline. Returns (bench, rescore_info_or_None, report_dict, out_path)."""
    bench = bench or rescore_judges._guess_bench(agent_dir)
    judge_model = judge_model or rescore_judges.DEFAULT_JUDGE
    rescore_info = rescore_judges.rescore(agent_dir, bench, judge_model=judge_model) if do_rescore else None
    rep = aggregate_report.build(agent_dir, bench)   # PURE-READ: reads the rescored Clinical Conduct Boundaries block
    out = out or os.path.join(agent_dir, "report.json")
    json.dump(rep, open(out, "w"), indent=1, ensure_ascii=False)
    return bench, rescore_info, rep, out


def dim_scores(rep):
    """The headline 7-dim scores from harness_dimensions (== harness_seven)."""
    hd = rep.get("harness_dimensions") or {}
    return {m: (hd[m].get("score") if isinstance(hd.get(m), dict) else None) for m in MODULES}


def _missing_dim_reasons(rep):
    """For a None dimension, surface WHY (adapter_admission + per-task governance evaluation_error)."""
    out = {}
    hd = rep.get("harness_dimensions") or {}
    for m in MODULES:
        cell = hd.get(m) or {}
        if isinstance(cell.get("score"), (int, float)):
            continue
        adm = cell.get("adapter_admission")
        reasons = set()
        if m == "Clinical Conduct Boundaries":
            for _t, a in (rep.get("governance_audit") or {}).items():
                e = (a or {}).get("evaluation_error")
                if e:
                    reasons.add(e)
        out[m] = {"adapter_admission": adm, "reasons": sorted(reasons)}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="One-shot scoring: rescore Clinical Conduct Boundaries + aggregate report -> full 7/7.")
    ap.add_argument("agent_dir")
    ap.add_argument("--bench", default=None)
    ap.add_argument("--judge-model", default=None, help="post-hoc Clinical Conduct Boundaries judge (default MH_JUDGE_MODEL or gpt-5.4)")
    ap.add_argument("--no-rescore", action="store_true", help="skip the post-hoc Clinical Conduct Boundaries rescore (yields 6/7)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    bench, rinfo, rep, out = score(a.agent_dir, a.bench, a.judge_model,
                                   do_rescore=not a.no_rescore, out=a.out)
    scores = dim_scores(rep)
    n = sum(1 for m in MODULES if isinstance(scores[m], (int, float)))
    judge = a.judge_model or rescore_judges.DEFAULT_JUDGE

    print("bench=%s  ->  %s" % (bench, out))
    if rinfo is not None:
        print("rescore(judge=%s): wrote Clinical Conduct Boundaries into %d bundle(s); cache_hits=%d judge_fail=%d refused=%d"
              % (judge, rinfo["written"], rinfo["cache_hits"], rinfo["judge_fail"], rinfo["refused_non_independent"]))
    else:
        print("rescore: SKIPPED (--no-rescore)")
    outc = rep.get("outcome") or {}
    print("outcome (dataset-native, independent axis): score=%s metric=%s" % (outc.get("score"), outc.get("metric")))
    print("%-15s %s" % ("DIMENSION", "SCORE"))
    for m in MODULES:
        print("  %-15s %s" % (m, scores[m]))
    print("=> %d/7 dimensions scored" % n)

    if n < 7:
        gaps = _missing_dim_reasons(rep)
        missing = list(gaps.keys())
        print("\n!! INCOMPLETE: %d/7 -- missing %s" % (n, ", ".join(missing)), file=sys.stderr)
        fixable = False
        for m, g in gaps.items():
            print("   %-13s adapter_admission=%r reasons=%s" % (m, g["adapter_admission"], g["reasons"]),
                  file=sys.stderr)
            if m == "Clinical Conduct Boundaries" and (a.no_rescore or any(r in ("not_rescored", "score_null") for r in g["reasons"])):
                fixable = True
        if fixable:
            if a.no_rescore:
                print("   -> Clinical Conduct Boundaries is missing because you passed --no-rescore. Re-run WITHOUT it.", file=sys.stderr)
            else:
                print("   -> Rescore ran but produced no Clinical Conduct Boundaries block. Check the judge model is reachable "
                      "(--judge-model / MH_JUDGE_MODEL=%s) and the gateway is up." % judge, file=sys.stderr)
            sys.exit(2)   # Clinical Conduct Boundaries missing for a fixable reason (--no-rescore or rescore failure) -> non-zero exit, never silent
        # else: the dataset legitimately produced no evidence for this dim on this run -> honest 6/7, exit 0.
