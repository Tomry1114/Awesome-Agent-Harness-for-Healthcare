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
from scoring import is_score_eligible
from scoring import compute_dim_status
from scoring import aggregate_dimension
try:
    from proxy_verifiers import proxy_dimensions, average_proxy
except Exception:
    proxy_dimensions = average_proxy = None

MODULES = ["Execution", "Tooling", "Context", "Lifecycle", "Observability", "Verification", "Governance"]


def _bundle_path(rp):
    """Codex #14: prefer the rescored layer (post-hoc judged: Governance etc.) over the IMMUTABLE raw
    result.json. raw stays untouched on disk; the report reflects the rescored view when present."""
    import os as _os
    rescored = _os.path.join(_os.path.dirname(rp), "result.rescored.json")
    return rescored if _os.path.exists(rescored) else rp
_ROOT = "benchmark_dataprocess"
CATEGORIES = {
    "task_competence": ["Execution", "Tooling", "Context", "Lifecycle"],   # 事做没做对
    "trustworthiness": ["Observability", "Verification", "Governance"],     # 能不能信任它
}


def _load(agent_dir):
    out = []
    for rp in sorted(glob.glob(os.path.join(agent_dir, "*", "result.json"))):
        try:
            out.append(json.load(open(_bundle_path(rp))))
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
        for c in (r.get("checkpoints") or []):
            if c.get("id") in idmap:
                c["dimension"], c["subdimension"], _w = idmap[c["id"]]
                c["weight"] = _w                      # carry task weight so aggregate_dimension is exact
        # Codex #1: report layer uses the SAME aggregate_dimension as raw/rescore — no second math.
        _dims = {m: aggregate_dimension([c for c in (r.get("checkpoints") or []) if c.get("dimension") == m]) for m in MODULES}
        r["dimension_scores"] = {m: _dims[m]["score_mean"] for m in MODULES}
        r["dimension_pass_rate"] = {m: _dims[m]["pass_rate"] for m in MODULES}
        r["dimension_stats"] = _dims
        _st, _rsn = compute_dim_status(r.get("checkpoints") or [], r["dimension_scores"], r.get("proxy_dimension_scores") or {})
        r["dimension_status"] = _st
        r["dimension_status_reason"] = _rsn
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
    import statistics as _st
    acc = {m: [] for m in MODULES}; prate = {m: [] for m in MODULES}
    for r in results:
        ds = r.get("dimension_scores") or {}; pr = r.get("dimension_pass_rate") or {}
        for m in MODULES:
            if ds.get(m) is not None: acc[m].append(ds[m])
            if pr.get(m) is not None: prate[m].append(pr[m])
    dims = {}
    for m in MODULES:
        v = acc[m]; covered = bool(v)
        # Codex #8 + rollup: distribution stats + tiered eligibility (the two semantics of score_eligible
        # split apart) + informativeness so a saturated dim is not mistaken for a discriminating one.
        std = round(_st.pstdev(v), 3) if len(v) > 1 else (0.0 if v else None)
        dims[m] = {"mean": round(sum(v) / len(v), 3) if v else None,
                   "pass_rate": round(sum(prate[m]) / len(prate[m]), 3) if prate[m] else None,
                   "n_scored": len(v), "n_tasks": len(results), "std": std,
                   "min": round(min(v), 3) if v else None, "max": round(max(v), 3) if v else None,
                   "zero_variance": (len(set(v)) == 1) if v else None,
                   "informativeness": ("saturated" if (v and len(set(v)) == 1) else ("discriminating" if v else "none")),
                   "evidence_tier": "strict" if covered else "not_evaluated",
                   "report_in_primary_profile": True, "formal_analysis_eligible": covered,
                   "status": "covered" if covered else "not_exercised_by_benchmark"}
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
    # per-dim spread so a SATURATED proxy (mean 1.0, var 0) is not mistaken for a discriminating one
    import statistics as _st2
    spread = {}
    for d in allp:
        vs = [t[d]["score"] for t in per_task if isinstance(t.get(d), dict) and isinstance(t[d].get("score"), (int, float))]
        if vs:
            spread[d] = {"std": round(_st2.pstdev(vs), 3) if len(vs) > 1 else 0.0,
                         "min": round(min(vs), 3), "max": round(max(vs), 3),
                         "zero_variance": len(set(vs)) == 1,
                         "informativeness": "saturated" if len(set(vs)) == 1 else "discriminating"}
    gap_only = {d: ({**v, **spread.get(d, {})} if isinstance(v, dict) else v)
                for d, v in allp.items() if d not in strict_covered}
    return {"kind": "trajectory_heuristic_soft", "score_eligible": False,
            "note": "gap-fill only; dims with strict coverage excluded",
            "by_dimension": gap_only}


def _tool_use_quality(results):
    """First-class harness-native Tooling metric (LLM judge, alternative-path tolerant). Reported
    standalone so it is not drowned by a benchmark's deterministic reference-chain checkpoints (which
    wrongly penalize legitimate alternative tool paths). Distinct from tool_execution_hygiene (proxy)."""
    subs = ["relevance", "necessity", "argument", "sequence", "evidence_use"]
    scores, sub_acc, unnec = [], {s: [] for s in subs}, []
    for r in results:
        for c in (r.get("checkpoints") or []):
            if c.get("id") == "cp_tool_use_quality":
                if isinstance(c.get("score"), (int, float)):
                    scores.append(c["score"])
                for s in subs:
                    if isinstance((c.get("subscores") or {}).get(s), (int, float)):
                        sub_acc[s].append(c["subscores"][s])
                if isinstance(c.get("unnecessary"), (int, float)):
                    unnec.append(c["unnecessary"])
    if not scores:
        return None
    return {"mean": round(sum(scores) / len(scores), 3), "n": len(scores),
            "subscore_means": {s: (round(sum(v) / len(v), 2) if v else None) for s, v in sub_acc.items()},
            "unnecessary_mean": round(sum(unnec) / len(unnec), 2) if unnec else None,
            "judge": "llm_judge (gpt-5.5), 0-2 per sub x5 -> [0,1]"}


def _experimental_evaluators(agent_dir, bench):
    """Step (b): deterministic state-machine Execution/Lifecycle (fault-injection validated). These FILL
    the Execution/Lifecycle dimension cells (replacing the coarse/deprecated proxies); tier=experimental
    until human-audited. Uses task policy (required_tool_groups) for required_operation_completion."""
    import statistics as _st
    try:
        import lifecycle_exec as _le
    except Exception:
        return {"note": "lifecycle_exec unavailable"}, {}, {}
    pol = {}
    tf = os.path.join(_ROOT, bench, "tasks_unified.jsonl")
    if os.path.exists(tf):
        for l in open(tf):
            t = json.loads(l); ref = t.get("reference") or {}
            pol[t.get("task_id")] = {"required_tool_groups": ref.get("required_tool_groups"),
                                     "prerequisites": (t.get("policy") or {}).get("required_tool_before_action"),
                                     "lifecycle_policy": t.get("lifecycle_policy")}
    ex_t, lc_t = {}, {}
    _lc_cov, _lc_unreportable = {}, []
    for tp in sorted(glob.glob(os.path.join(agent_dir, "*", "trajectory.jsonl"))):
        tid = os.path.basename(os.path.dirname(tp))
        try:
            evs = [json.loads(l) for l in open(tp) if l.strip()]
        except Exception:
            continue
        caps = None                                              # Review #1: per-task capability manifest
        rp = os.path.join(os.path.dirname(tp), "result.json")
        if os.path.exists(rp):
            try: caps = (json.load(open(rp)).get("provenance") or {}).get("capabilities")
            except Exception: caps = None
        e = _le.execution(evs, capabilities=caps, task_policy=pol.get(tid))
        l = _le.lifecycle(evs, task_policy=pol.get(tid), capabilities=caps)   # Review: Lifecycle ALSO gets capabilities
        if isinstance(e.get("score"), (int, float)): ex_t[tid] = e["score"]
        if isinstance(l.get("score"), (int, float)) and l.get("reportable_score"): lc_t[tid] = l["score"]
        for k, st in (l.get("submetric_status") or {}).items():
            _lc_cov.setdefault(k, {"valid": 0, "total": 0})
            _lc_cov[k]["total"] += 1; _lc_cov[k]["valid"] += 1 if st == "valid" else 0
        if not l.get("reportable_score"): _lc_unreportable.append(tid)
    def _agg(d):
        v = list(d.values())
        return {"mean": round(sum(v) / len(v), 3) if v else None, "n": len(v),
                "std": round(_st.pstdev(v), 3) if len(v) > 1 else (0.0 if v else None),
                "zero_variance": (len(set(v)) == 1) if v else None,
                "informativeness": ("saturated" if (v and len(set(v)) == 1) else ("discriminating" if v else "none")),
                "tier": "experimental_state_machine"}
    _life = _agg(lc_t)
    _life["submetric_coverage"] = {k: "%d/%d" % (v["valid"], v["total"]) for k, v in sorted(_lc_cov.items())}
    _life["n_unreportable_insufficient_coverage"] = len(_lc_unreportable)
    panel = {"tier": "experimental_fault_injection_validated", "deterministic": True,
             "promotion_path": "experimental -> human_audited -> strict",
             "Execution_sm": _agg(ex_t), "Lifecycle_sm": _life}
    return panel, ex_t, lc_t


def build(agent_dir, bench):
    results = _remap(_load(agent_dir), bench)
    hd = _harness_dims(results)
    strict_covered = {m for cat in hd["by_category"].values() for m, v in cat.items()
                      if v["status"] == "covered"}
    proxy = _proxy_dims(agent_dir, strict_covered)
    _exp_panel, _ex_t, _lc_t = _experimental_evaluators(agent_dir, bench)
    # Codex: FILL the Execution/Lifecycle dimension cells with the validated state-machine evaluator
    # (replaces the coarse Execution formula / deprecated info-gathering Lifecycle proxy).
    if isinstance(proxy.get("by_dimension"), dict):
        if _ex_t: proxy["by_dimension"]["Execution"] = _exp_panel["Execution_sm"]
        if _lc_t: proxy["by_dimension"]["Lifecycle"] = _exp_panel["Lifecycle_sm"]
    _integ = _integrity(results)
    _toc = (proxy.get("by_dimension") or {}).pop("trace_observation_coverage", None)   # Codex #7
    if _toc is not None:
        _integ["trace_observation_coverage"] = _toc
    _proxy_filled = sorted((set((proxy.get("by_dimension") or {}).keys()) & set(MODULES)) - strict_covered)
    _hd = hd["by_category"]
    _task_cov = {m: "%d/%d" % (d["n_scored"], d["n_tasks"]) for cat in _hd.values() for m, d in cat.items() if d["n_scored"]}
    coverage_summary = {
        "dimension_breadth": "%d/7 strict" % len(strict_covered), "strict_dimensions": sorted(strict_covered),
        "proxy_filled": "%d/7" % len(_proxy_filled), "proxy_dimensions": _proxy_filled,
        "task_eval_coverage": _task_cov,
        "caveat": ("dimension_breadth = how many dims HAVE an evaluator (NOT that they discriminate). "
                   "Only strict dims (formal_analysis_eligible) enter formal stats; proxy dims are shown "
                   "in the profile (report_in_primary_profile) but score_eligible=False. Check per-dim "
                   "evidence_tier / zero_variance / informativeness before averaging. Never report '7/7' unqualified.")}
    return {
        "source": agent_dir,
        "bench": bench,
        "n_tasks": len(results),
        "coverage_summary": coverage_summary,
        "native_metrics": _native_metrics(bench, results),
        "tool_use_quality": _tool_use_quality(results),
        "harness_dimensions": hd,
        "proxy_dimensions": proxy,
        "experimental_evaluators": _exp_panel,
        "integrity": _integ,
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
