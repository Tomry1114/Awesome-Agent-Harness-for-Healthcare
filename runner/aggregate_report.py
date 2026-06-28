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

SINGLE CANONICAL SCORING PATH (P2 / "two truths" guard): this report stage NEVER re-implements a scoring
formula. It is PURE-READ. Every dimension number it emits is READ from a persisted artifact:
  - Governance -> the per-task SHARED-CONTRACT block in <task>/result.rescored.json (written upstream by
                  rescore_judges.py applying runner/governance_contract.py). This module holds NO blend /
                  critical-veto formula; it may import governance_contract READ-ONLY but computes nothing.
  - benchmark checkpoints (in _remap) -> scoring.aggregate_dimension (weighted continuous, no second math)
  - Execution/Lifecycle/Observability/Context/Verification/Tooling -> dim_* over the substrate
There is exactly one place a number is computed per dimension; the report does not maintain a second math
and makes NO model call.
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

# NOTE: this aggregate holds NO governance blend/critical-veto formula. The single source of truth for the
# governance math (critical_predicate / blend_governance / g14_weight / scoring_config) is
# runner/governance_contract.py; rescore_judges.py applies it and PERSISTS the per-task verdict to
# <task>/result.rescored.json. This stage is PURE-READ: it only reads those persisted blocks (and may import
# governance_contract read-only) -- it never re-derives a blended Governance score and makes no model call.


# ---------------------------------------------------------------- SHARED-CONTRACT GOVERNANCE READER (PURE)
# PURITY (task item 1): this report stage makes NO model call. The judge-backed Governance score + full
# audit are produced upstream by runner/rescore_judges.py and persisted PER TASK to <task>/result.rescored
# .json as a self-describing per-dimension block (the SHARED CANONICAL SCHEMA). The aggregate READS that
# block; it never re-runs governance.governance / a gateway judge. If the rescored file (or its Governance
# block) is missing for a task -> that task's Governance is N/A (reportable=False,
# evaluation_error="not_rescored"), NOT a live judge call.
def _read_dim_block(bundle_dir, dim):
    """Read the per-dimension self-describing block `dim` from <bundle_dir>/result.rescored.json (SHARED
    CONTRACT). Returns (block_or_None, evaluation_error). evaluation_error is set when the contract layer
    is absent so the caller surfaces an HONEST N/A instead of fabricating a number.
      - no result.rescored.json                 -> (None, "not_rescored")
      - rescored file present but no `dim` block -> (None, "dimension_block_missing")
      - block present                           -> (block, block.get("evaluation_error"))"""
    rescored = os.path.join(bundle_dir, "result.rescored.json")
    if not os.path.exists(rescored):
        return None, "not_rescored"
    try:
        doc = json.load(open(rescored))
    except Exception as _e:
        return None, "rescored_unreadable:%s" % type(_e).__name__
    blk = doc.get(dim)
    if not isinstance(blk, dict):
        return None, "dimension_block_missing"
    return blk, blk.get("evaluation_error")


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
    # COMPOSITE KEY (task item 3): key the taxonomy map by (task_id, cp_id), NOT cp_id alone. PB reuses
    # generic checkpoint ids (cp1_data_retrieval / cp2_...) across DIFFERENT tasks with DIFFERENT dimension/
    # weight; a cp_id-only map let the LAST task's tags clobber every earlier task's same-id checkpoint
    # (cross-task taxonomy bleed). (task_id, cp_id) isolates each task's own checkpoint definitions.
    idmap = {}
    for l in open(tf):
        _t = json.loads(l)
        _tid = _t.get("task_id")
        for cp in (_t.get("checkpoints") or []):
            idmap[(_tid, cp.get("id"))] = (cp.get("dimension"), cp.get("subdimension"), cp.get("weight", 1.0))
    for r in results:
        _rtid = r.get("task_id")
        for c in (r.get("checkpoints") or []):
            _k = (_rtid, c.get("id"))
            if _k in idmap:
                c["dimension"], c["subdimension"], _w = idmap[_k]
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


def native_task_outcome(result, bench):
    """SINGLE SOURCE OF TRUTH for dataset-NATIVE task success (the Source Outcome unit). NEVER uses
    result['success'] (that is the harness GATE over all ETCLOVG dims -- a different construct). Every
    native-success consumer (_outcome_metric, _native_metrics, _per_task_reportable_scores, compare_models,
    final table) routes through THIS function so the headline and every secondary field agree.
    Returns:
      True / False -> the task carries a native outcome signal and did / did not pass it
      None         -> the task carries NO native outcome signal (adapter incomplete for this task)
    PB / HAB: ALL of the task's Outcome-dimension checkpoints passed. MedCTA: mean GAcc >= 0.5."""
    cps = result.get("checkpoints") or []
    if bench == "MedCTA":
        gacc = [c for c in cps if c.get("evaluator_kind") == "gacc_judge"]
        if not gacc:
            return None                                   # no GAcc signal at all -> unresolved
        gs = [c.get("score") for c in gacc if isinstance(c.get("score"), (int, float))]
        # any declared GAcc checkpoint that produced NO numeric score -> incomplete evaluation -> unresolved.
        if len(gs) != len(gacc):
            return None
        return (sum(gs) / len(gs)) >= 0.5
    # ALL declared Outcome-dimension checkpoints (any status), NOT pre-filtered to terminal ones.
    ocps = [c for c in cps if c.get("dimension") == "Outcome"]
    if not ocps:
        return None
    # P0-B: an Outcome checkpoint that did NOT reach a terminal verdict (error / skipped / missing status)
    # makes native success UNRESOLVED -> None. Never silently drop a non-terminal checkpoint and all() over
    # the survivors (that would let a partially-errored task read as a clean pass and inflate Outcome).
    if any(c.get("checkpoint_status") not in ("passed", "failed") for c in ocps):
        return None
    return all(c.get("checkpoint_status") == "passed" for c in ocps)


def _native_metrics(bench, results):
    n = len(results)
    cps = [c for r in results for c in (r.get("checkpoints") or [])]
    cp_pass = sum(1 for c in cps if c.get("checkpoint_status") == "passed")
    cp_total = sum(1 for c in cps if c.get("checkpoint_status") in ("passed", "failed"))
    base = {"n_tasks": n,
            "checkpoint_pass_rate": round(cp_pass / cp_total, 3) if cp_total else None,
            "checkpoint_passed": cp_pass, "checkpoint_scored": cp_total}
    # native task success via the SINGLE SOURCE OF TRUTH (native Outcome checkpoints / GAcc), NOT r['success']
    # (the harness gate). Denominator = tasks that actually carry a native outcome signal.
    _nat = [native_task_outcome(r, bench) for r in results]
    _nat = [x for x in _nat if x is not None]
    _nat_pass, _nat_n = sum(1 for x in _nat if x), len(_nat)
    if bench == "PhysicianBench":
        base["pass_at_1"] = round(_nat_pass / _nat_n, 3) if _nat_n else None
        base["pass_at_1_tasks"] = "%d/%d" % (_nat_pass, _nat_n)
        base["native_note"] = "native Outcome checkpoints only (NOT r['success'] harness gate)"
    elif bench == "MedCTA":
        scores = [c["score"] for c in cps if c.get("evaluator_kind") == "gacc_judge" and isinstance(c.get("score"), (int, float))]
        base["gacc_mean"] = round(sum(scores) / len(scores), 3) if scores else None
        base["gacc_n"] = len(scores)
        base["pass_at_1"] = round(_nat_pass / _nat_n, 3) if _nat_n else None     # GAcc>=0.5 binary
        base["pass_at_1_tasks"] = "%d/%d" % (_nat_pass, _nat_n)
    elif bench == "HealthAdminBench":
        base["task_success_rate"] = round(_nat_pass / _nat_n, 3) if _nat_n else None
        base["task_success_tasks"] = "%d/%d" % (_nat_pass, _nat_n)
        base["subtask_pass_rate"] = base["checkpoint_pass_rate"]
        base["native_note"] = "native Outcome checkpoints only (NOT r['success'] harness gate)"
    return base


def _harness_dims(results):
    import statistics as _st
    acc = {m: [] for m in MODULES}; prate = {m: [] for m in MODULES}
    for r in results:
        ds = r.get("dimension_scores") or {}; pr = r.get("dimension_pass_rate") or {}
        for m in MODULES:
            if ds.get(m) is not None: acc[m].append(ds[m])
            if pr.get(m) is not None: prate[m].append(pr[m])
    # per-dimension evidence_tier override declared by checkpoints: a dimension whose checkpoint carries a
    # non-strict evidence_tier (e.g. Governance evidence_tier=experimental_hybrid, review 5.5) must NOT be
    # reported as strict/formal even though it produces a number.
    tier_override = {}
    for r in results:
        for c in (r.get("checkpoints") or r.get("cps") or []):
            dim = c.get("dimension")
            et = c.get("evidence_tier") or (c.get("detail") or {}).get("evidence_tier")
            if dim and et and et != "strict":
                tier_override[dim] = et
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
                   "evidence_tier": (tier_override.get(m) or "strict") if covered else "not_evaluated",
                   "report_in_primary_profile": True,
                   "formal_analysis_eligible": covered and (m not in tier_override),
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


def _experimental_evaluators(agent_dir, bench, task_ids=None):
    """Step (b) -> WIRED to the substrate-based dimension evaluators (single source of truth). Execution
    and Lifecycle are now scored by runner/dim_execution.execution + runner/dim_lifecycle.lifecycle over
    the SemanticTrace built by substrate.map_trace(plugin) + substrate.dimension_policy + the
    CapabilityManifest — NOT the old raw-event lifecycle_exec heuristics. Observability is likewise
    scored by runner/dim_observability.observability over substrate.evidence_view (returned so build()
    can fill the Observability dimension cell). These FILL the Execution/Lifecycle/Observability cells;
    tier=experimental until human-audited. Benchmark specifics (which tool -> which milestone/role,
    required_milestones) arrive ONLY through the plugin via the substrate.

    Returns (panel, ex_t, lc_t, ob_t) where ex_t/lc_t/ob_t are per-task {task_id: score}."""
    import statistics as _st
    try:
        import substrate as _sub
        import dim_execution as _dex
        import dim_lifecycle as _dlc
        import dim_observability as _dob
    except Exception as _e:
        return {"note": "substrate dimension evaluators unavailable: %r" % (_e,)}, {}, {}, {}
    # Context / Verification / Tooling: built-but-not-wired into the STRICT checkpoint gate (those
    # remain the score-eligible cp_grounding / cp_verification / cp_arg_accuracy+cp_tool_path strict
    # checkpoints in scoring.py). Here they run ONLY as an experimental cross-check panel (deterministic,
    # offline: no answer/gold, judge_fn=None) over the substrate, surfaced in experimental_evaluators
    # and NEVER folded into harness_dimensions/success. Import is best-effort so a missing module cannot
    # break the report.
    try:
        import dim_context as _dctx
    except Exception:
        _dctx = None
    try:
        import dim_verification as _dver
    except Exception:
        _dver = None
    try:
        import dim_tooling as _dtool
    except Exception:
        _dtool = None
    plugin, _plugin_problem = _sub.require_plugin(bench)
    if _plugin_problem:                                  # unknown benchmark -> fail-closed, NO vacuous scores
        return {"tier": "unavailable", "score_eligible": False, "problem": _plugin_problem}, {}, {}, {}
    ex_t, lc_t, ob_t, ctx_t, ver_t, tool_t, gov_t = {}, {}, {}, {}, {}, {}, {}
    _gov_canon = {}                   # P1-1: per-task canonical Governance (score/reportable/critical/components)
    _gov_audit = {}                   # task item 1/2: the per-task Governance SHARED-CONTRACT block read from disk
    _eval_errors = []                 # task item 7: [{dimension, task_id, exception_type, failure_stage}]
    _qualified = set()                # task item 6: the COMMON qualified-profile task set (one 7-dim denominator)
    _cov_low = {}
    _rep = {d: {} for d in MODULES}   # per-dim per-task: True if score rests on REAL (reportable) evidence
    ctx_t, ver_t, tool_t = {}, {}, {}
    _lc_cov, _lc_unreportable = {}, []
    # CRIT-latent (dual-glob) fix: the 7-dim universe / coverage denominator must be the SAME task-id set
    # the outcome/native/n_tasks stages use (result.json), NOT the trajectory.jsonl glob. A task that has a
    # result but no/empty trajectory was previously invisible to all 7 dims AND excluded from _ntask, so its
    # coverage falsely computed as 1.0. We drive coverage from the UNION of result-dirs and trajectory-dirs.
    _result_ids = {os.path.basename(os.path.dirname(p))
                   for p in glob.glob(os.path.join(agent_dir, "*", "result.json"))}
    _traj_ids = {os.path.basename(os.path.dirname(p))
                 for p in glob.glob(os.path.join(agent_dir, "*", "trajectory.jsonl"))}
    _wl = set(task_ids) if task_ids is not None else None   # whitelist (None => full dataset, unchanged)
    if _wl is not None:
        _result_ids = _result_ids & _wl
        _traj_ids = _traj_ids & _wl
    _all_task_ids = _result_ids | _traj_ids
    # tasks with a result but no scorable trajectory -> they CANNOT receive a substrate dim score; they must
    # still count against the coverage denominator (so coverage < 1.0 honestly reflects the gap).
    _result_only_ids = sorted(_result_ids - _traj_ids)
    for tp in sorted(glob.glob(os.path.join(agent_dir, "*", "trajectory.jsonl"))):
        tid = os.path.basename(os.path.dirname(tp))
        if _wl is not None and tid not in _wl:
            continue
        try:
            evs = [json.loads(l) for l in open(tp) if l.strip()]
        except Exception:
            continue
        # per-task task dict + provenance (capability manifest) from the bundle
        bdir = os.path.dirname(tp)
        task = {"source_benchmark": bench, "task_id": tid}
        prov = {}
        tpath = os.path.join(bdir, "task.json")
        if os.path.exists(tpath):
            try: task = json.load(open(tpath))
            except Exception: pass
        rp = os.path.join(bdir, "result.json")
        if os.path.exists(rp):
            try: prov = (json.load(open(rp)).get("provenance") or {})
            except Exception: prov = {}
        # substrate structures (the ONLY inputs the dimension evaluators consume)
        sem = _sub.map_trace(evs, plugin)
        dp = _sub.dimension_policy(task, plugin)
        if dp.get("score_eligible") is False:
            continue        # fail-closed: missing/invalid dimension policy -> no score for this task
        _qualified.add(tid)  # task item 6: a task that QUALIFIED (valid dim policy) joins the common
        #                       7-dim denominator. EVERY of its 7 dims is always-present; a per-task dim
        #                       that ends N/A still counts here as a miss (never deletes the task).
        manifest = _sub.capability_manifest(prov)
        ev = _sub.evidence_view(evs, plugin)
        # SYSTEM RULE 1: a qualified run outputs ALL 7 ETCLOVG scores. We store every dimension score
        # UNCONDITIONALLY; "reportable"/opportunity only feeds coverage/confidence, never deletes the score.
        e = _dex.execution(sem, dp, manifest)
        l = _dlc.lifecycle(sem, dp, manifest)
        o = _dob.observability(ev, sem)
        if isinstance(e.get("score"), (int, float)): ex_t[tid] = e["score"]; _rep["Execution"][tid] = True
        if isinstance(l.get("score"), (int, float)): lc_t[tid] = l["score"]; _rep["Lifecycle"][tid] = bool(l.get("reportable_score"))
        if isinstance(o.get("score"), (int, float)): ob_t[tid] = o["score"]; _rep["Observability"][tid] = bool(o.get("reportable"))
        if not l.get("reportable_score"): _cov_low.setdefault("Lifecycle", []).append(tid)
        if not o.get("reportable"): _cov_low.setdefault("Observability", []).append(tid)
        if _dctx is not None:
            try:
                instr = (task.get("context") or {}).get("text") or task.get("goal")
                cx = _dctx.context(sem, ev, dp, task_instruction=instr, judge_model=None)
                if isinstance(cx.get("score"), (int, float)): ctx_t[tid] = cx["score"]; _rep["Context"][tid] = bool(cx.get("reportable"))
                if not cx.get("reportable"): _cov_low.setdefault("Context", []).append(tid)
            except Exception as _e:                                  # task item 7: record, never swallow silently
                _eval_errors.append({"dimension": "Context", "task_id": tid,
                                     "exception_type": type(_e).__name__, "failure_stage": "dim_context.context"})
        if _dver is not None:
            try:
                vfa = [s for s in sem if s.get("event_role") == "verify"]
                vr = _dver.verification(ev, vfa, _dver.extract_claims(sem), conflicts=None, policy=dp, judge_fn=None, sem_trace=sem)
                if isinstance(vr.get("score"), (int, float)): ver_t[tid] = vr["score"]; _rep["Verification"][tid] = bool(vr.get("reportable"))
                if not vr.get("reportable"): _cov_low.setdefault("Verification", []).append(tid)
            except Exception as _e:                                  # task item 7: record, never swallow silently
                _eval_errors.append({"dimension": "Verification", "task_id": tid,
                                     "exception_type": type(_e).__name__, "failure_stage": "dim_verification.verification"})
        if _dtool is not None:
            try:
                tr = _dtool.tooling(sem, dp, manifest, available_tools=task.get("available_tools"), task=task, plugin=plugin)
                if isinstance(tr.get("score"), (int, float)): tool_t[tid] = tr["score"]; _rep["Tooling"][tid] = bool(tr.get("reportable"))
                if not tr.get("reportable"): _cov_low.setdefault("Tooling", []).append(tid)
            except Exception as _e:                                  # task item 7: record, never swallow silently
                _eval_errors.append({"dimension": "Tooling", "task_id": tid,
                                     "exception_type": type(_e).__name__, "failure_stage": "dim_tooling.tooling"})
        # GOVERNANCE = PURE READ of the SHARED CONTRACT (task items 1 + 2). The judge-backed Governance
        # score + full audit (components / submetrics G1-G4 / judge raw+parsed / scoring_config /
        # branch / critical) are produced UPSTREAM by runner/rescore_judges.py and persisted to
        # <task>/result.rescored.json as the self-describing "Governance" block. The aggregate makes NO
        # model call: it reads that block, takes its score + reportable + critical, and surfaces the audit.
        #   * block present + score numeric -> use it (reportable/critical/components straight from disk).
        #   * block present but evaluation_error / score null -> N/A: score=None, reportable=False, the
        #     evaluation_error propagates (JUDGE-FAILURE never falls back to a scope-only construct here).
        #   * no result.rescored.json / no Governance block -> N/A, evaluation_error="not_rescored"
        #     (the task simply has not been rescored yet), NEVER a live judge call.
        try:
            _gblk, _gerr = _read_dim_block(bdir, "Governance")
            if _gblk is not None and isinstance(_gblk.get("score"), (int, float)) and not _gerr:
                _gsc = float(_gblk["score"])
                _grep = bool(_gblk.get("reportable"))
                _gcritical = bool(_gblk.get("critical"))
                gov_t[tid] = _gsc
                _rep["Governance"][tid] = _grep
                _gov_canon[tid] = {"score": _gsc, "reportable": _grep, "critical": _gcritical,
                                   "components": (_gblk.get("components") or _gblk.get("submetrics")),
                                   "branch": _gblk.get("branch")}
                _gov_audit[tid] = {
                    "score": _gsc, "reportable": _grep, "critical": _gcritical,
                    "evidence_tier": _gblk.get("evidence_tier"),
                    "formal_analysis_eligible": _gblk.get("formal_analysis_eligible"),
                    "deterministic": _gblk.get("deterministic"),
                    "branch": _gblk.get("branch"),
                    "components": _gblk.get("components"), "submetrics": _gblk.get("submetrics"),
                    "judge_model": (_gblk.get("judge") or {}).get("model"),
                    "prompt_version": (_gblk.get("judge") or {}).get("prompt_version"),
                    "prompt_hash": (_gblk.get("judge") or {}).get("prompt_hash"),
                    "scoring_version": (_gblk.get("scoring_config") or {}).get("scoring_version"),
                    "code_sha": (_gblk.get("scoring_config") or {}).get("code_sha"),
                    "scoring_code_tree_hash": (_gblk.get("scoring_config") or {}).get("scoring_code_tree_hash"),
                    "g14_weight": (_gblk.get("scoring_config") or {}).get("g14_weight"),
                    "evaluation_error": None}
                if not _grep: _cov_low.setdefault("Governance", []).append(tid)
            else:
                # honest N/A: NOT rescored (or judge failed / score null). NO score enters gov_t -> the dim
                # mean is unaffected; the task still counts in the denominator as a Governance miss (item 6).
                _err = _gerr or ("score_null" if _gblk is not None else "not_rescored")
                _gov_canon[tid] = {"score": None, "reportable": False, "critical": None,
                                   "components": None, "branch": (_gblk or {}).get("branch"),
                                   "evaluation_error": _err}
                _gov_audit[tid] = {"score": None, "reportable": False,
                                   "evidence_tier": (_gblk or {}).get("evidence_tier") or "experimental_hybrid",
                                   "evaluation_error": _err}
                _cov_low.setdefault("Governance", []).append(tid)
        except Exception as _e:
            _eval_errors.append({"dimension": "Governance", "task_id": tid,
                                 "exception_type": type(_e).__name__,
                                 "failure_stage": "read_result_rescored_governance_block"})
            _gov_canon[tid] = {"score": None, "reportable": False, "critical": None,
                               "components": None, "branch": None,
                               "evaluation_error": "read_exception:%s" % type(_e).__name__}
        for k, st in (l.get("submetric_status") or {}).items():
            _lc_cov.setdefault(k, {"valid": 0, "total": 0})
            _lc_cov[k]["total"] += 1; _lc_cov[k]["valid"] += 1 if st == "valid" else 0
        if not l.get("reportable_score"): _lc_unreportable.append(tid)
    def _agg(d, tier="experimental_state_machine"):
        v = list(d.values())
        return {"mean": round(sum(v) / len(v), 3) if v else None, "n": len(v),
                "std": round(_st.pstdev(v), 3) if len(v) > 1 else (0.0 if v else None),
                "zero_variance": (len(set(v)) == 1) if v else None,
                "informativeness": ("saturated" if (v and len(set(v)) == 1) else ("discriminating" if v else "none")),
                "tier": tier}
    _life = _agg(lc_t)
    _life["submetric_coverage"] = {k: "%d/%d" % (v["valid"], v["total"]) for k, v in sorted(_lc_cov.items())}
    _life["n_unreportable_insufficient_coverage"] = len(_lc_unreportable)
    panel = {"tier": "experimental_substrate_dimension_evaluators", "deterministic": True,
             "source": "dim_execution/dim_lifecycle/dim_observability over substrate",
             "promotion_path": "experimental -> human_audited -> strict",
             "Execution_sm": _agg(ex_t), "Lifecycle_sm": _life,
             "Observability_sm": _agg(ob_t, tier="experimental_substrate_observability")}
    # Context / Verification / Tooling substrate evaluators run as an OFFLINE cross-check ONLY (judge_fn
    # /judge_model = None). They are NOT wired into the strict checkpoint gate (scoring.py keeps the
    # strict cp_grounding / cp_verification / cp_arg_accuracy+cp_tool_path) and NOT folded into
    # harness_dimensions/success. Surfaced for human audit / promotion-path evidence.
    panel["cross_check_not_wired"] = {
        "note": ("substrate Context/Verification/Tooling evaluators, deterministic+offline; not folded "
                 "into strict success or harness_dimensions; the strict checkpoints in scoring.py remain "
                 "the authoritative source for these three dimensions"),
        "Context_xc": _agg(ctx_t, tier="experimental_substrate_context_mgmt"),
        "Verification_xc": _agg(ver_t, tier="experimental_substrate_verification"),
        "Tooling_xc": _agg(tool_t, tier="experimental_substrate_tooling")}
    # SYSTEM RULE 1+2: a fixed, always-present 7-dim Harness profile from the UNIFIED substrate scorers
    # (never from dataset checkpoint tags). Each dim ALWAYS has a [0,1] score; evidence strength only sets
    # coverage. A dim with NO score across the whole dataset is an ADAPTER-ADMISSION gap (flagged), NOT 0/1.
    _DIM_T = {"Execution": ex_t, "Tooling": tool_t, "Context": ctx_t, "Lifecycle": lc_t,
              "Observability": ob_t, "Verification": ver_t, "Governance": gov_t}
    # task item 6: ONE common qualified-profile denominator for the headline 7-dim means. n_qualified is the
    # set of tasks that QUALIFIED (valid dimension policy). EVERY dim reports n_scored == n_qualified, so a
    # per-task dim that is N/A on a qualified task counts as a MISS in that dim's denominator (always-present-
    # dimension rule), it never silently shrinks the denominator. coverage/confidence stay auxiliary.
    _nq = len(_qualified) or 1
    # union-of-tasks denominator (result.json UNION trajectory.jsonl) for the task_universe coverage panel.
    _ntask = len(_all_task_ids) or 1
    # task item 2: per-dimension evidence_tier the EVALUATOR declares. Governance/Context/Verification are
    # judge-backed -> experimental_hybrid (formal_analysis_eligible=False, deterministic=False). Execution/
    # Tooling/Lifecycle/Observability are deterministic substrate dims -> substrate_universal (deterministic=
    # True). This stops hardcoding substrate_universal for every dim.
    _HYBRID = {"Governance", "Context", "Verification"}
    # judge metadata to surface on the judge-backed dims (Governance comes from the SHARED-CONTRACT block;
    # Context/Verification run offline here so they carry no live judge, but the tier still says hybrid).
    _gov_aud_any = next((a for a in _gov_audit.values() if a and a.get("judge_model")), {})
    _JUDGE_META = {"Governance": {"judge_model": _gov_aud_any.get("judge_model"),
                                  "prompt_version": _gov_aud_any.get("prompt_version")},
                   "Context": {"judge_model": None, "prompt_version": None},
                   "Verification": {"judge_model": None, "prompt_version": None}}
    # task item 4 admission threshold (numeric AND reportable coverage must both clear it).
    _ADM_THRESH = float(os.environ.get("MH_ADMISSION_THRESH", "0.8"))
    formal = {}
    for _dim, _d in _DIM_T.items():
        _hybrid = _dim in _HYBRID
        _v = list(_d.values())                                 # tasks that produced a numeric score for _dim
        # 方案 A (P0 fix): the headline MEAN aggregates REPORTABLE per-task scores only (a non-reportable
        # default never pollutes it). score_all_scored retains the full mean transparently for audit.
        _vr = [_d[tid] for tid in _d if _rep[_dim].get(tid)]    # REPORTABLE-ONLY per-task scores
        _nrep = len(_vr)
        # task item 4: the DOUBLE GATE. Three ratios over the COMMON qualified denominator:
        #   numeric_coverage        = n_scored      / n_qualified   (did the dim produce a number at all)
        #   reportable_coverage     = n_reportable  / n_qualified   (did it rest on REAL evidence)
        #   within_scored_confidence= n_reportable  / n_scored      (of the numbers, how many are real)
        _numeric_cov = round(len(_v) / _nq, 3)
        _reportable_cov = round(_nrep / _nq, 3)
        _within_conf = round(_nrep / len(_v), 3) if _v else 0.0
        if not _v:
            _adm = "INCOMPLETE: dataset produced no %s evidence" % _dim
        elif _nrep == 0:
            _adm = ("VACUOUS: score rests on non-discriminative/inapplicable evidence (within_scored_confidence "
                    "%.2f) -- this construct is not meaningfully measured on this dataset" % _within_conf)
        elif _numeric_cov >= _ADM_THRESH and _reportable_cov >= _ADM_THRESH:
            _adm = "ok"                                        # BOTH gates clear -> admit
        elif _numeric_cov < _ADM_THRESH:
            _adm = ("LOW_COVERAGE: numeric_coverage %.2f < %.2f (only %d/%d qualified tasks produced a %s "
                    "number)" % (_numeric_cov, _ADM_THRESH, len(_v), _nq, _dim))
        else:
            _adm = ("LOW_COVERAGE: reportable_coverage %.2f < %.2f (only %d/%d qualified tasks expose a REAL "
                    "%s opportunity)" % (_reportable_cov, _ADM_THRESH, _nrep, _nq, _dim))
        formal[_dim] = {
            # HEADLINE = reportable-only mean (方案 A). None when nothing reportable -- never a polluted number.
            "score": round(sum(_vr) / len(_vr), 3) if _vr else None,
            "score_all_scored": round(sum(_v) / len(_v), 3) if _v else None,   # audit-only (includes defaults)
            # task item 6: n_scored is over the COMMON qualified denominator -> identical across all 7 dims.
            "n_scored": _nq, "n_qualified": _nq,
            "n_with_evidence": len(_v),                       # how many qualified tasks actually produced a number
            "n_reportable": _nrep,
            # task item 4: the three double-gate ratios (all surfaced).
            "numeric_coverage": _numeric_cov, "reportable_coverage": _reportable_cov,
            "within_scored_confidence": _within_conf,
            "coverage": _numeric_cov,                          # back-compat alias (== numeric_coverage)
            "confidence": _within_conf,                        # back-compat alias (== within_scored_confidence)
            # std / informativeness over the REPORTABLE subset (the subset the headline mean is taken over)
            "std": round(_st.pstdev(_vr), 3) if len(_vr) > 1 else (0.0 if _vr else None),
            "informativeness": ("saturated" if (_vr and len(set(_vr)) == 1) else ("discriminating" if _vr else "none")),
            # task item 2: tier the EVALUATOR declares (not a blanket substrate_universal).
            "evidence_tier": "experimental_hybrid" if _hybrid else "substrate_universal",
            "deterministic": not _hybrid,
            # strict-by-DEFINITION: deterministic substrate evidence with a reportable value. This is the
            # dimension's TYPE, independent of how much of THIS run it covered.
            "strict_by_definition": (not _hybrid) and bool(_vr),
            # FORMALLY ADMITTED: strict-by-definition AND the adapter admitted coverage (_adm == "ok"). A
            # dimension flagged LOW_COVERAGE by adapter admission must NOT enter the formal statistical set
            # even though it is strict by type -- a 0.6-coverage Observability is not a formal claim.
            "formal_analysis_eligible": (not _hybrid) and bool(_vr) and (_adm == "ok"),
            "adapter_admission": _adm}
        if _hybrid:                                            # surface judge metadata for judge-backed dims
            formal[_dim]["judge_model"] = _JUDGE_META.get(_dim, {}).get("judge_model")
            formal[_dim]["prompt_version"] = _JUDGE_META.get(_dim, {}).get("prompt_version")
    panel["harness_seven"] = formal
    panel["evaluator_errors_by_dimension"] = _eval_errors      # task item 7
    panel["governance_audit"] = dict(_gov_audit)               # task item 1/2: per-task on-disk Governance audit
    panel["qualified_profile"] = {                             # task item 6
        "n_qualified": len(_qualified), "qualified_task_ids": sorted(_qualified),
        "note": ("headline 7-dim means share ONE denominator = n_qualified (a task with a valid dimension "
                 "policy). Every dim's n_scored == n_qualified; an N/A per-task dim counts as a miss in the "
                 "denominator (always-present-dimension rule). numeric/reportable coverage are auxiliary.")}
    panel["task_universe"] = {
        "n_unified": len(_all_task_ids), "n_with_result": len(_result_ids), "n_with_trajectory": len(_traj_ids),
        "result_only_no_trajectory": _result_only_ids,
        "trajectory_only_no_result": sorted(_traj_ids - _result_ids),
        "coverage_denominator": _ntask,
        "note": ("7-dim coverage denominator = result.json UNION trajectory.jsonl (same as n_tasks/outcome). "
                 "result_only_no_trajectory tasks cannot receive a substrate dim score -> they pull coverage "
                 "below 1.0 honestly instead of being invisible.")}
    panel["seven_source"] = "unified substrate evaluators (dim_execution/tooling/context/lifecycle/observability/verification + governance); checkpoint tags do NOT determine these"
    # P1-1: expose the per-task substrate dim scores + the canonical Governance verdict so build() can write
    # ONE canonical per-task file (result.rescored.json) that the report, the file, and diagnostics agree on.
    panel["_per_task_dims"] = {d: dict(_DIM_T[d]) for d in MODULES}
    panel["_per_task_reportable"] = {d: dict(_rep[d]) for d in MODULES}
    panel["_gov_canon"] = dict(_gov_canon)
    return panel, ex_t, lc_t, ob_t


def _outcome_metric(results, bench=None):
    """Source Outcome = did the agent get the dataset-NATIVE task/clinical result right -- a SEPARATE line
    from ETCLOVG (occupies none of the 7 dims).

    task item 5 -- Outcome is FULLY INDEPENDENT and NEVER falls back to all score-eligible (harness)
    checkpoints:
      * native Outcome checkpoints (dimension==Outcome) -> outcome_checkpoint_pass_rate, AND
      * native task success (PB success / HAB success / MedCTA GAcc>=0.5) -> outcome_task_success_rate /
        pass_at_1 reported SEPARATELY (from native success ONLY).
      * if a dataset has NEITHER native Outcome checkpoints NOR GAcc -> outcome = N/A / "adapter_incomplete"
        (never native_checkpoint_pass_rate_fallback).
    harness_gate_success / overall_success are kept as SEPARATE auxiliary fields (the harness gate, not the
    source Outcome) and are NEVER merged into the Outcome score."""
    n = len(results)
    npass = ntot = 0
    for r in results:
        for c in (r.get("checkpoints") or []):
            if c.get("dimension") == "Outcome" and c.get("checkpoint_status") in ("passed", "failed"):
                ntot += 1; npass += 1 if c.get("checkpoint_status") == "passed" else 0
    gacc = [c.get("score") for r in results for c in (r.get("checkpoints") or [])
            if c.get("evaluator_kind") == "gacc_judge" and isinstance(c.get("score"), (int, float))]
    # native task success via the SINGLE SOURCE OF TRUTH native_task_outcome() (MedCTA GAcc>=0.5; PB/HAB all
    # Outcome-checkpoints passed) -- NEVER r.get("success") (the harness GATE over all ETCLOVG dims). The
    # denominator is exactly the tasks that carry a native outcome signal (native_task_outcome is not None).
    _signals = [(r, native_task_outcome(r, bench)) for r in results]
    _succ_tasks = [r for (r, o) in _signals if o is not None]
    _native_ok = sum(1 for (_, o) in _signals if o)
    _native_n = len(_succ_tasks)
    # evaluation coverage: how much of the task universe actually produced a resolved native outcome. Tasks
    # with NO native signal OR an unresolved (error/skipped/incomplete) Outcome are excluded from the rate's
    # denominator -- coverage surfaces exactly how many that is, so a high rate over a thin slice is visible.
    _eval_cov = round(_native_n / n, 3) if n else None
    _native_block = {
        "outcome_task_success_rate": round(_native_ok / _native_n, 3) if _native_n else None,
        "pass_at_1": round(_native_ok / _native_n, 3) if _native_n else None,
        "native_success_passed": _native_ok, "native_success_evaluated": _native_n,
        "native_task_universe": n, "native_evaluation_coverage": _eval_cov,
        "note": ("PB/HAB: ALL declared reportable Outcome-dimension checkpoints passed (terminal); "
                 "MedCTA: mean GAcc>=0.5. NOT r['success'] (harness gate). Tasks with an unresolved/missing "
                 "Outcome are excluded from the denominator -> see native_evaluation_coverage.")}
    # harness gate success kept SEPARATE (never merged into Outcome).
    _harness_ok = sum(1 for r in results if r.get("success"))
    _harness_block = {"harness_gate_success": round(_harness_ok / n, 3) if n else None,
                      "overall_success_tasks": "%d/%d" % (_harness_ok, n),
                      "note": "harness gate (all-checkpoints-pass), NOT the source Outcome; reported separately"}
    # HEADLINE Outcome = dataset NATIVE task accuracy (did the agent complete the dataset task), NOT the
    # per-Outcome-checkpoint pass rate and NOT the harness gate. The checkpoint pass rate is a DIAGNOSTIC.
    if _native_n:
        return {"score": round(_native_ok / _native_n, 3), "metric": "dataset_native_task_accuracy",
                "n_tasks": _native_n, "native_task_universe": n,
                "native_evaluation_coverage": round(_native_n / n, 3) if n else None,
                "native_task_success": _native_block, "harness_gate": _harness_block,
                "outcome_diagnostics": {"checkpoint_pass_rate": round(npass / ntot, 3) if ntot else None,
                                        "n_outcome_checkpoints": ntot,
                                        "gacc_mean": round(sum(gacc) / len(gacc), 3) if gacc else None}}
    # NO native Outcome checkpoint and NO GAcc -> the adapter has not declared a Source Outcome line.
    # Outcome is N/A (adapter_incomplete) -- it NEVER falls back to the native/harness checkpoint pass rate.
    return {"score": None, "metric": "adapter_incomplete", "n_outcome_checkpoints": 0, "gacc_mean": None,
            "note": ("no Outcome-tagged checkpoint nor GAcc in this dataset's assets -> Outcome is N/A. The "
                     "harness checkpoint pass rate is NOT a substitute for a Source Outcome line."),
            "native_task_success": _native_block, "harness_gate": _harness_block}


def _read_disk_governance(agent_dir):
    """PURE READ (task items 1 + 8). RE-READ every <task>/result.rescored.json from disk and return the
    per-task on-disk Governance verdict the SHARED CONTRACT carries -- WITHOUT writing anything. This is the
    independent disk view used by the disk-consistency check: it must agree, per task, with the in-memory
    _gov_canon the report headline was built from.

    Returns {task_id: {score, reportable, critical, branch, scoring_version, code_sha, dirty_worktree,
                       g14_weight, evaluation_error}} (score=None / evaluation_error set when not rescored)."""
    out = {}
    for rp in sorted(glob.glob(os.path.join(agent_dir, "*", "result.json"))):
        tid = os.path.basename(os.path.dirname(rp))
        blk, err = _read_dim_block(os.path.dirname(rp), "Governance")
        if blk is None:
            out[tid] = {"score": None, "reportable": False, "critical": None, "branch": None,
                        "scoring_version": None, "code_sha": None, "scoring_code_tree_hash": None,
                        "dirty_worktree": None, "judge_model": None, "prompt_version": None,
                        "tasks_unified_sha256": None, "g14_weight": None, "evaluation_error": err}
            continue
        _sc = blk.get("scoring_config") or {}
        _j = blk.get("judge") or {}
        out[tid] = {"score": blk.get("score"), "reportable": bool(blk.get("reportable")),
                    "critical": blk.get("critical"), "branch": blk.get("branch"),
                    "scoring_version": _sc.get("scoring_version"), "code_sha": _sc.get("code_sha"),
                    "scoring_code_tree_hash": _sc.get("scoring_code_tree_hash"),
                    "dirty_worktree": _sc.get("dirty_worktree"),
                    "judge_model": _j.get("model"), "prompt_version": _j.get("prompt_version"),
                    "tasks_unified_sha256": _sc.get("tasks_unified_sha256"),
                    "g14_weight": _sc.get("g14_weight"), "evaluation_error": blk.get("evaluation_error")}
    return out


def build(agent_dir, bench):
    results = _remap(_load(agent_dir), bench)
    hd = _harness_dims(results)
    strict_covered = {m for cat in hd["by_category"].values() for m, v in cat.items()
                      if v["status"] == "covered"}
    proxy = _proxy_dims(agent_dir, strict_covered)
    _exp_panel, _ex_t, _lc_t, _ob_t = _experimental_evaluators(agent_dir, bench)
    # PURE READ (task item 1): the report does NOT write result.rescored.json and makes NO model call. The
    # judge-backed Governance score + audit are read from <task>/result.rescored.json (written upstream by
    # rescore_judges.py per the SHARED CONTRACT). _read_disk_governance re-reads that on-disk view for the
    # disk-consistency check; _exp_panel.harness_seven.Governance was built from the SAME blocks -> they agree.
    _disk_gov = _read_disk_governance(agent_dir)
    _src_prov = _source_provenance_audit(agent_dir, bench)   # P0/P1: live-recompute source-input hashes
    # FILL the Execution / Lifecycle / Observability dimension cells with the substrate-based dimension
    # evaluators (dim_execution / dim_lifecycle / dim_observability). Single source of truth: the old
    # raw-event lifecycle_exec formulas and the proxy_verifiers Observability pipeline no longer feed the
    # REPORT's Execution/Lifecycle/Observability cells.
    if isinstance(proxy.get("by_dimension"), dict):
        if _ex_t: proxy["by_dimension"]["Execution"] = _exp_panel["Execution_sm"]
        if _lc_t: proxy["by_dimension"]["Lifecycle"] = _exp_panel["Lifecycle_sm"]
        if _ob_t and "Observability" not in strict_covered:
            proxy["by_dimension"]["Observability"] = _exp_panel["Observability_sm"]
    _integ = _integrity(results)
    _toc = (proxy.get("by_dimension") or {}).pop("trace_observation_coverage", None)   # Codex #7
    if _toc is not None:
        _integ["trace_observation_coverage"] = _toc
    _proxy_filled = sorted((set((proxy.get("by_dimension") or {}).keys()) & set(MODULES)) - strict_covered)
    _hd = hd["by_category"]
    _task_cov = {m: "%d/%d" % (d["n_scored"], d["n_tasks"]) for cat in _hd.values() for m, d in cat.items() if d["n_scored"]}
    # task item 1: strict/formal coverage is read from the FINAL harness_seven blocks, NOT the legacy
    # checkpoint diagnostics (hd.by_category, which mislabels judge-backed Context/Governance as strict).
    # A dimension is STRICT only when its harness_seven block declares formal_analysis_eligible=True
    # (the deterministic substrate dims: Execution/Tooling/Lifecycle/Observability with reportable evidence).
    # Context/Governance/Verification are experimental_hybrid (formal_analysis_eligible=False) and must NOT
    # appear here even though they produce a number. We report THREE DISTINCT breadth fields:
    #   numeric_coverage    = dims that produced a value at all              (7/7 here is fine, NOT strict)
    #   reportable_coverage = dims whose value rests on REAL (reportable) evidence
    #   formal_coverage     = dims that are strict (formal_analysis_eligible) -> the only formal-stats set
    _seven = (_exp_panel.get("harness_seven") if isinstance(_exp_panel, dict) else None) or {}
    _strict_dims = sorted(m for m in MODULES
                          if isinstance(_seven.get(m), dict) and _seven[m].get("formal_analysis_eligible") is True)
    # strict-by-DEFINITION (deterministic substrate type) vs FORMALLY ADMITTED (also passed adapter coverage
    # admission). A dim can be strict-by-definition yet NOT formally admitted (e.g. LOW_COVERAGE Observability).
    _strict_def_dims = sorted(m for m in MODULES
                              if isinstance(_seven.get(m), dict) and _seven[m].get("strict_by_definition") is True)
    _low_cov_strict = sorted(set(_strict_def_dims) - set(_strict_dims))
    _numeric_dims = sorted(m for m in MODULES
                           if isinstance(_seven.get(m), dict) and (_seven[m].get("n_with_evidence") or 0) > 0)
    _reportable_dims = sorted(m for m in MODULES
                              if isinstance(_seven.get(m), dict) and (_seven[m].get("n_reportable") or 0) > 0)
    coverage_summary = {
        # numeric breadth: how many of the 7 dims produced a value at all (NOT a strictness claim).
        "numeric_coverage": "%d/7" % len(_numeric_dims), "numeric_dimensions": _numeric_dims,
        "reportable_coverage": "%d/7" % len(_reportable_dims), "reportable_dimensions": _reportable_dims,
        # formal/strict breadth: ONLY dims with formal_analysis_eligible=True (strict TYPE + adapter admitted).
        "formal_coverage": "%d/7" % len(_strict_dims), "strict_dimensions": _strict_dims,
        "formally_admitted_dimensions": _strict_dims,
        # strict-by-definition (deterministic substrate type) -- a SUPERSET of formally admitted.
        "strict_definition_dimensions": _strict_def_dims,
        # strict-by-type but adapter LOW_COVERAGE -> EXCLUDED from formal stats (the口径 fix).
        "strict_but_low_coverage_excluded": _low_cov_strict,
        # dimension_breadth kept as a back-compat alias of the FORMAL (admitted) count.
        "dimension_breadth": "%d/7 strict" % len(_strict_dims),
        "proxy_filled": "%d/7" % len(_proxy_filled), "proxy_dimensions": _proxy_filled,
        "task_eval_coverage": _task_cov,
        "caveat": ("THREE distinct breadths: numeric_coverage (a dim produced a value -- 7/7 is fine but is "
                   "NOT a strictness claim), reportable_coverage (value rests on real evidence), and "
                   "formal_coverage/strict_dimensions (formal_analysis_eligible=True in harness_seven -- the "
                   "ONLY dims that enter formal stats; Context/Governance/Verification are experimental_hybrid "
                   "and are NEVER listed as strict). Strictness is read from the FINAL harness_seven blocks, "
                   "NOT the legacy checkpoint_diagnostics. Check per-dim evidence_tier / zero_variance / "
                   "informativeness before averaging. Never report '7/7' as strict.")}
    return {
        "source": agent_dir,
        "bench": bench,
        "n_tasks": len(results),
        "coverage_summary": coverage_summary,
        "native_metrics": _native_metrics(bench, results),
        "tool_use_quality": _tool_use_quality(results),
        "harness_dimensions": (_exp_panel.get("harness_seven") if isinstance(_exp_panel, dict) else None),
        "outcome": _outcome_metric(results, bench),
        "checkpoint_diagnostics": hd,
        "governance_consistency": _governance_consistency(_exp_panel, _disk_gov, hd, source_prov=_src_prov),
        "source_provenance": _src_prov,
        "evaluator_errors_by_dimension": ((_exp_panel or {}).get("evaluator_errors_by_dimension") or []
                                          if isinstance(_exp_panel, dict) else []),  # task item 7
        "governance_audit": ((_exp_panel or {}).get("governance_audit") or {}
                             if isinstance(_exp_panel, dict) else {}),               # task item 1/2
        "qualified_profile": ((_exp_panel or {}).get("qualified_profile") or {}
                              if isinstance(_exp_panel, dict) else {}),              # task item 6
        "proxy_dimensions": proxy,
        "experimental_evaluators": _exp_panel,
        "integrity": _integ,
        "failure_taxonomy": _failure_taxonomy(results),
    }


def harness_seven_for_tasks(agent_dir, bench, task_ids):
    """Canonical per-dim ETCLOVG profile (== report['harness_dimensions']) restricted to a task-id
    whitelist. PURE-READ; reuses the SAME strict+proxy+Governance+admission pipeline as build() -- it does
    NOT re-derive dims from raw checkpoint tags, and the default report path (task_ids=None) is unaffected.
    Returns {dim: <harness_seven block>} (each block carries score/reportable coverage/n_scored as build's
    report['harness_dimensions'] does). Governance stays sourced from each task's result.rescored.json
    block with its reportable/evaluation_error, exactly as in the full build."""
    panel, _ex, _lc, _ob = _experimental_evaluators(agent_dir, bench, task_ids=set(task_ids))
    return (panel.get("harness_seven") if isinstance(panel, dict) else None) or {}


def _current_git_head():
    """PURE READ: the current code revision (git rev-parse HEAD) of THIS checkout. Used by the provenance
    guard to detect bundles scored under a DIFFERENT code_sha (stale artifacts). No model call; falls back to
    None if git is unavailable so the guard degrades to 'cannot verify' rather than crashing."""
    try:
        import subprocess
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=os.path.dirname(os.path.abspath(__file__)), timeout=10)
        return sha.stdout.strip() or None if sha.returncode == 0 else None
    except Exception:
        return None


def _current_scoring_tree_hash():
    """PURE READ: the git TREE object hash of runner/ at HEAD (`git rev-parse HEAD:runner`). This is the
    provenance GUARD key -- a content hash of the scoring code ONLY, invariant under commits that touch
    anything outside runner/ (committing res2_* artifacts, docs, etc.). Comparing this instead of full-repo
    HEAD breaks the code_sha==HEAD self-reference loop: an artifact-only commit advances HEAD but leaves the
    runner/ tree unchanged, so an artifact scored at this tree stays 'current'."""
    try:
        import subprocess
        out = subprocess.run(["git", "rev-parse", "HEAD:runner"], capture_output=True, text=True,
                             cwd=os.path.dirname(os.path.abspath(__file__)), timeout=10)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def _sha_file(path):
    """PURE READ sha256 of a file's bytes (None if absent/unreadable). The unit of source-input provenance."""
    import hashlib
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except Exception:
        return None


def _current_tasks_unified_sha(bench):
    """sha256 of the LIVE tasks_unified.jsonl that _remap reads (checkpoint dimension/subdimension/WEIGHT).
    Editing it changes report scores WITHOUT changing the runner/ tree, so it is a distinct scoring-asset
    provenance key (P1-1). Same path resolution as _remap (_ROOT/<bench>/tasks_unified.jsonl)."""
    return _sha_file(os.path.join(_ROOT, bench, "tasks_unified.jsonl"))


REQUIRED_SOURCE_HASH_KEYS = ("raw_result_sha256", "trajectory_sha256", "task_json_sha256",
                             "checkpoint_set_sha256")


def _source_provenance_audit(agent_dir, bench):
    """P0/P1 SOURCE PROVENANCE. The rescored Governance is an overlay on a bundle whose raw inputs
    (result.json / trajectory.jsonl / task.json / the rescored CHECKPOINT SET / deliverable files) could be
    edited AFTER rescoring -- leaving a stale artifact the runner-tree guard cannot detect. Each block records
    Governance.input_provenance{...sha256}; here we RECOMPUTE from the LIVE files and compare. PURE READ.

    COMPLETENESS IS REQUIRED (P1): 'current' demands EVERY task (n_expected) carry a COMPLETE input_provenance
    (all REQUIRED_SOURCE_HASH_KEYS) that MATCHES. A task with no / partial provenance makes the bundle
    incomplete_input_provenance (NOT current) -- absence is 'unverified', never 'verified'. Same for the
    per-task tasks_unified_sha256."""
    import governance_contract as _gc                      # shared checkpoint_set_sha (one formula)
    live_tu = _current_tasks_unified_sha(bench)
    task_rps = sorted(glob.glob(os.path.join(agent_dir, "*", "result.json")))
    n_expected = len(task_rps)
    per_task = {}
    n_with_ip = n_complete_ip = n_src_match = 0
    n_tu_recorded = n_tu_match = 0
    for rp in task_rps:
        tid = os.path.basename(os.path.dirname(rp))
        bdir = os.path.dirname(rp)
        blk, _ = _read_dim_block(bdir, "Governance")
        ip = (blk or {}).get("input_provenance") or {}
        tu_recorded = ((blk or {}).get("scoring_config") or {}).get("tasks_unified_sha256")
        if not ip:
            per_task[tid] = {"has_input_provenance": False, "complete": False,
                             "source_match": None, "tasks_unified_match": None}
            continue
        n_with_ip += 1
        missing_keys = [k for k in REQUIRED_SOURCE_HASH_KEYS if not ip.get(k)]
        complete = (len(missing_keys) == 0)
        n_complete_ip += 1 if complete else 0
        live = {
            "raw_result_sha256": _sha_file(os.path.join(bdir, "result.json")),
            "trajectory_sha256": _sha_file(os.path.join(bdir, "trajectory.jsonl")),
            "task_json_sha256": _sha_file(os.path.join(bdir, "task.json")),
        }
        # checkpoint set recomputed from the RESCORED bundle (what the report actually reads via _bundle_path),
        # INCLUDING status/score -> catches edits to result.rescored.json checkpoints, not just tasks_unified.
        try:
            _rb = json.load(open(_bundle_path(rp)))
        except Exception:
            _rb = {}
        live["checkpoint_set_sha256"] = _gc.checkpoint_set_sha(_rb.get("checkpoints"))
        mism = [k for k, v in live.items() if k in ip and ip.get(k) != v]
        # deliverable files the judge read (recorded paths -> re-hash the live files).
        deliv = ip.get("deliverable_files_sha256") or {}
        deliv_mism = [p for p, h in deliv.items() if _sha_file(os.path.join(bdir, p)) != h]
        src_ok = complete and (len(mism) == 0) and (len(deliv_mism) == 0)
        n_src_match += 1 if src_ok else 0
        tu_ok = None
        if tu_recorded is not None:
            n_tu_recorded += 1
            tu_ok = (tu_recorded == live_tu)
            n_tu_match += 1 if tu_ok else 0
        per_task[tid] = {"has_input_provenance": True, "complete": complete, "missing_keys": missing_keys,
                         "source_match": src_ok, "source_mismatch_keys": mism,
                         "deliverable_mismatch": deliv_mism, "tasks_unified_match": tu_ok}
    # ---- rollups REQUIRING FULL COVERAGE over n_expected (absence != verified). ----
    if n_expected == 0:
        source_status = "no_rescored_artifact"
    elif n_with_ip == 0:
        source_status = "no_input_provenance"
    elif n_with_ip < n_expected or n_complete_ip < n_with_ip:
        source_status = "incomplete_input_provenance"      # some task missing / partial provenance
    elif n_src_match == n_expected:
        source_status = "current"
    else:
        source_status = "stale_input_artifact"
    if n_expected == 0:
        task_asset_status = "no_rescored_artifact"
    elif n_tu_recorded == 0:
        task_asset_status = "no_task_asset_provenance"
    elif n_tu_recorded < n_expected:
        task_asset_status = "incomplete_task_asset_provenance"
    elif n_tu_match == n_expected:
        task_asset_status = "current"
    else:
        task_asset_status = "stale_task_asset"
    return {"per_task": per_task, "live_tasks_unified_sha256": live_tu,
            "source_bundle_status": source_status, "task_asset_status": task_asset_status,
            "n_expected_tasks": n_expected, "n_with_input_provenance": n_with_ip,
            "n_complete_input_provenance": n_complete_ip, "n_source_match": n_src_match,
            "n_tasks_unified_recorded": n_tu_recorded, "n_tasks_unified_match": n_tu_match}


def _governance_consistency(exp_panel, disk_gov, hd, source_prov=None):
    """Single-source-of-truth audit. Reconciles the governance views:
      - report harness Governance  = harness_seven.Governance.score (reportable-only canonical mean)
      - in-memory per-task verdicts = _gov_canon (what the headline was built from)
      - ON-DISK per-task verdicts   = result.rescored.json Governance blocks RE-READ from disk (disk_gov)
      - checkpoint_diagnostics      = the OLD tag-based per-task aggregate (explicitly labelled legacy)
    task item 8 (DISK CONSISTENCY): the report's per-dim Governance mean MUST equal the mean of the ON-DISK
    per-task reportable Governance scores, AND branch / scoring_version / code_sha / g14_weight must agree
    across the on-disk blocks -> emit governance_consistency{disk_equals_report, ...}. (The old check only
    compared the in-memory _gov_canon to itself.)"""
    hs = ((exp_panel or {}).get("harness_seven") or {}).get("Governance") or {}
    report_score = hs.get("score")
    gc = exp_panel.get("_gov_canon") or {}
    # in-memory reportable-only mean re-derived from the per-task canonical verdicts (idempotent check).
    _rep_scores = [v["score"] for v in gc.values()
                   if v.get("reportable") and isinstance(v.get("score"), (int, float))]
    canon_file_mean = round(sum(_rep_scores) / len(_rep_scores), 3) if _rep_scores else None
    diag = (((hd or {}).get("by_category") or {}).get("trustworthiness") or {}).get("Governance") or {}
    agree = (report_score == canon_file_mean)
    n_crit = sum(1 for v in gc.values() if v.get("critical"))
    # ---- task item 8: re-read ON DISK and assert report == disk + metadata agreement ----
    disk_gov = disk_gov or {}
    _disk_rep = [v["score"] for v in disk_gov.values()
                 if v.get("reportable") and isinstance(v.get("score"), (int, float))]
    disk_mean = round(sum(_disk_rep) / len(_disk_rep), 3) if _disk_rep else None
    # branch / scoring_version / code_sha / g14_weight uniform across the on-disk REPORTABLE blocks (these
    # gate the SCORE mean). _uniq -> reportable-only.
    def _uniq(field):
        vals = {disk_gov[t].get(field) for t in disk_gov
                if disk_gov[t].get("reportable") and disk_gov[t].get(field) is not None}
        return sorted(v for v in vals if v is not None)
    # PROVENANCE fields (tree hash / dirty / scoring_version / judge model / prompt version) must be uniform
    # across ALL scored blocks -- INCLUDING non-reportable / judge-failed tasks (P1-1). A task that produced
    # no score but was scored under a DIFFERENT tree / dirty worktree / judge still pollutes the artifact set,
    # and it silently escaped the reportable-only check before. _uniq_all -> every block that recorded a value.
    def _uniq_all(field):
        vals = {disk_gov[t].get(field) for t in disk_gov if disk_gov[t].get(field) is not None}
        return sorted(v for v in vals if v is not None)
    _branches, _svers = _uniq("branch"), _uniq("scoring_version")
    _shas, _weights = _uniq("code_sha"), _uniq("g14_weight")
    _trees = _uniq_all("scoring_code_tree_hash")
    _judges, _pvers = _uniq_all("judge_model"), _uniq_all("prompt_version")
    # ---- task item 2: PROVENANCE GUARD (P0-2 fix). The staleness criterion is the SCORING-CODE TREE HASH
    # (git rev-parse HEAD:runner), NOT the full-repo HEAD. The tree hash is a content hash of runner/ only,
    # so it is INVARIANT under artifact-only commits (committing res2_* result.rescored.json does not touch
    # runner/). This breaks the old code_sha==HEAD self-reference loop where every artifact-only commit
    # advanced HEAD and re-marked all committed artifacts stale. code_sha / _head are kept for human audit
    # (which commit produced the artifact) but no longer gate artifact_status. A clean worktree is still
    # required. Legacy artifacts that predate the tree hash fall back to the old code_sha==HEAD check. ----
    _head = _current_git_head()
    _tree_head = _current_scoring_tree_hash()
    # dirty over ALL scored blocks (not just reportable) -- a dirty worktree on any scored task is suspect.
    _dirty_flags = sorted({bool(disk_gov[t].get("dirty_worktree")) for t in disk_gov
                           if disk_gov[t].get("dirty_worktree") is not None})
    _any_dirty = any(_dirty_flags)
    _sha_uniform = len(_shas) <= 1                              # reported for audit; no longer gating
    _sha_matches_head = bool(_head) and bool(_shas) and _shas == [_head]
    if _trees:                                                 # new-style artifacts carry the tree hash
        _tree_uniform = len(_trees) <= 1
        _tree_matches_head = bool(_tree_head) and _trees == [_tree_head]
        _code_current = _tree_matches_head                    # tree hash is the criterion
    else:                                                      # legacy artifacts -> fall back to code_sha==HEAD
        _tree_uniform = True
        _tree_matches_head = None
        _code_current = (not _shas) or _sha_matches_head
    # P1-2: judge model + prompt VERSION must be uniform across all scored blocks (a panel that silently
    # switched judges mid-dataset is not a comparable scoring run). prompt_version (a template id) is the
    # right granularity -- NOT each task's full prompt_hash, which legitimately differs per task content.
    _judge_uniform, _pver_uniform = len(_judges) <= 1, len(_pvers) <= 1
    # ---- SPLIT STATUSES (P0/P1 naming clarity). Three orthogonal provenance axes + an overall rollup: ----
    #   scoring_code_status : runner/ tree == HEAD:runner, uniform, clean worktree, judges/prompt uniform
    #   task_asset_status   : recorded tasks_unified_sha256 == live file (from source audit)
    #   source_bundle_status: recorded input hashes == live bundle files  (from source audit)
    _sp = source_prov or {}
    if not _shas and not _trees:
        scoring_code_status = "no_rescored_artifact"
    elif _any_dirty or not _code_current or not _tree_uniform or not _judge_uniform or not _pver_uniform:
        scoring_code_status = "stale_code_version"
    else:
        scoring_code_status = "current"
    task_asset_status = _sp.get("task_asset_status", "no_task_asset_provenance")
    source_bundle_status = _sp.get("source_bundle_status", "no_input_provenance")
    # overall (P1 fix): 'current' demands ALL THREE axes be EXPLICITLY current. Absence of provenance is
    # 'unverified', NOT 'verified' -> it can never yield current. Precedence: nothing rescored -> that; any
    # stale_* axis -> stale wins (surface which); all three current -> current; otherwise (some no_*/incomplete
    # axis, none stale) -> incomplete_provenance (unverified, distinctly NOT current).
    _axes = [scoring_code_status, task_asset_status, source_bundle_status]
    _stale = [a for a in _axes if a.startswith("stale")]
    if scoring_code_status == "no_rescored_artifact" and all(a in ("no_rescored_artifact",) for a in _axes):
        overall_artifact_status = "no_rescored_artifact"
    elif _stale:
        overall_artifact_status = _stale[0]
    elif all(a == "current" for a in _axes):
        overall_artifact_status = "current"
    else:
        overall_artifact_status = "incomplete_provenance"
    # back-compat: artifact_status now mirrors the OVERALL rollup (was scoring-code only).
    _artifact_status = overall_artifact_status
    # metadata_agrees: reportable branch/version/weight uniform + tree uniform + current code + clean +
    # judge/prompt uniform + task-asset & source axes not stale.
    _meta_agree = bool(all(len(s) <= 1 for s in (_branches, _svers, _weights))
                       and _tree_uniform and _code_current and not _any_dirty
                       and _judge_uniform and _pver_uniform
                       and task_asset_status != "stale_task_asset"
                       and source_bundle_status != "stale_input_artifact")
    # disk_equals_report: the headline equals the freshly re-read on-disk reportable mean. When NO task has
    # been rescored yet (disk_mean is None AND report_score is None) the two trivially agree (both N/A).
    disk_equals_report = bool(report_score == disk_mean and _meta_agree)
    return {
        "report_harness_governance": report_score,
        "canonical_per_task_file_mean": canon_file_mean,
        "report_equals_canonical_file": agree,
        "n_reportable": len(_rep_scores), "n_critical_veto": n_crit,
        # task item 8 block
        "disk_equals_report": disk_equals_report,
        "disk_reportable_mean": disk_mean,
        "n_disk_reportable": len(_disk_rep),
        "metadata_agrees": _meta_agree,
        # provenance guard outputs. artifact_status = OVERALL rollup (back-compat); the three axes are split
        # out so 'current' has a precise meaning (P0/P1 naming clarity).
        "artifact_status": _artifact_status,
        "overall_artifact_status": overall_artifact_status,
        "scoring_code_status": scoring_code_status,
        "task_asset_status": task_asset_status,
        "source_bundle_status": source_bundle_status,
        "artifact_status_meaning": ("'current' iff scoring_code (runner/ tree == HEAD:runner, uniform judges/"
                                    "prompt, clean worktree) AND task assets (tasks_unified.jsonl unchanged) "
                                    "AND source bundle (result/trajectory/task unchanged) all verify."),
        "provenance_criterion": "scoring_code_tree_hash" if _trees else "code_sha_legacy_fallback",
        "current_scoring_tree_hash": _tree_head,
        "scoring_tree_matches_head": _tree_matches_head,
        "scoring_tree_uniform": _tree_uniform,
        "disk_scoring_tree_hashes": _trees,
        "disk_judge_models": _judges, "judge_model_uniform": _judge_uniform,
        "disk_prompt_versions": _pvers, "prompt_version_uniform": _pver_uniform,
        # audit-only (NOT used for artifact validity): which commit scored vs current repo HEAD.
        "scorer_commit_code_sha": _shas[0] if _shas else None,
        "current_repository_head": _head,
        "code_sha_matches_head": _sha_matches_head,
        "code_sha_matches_head_note": "audit only; NOT a validity criterion (artifact-only commits advance HEAD)",
        "code_sha_uniform": _sha_uniform,
        "dirty_worktree_any": _any_dirty,
        "live_tasks_unified_sha256": _sp.get("live_tasks_unified_sha256"),
        "n_source_match": _sp.get("n_source_match"), "n_with_input_provenance": _sp.get("n_with_input_provenance"),
        "disk_branches": _branches, "disk_scoring_versions": _svers,
        "disk_code_shas": _shas, "disk_g14_weights": _weights,
        "disk_not_rescored": sorted(t for t in disk_gov
                                    if (disk_gov[t].get("evaluation_error") == "not_rescored")),
        "legacy_checkpoint_diagnostics_mean": diag.get("mean"),
        "legacy_note": ("checkpoint_diagnostics.Governance is the OLD tag-based per-task aggregate "
                        "(NOT critical-veto aware, NOT subject-scope aware). The CANONICAL governance number "
                        "is harness_dimensions.Governance == disk_reportable_mean; they agree by construction "
                        "(the report reads the SAME on-disk result.rescored.json Governance blocks). "
                        "Diagnostics may differ and is retained only as a legacy cross-check."),
        "canonical_path": "result.rescored.json -> Governance (SHARED CONTRACT per-dimension block)"}


# ============================================================================== P1-2 PAIRED MODEL COMPARISON
def _per_task_reportable_scores(agent_dir, bench, metric="native_success"):
    """Per-task {task_id: (score, reportable)} for the requested metric, the unit of the paired comparison.
      metric='native_success'  -> source-outcome correctness (PB success / MedCTA GAcc / HAB success), the
                                  headline a final table reports; reportable = the task was actually evaluated
                                  (evaluation_status complete/partial, not error/not_evaluated).
      metric='governance'       -> the CANONICAL per-task Governance (critical-veto aware) + its reportability.
    A task that errored / produced no eligible score is reportable=False; it is excluded from the mean but
    still counts in n_tasks -> reportability_rate exposes how thin the reportable subset is."""
    out = {}
    if metric == "governance":
        for rp in sorted(glob.glob(os.path.join(agent_dir, "*", "result.json"))):
            tid = os.path.basename(os.path.dirname(rp))
            bdir = os.path.dirname(rp)
            # SOLE TRUTH: the shared-contract top-level "Governance" block. The legacy canonical.governance
            # fallback is REMOVED -- a paired comparison must never silently read a stale shadow score. A
            # missing / invalid top-level block -> (None, False) = N/A, exactly like the main report. NO call.
            gblk, _err = _read_dim_block(bdir, "Governance")
            if gblk and isinstance(gblk.get("score"), (int, float)):
                out[tid] = (gblk["score"], bool(gblk.get("reportable")))
            else:
                out[tid] = (None, False)
        return out
    # native_success
    for rp in sorted(glob.glob(os.path.join(agent_dir, "*", "result.json"))):
        tid = os.path.basename(os.path.dirname(rp))
        try:
            r = json.load(open(_bundle_path(rp)))
        except Exception:
            out[tid] = (None, False); continue
        es = r.get("evaluation_status")
        # native_success unit via the SINGLE SOURCE OF TRUTH native_task_outcome() (binary; MedCTA GAcc>=0.5,
        # PB/HAB all-Outcome-checkpoints-passed) -- NOT r['success']. Headline and paired comparison agree.
        if bench == "MedCTA":
            o = native_task_outcome(r, bench)           # MedCTA signal is GAcc, independent of es
            out[tid] = ((1.0 if o else 0.0), True) if o is not None else (None, False)
        else:
            if es in ("complete", "partial", "proxy_partial"):
                o = native_task_outcome(r, bench)
                out[tid] = ((1.0 if o else 0.0), True) if o is not None else (None, False)
            else:
                out[tid] = (None, False)        # error / not_evaluated -> not reportable
    return out


def compare_models(agent_dir_a, agent_dir_b, bench, label_a=None, label_b=None, metric="native_success"):
    """P1-2: per-dataset PAIRED model comparison. With 5-task subsets, WHO enters the mean shifts the number
    ~20%, so we surface three views instead of one:
      paired_common_task_score : mean over the SAME task ids reportable in BOTH models (apples-to-apples)
      all_task_score           : each model's own reportable mean (its full reportable subset)
      reportability_rate       : n_reportable / n_tasks per model (how thin/biased the subset is)
    Returns a dict ready to drop into a report panel / final table."""
    la = label_a or os.path.basename(agent_dir_a.rstrip("/"))
    lb = label_b or os.path.basename(agent_dir_b.rstrip("/"))
    A = _per_task_reportable_scores(agent_dir_a, bench, metric)
    B = _per_task_reportable_scores(agent_dir_b, bench, metric)
    def _mean(d):
        v = [s for (s, rep) in d.values() if rep and isinstance(s, (int, float))]
        return round(sum(v) / len(v), 3) if v else None
    def _rep_ids(d):
        return {t for t, (s, rep) in d.items() if rep and isinstance(s, (int, float))}
    rep_a, rep_b = _rep_ids(A), _rep_ids(B)
    common = sorted(rep_a & rep_b)
    pa = round(sum(A[t][0] for t in common) / len(common), 3) if common else None
    pb = round(sum(B[t][0] for t in common) / len(common), 3) if common else None
    n_tasks = len(set(A) | set(B))
    # P1-2 CONTRACT COMPATIBILITY: a paired comparison is valid ONLY if both models were scored under the
    # SAME scoring contract. For governance, compare scoring_version / scoring_code_tree_hash / g14_weight /
    # judge model / prompt version across the two bundles. If they diverge (or vary within a model), the
    # paired number compares apples-to-oranges -> NULL the paired score and surface why; never report it.
    comp_status, comp_detail = "compatible", {}
    if metric == "governance":
        ga, gb = _read_disk_governance(agent_dir_a), _read_disk_governance(agent_dir_b)
        def _uval(g, f):
            return sorted({g[t].get(f) for t in g if g[t].get(f) is not None})
        def _has_none(g, f):                               # a REPORTABLE task missing this contract field
            return any(g[t].get("reportable") and g[t].get(f) is None for t in g)
        # contract fields now INCLUDE tasks_unified_sha256 (the task weights/dimension mapping aggregate uses).
        for f in ("scoring_version", "scoring_code_tree_hash", "g14_weight", "judge_model", "prompt_version",
                  "tasks_unified_sha256"):
            va, vb = _uval(ga, f), _uval(gb, f)
            if len(va) > 1 or len(vb) > 1 or va != vb or _has_none(ga, f) or _has_none(gb, f):
                comp_status = "incompatible_scoring_contract"
                comp_detail[f] = {la: va, lb: vb}
        # BOTH bundles must be source/asset CURRENT -- a paired number over a stale/incomplete artifact is void.
        sa, sb = _source_provenance_audit(agent_dir_a, bench), _source_provenance_audit(agent_dir_b, bench)
        for nm, aud in ((la, sa), (lb, sb)):
            if aud.get("source_bundle_status") != "current" or aud.get("task_asset_status") != "current":
                if comp_status == "compatible":
                    comp_status = "stale_or_incomplete_artifact"
                comp_detail.setdefault("artifact_status", {})[nm] = {
                    "source_bundle_status": aud.get("source_bundle_status"),
                    "task_asset_status": aud.get("task_asset_status")}
    _descriptive_only = (comp_status != "compatible")
    if _descriptive_only:
        pa = pb = None                                     # paired score is VOID under an invalid contract
    return {
        "bench": bench, "metric": metric,
        "models": [la, lb],
        "n_tasks": n_tasks,
        "comparison_status": comp_status, "contract_incompatibilities": comp_detail,
        "all_task_score_descriptive_only": _descriptive_only,
        "paired_common_task_ids": common, "n_paired_common": len(common),
        "paired_common_task_score": {la: pa, lb: pb},
        "all_task_score": {la: _mean(A), lb: _mean(B)},
        "reportability_rate": {
            la: round(len(rep_a) / (len(A) or 1), 3), lb: round(len(rep_b) / (len(B) or 1), 3)},
        "n_reportable": {la: len(rep_a), lb: len(rep_b)},
        "note": ("paired_common_task_score compares the SAME reportable task ids in both models "
                 "(apples-to-apples); all_task_score is each model's own reportable mean; they DIVERGE when "
                 "reportability differs across models -- always show both, never collapse to one number. "
                 "paired score is NULLED when comparison_status != compatible (incompatible scoring contract).")}


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
    ap.add_argument("--compare", default=None,
                    help="P1-2: second agent_dir to PAIR against -> emits paired vs all-task comparison")
    ap.add_argument("--metric", default="native_success", choices=["native_success", "governance"])
    a = ap.parse_args()
    bench = a.bench or _guess_bench(a.agent_dir, None)
    if a.compare:
        # both reports must be built first so result.rescored.json (canonical governance) exists for both.
        build(a.agent_dir, bench)
        build(a.compare, bench)
        cmp = compare_models(a.agent_dir, a.compare, bench, metric=a.metric)
        print(json.dumps(cmp, indent=1, ensure_ascii=False))
        sys.exit(0)
    rep = build(a.agent_dir, bench)
    out = a.out or os.path.join(a.agent_dir, "report.json")
    json.dump(rep, open(out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    print("\n-> wrote", out, file=sys.stderr)
