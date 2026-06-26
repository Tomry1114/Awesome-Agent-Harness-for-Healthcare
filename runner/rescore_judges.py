#!/usr/bin/env python3
"""Canonical, post-hoc Governance RE-SCORER (no agent re-run, no GPU). This is the SINGLE place in the
harness allowed to call a judge model for Governance: it runs governance.governance(...) ONCE per task
bundle, applies the SHARED GOVERNANCE BLEND CONTRACT, and WRITES <task>/result.rescored.json's
`Governance` block with the FULL audit schema. Moving the judge call HERE lets aggregate_report.py be
PURE-READ again (it reads result.rescored.json['Governance'] instead of re-judging).

SHARED GOVERNANCE CONTRACT (the block this writer produces and aggregate consumes verbatim):
  result.rescored.json["Governance"] = {
    "score": float|None, "reportable": bool, "evaluation_error": str|None,
    "evidence_tier": "experimental_hybrid", "formal_analysis_eligible": False, "deterministic": False,
    "components": {"g1_g4_unified": float, "subject_binding_completion": float,
                   "cross_subject_exclusivity": 0|1|None},
    "submetrics": {"G1": {...}, "G2": {...}, "G3": {...}, "G4": {...}},
    "judge": {"model","prompt_version","prompt_hash","raw_response","parsed_response"},
    "output_extraction": {"source_fields": [...], "source_files": [...]},
    "scoring_config": {"g14_weight","subject_scope_weight","scoring_version","code_sha",
                       "dirty_worktree","judge_model","judge_prompt_hash"},   # from governance_contract
    "branch": "hab_unified_g1g4_blend"|"pb_unified_g1g4_blend"|"medcta_governance_4rule",
    "critical": bool, "critical_reason": str|None,
    "judge_independence": "independent"|"shared_model_with_agent_or_tool",
    "subject_scope_diagnostic": float|None   # scope-only value; NEVER used as a judge-fail fallback
  }

BLEND (imported from governance_contract -- the ONE blend implementation in the harness; this writer no
longer carries a private copy):
  governance_contract.blend_governance(gov, scope, gcps): critical (cross_subject_exclusivity breach OR a
  real hard policy breach from the unified governance critical set, EXCLUDING the GUI-substrate
  concealed_critical_failure artifact, OR a critical benchmark governance checkpoint -- a persisted
  cross_patient_access checkpoint DEFERS to the FRESH scope) -> 0.0; else
  g14_weight()*g1_g4_unified + (1-g14_weight())*subject_binding_completion.
JUDGE-FAILURE -> score=None, reportable=False, evaluation_error set; we NEVER fall back to scope-only as the
Governance SCORE (that is a different construct, saved only as subject_scope_diagnostic).

JUDGE-INDEPENDENCE (fail-closed): if the judge model id == the agent model id (exact identity, the harness
convention -- gpt-5.4 judging gpt-5.4-mini stays INDEPENDENT), refuse: score=None, reportable=False,
evaluation_error="judge_not_independent". judge_independence is recorded either way.

IDEMPOTENT + CACHED: the raw judge response is cached on disk keyed by
(task_id, output_hash, judge_model, prompt_hash). A second run with the cache present makes NO new model
call (the cached raw response is replayed into governance()). Cache lives under <agent_dir>/.judge_cache/.

DIMS PERSISTED: this writer persists ONLY the Governance block into result.rescored.json (merging onto any
pre-existing rescored layer). Context/Verification remain aggregate's responsibility (their judge_fn is
None / deterministic there) -- documented, not silently dropped.
"""
import json, os, sys, glob, hashlib, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import governance as _gov
import governance_contract as _contract   # SHARED CONTRACT: the ONLY blend/critical/g14_weight/config
import scoring as _scoring
import substrate as _sub
import gateway

PROMPT_VERSION = "governance-g1g4-v3"
SCORING_VERSION = _contract.SCORING_VERSION
DEFAULT_JUDGE = os.environ.get("MH_JUDGE_MODEL", "gpt-5.4")

# branch tag per benchmark (matches aggregate_report's _gov_canon branch vocabulary)
_BRANCH = {"PhysicianBench": "pb_unified_g1g4_blend",
           "HealthAdminBench": "hab_unified_g1g4_blend",
           "MedCTA": "medcta_governance_4rule"}


def _guess_bench(agent_dir):
    ids = " ".join(os.path.basename(os.path.dirname(p))
                   for p in glob.glob(os.path.join(agent_dir, "*", "result.json")))
    if "PB-" in ids:
        return "PhysicianBench"
    if "MCTA-" in ids or "MedCTA" in ids:
        return "MedCTA"
    if "HAB-" in ids:
        return "HealthAdminBench"
    return "Unknown"


def _agent_model_id(prov):
    """The agent's model id, normalized like run.py's _ind_str ('gpt-5.5 (api brain)' -> 'gpt-5.5')."""
    am = str((prov or {}).get("agent_model") or "").split(" (")[0].strip()
    return am or None


def _allowed_tool_names(task):
    out = []
    for t in (task.get("available_tools") or []):
        if isinstance(t, dict) and t.get("name"):
            out.append(t["name"])
        elif isinstance(t, str):
            out.append(t)
    return out or None


def _sha(s):
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


# ---- judge cache + a gateway.chat wrapper so governance()'s internal judge call is CACHED -------------
class _JudgeCache:
    """Disk cache for the ONE governance judge call per task. Key = (task_id, output_hash, judge_model,
    prompt_hash). prompt_hash is computed from the EXACT (system+user) messages governance() builds, so the
    cache is sound even though governance owns the prompt. A second run replays the cached raw response into
    governance() (no model call); the cache also records the prompt_hash actually used so the audit block can
    publish it."""
    def __init__(self, agent_dir, task_id, judge_model):
        self.dir = os.path.join(agent_dir, ".judge_cache")
        os.makedirs(self.dir, exist_ok=True)
        self.task_id = task_id
        self.judge_model = judge_model
        self.output_hash = None       # set per-task before the governance call
        self.hit = False              # did THIS task's judge call come from cache?
        self.prompt_hash = None       # prompt_hash actually used (cache key or live)
        self.raw_response = None      # raw judge content actually used

    def _path(self, prompt_hash):
        key = "|".join([self.task_id, self.output_hash or "", self.judge_model, prompt_hash])
        return os.path.join(self.dir, _sha(key) + ".json")

    def wrapped_chat(self, _orig_chat):
        def chat(messages, model, max_tokens=1024, judge=False, timeout=None, retries=None,
                 image_path=None, extra=None):
            # governance builds [{system},{user}]; key on the concatenated prompt content
            prompt_blob = "\n\n".join((m.get("content") if isinstance(m.get("content"), str) else
                                       json.dumps(m.get("content"), ensure_ascii=False)) for m in messages)
            ph = _sha(prompt_blob)
            self.prompt_hash = ph
            p = self._path(ph)
            if os.path.exists(p):
                try:
                    cached = json.load(open(p))
                    self.hit = True
                    self.raw_response = cached.get("content")
                    return {"ok": True, "content": cached.get("content"),
                            "error_type": None, "raw": (cached.get("content") or "")[:300]}
                except Exception:
                    pass
            res = _orig_chat(messages, model, max_tokens=max_tokens, judge=judge, timeout=timeout,
                             retries=retries, image_path=image_path, extra=extra)
            self.raw_response = res.get("content")
            if res.get("ok"):
                try:
                    json.dump({"content": res.get("content"), "model": model,
                               "task_id": self.task_id, "output_hash": self.output_hash,
                               "prompt_hash": ph, "judge_model": self.judge_model},
                              open(p, "w"), ensure_ascii=False)
                except Exception:
                    pass
            return res
        return chat


def _run_unified(evs, bench, task, prov, dp, manifest, bundle_dir, judge_model, cache):
    """Run governance.governance(...) ONCE for this task with the judge call routed through `cache`.
    The agent's REAL submitted output is extracted with the FULL extractor (PB reads the task-manifest's
    DECLARED deliverable path), and the REAL output_extraction provenance ({source_fields, source_files})
    is returned so the persisted Governance block reflects what was actually read -- not a static string.
    Returns (gov_dict_or_None, answer_used, extraction)."""
    policy = _gov._resolve_policy(bench)
    answer, extraction = _gov._agent_final_output_ex(
        evs, policy=policy, bundle_dir=bundle_dir, task_manifest=task)
    cache.output_hash = _sha(answer)
    _orig = gateway.chat
    gateway.chat = cache.wrapped_chat(_orig)
    # ensure governance uses the configured judge model + judge enabled
    _prev_model = os.environ.get("MH_JUDGE_MODEL")
    _prev_judge = os.environ.get("MH_GOV_JUDGE")
    os.environ["MH_JUDGE_MODEL"] = judge_model
    os.environ["MH_GOV_JUDGE"] = "1"
    try:
        gov = _gov.governance(
            evs, policy=bench, question=(task.get("goal") or ""), answer=answer,
            allowed_tools=_allowed_tool_names(task),
            provenance=(prov or {}).get("prompt_provenance"),
            dimension_policy=dp, manifest=manifest, use_judge=True, task_manifest=task)
    except Exception as e:
        gov = {"_error": repr(e)}
    finally:
        gateway.chat = _orig
        if _prev_model is None:
            os.environ.pop("MH_JUDGE_MODEL", None)
        else:
            os.environ["MH_JUDGE_MODEL"] = _prev_model
        if _prev_judge is None:
            os.environ.pop("MH_GOV_JUDGE", None)
        else:
            os.environ["MH_GOV_JUDGE"] = _prev_judge
    return gov, answer, extraction


def _submetric_blocks(gov):
    """Build the per-rule submetric audit (G1-G4) with evidence/judge_reason from the gov dict."""
    subs = (gov.get("submetrics") or {}) if isinstance(gov, dict) else {}
    g1d = gov.get("g1_detail") or {}
    g2d = gov.get("g2_detail") or {}
    raw = gov.get("judge_raw") or ""
    return {
        "G1": {"score": subs.get("G1_information_access_compliance"),
               "evidence": {"violations": g1d.get("violations"),
                            "detection_method": g1d.get("detection_method"),
                            "score_eligible": gov.get("g1_score_eligible"),
                            "excluded_score": gov.get("g1_excluded_score")}},
        "G2": {"score": subs.get("G2_provenance_honesty"),
               "evidence": {"violations": g2d.get("violations")}},
        "G3": {"score": subs.get("G3_scope_and_risk_boundary"), "judge_reason": raw},
        "G4": {"score": subs.get("G4_failure_handling_compliance"),
               "judge_reason": raw,
               "structured_failure_state": gov.get("structured_failure_state")},
    }


def _build_governance_block(bench, gov, scope, gcps, judge_model, agent_model, extraction,
                            cache, evaluation_error=None):
    """Assemble the FULL SHARED-CONTRACT Governance block. evaluation_error set -> fail-closed
    (score=None, reportable=False) regardless of any computed number."""
    # judge independence (exact-id convention, matching run.py._ind_str)
    independence = ("independent" if (judge_model and agent_model and judge_model != agent_model)
                    else ("shared_model_with_agent_or_tool" if agent_model else "unknown"))
    if independence == "shared_model_with_agent_or_tool" and not evaluation_error:
        evaluation_error = "judge_not_independent"

    g14 = gov.get("score") if isinstance(gov, dict) else None
    binding = None
    cross_excl = None
    scope_diag = None
    if isinstance(scope, dict):
        binding = scope.get("subject_binding_completion")
        if binding is None and isinstance(scope.get("score"), (int, float)):
            binding = scope.get("score")
        cross_excl = scope.get("cross_subject_exclusivity")
        scope_diag = scope.get("score")

    if evaluation_error:
        score, reportable, critical, critical_reason = None, False, False, None
    else:
        # SHARED CONTRACT: the ONLY blend implementation.
        score, reportable, critical, critical_reason = _contract.blend_governance(gov, scope, gcps)
        if score is None and critical is False:
            # blend returned a JUDGE FAILURE (g1-g4 not reportable) -> fail-closed
            evaluation_error = "judge_unavailable_or_unparseable"
            reportable = False

    block = {
        "score": score, "reportable": bool(reportable), "evaluation_error": evaluation_error,
        "evidence_tier": "experimental_hybrid", "formal_analysis_eligible": False, "deterministic": False,
        "components": {
            "g1_g4_unified": (round(float(g14), 3) if isinstance(g14, (int, float)) else None),
            "subject_binding_completion": (float(binding) if isinstance(binding, (int, float)) else None),
            "cross_subject_exclusivity": cross_excl},
        "submetrics": _submetric_blocks(gov if isinstance(gov, dict) else {}),
        "judge": {
            "model": judge_model, "prompt_version": PROMPT_VERSION,
            "prompt_hash": cache.prompt_hash,
            "raw_response": cache.raw_response,
            "parsed_response": {"G3": (gov.get("submetrics") or {}).get("G3_scope_and_risk_boundary")
                                if isinstance(gov, dict) else None,
                                "G4": (gov.get("submetrics") or {}).get("G4_failure_handling_compliance")
                                if isinstance(gov, dict) else None}},
        # REAL extraction returned by the governance extractor (PB: the declared deliverable path the task
        # actually named), not a static per-benchmark string.
        "output_extraction": (extraction if isinstance(extraction, dict)
                              else {"source_fields": [], "source_files": []}),
        # SHARED CONTRACT scoring_config: code_sha=git HEAD + dirty_worktree from runner/*.py porcelain.
        "scoring_config": _contract.scoring_config(judge_model, cache.prompt_hash),
        "branch": _BRANCH.get(bench, "unknown"),
        "critical": bool(critical), "critical_reason": critical_reason,
        "judge_independence": independence,
        "subject_scope_diagnostic": (float(scope_diag) if isinstance(scope_diag, (int, float)) else None),
    }
    return block


def rescore(agent_dir, bench, judge_model=DEFAULT_JUDGE):
    plugin, problem = _sub.require_plugin(bench)
    if problem:
        raise SystemExit("no substrate plugin for bench %r: %r" % (bench, problem))
    n_written = n_cache_hits = n_judge_fail = n_refused = 0
    sample = None
    for rp in sorted(glob.glob(os.path.join(agent_dir, "*", "result.json"))):
        bdir = os.path.dirname(rp)
        tid = os.path.basename(bdir)
        traj = os.path.join(bdir, "trajectory.jsonl")
        if not os.path.exists(traj):
            continue
        try:
            evs = [json.loads(l) for l in open(traj) if l.strip()]
        except Exception:
            continue
        res = {}
        try:
            res = json.load(open(rp))
        except Exception:
            pass
        prov = res.get("provenance") or {}
        task = {"source_benchmark": bench, "task_id": tid}
        tpath = os.path.join(bdir, "task.json")
        if os.path.exists(tpath):
            try:
                task = json.load(open(tpath))
            except Exception:
                pass
        dp = _sub.dimension_policy(task, plugin)
        manifest = _sub.capability_manifest(prov)
        scope = _scoring.governance_subject_scope(evs, dp, task)
        gcps = [c for c in (res.get("checkpoints") or []) if c.get("dimension") == "Governance"
                and c.get("checkpoint_status") in ("passed", "failed") and c.get("score_eligible")]
        agent_model = _agent_model_id(prov)

        cache = _JudgeCache(agent_dir, tid, judge_model)
        # JUDGE-INDEPENDENCE fail-closed: refuse BEFORE spending a model call.
        refused = bool(agent_model and judge_model == agent_model)
        if refused:
            gov = {}
            cache.output_hash = None
            eval_err = "judge_not_independent"
            # still record WHAT would have been read for the judge (real extractor, no model call)
            _ans, extraction = _gov._agent_final_output_ex(
                evs, policy=_gov._resolve_policy(bench), bundle_dir=bdir, task_manifest=task)
            n_refused += 1
        else:
            gov, _ans, extraction = _run_unified(
                evs, bench, task, prov, dp, manifest, bdir, judge_model, cache)
            eval_err = None
            if cache.hit:
                n_cache_hits += 1

        block = _build_governance_block(bench, gov, scope, gcps, judge_model, agent_model,
                                        extraction, cache, evaluation_error=eval_err)
        if block.get("evaluation_error"):
            if block["evaluation_error"] == "judge_not_independent" and refused:
                pass
            else:
                n_judge_fail += 1

        # MERGE onto any existing rescored layer (preserve other agents' blocks); never touch raw result.json
        rescored_path = os.path.join(bdir, "result.rescored.json")
        merged = {}
        if os.path.exists(rescored_path):
            try:
                merged = json.load(open(rescored_path))
            except Exception:
                merged = {}
        if not merged:
            merged = dict(res)            # seed from raw so the rescored file is a complete bundle
        merged["Governance"] = block
        merged.setdefault("_rescore_dims_persisted", [])
        if "Governance" not in merged["_rescore_dims_persisted"]:
            merged["_rescore_dims_persisted"].append("Governance")
        json.dump(merged, open(rescored_path, "w"), indent=1, ensure_ascii=False)
        n_written += 1
        if sample is None:
            sample = (tid, block)
    return {"written": n_written, "cache_hits": n_cache_hits, "judge_fail": n_judge_fail,
            "refused_non_independent": n_refused, "sample": sample}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Canonical post-hoc Governance re-scorer (writes "
                                             "result.rescored.json['Governance']).")
    ap.add_argument("agent_dir")
    ap.add_argument("--bench", default=None)
    ap.add_argument("--judge-model", default=DEFAULT_JUDGE)
    a = ap.parse_args()
    bench = a.bench or _guess_bench(a.agent_dir)
    out = rescore(a.agent_dir, bench, judge_model=a.judge_model)
    print("bench=%s judge=%s" % (bench, a.judge_model))
    print("wrote Governance into %d bundle(s); cache_hits=%d judge_fail=%d refused_non_independent=%d"
          % (out["written"], out["cache_hits"], out["judge_fail"], out["refused_non_independent"]))
    if out["sample"]:
        tid, blk = out["sample"]
        print("sample task:", tid)
        print(json.dumps(blk, ensure_ascii=False, indent=1)[:1800])
