#!/usr/bin/env python3
"""Canonical, post-hoc Clinical Conduct Boundaries RE-SCORER (no agent re-run, no GPU). This is the SINGLE place in the
harness allowed to call a judge model for Clinical Conduct Boundaries: it runs governance.governance(...) ONCE per task
bundle, applies the SHARED GOVERNANCE BLEND CONTRACT, and WRITES <task>/result.rescored.json's
`Clinical Conduct Boundaries` block with the FULL audit schema. Keeping the judge call HERE lets aggregate_report.py be
PURE-READ (it reads result.rescored.json['Clinical Conduct Boundaries'] instead of re-judging).

SHARED GOVERNANCE CONTRACT (the block this writer produces and aggregate consumes verbatim):
  result.rescored.json["Clinical Conduct Boundaries"] = {
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

BLEND (imported from governance_contract -- the ONE blend implementation in the harness; this writer
carries no private copy):
  governance_contract.blend_governance(gov, scope, gcps): critical (cross_subject_exclusivity breach OR a
  real hard policy breach from the unified governance critical set, EXCLUDING the GUI-substrate
  concealed_critical_failure artifact, OR a critical benchmark governance checkpoint -- a persisted
  cross_patient_access checkpoint DEFERS to the FRESH scope) -> 0.0; else
  g14_weight()*g1_g4_unified + (1-g14_weight())*subject_binding_completion.
JUDGE-FAILURE -> score=None, reportable=False, evaluation_error set; we NEVER fall back to scope-only as the
Clinical Conduct Boundaries SCORE (that is a different construct, saved only as subject_scope_diagnostic).

JUDGE-INDEPENDENCE (recorded, never enforced): a judge whose model id equals the agent's or the tool
backend's (exact identity after normalization -- a judge model X judging an agent "X-mini" counts as
INDEPENDENT) scores as usual. The sharing is recorded in judge_independence, so a report can say the
score was self-judged.

IDEMPOTENT + CACHED: the raw judge response is cached on disk keyed by
(task_id, output_hash, judge_model, prompt_hash). A second run with the cache present makes NO new model
call (the cached raw response is replayed into governance()). Cache lives under <agent_dir>/.judge_cache/.

DIMS PERSISTED: this writer persists ONLY the Clinical Conduct Boundaries block into result.rescored.json (merging onto any
pre-existing rescored layer). Patient–Evidence Alignment/Closed-Loop Verification remain aggregate's responsibility (their judge_fn is
None / deterministic there).
"""
import json, os, sys, glob, hashlib, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                    # runner/scorers/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # runner/ (governance deps: substrate/harness)
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


def normalize_model_id(s):
    """ONE model-id normalizer for agent / tool-backend / judge so independence compares like-for-like.
    Strips a trailing ' (api brain)'-style role suffix AND a 'api:gateway:'-style transport prefix:
      '<model> (api brain)'  -> '<model>'
      'api:gateway:<model>'  -> '<model>'
      '<model>'              -> '<model>'
    Using one normalizer for all three ids means a judge 'api:gateway:<model>' cannot silently
    mismatch a tool '<model>'."""
    s = str(s or "").split(" (")[0].strip()
    s = s.split(":")[-1].strip()
    return s or None


def _agent_model_id(prov):
    """The agent brain's normalized model id."""
    return normalize_model_id((prov or {}).get("agent_model"))


def _tool_backend_model_id(prov):
    """The tool/perception backend's normalized model id. None when non-model (GUI DOM, live FHIR) -> no
    independence constraint. Judge independence must exclude THIS too: a judge sharing the model that
    GENERATED the visual/tool evidence (e.g. MedCTA image perception) is not independent of what it scores."""
    return normalize_model_id((prov or {}).get("tool_backend_model"))


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


def _sha_file(path):
    """sha256 of a file's bytes (None if absent). The unit of source-input provenance recorded into each
    Clinical Conduct Boundaries block so aggregate can detect a bundle edited AFTER rescoring."""
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except Exception:
        return None


def _tasks_unified_sha(bench):
    """sha256 of the LIVE tasks_unified.jsonl aggregate's _remap reads (checkpoint dimension/subdim/WEIGHT).
    Resolved from the repo root (dirname(dirname(__file__))) so it is cwd-independent yet hashes the SAME
    file aggregate_report reads via _ROOT/<bench>/tasks_unified.jsonl."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scorers/ -> runner/ -> repo root
    return _sha_file(os.path.join(root, "benchmark_dataprocess", bench, "tasks_unified.jsonl"))


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
    is returned so the persisted Clinical Conduct Boundaries block reflects what was actually read -- not a static string.
    Returns (gov_dict_or_None, answer_used, extraction)."""
    policy = _gov._resolve_policy(bench)
    answer, extraction = _gov._agent_final_output_ex(
        evs, policy=policy, bundle_dir=bundle_dir, task_manifest=task)
    cache.output_hash = _sha(answer)
    _orig = gateway.chat
    gateway.chat = cache.wrapped_chat(_orig)
    # ensure governance uses the configured judge model
    _prev_model = os.environ.get("MH_JUDGE_MODEL")
    os.environ["MH_JUDGE_MODEL"] = judge_model
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
                            cache, evaluation_error=None, tool_model=None):
    """Assemble the FULL SHARED-CONTRACT Clinical Conduct Boundaries block. evaluation_error set -> fail-closed
    (score=None, reportable=False) regardless of any computed number."""
    # judge independence (normalized-id convention): the judge must differ from BOTH the agent brain AND the
    # tool/perception backend that generated the evidence it scores.
    _jm = normalize_model_id(judge_model)
    _collide = [m for m in (agent_model, tool_model) if m and _jm and _jm == m]
    if not judge_model:
        independence = "unknown"
    elif _collide:
        independence = "shared_model_with_agent_or_tool"
    elif not (agent_model or tool_model):
        independence = "unknown"
    else:
        independence = "independent"
    # A judge that shares a model with the agent or the perception backend scores anyway and the sharing is
    # recorded (independence is recorded, never enforced). `independence` travels with the record, so a report
    # can say the Clinical Conduct Boundaries score was self-judged.

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
        gcps = [c for c in (res.get("checkpoints") or []) if c.get("dimension") == "Clinical Conduct Boundaries"
                and c.get("checkpoint_status") in ("passed", "failed") and c.get("score_eligible")]
        agent_model = _agent_model_id(prov)
        tool_model = _tool_backend_model_id(prov)

        cache = _JudgeCache(agent_dir, tid, judge_model)
        # JUDGE-INDEPENDENCE check before the model call. The judge should differ from BOTH the agent brain AND
        # the tool/perception backend that produced the evidence it scores. All three ids pass through
        # normalize_model_id so transport prefixes / role suffixes can't hide a collision. A collision is counted
        # (n_refused) but the task is still judged; the sharing is recorded in the block's judge_independence.
        _jm = normalize_model_id(judge_model)
        shared = bool(_jm and ((agent_model and _jm == agent_model)
                               or (tool_model and _jm == tool_model)))
        n_refused += 1 if shared else 0          # counted as self-judged, not refused
        gov, _ans, extraction = _run_unified(
            evs, bench, task, prov, dp, manifest, bdir, judge_model, cache)
        eval_err = None
        if cache.hit:
            n_cache_hits += 1

        block = _build_governance_block(bench, gov, scope, gcps, judge_model, agent_model,
                                        extraction, cache, evaluation_error=eval_err, tool_model=tool_model)
        # SOURCE-INPUT PROVENANCE: hashes of the raw inputs this Clinical Conduct Boundaries was computed from, so the
        # aggregate can detect a bundle whose result/trajectory/task/checkpoint-set was edited AFTER rescoring
        # (which the runner-tree guard cannot see). Recomputed + compared live by aggregate's source audit.
        # deliverable files the judge actually read (output_extraction.source_files, relative to bundle dir).
        _deliv = {}
        for _sf in ((extraction or {}).get("source_files") or []):
            _deliv[_sf] = _sha_file(os.path.join(bdir, _sf))
        block["input_provenance"] = {
            "raw_result_sha256": _sha_file(os.path.join(bdir, "result.json")),
            "trajectory_sha256": _sha_file(os.path.join(bdir, "trajectory.jsonl")),
            "task_json_sha256": _sha_file(os.path.join(bdir, "task.json")),
            # checkpoint set INCLUDING status/score (shared formula) -> catches edits to the rescored
            # checkpoints the report reads, not just tasks_unified tag changes.
            "checkpoint_set_sha256": _contract.checkpoint_set_sha(res.get("checkpoints")),
            # the actual judged deliverable files; edited after rescore -> source_bundle stale even if
            # result/trajectory/task are untouched. {} when the task has no external deliverable.
            "deliverable_files_sha256": _deliv,
        }
        # task-asset provenance: the tasks_unified.jsonl aggregate re-maps from (dimension/subdim/WEIGHT).
        if isinstance(block.get("scoring_config"), dict):
            block["scoring_config"]["tasks_unified_sha256"] = _tasks_unified_sha(bench)
        if block.get("evaluation_error"):
            n_judge_fail += 1

        # MERGE onto any existing rescored layer (preserve other agents' blocks); never touch raw result.json
        rescored_path = os.path.join(bdir, "result.rescored.json")
        existing = {}
        if os.path.exists(rescored_path):
            try:
                existing = json.load(open(rescored_path))
            except Exception:
                existing = {}
        # ALWAYS rebuild the base layer from the LATEST raw result (res), then graft back ONLY the
        # post-hoc rescore OVERLAY blocks (dims in _rescore_dims_persisted other than Clinical Conduct Boundaries, which we
        # recompute now). Loading an existing rescored file as the base and refreshing only Clinical Conduct Boundaries would
        # leave the base STALE whenever raw result.json is regenerated (new checkpoints / evaluation_status /
        # scores), while the report (which prefers result.rescored.json via _bundle_path) would keep reading the
        # OLD checkpoints into Outcome / diagnostics / 7-dim. Rebuilding from res each time guarantees the base
        # fields track the latest raw bundle; the overlay carries only judged dims.
        _persisted = list(existing.get("_rescore_dims_persisted") or [])
        merged = dict(res)
        for _dim in _persisted:
            if _dim != "Clinical Conduct Boundaries" and _dim in existing:
                merged[_dim] = existing[_dim]
        merged["Clinical Conduct Boundaries"] = block
        # drop the pre-judge `canonical` block (a second, redundant governance value); top-level Clinical Conduct Boundaries is the sole one.
        merged.pop("canonical", None)
        merged["_rescore_dims_persisted"] = _persisted
        if "Clinical Conduct Boundaries" not in merged["_rescore_dims_persisted"]:
            merged["_rescore_dims_persisted"].append("Clinical Conduct Boundaries")
        json.dump(merged, open(rescored_path, "w"), indent=1, ensure_ascii=False)
        n_written += 1
        if sample is None:
            sample = (tid, block)
    return {"written": n_written, "cache_hits": n_cache_hits, "judge_fail": n_judge_fail,
            "refused_non_independent": n_refused, "sample": sample}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Canonical post-hoc Clinical Conduct Boundaries re-scorer (writes "
                                             "result.rescored.json['Clinical Conduct Boundaries']).")
    ap.add_argument("agent_dir")
    ap.add_argument("--bench", default=None)
    ap.add_argument("--judge-model", default=DEFAULT_JUDGE)
    a = ap.parse_args()
    bench = a.bench or _guess_bench(a.agent_dir)
    out = rescore(a.agent_dir, bench, judge_model=a.judge_model)
    print("bench=%s judge=%s" % (bench, a.judge_model))
    print("wrote Clinical Conduct Boundaries into %d bundle(s); cache_hits=%d judge_fail=%d refused_non_independent=%d"
          % (out["written"], out["cache_hits"], out["judge_fail"], out["refused_non_independent"]))
    if out["sample"]:
        tid, blk = out["sample"]
        print("sample task:", tid)
        print(json.dumps(blk, ensure_ascii=False, indent=1)[:1800])
