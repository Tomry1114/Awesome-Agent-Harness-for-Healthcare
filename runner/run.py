#!/usr/bin/env python3
"""Unified Medical Harness runner (skeleton).

load unified task -> environment adapter -> agent loop -> unified trajectory -> scorer -> result JSON.
Runs end-to-end with the stub agent (no API key). native_pytest executes via subprocess pytest;
deterministic/llm_judge are skipped (skip_reason) pending B-line. Result validated vs spec/result.schema.json.
"""
import os, sys, json, glob, argparse, shutil, datetime, hashlib
import canonical_schema as _canon
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import environments, agents, scoring

def _next_feedback(hr, stage):
    """ADDITIVE structured feedback handed to the agent next turn. Pass the FULL harness feedback dict
    (suggested_capabilities, evidence_needed, avoid_capabilities, do_not_repeat, ... -- NOT a hand-picked
    subset) so the agent receives the actionable repair signal, plus the decision + stage. The richer the
    signal, the more the agent can hill-climb (harness-engineering: feedback quality is the lever)."""
    fb = dict(hr.feedback or {})
    fb["reason"] = fb.get("reason") or fb.get("message")
    fb["decision"] = hr.type
    fb["stage"] = stage
    return fb


def _resolve_git_sha():
    # the code SHA this run executed -> result.json provenance -> cmp_report flags a MIXED-SHA bundle.
    try:
        import subprocess
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=_root,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return os.environ.get("MH_GIT_SHA")
from pb_policy import DeliverableScaffold
from environments import _canon_fhir_tool


def _state_snapshot(env):
    """The STRUCTURED canonical state (env.state_summary() / full_state) — given to the harness so field-
    level postconditions (field_equals/object_exists/...) can be evaluated, not only a state-change hash.
    None for stateless envs."""
    try:
        return env.state_summary() if hasattr(env, "state_summary") else getattr(env, "full_state", None)
    except Exception:
        return None


def _state_hash(env):
    """Tamper-evident digest of the structured state — for the trajectory state_record (audit)."""
    st = _state_snapshot(env)
    if st is None:
        return None
    try:
        return hashlib.sha256(json.dumps(st, sort_keys=True, default=str).encode("utf-8", "replace")).hexdigest()[:12]
    except Exception:
        return None


def load_task(bench, task_id):
    p = os.path.join(ROOT, "benchmark_dataprocess", bench, "tasks_unified.jsonl")
    for line in open(p):
        t = json.loads(line)
        if t["task_id"] == task_id:
            return t
    raise SystemExit(f"task {task_id} not found in {p}")

def _truthy_env(name):
    """An env var is 'set/on' if present and not an explicit off value (empty/0/false/no/off)."""
    v = os.environ.get(name)
    return v is not None and v.strip().lower() not in ("", "0", "false", "no", "off")

def schema_strict_enabled():
    """Schema-validation strictness gate (V6 formal-run default). Strict (refuse to emit a result that
    fails spec/result.schema.json, including the jsonschema-missing fail-closed case) is ON when ANY of:
      - MH_SCHEMA_STRICT  (the explicit, pre-existing knob)
      - MH_FORMAL         (this is a formal benchmark run -> strict by default)
      - MH_BENCH_STRICT    (formal-run alias)
    Otherwise strict is OFF (exploratory: warn but still emit). Exposed as a function so conformance
    can test the wiring without re-deriving the env logic."""
    return any(_truthy_env(k) for k in ("MH_SCHEMA_STRICT", "MH_FORMAL", "MH_BENCH_STRICT"))

def validate_result(result):
    # FAIL CLOSED (V6): a missing jsonschema dependency means we CANNOT prove the result conforms to
    # spec/result.schema.json, so we must NOT silently treat it as valid. Return a structured
    # {valid: False, errors:[...]} (NOT a bare string that the caller coerces to valid=True). This makes
    # the MH_SCHEMA_STRICT / formal-run path correctly refuse to emit an unvalidated result.
    try:
        from jsonschema import Draft7Validator, RefResolver
    except Exception:
        return {"valid": False, "errors": ["jsonschema dependency missing"]}
    spec = os.path.join(ROOT, "spec"); store = {}
    for f in glob.glob(os.path.join(spec, "*.json")):
        s = json.load(open(f))
        if "$id" in s: store[s["$id"]] = s
    rs = json.load(open(os.path.join(spec, "result.schema.json")))
    v = Draft7Validator(rs, resolver=RefResolver(base_uri=rs["$id"], referrer=rs, store=store))
    errs = list(v.iter_errors(result))
    return {"valid": not errs, "errors": [f"{list(e.path)}:{e.message}" for e in errs[:8]]}

def reset_fhir(mode):
    """reset-mode none|restore_pristine|per_task. restore_pristine re-extracts the pristine H2 from
    the OCI layer and restarts (safe — NOT a hot-copy). Raises if the reset script fails (#2)."""
    if mode in (None, "none"):
        return
    if mode in ("restore_pristine", "per_task"):
        import subprocess
        subprocess.run(["bash", os.path.join(ROOT, "benchmark_dataprocess", "PhysicianBench",
                       "augmentation", "restore_pristine_h2.sh")], check=True)

def cleanup_stub(base):
    """Light per-task cleanup: delete agent-created resources tagged stub-run (avoids FHIR pollution
    without a full pristine restore). Returns {deleted, error}. Errors are surfaced, NOT swallowed,
    so a failed cleanup (= possible pollution) is visible (#6)."""
    import urllib.request, json as _j
    tag = "http://medical-harness/tags|stub-run"  # system|code to avoid deleting other stub-run tags
    n = 0; err = None
    for rt in ("MedicationRequest",):
        try:
            req = urllib.request.Request(f"{base}/{rt}?_tag={tag}&_count=200", headers={"Accept": "application/fhir+json"})
            b = _j.load(urllib.request.urlopen(req, timeout=30))
            for e in b.get("entry", []):
                rid = e.get("resource", {}).get("id")
                if rid:
                    urllib.request.urlopen(urllib.request.Request(f"{base}/{rt}/{rid}", method="DELETE"), timeout=30); n += 1
        except Exception as ex:
            err = repr(ex)
    return {"deleted": n, "error": err}

def run_task(bench, task_id, agent_name="stub", fhir_base=None, max_steps=12, job_dir=None, cleanup=True):
    # NOTE: in PhysicianBench the FHIR Patient.id == MRN, so context.patient_ref (an MRN) is also a
    # valid resource id and Patient/{patient_ref} resolves. If a future bench uses non-MRN ids,
    # split into patient_ref (resource id) + patient_identifier (MRN) before reusing this.
    task = load_task(bench, task_id)
    env_type = (task.get("environment") or {}).get("type")  # needed inside the step loop (canonical_action); was only set post-loop -> UnboundLocalError
    aug = os.path.join(ROOT, "benchmark_dataprocess", "PhysicianBench", "augmentation")
    pb_repo = os.path.join(ROOT, "benchmark", "PhysicianBench", "PhysicianBench")
    job_dir = job_dir or os.path.join("/tmp/mh_jobs", bench, task_id)
    shutil.rmtree(job_dir, ignore_errors=True)
    os.makedirs(os.path.join(job_dir, "workspace", "output"), exist_ok=True)
    os.makedirs(os.path.join(job_dir, "logs", "agent"), exist_ok=True)

    env = environments.make_env(task, fhir_base=fhir_base, aug_dir=aug,
                                workspace=os.path.join(job_dir, "workspace", "output"))
    env.reset()
    if getattr(env, "type", None) == "fhir":  # B: expose upstream granular FHIR tools (precise retrieval + native cp tool-name match)
        task["available_tools"] = environments.FHIR_GRANULAR_TOOLS
    agent = agents.make_agent(agent_name, task)

    # ---- Clinical Process Harness (Part 2), gated by MH_HARNESS_MODE (default 'off' -> _harness=None,
    #      ZERO behavior change). A harness build/runtime error must never break a run (fail open).
    _harness = None
    _harness_requested_mode = "off"     # the mode the operator REQUESTED (before any build failure)
    _harness_runtime_errors = []        # build/hook errors — surfaced in result["harness"]["runtime_errors"]
    _harness_build_failed = False
    try:
        import harness as _Harness
        _harness_requested_mode = _Harness.resolve_mode(os.environ.get("MH_HARNESS_MODE"))
        if _harness_requested_mode != "off":
            # substrate from env_type; a specific environment ADAPTER may be named per-task (two datasets of
            # the same substrate with different tool names) — task.environment.adapter, else the default.
            _adapter = (task.get("environment") or {}).get("adapter") if isinstance(task, dict) else None
            _tool_model = None
            if env_type == "tool_sandbox":   # perception tool backend (VLM) -> the ACTUAL model the tools call
                try:
                    import vlm_backend as _vb
                    _bk = _vb.get_backend()
                    _tool_model = getattr(_bk, "model", None) or getattr(_bk, "model_path", None)
                except Exception:
                    _tool_model = None
            _harness = _Harness.build_kernel(task, env_type=env_type, adapter=_adapter,
                                             mode=os.environ.get("MH_HARNESS_MODE"),
                                             agent_model=getattr(agent, "model", None), tool_model=_tool_model)
    except Exception as _he:
        _harness = None
        _harness_build_failed = True
        _harness_runtime_errors.append("build_kernel: %r" % _he)
    if _harness is not None and getattr(_harness, "contract", None) is not None and _harness.contract.meta is not None:
        try:
            from harness.engines.semantic import compile_goal_spec as _cgs
            if "goal_spec" not in _harness.contract.meta:
                _gspec = _cgs(_harness.contract.meta.get("goal"), _harness.contract.meta.get("public_context"),
                              getattr(_harness.ctx, "judge_fn", None))
                if _gspec:
                    _harness.contract.meta["goal_spec"] = _gspec
                _harness.contract.meta.setdefault("available_tools", task.get("available_tools"))
                _harness.contract.meta.setdefault("task_id", task.get("task_id"))
                # item 3: propagate the substrate manifest's repair_targets vocabulary (writable-path
                # equivalences) so the effect verifier can resolve a finding the judge mis-located.
                _harness.contract.meta.setdefault("repair_targets",
                    (getattr(_harness.ctx, "manifest", {}) or {}).get("repair_targets") or [])
        except Exception as _ge:
            _harness_runtime_errors.append("goal_spec: %r" % _ge)

    trajectory, last_obs, last_res, finished = [], None, None, False
    pending_harness_feedback = None   # CONTRACT(1): ADDITIVE harness feedback handed to the agent next turn
    max_repair_turns = int(os.environ.get("MH_MAX_REPAIR_TURNS", "4"))   # CONTRACT(2): cap on REVISE/BLOCK turns
    _repair_turns = 0
    _env_actions = 0   # CONTRACT(2): EXECUTED environment actions; repair turns do NOT count against this
    _insufficient_seen = {}   # CONTRACT: INSUFFICIENT grounding gets at most ONE revise PER evidence_version
    _answer_attempts = 0   # count EVERY final answer the agent submits (not just the one that lands)
    _repair_mode = os.environ.get("MH_REPAIR", "hard")   # ablation: off|hard|soft|select|full (answer layer)
    _pending_iv = None   # Unified PendingIntervention (answer surface): held root A + pre-intervention evidence baseline, awaiting candidate B
    _pending_artifact = None   # artifact surface: held root deliverable content A + evidence baseline, awaiting candidate B
    _reconcile = None   # P0.1: an UNKNOWN commit under restricted read-back recovery (action,res,budget)
    _forced_action = None   # harness-dispatched read-only acquisition, run through the NORMAL tool pipeline
    _plan_patched = set()   # Phase 3: deliverable paths already goal-completeness-patched (fire once each)
    _effect_done = set()   # Phase 4b: deliverable paths whose committed-order completion already ran
    _root_agent_content = {}   # Phase 4b PROVENANCE SEAL: path -> ORIGINAL agent content, captured BEFORE any harness patch
    _deliv_writes_used = 0   # at most ONE reserved over-budget deliverable write (off/enforce budget parity)
    deliv = DeliverableScaffold(task)  # PB deliverable scaffolding (Codex #1: extracted from the generic runner; no-op for non-PB)
    _fail_sig = None; _fail_n = 0; _aborted = False  # circuit breaker: abort on repeated identical failing call
    _last_tool_ev = None  # for consumed_by_agent backfill (#review: rendered != consumed)
    # FAIL-CLOSED: if assist/enforce was requested but the harness FAILED to build, do NOT run the agent
    # unprotected and admit it afterward — abort BEFORE the first action (the task is an infrastructure
    # error, excluded from mode comparison). observe/off tolerate a degraded/absent harness.
    _harness_infra_error = _harness_build_failed and _harness_requested_mode in ("assist", "enforce")
    if _harness_infra_error:
        trajectory.append({"step": 0, "event_type": "harness_infrastructure_error", "status": "error",
                           "requested_mode": _harness_requested_mode, "errors": list(_harness_runtime_errors)})
        _aborted = True
    if hasattr(env, "initial_observation"):
        _io = env.initial_observation()
        if _io is not None:
            last_res = _io; last_obs = json.dumps(_io, ensure_ascii=False)[:int(os.environ.get("MH_OBS_MAX_LEN", "10000"))]
    from harness.executor import ActionExecutor
    _executor = ActionExecutor(_canon, env_type, _state_hash, _state_snapshot)
    _recovery_orch = None   # C5b: RecoveryOrchestrator (lazy, per-task; episode registry persists across the task)
    for step in range(0 if _harness_infra_error else (max_steps + max_repair_turns + 2)):
        _env_budget_spent = _env_actions >= max_steps   # CONTRACT(2): tool budget used; still allow a final-only turn
        _bw = deliv.budget_warning(env, _env_actions, max_steps, trajectory)
        if _bw is not None: last_res, last_obs = _bw
        if _last_tool_ev is not None:          # the previous tool observation is about to be consumed
            _last_tool_ev["delivery_record"]["consumed_by_agent"] = True
            _last_tool_ev = None
        if _forced_action is not None:   # harness-dispatched read-only acquisition -> run via the NORMAL
            action = _forced_action; _forced_action = None   # tool pipeline (real tool_call, error/localization
            trajectory.append({"step": step, "event_type": "harness_acquire_dispatch",   # normalization,
                               "tool": action.get("tool"), "args": action.get("args"), "status": "ok"})
        else:
            action = agent.act({"goal": task.get("goal"), "context": task.get("context"),
                                "tools": env.available_tools(), "last_observation": last_obs, "last_result": last_res,
                                "harness_feedback": pending_harness_feedback})
            pending_harness_feedback = None   # consumed this turn
        if not isinstance(action, dict) or "type" not in action:  # #7 agent contract violation
            trajectory.append({"step": step, "event_type": "agent_error", "error": "invalid_action", "raw": str(action)[:200], "status": "error"})
            finished = True; break
        if action["type"] == "final":
            _pn = deliv.pre_final_nudge(env, step, trajectory)
            if _pn is not None:
                last_res, last_obs = _pn
                continue
            _answer_attempts += 1
            _ahash = hashlib.sha1((action.get("answer", "") or "").encode("utf-8")).hexdigest()[:12]
            trajectory.append({"step": step, "event_type": "answer_attempt", "attempt_index": _answer_attempts,
                               "answer_hash": _ahash, "answer": (action.get("answer", "") or "")[:500], "status": "ok"})
            if _pending_iv is not None and _harness is not None:   # consume this final as candidate B for the active intervention
                from harness.engines.semantic import evaluate_candidate   # STRENGTHENED AnswerRetention gate
                _A = _pending_iv.get("root_answer", ""); _B = action.get("answer", "")
                _cm = (_harness.contract.meta if (_harness.contract and _harness.contract.meta) else {})
                _base_ids = _pending_iv.get("baseline_ev_ids") or set()
                # evidence DELTA = validated, non-foreign evidence acquired AFTER the intervention started
                _new_ev = [e for e in _harness.ledger.evidence if e.get("evidence_id") not in _base_ids
                           and e.get("status") == "VALIDATED" and e.get("scope_relation") != "foreign"]
                try:
                    _decision, _disp = evaluate_candidate(
                        _pending_iv, _B, _cm, list(_harness.ledger.evidence), _new_ev,
                        judge_fn=getattr(_harness.ctx, "judge_fn", None),
                        reverify_fn=(lambda _ans: _harness.before_final(_ans, step=step)))
                except Exception as _re:
                    _harness_runtime_errors.append("evaluate_candidate: %r" % _re)
                    _decision, _disp = "KEEP_A", "kept_original_gate_error"
                if _decision == "ADOPT_B":
                    _chosen = _B; _rv_clean = True   # B already re-verified ALLOW + committed inside evaluate_candidate
                else:
                    _chosen = _A   # keep root A; commit it (flagged if not itself clean -> == off baseline)
                    _rv = None
                    try: _rv = _harness.before_final(_chosen, step=step)
                    except Exception as _re: _harness_runtime_errors.append("keepA re-verify: %r" % _re)
                    _rv_clean = (_rv is not None and _rv.type == "ALLOW")
                    if not _rv_clean:
                        try: _harness.record_flagged_final(_chosen, flag=_disp, step=step)
                        except Exception as _re: _harness_runtime_errors.append("record_flagged_final: %r" % _re)
                trajectory.append({"step": step, "event_type": "final_answer", "thought": _chosen, "status": "ok",
                                   "final_disposition": _disp, "intervention_type": _pending_iv.get("type"),
                                   "answer_attempt_index": _answer_attempts, "reverified": _rv_clean,
                                   "canonical_action": _canon.canonical_action({"type": "final", "answer": _chosen}, env_type)})
                _pending_iv = None; finished = True; break
            if _harness is not None:                    # final answer is a commit point
                try:
                    _hf = _harness.before_final(action.get("answer", ""), step=step)
                except Exception as _he:
                    _hf = None
                    _harness_runtime_errors.append("before_final: %r" % _he)
                    if _harness_requested_mode == "enforce":
                        trajectory.append({"step": step, "event_type": "harness_escalation", "stage": "before_final",
                                           "reason": "hook_error", "status": "error"}); _aborted = True; break
                if _hf is not None:
                    trajectory.extend(_hf.events)
                    if _hf.type == "ACQUIRE":   # ACTIVE read-only evidence acquisition: dispatch through the
                        # NORMAL tool pipeline (P0-1) -> a real tool_call event (scorers count it), error +
                        # localization normalization (P0-2: failed/fallback obs is NOT marked valid), after_action
                        # observation + provenance, and last_obs update -> the agent re-reasons on a NORMAL
                        # observation, not a truncated authority message (P0-6). Read-only tools only.
                        _na = ((getattr(_hf, "raw", None) and getattr(_hf.raw, "extra", None)) or {}).get("next_action") or {}
                        if _na.get("tool") and _na.get("read_only"):
                            _forced_action = {"type": "tool_call", "tool": _na["tool"], "args": _na.get("args") or {},
                                              "_harness_acquire": True}
                            try:
                                _harness.ledger.acquire_count = getattr(_harness.ledger, "acquire_count", 0) + 1
                            except Exception:
                                pass
                            if _pending_iv is None:   # PRE-acquisition baseline -> post-acquire B judged vs the evidence delta
                                _pending_iv = {"surface": "answer", "type": "ACQUIRE",
                                               "root_answer": action.get("answer", ""),
                                               "baseline_ev_ids": {e.get("evidence_id") for e in _harness.ledger.evidence}}
                            _aq = _na.get("args") or {}
                            pending_harness_feedback = {"decision": "ACQUIRE", "stage": "before_final",
                                "reason": "A targeted observation of %s (%s) was just performed for you; its full "
                                          "result is in your last observation. Reassess your answer using it, and "
                                          "change your answer only if it contradicts your current one."
                                          % (_aq.get("region"), _aq.get("attribute")), "missing_obligations": []}
                            continue   # the acquisition runs via the normal pipeline next iteration, then re-answer
                    if _hf.type in ("REVISE", "BLOCK"):   # ADDITIVE: keep last_obs/last_res (env state) intact
                        _extra_hf = ((getattr(_hf, "raw", None) and getattr(_hf.raw, "extra", None)) or {})
                        if _extra_hf.get("candidate"):       # Layer-2 candidate mode: keep A, ask for revised B
                            _repair_turns += 1
                            pending_harness_feedback = _next_feedback(_hf, "before_final")
                            if _repair_turns > max_repair_turns:   # no candidate produced in budget -> keep A
                                trajectory.append({"step": step, "event_type": "final_answer", "thought": action.get("answer", ""),
                                    "status": "ok", "verification_flag": "unverified_grounding",
                                    "final_disposition": "kept_original_no_candidate", "answer_attempt_index": _answer_attempts,
                                    "harness_feedback": _hf.feedback, "canonical_action": _canon.canonical_action(action, env_type)})
                                try: _harness.record_flagged_final(action.get("answer", ""), flag="unverified_grounding", step=step)
                                except Exception as _re: _harness_runtime_errors.append("record_flagged_final: %r" % _re)
                                finished = True; break
                            _pending_iv = {"surface": "answer", "type": "REVISE",
                                           "root_answer": action.get("answer", ""), "critique": _extra_hf.get("critique"),
                                           "baseline_ev_ids": {e.get("evidence_id") for e in _harness.ledger.evidence}}
                            continue
                        _repair_turns += 1
                        _insuff_exceeded = False
                        if getattr(getattr(_hf, "raw", None), "reason_code", None) == "insufficient_grounding":
                            _evk = (_harness.ledger.validated_evidence_version if _harness is not None else 0)
                            _insufficient_seen[_evk] = _insufficient_seen.get(_evk, 0) + 1
                            _insuff_exceeded = _insufficient_seen[_evk] > 1   # a 2nd insufficient on the SAME evidence
                        pending_harness_feedback = _next_feedback(_hf, "before_final")
                        _mr_viol = bool(((getattr(_hf, "raw", None) and getattr(_hf.raw, "extra", None)) or {}).get("must_resolve"))
                        _budget_out = _repair_turns > max_repair_turns
                        # MUST-RESOLVE (high-confidence, LOCALIZED evidence contradiction): a confirmed conflict
                        # is NOT eligible for unverified_grounding flagged delivery. Bounded repair; on
                        # exhaustion the original (refuted) answer is WITHHELD -> safe abstention, never the
                        # violating answer. (INSUFFICIENT/low-confidence keep their flagged-delivery path.)
                        if _mr_viol:
                            if _budget_out:
                                trajectory.append({"step": step, "event_type": "final_answer",
                                    "thought": "[WITHHELD] The answer asserted a claim refuted by the validated "
                                               "evidence and the conflict was not resolved; no safe answer "
                                               "could be committed.",
                                    "status": "ok", "verification_flag": "abstained_unresolved_contradiction",
                                    "final_disposition": "abstained_unresolved_violation",
                                    "answer_attempt_index": _answer_attempts, "harness_feedback": _hf.feedback,
                                    "canonical_action": _canon.canonical_action(action, env_type)})
                                try: _harness.record_flagged_final("[WITHHELD: unresolved evidence contradiction]",
                                                                   flag="abstained_unresolved_contradiction", step=step)
                                except Exception as _re: _harness_runtime_errors.append("record_flagged_final: %r" % _re)
                                finished = True; break
                            continue                     # within budget -> agent gets the targeted REVISE
                        # INSUFFICIENT -> ONE revise per evidence_version, else flagged delivery (advisory).
                        if _insuff_exceeded or _budget_out:
                            trajectory.append({"step": step, "event_type": "final_answer", "thought": action.get("answer", ""),
                                               "status": "ok", "verification_flag": "unverified_grounding",
                                               "final_disposition": "flagged_delivery",
                                               "answer_attempt_index": _answer_attempts,
                                               "harness_feedback": _hf.feedback,
                                               "canonical_action": _canon.canonical_action(action, env_type)})
                            try: _harness.record_flagged_final(action.get("answer", ""), flag="unverified_grounding", step=step)
                            except Exception as _re: _harness_runtime_errors.append("record_flagged_final: %r" % _re)
                            finished = True; break
                        continue
                    if _hf.type == "ESCALATE":
                        _xe = (getattr(_hf.raw, "extra", None) or {}) if getattr(_hf, "raw", None) is not None else {}
                        if not _xe.get("side_effecting"):   # CONTRACT(5): epistemic answer (no env side effect) -> DELIVER with flag
                            trajectory.append({"step": step, "event_type": "final_answer", "thought": action.get("answer", ""),
                                               "status": "ok", "verification_flag": _xe.get("verification_flag", "unresolved_risk"),
                                               "harness_feedback": _hf.feedback,
                                               "canonical_action": _canon.canonical_action(action, env_type)})
                            try: _harness.record_flagged_final(action.get("answer", ""), flag=_xe.get("verification_flag", "unresolved_risk"), step=step)
                            except Exception as _re: _harness_runtime_errors.append("record_flagged_final: %r" % _re)
                            finished = True; break
                        trajectory.append({"step": step, "event_type": "harness_escalation",   # operational write -> fail-closed
                                           "feedback": _hf.feedback, "status": "error"})
                        _aborted = True; break
            trajectory.append({"step": step, "event_type": "final_answer", "thought": action.get("answer", ""), "status": "ok",
                               "final_disposition": "clean_commit", "answer_attempt_index": _answer_attempts,
                               "canonical_action": _canon.canonical_action(action, env_type)})
            finished = True; break
        if action["type"] == "tool_call_truncated":  # cut-off tool call (e.g. oversized write_file) -> ask to re-issue
            trajectory.append({"step": step, "event_type": "agent_error", "error": "truncated_tool_call",
                               "raw": action.get("raw", "")[:200], "status": "error"})
            last_res = {"feedback": "Your previous tool call was CUT OFF before the JSON closed (content too long). "
                        "Re-issue the SAME write_file tool call but keep the content focused so the JSON completes."}
            last_obs = "truncated_tool_call"; continue
        if action["type"] != "tool_call" or not action.get("tool"):
            trajectory.append({"step": step, "event_type": "agent_error", "error": "bad_action_type", "raw": str(action)[:200], "status": "error"})
            finished = True; break
        if _reconcile is not None:                 # RESTRICTED recovery: only read/inspect until the commit is reconciled
            _rst = (_canon.canonical_action(action, env_type) or {}).get("semantic_type")
            if _rst in ("create", "update", "submit"):
                trajectory.append({"step": step, "event_type": "reconcile_block", "tool": action.get("tool"),
                                   "reason": "unconfirmed commit -- read back before any new write", "status": "ok"})
                last_res = {"feedback": "You have an UNCONFIRMED prior commit (it timed out). Use a READ/inspect "
                            "tool to read back whether it landed BEFORE any new write."}
                last_obs = "reconcile_pending"; continue
        _deliv_write = (deliv.is_required_write(action) and deliv._missing(env) and _deliv_writes_used < 1)
        if _env_budget_spent and _deliv_write:
            _deliv_writes_used += 1   # consume the single reserved deliverable slot (success or fail)
        if _env_budget_spent and not _deliv_write and not action.get("_harness_acquire"):   # past the tool budget: allow ONLY the one exact deliverable write (harness read-only acquisition exempt)
            _repair_turns += 1
            pending_harness_feedback = {"decision": "REVISE", "stage": "runtime_budget", "missing_obligations": [],
                "reason": "Environment-action budget is exhausted. Do NOT call another tool; give your best "
                          "final answer from the evidence already gathered."}
            if _repair_turns > max_repair_turns:
                trajectory.append({"step": step, "event_type": "tool_budget_exhausted", "status": "escalated"}); _aborted = True; break
            continue
        if _harness is not None:
            try:
                _hb = _harness.before_action(action, _state_snapshot(env), step=step)
            except Exception as _he:
                _hb = None
                _harness_runtime_errors.append("before_action: %r" % _he)
                if _harness_requested_mode == "enforce":   # enforce never proceeds past a harness failure
                    trajectory.append({"step": step, "event_type": "harness_escalation", "stage": "before_action",
                                       "reason": "hook_error", "status": "error"}); _aborted = True; break
            if _hb is not None:
                trajectory.extend(_hb.events)
                if _hb.type == "ACQUIRE":   # AMPLIFICATION: acquire a required read-only record BEFORE this
                    # commit, then defer the commit so the agent re-decides WITH the evidence. Runs via the NORMAL
                    # tool pipeline (real tool_call, provenance, last_obs update); read-only tools only.
                    _nab = ((getattr(_hb, "raw", None) and getattr(_hb.raw, "extra", None)) or {}).get("next_action") or {}
                    if _nab.get("tool") and _nab.get("read_only"):
                        _forced_action = {"type": "tool_call", "tool": _nab["tool"], "args": _nab.get("args") or {},
                                          "_harness_acquire": True}
                        try: _harness.ledger.acquire_count = getattr(_harness.ledger, "acquire_count", 0) + 1
                        except Exception: pass
                        pending_harness_feedback = {"decision": "ACQUIRE", "stage": "before_action",
                            "reason": "A required record (%s) was read for you; its result is in your next "
                                      "observation. Use it, then proceed with your action."
                                      % ((_nab.get("args") or {}).get("resourceType")), "missing_obligations": []}
                        trajectory.append({"step": step, "event_type": "harness_acquire_predispatch",
                                           "tool": _nab.get("tool"), "args": _nab.get("args"), "status": "ok"})
                        if _pending_artifact is None and action.get("tool") == "write_file" and (action.get("args") or {}).get("content"):
                            _pending_artifact = {"root_payload": action["args"]["content"],   # ArtifactRetention: hold A
                                                 "path": (action.get("args") or {}).get("path"),
                                                 "baseline_ev_ids": {e.get("evidence_id") for e in _harness.ledger.evidence}}
                        continue   # the acquisition runs next iteration; the original commit is deferred
                if _hb.type in ("REVISE", "BLOCK"):     # do NOT execute the tool; ADDITIVE feedback (env obs intact)
                    _repair_turns += 1
                    pending_harness_feedback = _next_feedback(_hb, "before_action")
                    if _repair_turns > max_repair_turns:
                        trajectory.append({"step": step, "event_type": "repair_budget_exhausted", "stage": "before_action", "status": "escalated", "reason": "repair_budget_exhausted"}); _aborted = True; break
                    continue
                if _hb.type == "ESCALATE":
                    trajectory.append({"step": step, "event_type": "harness_escalation",
                                       "feedback": _hb.feedback, "status": "error"})
                    _aborted = True; break
        _pending_auth = None   # C3: the mutation authorization the executor will dispatch for THIS action
        if (_harness is not None and _hb is not None and _hb.type == "ALLOW"
                and getattr(_harness.ledger, "pending_authorization", None) is not None):
            _cand_auth = _harness.ledger.pending_authorization
            if _harness.ledger.reserve_authorization(_cand_auth):   # AVAILABLE -> RESERVED (combined winner is ALLOW)
                _pending_auth = _cand_auth
            else:   # C3.1 fail-closed: a matched auth that is NOT AVAILABLE cannot be reserved -> do NOT dispatch/execute
                trajectory.append({"step": step, "event_type": "authorization_reserve_failed",
                                   "authorization_id": getattr(_cand_auth, "authorization_id", None),
                                   "auth_status": getattr(_cand_auth, "status", None), "status": "ok"})
                pending_harness_feedback = {"decision": "BLOCK", "stage": "authorization",
                                            "reason": "authorization could not be reserved (not AVAILABLE)", "missing_obligations": []}
                continue
        if (_pending_artifact is not None and _harness is not None and action.get("tool") == "write_file"
                and (action.get("args") or {}).get("content")
                and (action.get("args") or {}).get("path") == _pending_artifact.get("path")):
            # ARTIFACT PROMOTION (dual-surface): the deliverable is epistemic content (promote via the SAME
            # evidence-grounded gate as the answer surface) persisted operationally. Adopt the re-written plan B
            # ONLY if its core change is supported by the newly-acquired evidence delta; else retain root A.
            from harness.engines.semantic import evaluate_candidate
            _Aart = _pending_artifact.get("root_payload", ""); _Bart = action["args"]["content"]
            _cm = (_harness.contract.meta if (_harness.contract and _harness.contract.meta) else {})
            _base = _pending_artifact.get("baseline_ev_ids") or set()
            _newev = [e for e in _harness.ledger.evidence if e.get("evidence_id") not in _base
                      and e.get("status") == "VALIDATED" and e.get("scope_relation") != "foreign"]
            try:
                _adec, _adisp = evaluate_candidate({"type": "ACQUIRE", "root_answer": _Aart}, _Bart, _cm,
                                                   list(_harness.ledger.evidence), _newev,
                                                   judge_fn=getattr(_harness.ctx, "judge_fn", None), reverify_fn=None)
            except Exception as _ae:
                _harness_runtime_errors.append("evaluate_artifact: %r" % _ae); _adec, _adisp = "KEEP_A", "kept_original_gate_error"
            if _adec != "ADOPT_B":
                action["args"]["content"] = _Aart    # retain root plan A (B not evidence-justified)
            _chosen_content = action["args"].get("content", "")
            trajectory.append({"step": step, "event_type": "artifact_promotion", "path": _pending_artifact.get("path"),
                               "decision": _adec, "final_disposition": _adisp, "surface": "artifact",
                               "n_new_evidence": len(_newev),
                               "content_sha": hashlib.sha1((_chosen_content or "").encode("utf-8")).hexdigest()[:12],
                               "status": "ok"})
            _pending_artifact = None
        if (action.get("tool") == "write_file" and (action.get("args") or {}).get("content") is not None):
            _rp = (action.get("args") or {}).get("path")
            if _rp not in _root_agent_content:   # first write of this path = the agent's own content
                _root_agent_content[_rp] = action["args"]["content"]
        # PLAN COMPLETENESS (Phase 3): before the DELIVERABLE commit lands, fill any slot the PUBLIC GOAL
        # requires but the draft omits -- localized, append-only, slot-level-promoted. Oracle-blind (goal +
        # draft only). Gated on the manifest-declared commit (not scratch writes) + a per-path fire-once guard.
        if (_harness is not None and os.environ.get("MH_REPAIR", "hard") in ("soft", "select", "full")
                and os.environ.get("MH_PLAN_COMPLETENESS", "1") != "0"
                and action.get("tool") == "write_file" and (action.get("args") or {}).get("content")):
            _pc_sem = getattr(_harness.ctx, "sem", None)
            _pc_path = (action.get("args") or {}).get("path")
            if (_pc_sem is not None and _pc_sem.is_commit() and _pc_path not in _plan_patched):
                _plan_patched.add(_pc_path)
                try:
                    from harness.plan_completeness import compute_completeness_patch
                    _cm = (_harness.contract.meta if (_harness.contract and _harness.contract.meta) else {})
                    _pc = compute_completeness_patch(action["args"]["content"], _cm.get("goal"),
                                                     getattr(_harness.ctx, "judge_fn", None))
                except Exception as _pce:
                    _harness_runtime_errors.append("plan_completeness: %r" % _pce); _pc = None
                if _pc and _pc.get("applied"):
                    action["args"]["content"] = _pc["merged_content"]
                if _pc is not None:
                    trajectory.append({"step": step, "event_type": "plan_completeness_patch", "path": _pc_path,
                                       "applied": bool(_pc.get("applied")), "reason": _pc.get("reason"),
                                       "required": _pc.get("required"), "missing": _pc.get("missing"),
                                       "filled": _pc.get("filled"), "surface": "artifact",
                                       "content_sha": hashlib.sha1((action["args"].get("content") or "").encode("utf-8")).hexdigest()[:12],
                                       "status": "ok"})
        _outcome = _executor.execute_and_normalize(action, env, ledger=(_harness.ledger if _harness is not None else None), auth=_pending_auth)   # Commit B/C3
        res = _outcome.res; _err = _outcome.err; _recon = _outcome.recon
        _state_before = _outcome.state_before; _state_after = _outcome.state_after
        _snap_before = _outcome.snap_before; _snap_after = _outcome.snap_after
        _result_status = _outcome.result_status
        _env_actions += 1   # the environment call HAPPENED (count before any post-action escalate / circuit-break)
        if _reconcile is not None:                   # a read executed under recovery -> try to resolve the pending commit
            try:
                _rr = env.reconcile_write(_reconcile["action"]["tool"], _reconcile["action"].get("args", {}), _reconcile["res"])
            except Exception as _rrx:
                _rr = {"confirmed": None, "detail": "reconcile_error:%r" % (_rrx,)}
            if _rr and _rr.get("confirmed") is True:
                trajectory.append({"step": step, "event_type": "reconcile_resolved", "outcome": "committed", "detail": _rr.get("detail"), "status": "ok"}); _reconcile = None
            elif _rr and _rr.get("confirmed") is False:
                trajectory.append({"step": step, "event_type": "reconcile_resolved", "outcome": "failed", "detail": _rr.get("detail"), "status": "ok"})
                last_res = {"feedback": "Read-back confirms the prior commit did NOT land -- you may re-attempt it."}; _reconcile = None
            else:
                _reconcile["budget"] -= 1
                if _reconcile["budget"] <= 0:
                    trajectory.append({"step": step, "event_type": "harness_escalation", "stage": "reconcile", "reason": "reconcile_budget_exhausted", "status": "error"}); _aborted = True; break
        if _recon and _recon.get("confirmed") is not None:
            trajectory.append({"step": step, "event_type": "reconciliation", "tool": action["tool"],
                               "confirmed": _recon.get("confirmed"), "detail": _recon.get("detail"), "status": "ok"})
        _hpost = None
        if _harness is not None:
            try:
                _hpost = _executor.run_after_action(_harness, action, _outcome, step)
            except Exception as _he:
                _hpost = None
                _harness_runtime_errors.append("after_action: %r" % _he)
                if _harness_requested_mode == "enforce":
                    trajectory.append({"step": step, "event_type": "harness_escalation", "stage": "after_action",
                                       "reason": "hook_error", "status": "error"}); _aborted = True; break
        if _pending_auth is not None and _harness is not None:   # C3.1: finalize the DISPATCHED auth AFTER after_action + commit verification (never VERIFIED on a read-back alone)
            _ver = getattr(_harness.ctx, "verification", None)
            _aeff = (_hpost.type if _hpost is not None else "ALLOW")
            _cf = _recon.get("confirmed") if _recon is not None else None
            if _result_status == "failed" and _cf is False:
                _harness.ledger.fail_authorization(_pending_auth)             # explicit failure + read-back: NOT landed
            elif _result_status == "unknown" or _aeff == "RECONCILE" or _cf is None:
                _harness.ledger.unknown_authorization(_pending_auth)         # ambiguous -> reconcile only, never reuse
            elif _cf is True and _ver is True and _aeff == "ALLOW":
                _harness.ledger.verify_authorization(_pending_auth)          # landed AND commit verified AND no post-action veto
            elif _ver is False:
                _harness.ledger.fail_authorization(_pending_auth)            # commit verification says the effect is wrong
            else:
                _harness.ledger.unknown_authorization(_pending_auth)
        _ev, obs = _executor.build_event(action, _outcome, step)   # Commit B: single event builder
        trajectory.append(_ev)
        if _hpost is not None:
            trajectory.extend(_hpost.events)
            if _hpost.type == "RECONCILE":         # UNKNOWN commit -> RECOVER (read back), do NOT terminate
                _reconcile = {"action": action, "res": res, "budget": int(os.environ.get("MH_RECONCILE_BUDGET", "3"))}
                trajectory.append({"step": step, "event_type": "reconcile_enter", "stage": "after_action",
                                   "feedback": _hpost.feedback, "status": "ok"})
                pending_harness_feedback = _next_feedback(_hpost, "after_action")
            elif _hpost.type == "ESCALATE":        # retrospective escalation TERMINATES the run
                trajectory.append({"step": step, "event_type": "harness_escalation",
                                   "stage": "after_action", "feedback": _hpost.feedback, "status": "error"})
                _aborted = True; break
            if _hpost.type in ("REVISE", "BLOCK") and _hpost.feedback:   # fold into next obs (RESERVE room)
                # external channel routes the finding via act_fc as an external-reviewer claim; skip the
                # duplicate in-context "[HARNESS]" obs copy so the signal is not also posed as self-distrust.
                if os.environ.get("MH_REPAIR_CHANNEL", "inline").strip().lower() != "external":
                    _max = int(os.environ.get("MH_OBS_MAX_LEN", "10000"))
                    _htxt = "\n[HARNESS] " + json.dumps(_hpost.feedback, ensure_ascii=False)
                    obs = (_htxt[:_max] if len(_htxt) >= _max else obs[:_max - len(_htxt)] + _htxt)
                pending_harness_feedback = _next_feedback(_hpost, "after_action")
        # EFFECT COMPLETION (Phase 4b, GUARDED): realize an order the AGENT committed to in its ROOT deliverable
        # but never executed. Fail-CLOSED throughout. Runs ONLY after the deliverable write is CONFIRMED
        # (after_action processed, no RECONCILE/ESCALATE, read-back not failed). Provenance-sealed (ROOT agent
        # content only), governance-first via EvidenceState, existing-effect fail-closed, and the fhir_create
        # routes through before_action/after_action so the MutationAuthorization is a real execution boundary.
        if (_harness is not None and os.environ.get("MH_COMPLETE_EFFECT", "0") == "1" and not _err
                and _reconcile is None and (_recon is not None and _recon.get("confirmed") is True)   # #8 STRICT: only run when the deliverable write is POSITIVELY read-back-confirmed
                and action.get("tool") == "write_file" and (action.get("args") or {}).get("content")):
            _ec_sem = getattr(_harness.ctx, "sem", None)
            _ec_path = (action.get("args") or {}).get("path")
            if (_ec_sem is not None and _ec_sem.is_commit() and _ec_path not in _effect_done):
                _effect_done.add(_ec_path)
                try:
                    from harness.effect_completion import (context_refs, inspect_existing_effect,
                        build_order_resource, resource_type_for_category)
                    from harness.evidence_state import PRESENT, UNKNOWN
                    from harness.engines.semantic import extract_committed_orders
                    from harness.effect_reconciliation import is_realized
                    from harness.semantics import canonicalize
                    from harness.authorization import action_target_path
                    from harness.recovery_orchestrator import RecoveryOrchestrator, EffectCompletionKey
                    from harness.run_driver import RunDriver
                    if _recovery_orch is None:
                        _recovery_orch = RecoveryOrchestrator(RunDriver(_harness, _executor, env, task, _state_snapshot))
                    _recovery_orch.d.step = step; _recovery_orch.d.trajectory = trajectory
                    _root_content = _root_agent_content.get(_ec_path, action["args"].get("content"))   # PROVENANCE: root only
                    _cm = (_harness.contract.meta if (_harness.contract and _harness.contract.meta) else {})
                    _committed = extract_committed_orders(_root_content, _cm.get("goal"), getattr(_harness.ctx, "judge_fn", None))
                    _refs = context_refs(task)
                    _art_hash = hashlib.sha1((_root_content or "").encode("utf-8")).hexdigest()[:12]
                    if _committed and _refs.get("subject"):
                        for _u in _committed[:2]:
                            _otext = _u.get("text"); _ocat = _u.get("category") or "other"
                            _rt3 = resource_type_for_category(_ocat)
                            _insp = inspect_existing_effect(env, _rt3, _refs["subject"])   # fail-closed existing-effect probe
                            if _insp["state"] == UNKNOWN:
                                trajectory.append({"step": step, "event_type": "effect_completion_blocked", "surface": "state",
                                    "order_text": _otext, "resource_type": _rt3, "reason": "existing_effect_unknown", "status": "ok"}); continue
                            if _insp["state"] == PRESENT and is_realized(_otext, _insp["texts"]):
                                trajectory.append({"step": step, "event_type": "effect_completion", "surface": "state",
                                    "order_text": _otext, "resource_type": _rt3, "reason": "already_realized", "created_id": None, "status": "ok"}); continue
                            _rtb, _resource = build_order_resource({"text": _otext, "category": _ocat}, _refs)
                            if not _resource:
                                continue
                            _mact = {"type": "tool_call", "tool": "fhir_create", "args": {"resource": _resource}}
                            _fsem = canonicalize(_mact, getattr(_harness, "manifest", None) or {})
                            _scope = {"allowed_semantic_type": _fsem.semantic_type, "allowed_tool": "fhir_create",
                                      "allowed_effect": _fsem.effect, "target_path": action_target_path(_fsem, _mact),
                                      "expected_postcondition": {"resource": _rtb, "status": "active", "verify": "server_readback"}}
                            _key = EffectCompletionKey(_refs["subject"], _art_hash, (_otext or "")[:120].strip().lower(), _rtb)
                            # the orchestrator drives ACQUIRE-prereq (governance via RequiredContext, through the
                            # SAME executor -> binds evidence) + re-evaluate + authorized create + verify.
                            _res = _recovery_orch.realize(_mact, _scope, key=_key)
                            _harness.ledger.clear_mutation_hold()
                            trajectory.append({"step": step, "event_type": "effect_completion", "surface": "state",
                                "order_text": _otext, "category": _ocat, "resource_type": _rtb, "episode_state": _res.state,
                                "created_id": _res.created_id, "reason": _res.reason, "auth_status": _res.auth_status,
                                "prereq_rounds": _res.prereq_rounds, "status": ("ok" if _res.realized else "blocked")})
                except Exception as _ece:
                    _harness_runtime_errors.append("effect_completion: %r" % _ece)
                    try: _harness.ledger.clear_mutation_hold()
                    except Exception: pass
        _last_tool_ev = _ev   # mark consumed only if a later agent.act() runs (circuit-break -> stays False)
        if _err:
            _sig = (action["tool"], json.dumps(action.get("args", {}), sort_keys=True), _ev.get("error_type"))
            _fail_n = _fail_n + 1 if _sig == _fail_sig else 1; _fail_sig = _sig
            if _fail_n >= 3:  # same (tool,args,error) 3x -> stuck; abort instead of burning to max_steps (cf. upstream mini_agent)
                trajectory.append({"step": step, "event_type": "circuit_breaker", "error": "repeated_failing_call",
                                   "tool": action["tool"], "repeats": _fail_n, "status": "error"})
                _aborted = True; break
        else:
            _fail_sig = None; _fail_n = 0
        last_obs = obs; last_res = res
        if action.get("_harness_acquire") and _pending_artifact is not None and _harness is not None:
            # structured EvidenceDeltaPack: help the (weak) agent CONSUME the record it just read -> which plan
            # sections to revise vs preserve, before it re-writes the deliverable. Generic; not an answer.
            try:
                from harness.engines.semantic import build_evidence_delta_pack
                _pbase = _pending_artifact.get("baseline_ev_ids") or set()
                _pnew = [e for e in _harness.ledger.evidence if e.get("evidence_id") not in _pbase
                         and e.get("status") == "VALIDATED" and e.get("scope_relation") != "foreign"]
                _pack = build_evidence_delta_pack(_pending_artifact.get("root_payload", ""), _pnew,
                                                  (_harness.contract.meta or {}).get("goal") if _harness.contract else None,
                                                  judge_fn=getattr(_harness.ctx, "judge_fn", None))
                if _pack:
                    pending_harness_feedback = {"decision": "ACQUIRE", "stage": "evidence_delta_pack",
                                                "evidence_delta_pack": _pack, "missing_obligations": [],
                                                "reason": _pack.get("instruction")}
                    trajectory.append({"step": step, "event_type": "evidence_delta_pack",
                                       "affected_sections": _pack.get("affected_sections"),
                                       "preserve_sections": _pack.get("preserve_sections"), "status": "ok"})
            except Exception as _pe:
                _harness_runtime_errors.append("evidence_delta_pack: %r" % _pe)
    if not finished and not _aborted:  # #8 ran out of steps without a final answer (circuit-breaker abort logs its own event)
        trajectory.append({"step": max_steps, "event_type": "agent_error", "error": "max_steps_exceeded", "status": "error"})
    # deliverable enforcement (final): guarantee ONE write attempt if the required file is still missing
    # (covers both "finished early" and "ran out of steps"), then normalize a mis-named single file.
    deliv.enforce(env, agent, task, trajectory, max_steps, harness=_harness, state_snapshot=_state_snapshot)
    try:
        _caps = env.capabilities()   # Codex #10: four-state manifest captured while env is alive (pre-teardown)
    except Exception:
        _caps = {}
    env.teardown()

    # upstream-format trajectory.log for native_pytest
    with open(os.path.join(job_dir, "logs", "agent", "trajectory.log"), "w") as f:
        for ev in trajectory:
            if ev.get("event_type") == "tool_call":
                f.write(json.dumps({"timestamp": datetime.datetime.now().isoformat(), "type": "tool_call",
                                    "content": ev["tool"], "metadata": {"tool_name": _canon_fhir_tool(ev["tool"], ev.get("args")),
                                    "input": ev.get("args"), "output": (ev.get("result") if isinstance(ev.get("result"), str) else json.dumps(ev.get("result"), ensure_ascii=False))}}) + "\n")

    # text sources for recommendation/documentation safety
    final_texts = [ev.get("thought", "") for ev in trajectory if ev.get("event_type") == "final_answer"]
    note_texts = [ev.get("args", {}).get("content", "") for ev in trajectory
                  if ev.get("event_type") == "tool_call" and ev.get("tool") == "write_file"]
    outdir = os.path.join(job_dir, "workspace", "output")
    if os.path.isdir(outdir):
        for fn in os.listdir(outdir):
            try: note_texts.append(open(os.path.join(outdir, fn)).read())
            except Exception: pass
    # tool-call sequences for MedCTA ToolAcc/ArgAcc — normalize args (str|dict) on BOTH sides (#2)
    agent_tool_calls = [(ev["tool"], scoring.parse_args(ev.get("args", {}))) for ev in trajectory if ev.get("event_type") == "tool_call"]
    ref_tool_calls = []
    for ev in (task.get("reference") or {}).get("reference_trace", []) or []:
        if ev.get("role") == "assistant" and ev.get("tool_calls"):
            for tc in ev["tool_calls"]:
                fn = tc.get("function", {}); ref_tool_calls.append((fn.get("name"), scoring.parse_args(fn.get("arguments"))))
    # #5 lazy: only load PB drug-safety verifiers if a checkpoint needs them
    needs_pb = any((cp.get("check") or {}).get("verifier", "").startswith(("augmentation/drug_safety_check", "drug_safety_check"))
                   for cp in task.get("checkpoints", []))
    _judge_on = os.environ.get("MH_JUDGE", "").lower() in ("1", "qwen", "local", "on", "true")
    _judge_id = ("qwen3vl_judge:%s" % os.path.basename(os.environ.get("MH_VLM_PATH", "Qwen3-VL-2B-Instruct"))) if _judge_on else None
    _judge_fn = None
    if _judge_on:
        import judge_backend; _judge_fn = judge_backend.judge
    _mm_on = os.environ.get("MH_MM_JUDGE", "").lower() in ("1", "on", "true") or bool(os.environ.get("MH_MM_JUDGE_MODEL"))
    _mm_fn = None
    if _mm_on:
        import mm_judge_backend; _mm_fn = mm_judge_backend.judge_grounding
    _gacc_on = os.environ.get("MH_GACC", "").lower() in ("1", "on", "true") or bool(os.environ.get("MH_GACC_MODEL"))
    _gacc_fn = None
    if _gacc_on:
        import gacc_judge; _gacc_fn = gacc_judge.score
    # PB content checkpoints (upstream eval_helpers.llm_judge) need an OpenAI-compatible judge. Opt-in by
    # setting LLM_JUDGE_MODEL; auto-fill key/base from the gateway key file (default micuapi) so only the model is chosen.
    # (native_pytest propagates os.environ to the pytest subprocess.) Judge-model deviation -> passport.
    if os.environ.get("LLM_JUDGE_MODEL") and not os.environ.get("OPENAI_API_KEY"):
        _kf = os.path.expanduser("~/.xbai_key")
        if os.path.exists(_kf):
            os.environ["OPENAI_API_KEY"] = open(_kf).read().strip()
            _jb = (os.environ.get("MH_JUDGE_BASE") or os.environ.get("MH_OPENAI_BASE", "https://www.micuapi.ai")).rstrip("/")
            if _jb.endswith("/v1"): _jb = _jb[:-3].rstrip("/")
            os.environ.setdefault("OPENAI_BASE_URL", _jb + "/v1")
    # G3: prompt-source provenance -- record which AGENT-VISIBLE segments fed the agent and whether the
    # hidden reference (gold) leaked into any of them (system/user prompt path), so Governance G1 reads a real
    # access signal instead of guessing from answer<->gold similarity. reference is scorer-only and never put
    # in goal/context, so this is normally False; True means a real leak bug.
    _gold = str((task.get("reference") or {}).get("gold_answer") or "").strip()
    _vis = {"goal": str(task.get("goal") or ""),
            "task_context": json.dumps(task.get("context") or {}, ensure_ascii=False),
            "constraints": json.dumps(task.get("constraints") or {}, ensure_ascii=False)}
    _prompt_prov = {"system_sources": ["base_system_prompt", "benchmark_policy"],
                    "user_sources": list(_vis.keys()),
                    "segment_hashes": {k: hashlib.sha256(v.encode("utf-8", "replace")).hexdigest()[:12] for k, v in _vis.items()},
                    "hidden_reference_exposed_to_agent": bool(_gold) and len(_gold) >= 16 and any(_gold.lower() in v.lower() for v in _vis.values())}
    ctx = {"base": getattr(env, "base", None), "mrn": (task.get("context") or {}).get("patient_ref"),
           "trajectory": trajectory, "created_meds": scoring.created_meds(trajectory),
           "final_texts": final_texts, "note_texts": note_texts, "full_state": getattr(env, "full_state", None),
           "reference": task.get("reference", {}), "agent_tool_calls": agent_tool_calls, "ref_tool_calls": ref_tool_calls,
           "verifiers": scoring._load_verifiers() if needs_pb else None, "pb_repo": pb_repo, "job_dir": job_dir,
           "judge": _judge_fn, "judge_id": _judge_id,
           "mm_judge": _mm_fn, "medcta_img": getattr(env, "image_path", None),
           "medcta_question": (task.get("context") or {}).get("text"), "gacc": _gacc_fn,
           "available_tools": [t.get("name") for t in (task.get("available_tools") or []) if t.get("name")],
           "prompt_provenance": _prompt_prov,
           "task": task, "source_benchmark": task.get("source_benchmark")}
    results = [scoring.run_checkpoint(cp, ctx) for cp in task.get("checkpoints", [])]
    env_type = (task.get("environment") or {}).get("type")
    env_cls = type(env).__name__
    vlm = os.path.basename(os.environ.get("MH_VLM_PATH", "Qwen3-VL-2B-Instruct"))
    # ---- role-separated provenance: brain != tool-backend != judge. Record EACH even if same model,
    #      so nobody mistakes "agent saw the image" for "an image tool (also Qwen3-VL) saw it". ----
    real_ts = os.environ.get("MH_TOOL_MODE", "real") != "replay"
    if env_type == "fhir":
        tool_backend = {"fhir": "live_hapi", "rxnorm": "frozen"}; tool_backend_model = None
    elif env_type == "gui":
        tool_backend = {"gui": "real_playwright_portal" if env_cls == "GuiEnvReal" else "mock_inmemory"}
        tool_backend_model = None  # DOM actions, no model
    elif env_type == "tool_sandbox":
        tool_backend = {"tool_sandbox": "real_vlm_tools" if real_ts else "replay_cache"}
        # tool_backend_model = the ACTUAL VLM perception backend at runtime (NOT a hardcoded default).
        # MedCTA now defaults to gpt-5.5 via the gateway (vlm_backend.get_backend() -> ApiVLM); a local
        # run is Qwen3-VL. Resolve defensively from the live backend; fall back to MH_VLM_PATH basename.
        tool_backend_model = None
        if real_ts:
            try:
                import vlm_backend as _vb
                _be = _vb.get_backend()
                _bm = getattr(_be, "model", None) or getattr(_be, "model_path", None)
                if _bm:
                    _bm = os.path.basename(str(_bm))
                tool_backend_model = ("%s:%s" % (getattr(_be, "name", "vlm"), _bm)) if _bm else getattr(_be, "name", vlm)
            except Exception:
                tool_backend_model = vlm  # last-resort: configured local path basename
    else:
        tool_backend = {env_type: "unknown"}; tool_backend_model = None
    aname = getattr(agent, "name", "") or agent_name
    if aname == "qwen":
        agent_model = "%s (text-only brain)" % vlm; uses_hidden_ref = False; validation_only = False
    elif aname == "gpt5":
        agent_model = "%s (api brain)" % (os.environ.get("MH_API_MODEL") or os.environ.get("MH_OPENAI_MODEL", "gpt-5.5")); uses_hidden_ref = False; validation_only = False
    elif aname == "replay":
        agent_model = "gold_replay:reference_trace"; uses_hidden_ref = True; validation_only = True
    elif aname == "scripted":
        agent_model = "scripted:gold_path"; uses_hidden_ref = True; validation_only = True
    else:
        agent_model = "stub:%s" % aname; uses_hidden_ref = False; validation_only = True
    # Role-specific judges: outcome via Gacc (MH_GACC) | local Qwen (MH_JUDGE) | offline proxy;
    # grounding via multimodal judge (MH_MM_JUDGE) | local Qwen. Record EACH truthfully — result.json
    # must say who judged the answer and who judged grounding (reproducibility / integrity gate).
    def _ind_str(jm):
        # EXACT model identity, NOT substring: gpt-5.4 judging gpt-5.4-mini IS independent (different
        # models) — the old `jm not in agent_model` substring wrongly flagged gpt-5.4 in gpt-5.4-mini.
        am = str(agent_model or "").split(" (")[0].strip()          # "gpt-5.4-mini (api brain)" -> "gpt-5.4-mini"
        tb = str(tool_backend_model or "").split(":")[-1].strip()   # "api:gateway:gpt-5.5" -> "gpt-5.5"
        return "independent" if (jm and jm != am and jm != tb) else "shared_model_with_agent_or_tool"
    _TIER = {"gacc_judge": "gacc_semantic", "multimodal_judge": "multimodal_judge",
             "llm_judge": "local_model_judge", "proxy": "offline_whitelist_proxy"}
    # judges are derived from what ACTUALLY ran (per-checkpoint evaluator_kind), NOT from env flags, so a
    # stray MH_GACC_MODEL during a native_pytest (PB) run cannot falsely claim "deepseek judged the answer".
    def _judge_for(subdim):
        for r in results:
            if r.get("subdimension") == subdim and r.get("evaluator_kind"):
                ek = r["evaluator_kind"]; jb = r.get("judge_backend") or ek
                return {"model": jb, "tier": _TIER.get(ek, ek),
                        "independence": "n/a" if ek == "proxy" else _ind_str(jb)}
        return None
    judges = {}
    _o = _judge_for("result_verification") or _judge_for("clinical_task_success")
    if _o: judges["outcome"] = _o
    _g = _judge_for("context_grounding")
    if _g: judges["grounding"] = _g
    # Codex #7: judge independence is ENFORCED at scoring time, not merely recorded. A judge sharing the
    # agent's or tool-backend's model is not a valid scorer -> fail-closed: demote the checkpoints it
    # scored OUT of the main score (score_eligible=False) BEFORE build_result aggregates, so dimension
    # scores + both report layers stay consistent. Exploratory runs opt in via MH_ALLOW_SHARED_JUDGE=1
    # (the non_independent_judge qualification tag still records the deviation).
    _allow_shared = os.environ.get("MH_ALLOW_SHARED_JUDGE", "").lower() in ("1", "on", "true")
    _shared_subs = {sub for sub, j in (("result_verification", judges.get("outcome")),
                                       ("clinical_task_success", judges.get("outcome")),
                                       ("context_grounding", judges.get("grounding")))
                    if j and j.get("independence") == "shared_model_with_agent_or_tool"}
    if _shared_subs and not _allow_shared:
        for _r in results:
            if _r.get("subdimension") in _shared_subs and _r.get("score_eligible") is True:
                _r["score_eligible"] = False
                _r["score_demoted_reason"] = "non_independent_judge"
    _judge_policy = "exploratory_allowed_shared" if _allow_shared else "fail_closed"
    _outc = judges.get("outcome", {})
    judge_model = _outc.get("model", "none")
    judge_decoding = ({"max_tokens": 1024} if _outc.get("tier") == "gacc_semantic" else  # matches gacc_judge actual (no temperature)
                      ({"temperature": 0, "do_sample": False, "max_new_tokens": 220} if _outc.get("tier") == "local_model_judge" else None))
    # ---- per-run fidelity provenance: native-fidelity is a RECORDED field of THIS run, not a separate
    #      evaluation system. Derived from source benchmark + env type + protocol/prompt env vars. ----
    _bench_name = {"PhysicianBench": "PhysicianBench", "MedCTA": "MedCTA",
                   "HealthAdminBench": "HealthAdminBench"}.get(bench, bench)
    # prompt fidelity: PB replays the upstream verbatim SYSTEM_PROMPT; MedCTA/HAB are semantically aligned.
    # MH_PROMPT_TRACK=native -> verbatim upstream prompt scaffolding for any bench.
    _prompt_track = os.environ.get("MH_PROMPT_TRACK", "harness")
    _prompt_fidelity = "verbatim" if (env_type == "fhir" or _prompt_track == "native") else "semantic_aligned"
    # protocol fidelity: unified canonical protocol unless MH_PROTOCOL=function_calling (native FC).
    _protocol_fidelity = "native_function_calling" if os.environ.get("MH_PROTOCOL") == "function_calling" else "canonicalized"
    # environment fidelity: real substrate (live FHIR / real playwright portal / real VLM tool sandbox)
    # downgrades to "mock"/"replay" when not using the real backend.
    if env_type == "fhir":
        _env_fidelity = "full"
    elif env_type == "gui":
        _env_fidelity = "full" if env_cls == "GuiEnvReal" else "mock"
    elif env_type == "tool_sandbox":
        _env_fidelity = "full" if real_ts else "replay"
    else:
        _env_fidelity = "unknown"
    # MedCTA single-system config (which of the 3 modes this run used) -> self-describing provenance.
    _medcta_cfg = None
    if env_type == "tool_sandbox":
        _iv = getattr(agent, "medcta_image_visible", False); _te = getattr(agent, "medcta_tools_enabled", True)
        _mode = ("tool_mediated" if (not _iv and _te) else "pure_vqa" if (_iv and not _te)
                 else "sighted_with_tools" if (_iv and _te) else "blind_no_tools")
        _medcta_cfg = {"image_visible": _iv, "tools_enabled": _te, "mode": _mode,
                       "is_default_faithful": (not _iv and _te)}
    fidelity = {"source_benchmark": _bench_name,
                "prompt_fidelity": _prompt_fidelity,
                "protocol_fidelity": _protocol_fidelity,
                "environment_fidelity": _env_fidelity,
                "metric_definition": "native_plus_harness"}
    provenance = {"agent_model": agent_model, "tool_backend": tool_backend,
                  "tool_backend_model": tool_backend_model, "judge_model": judge_model,
                  "judge_tier": _outc.get("tier", "none"),
                  "judge_independence": _outc.get("independence", "n/a"),
                  "judge_independence_policy": _judge_policy,
                  "judge_decoding": judge_decoding, "judges": judges,
                  "uses_hidden_reference": uses_hidden_ref, "scorer_validation_only": validation_only,
                  "fidelity": fidelity, "medcta_config": _medcta_cfg, "capabilities": _caps,
                  "prompt_provenance": _prompt_prov, "git_sha": _resolve_git_sha()}
    result = scoring.build_result(task, trajectory, results, provenance)
    # Harness RUNTIME STATUS — always honest about what the harness actually did this run:
    #   requested_mode : the MH_HARNESS_MODE the operator asked for (default 'off').
    #   effective_mode : the kernel's live mode, or 'off' when no kernel was built.
    #   status         : 'active' / 'degraded' (a capability error occurred) / 'failed' (enforce was
    #                    requested but the kernel failed to build -> fail closed, do NOT pretend active)
    #                    / 'off' (nothing was requested).
    #   runtime_errors : every build/hook error string collected during the run.
    if _harness is not None:
        try:
            from harness.ledger import governance as _hgov
            _h_events = [ev for ev in trajectory if str(ev.get("event_type", "")).startswith("harness_")]
            _h_audit = _harness.audit()
            _h_status = _h_audit.get("status", "active")
            result["harness"] = {"mode": _harness.mode,
                                 "requested_mode": _harness_requested_mode,
                                 "effective_mode": _harness.mode,
                                 "status": "degraded" if (_h_status == "degraded" or _harness_runtime_errors) else "active",
                                 "runtime_errors": _harness_runtime_errors,
                                 "audit": _h_audit,
                                 "metrics": _hgov.summarize(_harness.ledger, _h_events, mode=_harness.mode)}
        except Exception as _he:
            _harness_runtime_errors.append("audit/summarize: %r" % _he)
            result["harness"] = {"mode": getattr(_harness, "mode", _harness_requested_mode),
                                 "requested_mode": _harness_requested_mode,
                                 "effective_mode": getattr(_harness, "mode", _harness_requested_mode),
                                 "status": "degraded", "runtime_errors": _harness_runtime_errors}
    elif _harness_requested_mode != "off" or _harness_runtime_errors:
        # a harness WAS requested (or errored) but no kernel is live -> report it, never silently 'off'.
        result["harness"] = {"mode": "off", "requested_mode": _harness_requested_mode,
                             "effective_mode": "off",
                             "status": "failed" if _harness_infra_error else "off",
                             "runtime_errors": _harness_runtime_errors}
    if _harness_infra_error:
        # assist/enforce requested but the harness could not be built -> the task ran NO agent actions; it
        # is an infrastructure error, excluded from mode comparison (never mixed into a baseline bundle).
        result["evaluation_status"] = "infrastructure_error"
    _sv = validate_result(result)
    # validate_result is contracted to return a dict (valid result, schema errors, OR fail-closed
    # 'jsonschema dependency missing'). If anything unexpected leaks through, treat it as NOT valid
    # (fail closed) rather than coercing to valid=True -- an unprovable result must never pass silently.
    _sv = _sv if isinstance(_sv, dict) else {"valid": False, "errors": ["validator returned non-dict: " + str(_sv)]}
    result["schema_validation"] = _sv                 # NON-underscore -> survives --out / run_batch
    result["_schema"] = "OK" if _sv.get("valid") else "INVALID: " + "; ".join(_sv.get("errors", []))
    # Formal-run default: a formal/benchmark run must always be strict, WITHOUT the operator having to
    # remember MH_SCHEMA_STRICT. Wiring (least surprising): MH_SCHEMA_STRICT stays the explicit knob;
    # MH_FORMAL / MH_BENCH_STRICT additionally turn strict ON (any truthy value). So strict is the
    # default for formal runs and opt-in otherwise.
    _strict = schema_strict_enabled()
    if not _sv.get("valid"):
        print("SCHEMA INVALID:", result["_schema"])
        if _strict:                                   # fail-closed: refuse to emit a protocol-violating result
            raise SystemExit("result fails spec/result.schema.json (strict): " + result["_schema"])
    result["_trajectory"] = trajectory
    result["_job_dir"] = job_dir
    # ---- qualification: downgrade ONLY for mock / replay / proxy / hidden-reference — NOT by substrate.
    #      A real Playwright GUI run or real VLM-tool run is NOT auto-"stub". ----
    quals = []
    if uses_hidden_ref: quals.append("uses_hidden_reference")
    if validation_only: quals.append("scorer_validation_only")
    if env_type == "gui" and env_cls != "GuiEnvReal": quals.append("mock_env")
    if env_type == "tool_sandbox" and not real_ts: quals.append("replay_tool_backend")
    if any(c.get("evaluator_kind") == "proxy" and c.get("subdimension") in ("clinical_task_success", "result_verification")
           for c in result.get("checkpoints", [])): quals.append("outcome_proxy")
    if any(j.get("independence") == "shared_model_with_agent_or_tool" for j in judges.values()):
        quals.append("non_independent_judge")
    if any(c.get("score_eligible") is False for c in result.get("checkpoints", [])): quals.append("proxy_scored_checkpoints")
    if any("API_BRAIN_ERROR" in str(ev.get("thought", "")) for ev in trajectory): quals.append("api_backend_error")
    quals = sorted(set(quals))
    if any(not (c or {}).get("healthy", True) for c in (_caps or {}).values()):
        quals.append("degraded_tool_health")   # Codex #10: a down backing service is NOT agent incompetence
    quals = sorted(set(quals))
    result["qualification"] = quals  # F1: non-underscore -> survives --out / run_batch result.json (meta.qualification_integrity)
    if quals:
        result["_warning"] = "qualified: " + ", ".join(quals)
    # light per-task cleanup of agent-created (stub-tagged) resources to avoid FHIR pollution
    if cleanup and env_type == "fhir" and ctx.get("base"):
        cl = cleanup_stub(ctx["base"])
        result["_cleanup_deleted"] = cl["deleted"]
        if cl["error"]:
            result["_cleanup_error"] = cl["error"]
            result["failure_tags"] = sorted(set(result.get("failure_tags", [])) | {"environment_error"})
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--agent", default="stub")
    ap.add_argument("--fhir-base", default=None)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--job-dir", default=None)
    ap.add_argument("--reset-mode", default="none", choices=["none", "restore_pristine", "per_task"])
    ap.add_argument("--unsafe-no-reset", action="store_true", help="skip per-task stub-resource cleanup")
    ap.add_argument("--formal", action="store_true", help="formal benchmark run: schema-strict (refuse to emit a protocol-violating result)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.formal:
        os.environ["MH_FORMAL"] = "1"   # formal run -> strict validation without needing the env var set
    reset_fhir(args.reset_mode)
    result = run_task(args.bench, args.task, args.agent, args.fhir_base, args.max_steps, args.job_dir,
                      cleanup=not args.unsafe_no_reset)
    if result.get("_warning"): print("WARNING:", result["_warning"])
    print(json.dumps({"task": result["task_id"], "success": result["success"],
                      "evaluation_status": result["evaluation_status"],
                      "dimension_scores": result["dimension_scores"],
                      "failure_tags": result["failure_tags"],
                      "checkpoints": [(c["id"], c["checkpoint_status"]) for c in result["checkpoints"]],
                      "schema": result["_schema"]}, indent=1, ensure_ascii=False))
    if args.out:
        r = {k: v for k, v in result.items() if not k.startswith("_")}
        json.dump(r, open(args.out, "w"), indent=1, ensure_ascii=False); print("wrote", args.out)

if __name__ == "__main__":
    main()
