#!/usr/bin/env python3
"""Unified Medical Harness runner (skeleton).

load unified task -> environment adapter -> agent loop -> unified trajectory -> scorer -> result JSON.
Runs end-to-end with the stub agent (no API key). native_pytest executes via subprocess pytest;
deterministic/llm_judge are skipped (skip_reason) pending B-line. Result validated vs spec/result.schema.json.
"""
import os, sys, json, glob, argparse, shutil, datetime
import canonical_schema as _canon
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import environments, agents, scoring

def load_task(bench, task_id):
    p = os.path.join(ROOT, "benchmark_dataprocess", bench, "tasks_unified.jsonl")
    for line in open(p):
        t = json.loads(line)
        if t["task_id"] == task_id:
            return t
    raise SystemExit(f"task {task_id} not found in {p}")

def validate_result(result):
    try:
        from jsonschema import Draft7Validator, RefResolver
    except Exception:
        return "(jsonschema missing)"
    spec = os.path.join(ROOT, "spec"); store = {}
    for f in glob.glob(os.path.join(spec, "*.json")):
        s = json.load(open(f))
        if "$id" in s: store[s["$id"]] = s
    rs = json.load(open(os.path.join(spec, "result.schema.json")))
    v = Draft7Validator(rs, resolver=RefResolver(base_uri=rs["$id"], referrer=rs, store=store))
    errs = list(v.iter_errors(result))
    return "OK" if not errs else "; ".join(f"{list(e.path)}:{e.message}" for e in errs[:5])

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

_FHIR_CANON_SEARCH = {"Patient":"fhir_patient_search_demographics","Condition":"fhir_condition_search_problems","MedicationRequest":"fhir_medication_request_search_orders","Procedure":"fhir_procedure_search_orders","DocumentReference":"fhir_document_reference_search_clinical_notes","ServiceRequest":"fhir_service_request_search"}
_FHIR_CANON_CREATE = {"MedicationRequest":"fhir_medication_request_create","ServiceRequest":"fhir_service_request_create","Communication":"fhir_communication_create_message","Appointment":"fhir_appointment_create"}
def _canon_fhir_tool(tool, args):
    """Map our generic fhir_search/read/create(resourceType=X) to the upstream PhysicianBench granular
    tool_name so native test_outputs.py checkpoints (which match metadata.tool_name) recognize the query.
    Conservative: a category-less Observation search -> labs only (never auto-credits vitals/social)."""
    a = args or {}
    if tool == "fhir_create":
        res = a.get("resource", a) or {}
        rt = res.get("resourceType", a.get("resourceType", ""))
        return _FHIR_CANON_CREATE.get(rt, tool)
    if tool in ("fhir_search", "fhir_read"):
        rt = a.get("resourceType", ""); cat = str(a.get("category", "")).lower()
        if rt == "Observation":
            if "vital" in cat: return "fhir_observation_search_vitals"
            if "social" in cat: return "fhir_observation_search_social_history"
            return "fhir_observation_search_labs"
        return _FHIR_CANON_SEARCH.get(rt, tool)
    return tool


def run_task(bench, task_id, agent_name="stub", fhir_base=None, max_steps=12, job_dir=None, cleanup=True):
    # NOTE: in PhysicianBench the FHIR Patient.id == MRN, so context.patient_ref (an MRN) is also a
    # valid resource id and Patient/{patient_ref} resolves. If a future bench uses non-MRN ids,
    # split into patient_ref (resource id) + patient_identifier (MRN) before reusing this.
    task = load_task(bench, task_id)
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

    trajectory, last_obs, last_res, finished = [], None, None, False
    import re as _re
    _dm = _re.search(r"(?:/?workspace/)?output/[\w.\-]+", task.get("goal", "") or "")  # match output/X AND workspace/output/X
    _track_ds = os.environ.get("MH_PROMPT_TRACK", "harness")
    _scaffold = os.environ.get("MH_DELIV_SCAFFOLD", "0" if _track_ds == "native" else "1") != "0"
    _deliverable = (_dm.group(0) if _dm else None) if _scaffold else None  # native: no deliverable nudge/budget/forced-turn scaffolding (obs-bug fixed)
    _deliv_nudges = 0
    _fail_sig = None; _fail_n = 0; _aborted = False  # circuit breaker: abort on repeated identical failing call
    _budget_nudged = False  # one-shot deliverable warning when steps run low
    if hasattr(env, "initial_observation"):
        _io = env.initial_observation()
        if _io is not None:
            last_res = _io; last_obs = json.dumps(_io, ensure_ascii=False)[:int(os.environ.get("MH_OBS_MAX_LEN", "10000"))]
    for step in range(max_steps):
        if _deliverable and not _budget_nudged and (max_steps - step) <= 8:
            _bws = getattr(env, "workspace", "") or ""
            _bwant = os.path.join(_bws, os.path.basename(_deliverable)) if _bws else ""
            if _bwant and not (os.path.isfile(_bwant) and os.path.getsize(_bwant) > 0):
                last_res = {"feedback": ("You are RUNNING OUT OF STEPS (only %d left) and have NOT written the "
                            "required deliverable. STOP retrieving NOW. Immediately call write_file with EXACTLY "
                            "path=\"%s\" and content = your full clinical assessment and management plan from the "
                            "data already retrieved." % (max_steps - step, _deliverable))}
                last_obs = "budget_warning"; _budget_nudged = True
                trajectory.append({"step": step, "event_type": "deliverable_budget_warning",
                                   "remaining": max_steps - step, "status": "ok"})
        action = agent.act({"goal": task.get("goal"), "context": task.get("context"),
                            "tools": env.available_tools(), "last_observation": last_obs, "last_result": last_res})
        if not isinstance(action, dict) or "type" not in action:  # #7 agent contract violation
            trajectory.append({"step": step, "event_type": "agent_error", "error": "invalid_action", "raw": str(action)[:200], "status": "error"})
            finished = True; break
        if action["type"] == "final":
            if _deliverable and _deliv_nudges < 3:
                _fp = os.path.join(getattr(env, "workspace", "") or "", os.path.basename(_deliverable))
                if not (os.path.isfile(_fp) and os.path.getsize(_fp) > 0):
                    _deliv_nudges += 1
                    _fb = ("STOP. The task REQUIRES a deliverable file and it is NOT written yet. You MUST "
                           "call write_file NOW with EXACTLY path=\"%s\" (this EXACT filename, no other name) "
                           "and content = your full clinical assessment and management plan. Do not answer in "
                           "chat. Do not use any other filename. After write_file succeeds, then finish." % _deliverable)
                    trajectory.append({"step": step, "event_type": "deliverable_nudge", "path": _deliverable,
                                       "attempt": _deliv_nudges, "status": "ok"})
                    last_res = {"feedback": _fb}; last_obs = _fb
                    continue
            trajectory.append({"step": step, "event_type": "final_answer", "thought": action.get("answer", ""), "status": "ok", "canonical_action": _canon.canonical_action(action, env_type)})
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
        try:
            res = env.call_tool(action["tool"], action.get("args", {}))
        except Exception as _e:
            res = {"error": repr(_e)}
        obs = json.dumps(res, ensure_ascii=False)[:int(os.environ.get("MH_OBS_MAX_LEN", "10000"))]  # official mini_agent caps tool output to LLM at 10k (was 200 -> agent could not see search results -> over-read)
        _err = res.get("error") if isinstance(res, dict) else None
        if not _err and isinstance(res, dict):  # F2: tool_sandbox tools report errors as a bracketed string in "output"
            _out = res.get("output")
            if isinstance(_out, str) and _out.lstrip().startswith("["):
                _marker = (_out[_out.find("[") + 1:_out.find("]")] if "]" in _out else _out[:40]).lower()
                if any(w in _marker for w in ("error", "unknown", "invalid", "fail")):
                    _err = _out[:120]
        _ev = {"step": step, "event_type": "tool_call", "tool": action["tool"],
               "args": action.get("args", {}), "result": res, "observation": obs, "ts": str(step),
               "status": "error" if _err else "ok",
               "canonical_action": _canon.canonical_action(action, env_type),
               "canonical_result": _canon.canonical_result(res)}  # F2: explicit per-action status (no obs-substring heuristic downstream)
        if _err:
            _es = str(_err)
            _ev["error_type"] = next(("http_" + c for c in ("400", "401", "403", "404", "409", "422", "500", "502", "503")
                                      if ("HTTP " + c) in _es),
                                     "exception" if any(k in _es for k in ("Error", "Exception", "Traceback")) else "tool_error")
        trajectory.append(_ev)
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
    if not finished and not _aborted:  # #8 ran out of steps without a final answer (circuit-breaker abort logs its own event)
        trajectory.append({"step": max_steps, "event_type": "agent_error", "error": "max_steps_exceeded", "status": "error"})
    # deliverable enforcement (final): guarantee ONE write attempt if the required file is still missing
    # (covers both "finished early" and "ran out of steps"), then normalize a mis-named single file.
    if _deliverable:
        _ws = getattr(env, "workspace", "") or ""
        _want = os.path.join(_ws, os.path.basename(_deliverable)) if _ws else ""
        _missing = lambda: not (_want and os.path.isfile(_want) and os.path.getsize(_want) > 0)
        if _ws and _missing():
            _fb = ("You must NOW save the required deliverable. Call write_file with EXACTLY path=\"%s\" and "
                   "content = your full clinical assessment and management plan." % _deliverable)
            try:
                _a = agent.act({"goal": task.get("goal"), "context": task.get("context"),
                                "tools": env.available_tools(), "last_observation": _fb,
                                "last_result": {"feedback": _fb}})
                if isinstance(_a, dict) and _a.get("type") == "tool_call" and _a.get("tool") == "write_file":
                    _wr = env.call_tool("write_file", _a.get("args", {}))
                    trajectory.append({"step": max_steps, "event_type": "tool_call", "tool": "write_file",
                                       "args": _a.get("args", {}), "result": _wr,
                                       "observation": json.dumps(_wr)[:200], "ts": str(max_steps),
                                       "status": "ok", "forced_deliverable": True})
            except Exception as _fe:
                trajectory.append({"step": max_steps, "event_type": "agent_error",
                                   "error": "forced_deliverable_failed", "raw": repr(_fe)[:120], "status": "error"})
        if _ws and os.path.isdir(_ws) and _missing():  # mis-named single file -> normalize (transparent)
            _cands = [f for f in glob.glob(os.path.join(_ws, "*")) if os.path.isfile(f) and os.path.getsize(f) > 0]
            if _cands:  # >=1 candidate: copy the largest non-empty file to the required name (handles >=2 too)
                _best = max(_cands, key=os.path.getsize)
                shutil.copyfile(_best, _want)
                trajectory.append({"step": max_steps, "event_type": "deliverable_renamed",
                                   "from": os.path.basename(_best), "to": os.path.basename(_deliverable),
                                   "n_candidates": len(_cands), "status": "ok"})
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
    ctx = {"base": getattr(env, "base", None), "mrn": (task.get("context") or {}).get("patient_ref"),
           "trajectory": trajectory, "created_meds": scoring.created_meds(trajectory),
           "final_texts": final_texts, "note_texts": note_texts, "full_state": getattr(env, "full_state", None),
           "reference": task.get("reference", {}), "agent_tool_calls": agent_tool_calls, "ref_tool_calls": ref_tool_calls,
           "verifiers": scoring._load_verifiers() if needs_pb else None, "pb_repo": pb_repo, "job_dir": job_dir,
           "judge": _judge_fn, "judge_id": _judge_id,
           "mm_judge": _mm_fn, "medcta_img": getattr(env, "image_path", None),
           "medcta_question": (task.get("context") or {}).get("text"), "gacc": _gacc_fn}
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
        agent_model = "%s (api brain)" % os.environ.get("MH_OPENAI_MODEL", "gpt-5.5"); uses_hidden_ref = False; validation_only = False
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
        return "independent" if (jm and (jm not in (agent_model or "")) and (jm != (tool_backend_model or ""))) \
               else "shared_model_with_agent_or_tool"
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
    _o = _judge_for("clinical_task_success")
    if _o: judges["outcome"] = _o
    _g = _judge_for("context_grounding")
    if _g: judges["grounding"] = _g
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
    fidelity = {"source_benchmark": _bench_name,
                "prompt_fidelity": _prompt_fidelity,
                "protocol_fidelity": _protocol_fidelity,
                "environment_fidelity": _env_fidelity,
                "metric_definition": "native_plus_harness"}
    provenance = {"agent_model": agent_model, "tool_backend": tool_backend,
                  "tool_backend_model": tool_backend_model, "judge_model": judge_model,
                  "judge_tier": _outc.get("tier", "none"),
                  "judge_independence": _outc.get("independence", "n/a"),
                  "judge_decoding": judge_decoding, "judges": judges,
                  "uses_hidden_reference": uses_hidden_ref, "scorer_validation_only": validation_only,
                  "fidelity": fidelity}
    result = scoring.build_result(task, trajectory, results, provenance)
    result["_schema"] = validate_result(result)
    result["_trajectory"] = trajectory
    result["_job_dir"] = job_dir
    # ---- qualification: downgrade ONLY for mock / replay / proxy / hidden-reference — NOT by substrate.
    #      A real Playwright GUI run or real VLM-tool run is NOT auto-"stub". ----
    quals = []
    if uses_hidden_ref: quals.append("uses_hidden_reference")
    if validation_only: quals.append("scorer_validation_only")
    if env_type == "gui" and env_cls != "GuiEnvReal": quals.append("mock_env")
    if env_type == "tool_sandbox" and not real_ts: quals.append("replay_tool_backend")
    if any(c.get("evaluator_kind") == "proxy" and c.get("subdimension") == "clinical_task_success"
           for c in result.get("checkpoints", [])): quals.append("outcome_proxy")
    if any(j.get("independence") == "shared_model_with_agent_or_tool" for j in judges.values()):
        quals.append("non_independent_judge")
    if any(c.get("score_eligible") is False for c in result.get("checkpoints", [])): quals.append("proxy_scored_checkpoints")
    if any("API_BRAIN_ERROR" in str(ev.get("thought", "")) for ev in trajectory): quals.append("api_backend_error")
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
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
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
