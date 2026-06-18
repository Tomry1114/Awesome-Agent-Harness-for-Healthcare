#!/usr/bin/env python3
"""Unified Medical Harness runner (skeleton).

load unified task -> environment adapter -> agent loop -> unified trajectory -> scorer -> result JSON.
Runs end-to-end with the stub agent (no API key). native_pytest executes via subprocess pytest;
deterministic/llm_judge are skipped (skip_reason) pending B-line. Result validated vs spec/result.schema.json.
"""
import os, sys, json, glob, argparse, shutil, datetime
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
    agent = agents.make_agent(agent_name, task)

    trajectory, last_obs, last_res, finished = [], None, None, False
    if hasattr(env, "initial_observation"):
        _io = env.initial_observation()
        if _io is not None:
            last_res = _io; last_obs = json.dumps(_io)[:200]
    for step in range(max_steps):
        action = agent.act({"goal": task.get("goal"), "context": task.get("context"),
                            "tools": env.available_tools(), "last_observation": last_obs, "last_result": last_res})
        if not isinstance(action, dict) or "type" not in action:  # #7 agent contract violation
            trajectory.append({"step": step, "event_type": "agent_error", "error": "invalid_action", "raw": str(action)[:200]})
            finished = True; break
        if action["type"] == "final":
            trajectory.append({"step": step, "event_type": "final_answer", "thought": action.get("answer", "")})
            finished = True; break
        if action["type"] != "tool_call" or not action.get("tool"):
            trajectory.append({"step": step, "event_type": "agent_error", "error": "bad_action_type", "raw": str(action)[:200]})
            finished = True; break
        try:
            res = env.call_tool(action["tool"], action.get("args", {}))
        except Exception as _e:
            res = {"error": repr(_e)}
        obs = json.dumps(res)[:200]
        trajectory.append({"step": step, "event_type": "tool_call", "tool": action["tool"],
                           "args": action.get("args", {}), "result": res, "observation": obs, "ts": str(step)})
        last_obs = obs; last_res = res
    if not finished:  # #8 ran out of steps without a final answer
        trajectory.append({"step": max_steps, "event_type": "agent_error", "error": "max_steps_exceeded"})
    env.teardown()

    # upstream-format trajectory.log for native_pytest
    with open(os.path.join(job_dir, "logs", "agent", "trajectory.log"), "w") as f:
        for ev in trajectory:
            if ev.get("event_type") == "tool_call":
                f.write(json.dumps({"timestamp": datetime.datetime.now().isoformat(), "type": "tool_call",
                                    "content": ev["tool"], "metadata": {"tool_name": ev["tool"],
                                    "input": ev.get("args"), "output": ev.get("result")}}) + "\n")

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
    ctx = {"base": getattr(env, "base", None), "mrn": (task.get("context") or {}).get("patient_ref"),
           "trajectory": trajectory, "created_meds": scoring.created_meds(trajectory),
           "final_texts": final_texts, "note_texts": note_texts, "full_state": getattr(env, "full_state", None),
           "reference": task.get("reference", {}), "agent_tool_calls": agent_tool_calls, "ref_tool_calls": ref_tool_calls,
           "verifiers": scoring._load_verifiers() if needs_pb else None, "pb_repo": pb_repo, "job_dir": job_dir}
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
        tool_backend_model = vlm if real_ts else None  # ImageDescription/RegionAttribute/OCR run a VLM
    else:
        tool_backend = {env_type: "unknown"}; tool_backend_model = None
    aname = getattr(agent, "name", "") or agent_name
    if aname == "qwen":
        agent_model = "%s (text-only brain)" % vlm; uses_hidden_ref = False; validation_only = False
    elif aname == "replay":
        agent_model = "gold_replay:reference_trace"; uses_hidden_ref = True; validation_only = True
    elif aname == "scripted":
        agent_model = "scripted:gold_path"; uses_hidden_ref = True; validation_only = True
    else:
        agent_model = "stub:%s" % aname; uses_hidden_ref = False; validation_only = True
    # no LLM judge wired; MedCTA outcome uses an offline whitelist proxy until a real Gacc judge lands
    judge_model = "offline_whitelist_proxy" if env_type == "tool_sandbox" else "none"
    provenance = {"agent_model": agent_model, "tool_backend": tool_backend,
                  "tool_backend_model": tool_backend_model, "judge_model": judge_model,
                  "uses_hidden_reference": uses_hidden_ref, "scorer_validation_only": validation_only}
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
    if judge_model.startswith("offline"): quals.append("outcome_proxy")
    if any(c.get("score_eligible") is False for c in result.get("checkpoints", [])): quals.append("proxy_scored_checkpoints")
    quals = sorted(set(quals))
    if quals:
        result["_qualification"] = quals
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
