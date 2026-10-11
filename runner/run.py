#!/usr/bin/env python3
"""Unified Medical Harness runner.

load unified task -> environment adapter -> agent loop -> unified trajectory -> scorer -> result JSON.
Runs end-to-end with the stub agent (no API key). Checkpoints are scored through runner/scorers/ (native_pytest
executes via subprocess pytest). Result validated vs spec/result.schema.json.
"""
import os
from harness import actuation as _act     # ONE actuation layer for ACQUIRE / REVISE / DEFER_FINAL
import re, sys, json, glob, argparse, shutil, datetime, hashlib
import canonical_schema as _canon
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scorers"))   # scoring machinery lives in runner/scorers/
import environments, agents, scoring
import credit as _credit        # agent vs harness scoring views and checkpoint credit (runner/scorers/credit.py)
from harness import origin as _origin   # the origin model (runner/harness/origin.py)

MAX_REPAIR_TURNS = 4          # cap on REVISE/BLOCK repair turns per task
RECONCILE_BUDGET = 3          # reads allowed to reconcile an UNKNOWN commit before the run aborts
OBS_MAX_LEN = 10000           # characters of an observation handed to the agent

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


def _emit(decision, stage, reason, obs, **extra):
    """The (feedback, last_res, last_obs) triple EVERY uniform intervention site sets --
    a structured feedback dict handed to the agent next turn + the mirrored last_res / last_obs. Shared by
    the PB / HAB / MedCTA / generic gate dispatch so this apply protocol lives in ONE place, not 10 copies."""
    fb = {"decision": decision, "stage": stage, "reason": reason}
    fb.update(extra)
    return fb, {"feedback": reason}, obs


def _resolve_git_sha():
    # the code SHA this run executed -> result.json provenance, so a set of bundles made with different code can be detected.
    try:
        import subprocess
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=_root,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


from adapters import get_adapter
from harness.dataset_adapter import HarnessAdapter
from environments import _canon_fhir_tool


def _state_snapshot(env):
    """The STRUCTURED canonical state (env.state_summary() / full_state) — given to the harness so field-
    level postconditions (field_equals/object_exists/...) can be evaluated, not only a state-change hash.
    None for stateless envs."""
    try:
        return env.state_summary() if hasattr(env, "state_summary") else getattr(env, "full_state", None)
    except Exception:
        return None


def _harness_observation(env):
    """What the HARNESS sees at before_action: the structured state PLUS the live control model.

    Deliberately not _state_snapshot(): that one is hashed into state_record (see _state_hash) and its
    contract is that a read-only navigate does not change it. The control list changes on every page, so it
    belongs here and nowhere near the hash. Adapters without list_controls get the plain snapshot back.
    """
    st = _state_snapshot(env)
    lc = getattr(env, "list_controls", None)
    if not callable(lc):
        return st
    try:
        controls = lc() or []
    except Exception:
        controls = []
    if not controls:
        return st
    out = dict(st) if isinstance(st, dict) else {"state": st}
    out["controls"] = controls
    return out


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


def validate_result(result):
    # FAIL CLOSED: a missing jsonschema dependency means we CANNOT prove the result conforms to
    # spec/result.schema.json, so we must NOT silently treat it as valid. Return a structured
    # {valid: False, errors:[...]} (NOT a bare string that the caller coerces to valid=True), so the
    # recorded schema_validation verdict never claims a result conforms when it was never checked.
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
    the OCI layer and restarts (safe — NOT a hot-copy). Raises if the reset script fails."""
    if mode in (None, "none"):
        return
    if mode in ("restore_pristine", "per_task"):
        import subprocess
        # The script starts the HAPI server, which backgrounds a java process. A backgrounded grandchild
        # INHERITS whatever stdout/stderr it was handed and holds them open for its entire life, so a
        # subprocess.run that passes the caller's pipes straight through waits for an EOF that only comes
        # when the FHIR server dies: the task would hang with no trajectory and no error, and end with a score
        # that looks like an agent failure. Own the pipes here, deny stdin, and put a ceiling on it so a stuck
        # reset fails loudly.
        _p = subprocess.run(["bash", os.path.join(ROOT, "benchmark_dataprocess", "PhysicianBench",
                            "augmentation", "restore_pristine_h2.sh")],
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=900, check=False)
        _out = (_p.stdout or b"").decode("utf-8", "replace")
        if _p.returncode != 0:
            raise RuntimeError("restore_pristine failed (rc=%d):\n%s" % (_p.returncode, _out[-1500:]))
        print("[reset_fhir] restore_pristine ok\n%s" % _out[-400:], flush=True)

def cleanup_stub(base):
    """Light per-task cleanup: delete agent-created resources tagged stub-run (avoids FHIR pollution
    without a full pristine restore). Returns {deleted, error}. Errors are surfaced, NOT swallowed,
    so a failed cleanup (= possible pollution) is visible."""
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

def resolve_workspace(bench, task, workspace=None):
    """The run's workspace, resolved once: -> (absolute path, identity) with identity = workspace.identity (the
    result.json `workspace` block), or (None, None).

    `workspace` is the directory the run names. When it names none, the run takes the default workspace of its task's
    substrate (workspace.H0: workspaces/<bench key>/h0); this is the one place a default workspace is chosen, and every
    reader of workspace content (agent program, host and kernel middleware, policies, completion obligations,
    workspace-relative modules, instrumentation) receives the path from the run. (None, None) when none is named and the
    substrate has no default workspace. Refuses (WorkspaceError) a workspace whose manifest does not declare the run's
    benchmark and env_type, a symlink under it and a path resolving outside it."""
    import workspace as W
    env_type = (task.get("environment") or {}).get("type") if isinstance(task, dict) else None
    path = workspace if workspace is not None else W.H0.get(env_type)
    if path is None:
        return None, None
    return os.path.abspath(path), W.identity(path, bench, env_type)


def resume_key(task_id, workspace_digest, binding, agent_name, max_steps):
    """The key a stored result may be reused under: task id, workspace digest, the bound judge configuration (the
    sha256 of runner/judges.py signature(binding)), agent model, protocol and max_steps.
    run_task records it in result.json; run_batch reuses a stored result only when the key it records equals this key
    for the run it would replace. An API agent's model and protocol are the model table's (runner/models.yaml; a model
    it does not list raises UnknownModelError); another agent's model is its name and its protocol None."""
    import judges as _J
    if agent_name in agents.API_AGENTS:
        import models as _M
        model = _M.agent_model()
        protocol = _M.protocol(model)
    else:
        model, protocol = agent_name, None
    sig = json.dumps(_J.signature(binding), sort_keys=True)
    return {"task_id": task_id, "workspace_digest": workspace_digest,
            "judge_config_digest": "sha256:" + hashlib.sha256(sig.encode("utf-8")).hexdigest(),
            "agent_model": model, "protocol": protocol, "max_steps": max_steps}


def run_task(bench, task_id, agent_name="stub", fhir_base=None, max_steps=12, job_dir=None, cleanup=True,
             workspace=None):
    """Run one task and return its result. Three guarantees wrap the episode (_run_task):

    * JUDGES FIRST. Every judge the benchmark needs (runner/judges.py, harness judge included) is bound from
      configuration before anything else happens; a missing one raises JudgeConfigError -- no environment, no
      agent, no step. The binding is recorded in result["run_config"]["judges"].
    * THE WORKSPACE IS AN INPUT. `workspace` (None: the substrate's default workspace) is resolved once
      (resolve_workspace) and refused before anything runs when its manifest does not match the run; its path reaches
      every reader. result.json records result["workspace"] = {path, digest, component_digests, parent_digest},
      run_config.workspace_digest and the resume_key the run was made under.
    * A CRASH IS A RESULT. An exception out of the episode (a workspace hook, the kernel, the loop, the scorer)
      returns a result with end_reason=harness_error, outcome_status=harness_error, no checkpoints and a
      traceback digest (crash_result.py), instead of propagating and leaving the task with no result.json."""
    import judges as _J
    _binding = _J.require(bench)
    _ws_path, _ws = resolve_workspace(bench, load_task(bench, task_id), workspace)
    _key = resume_key(task_id, (_ws or {}).get("digest"), _binding, agent_name, max_steps)
    _partial = {}
    try:
        result = _run_task(bench, task_id, agent_name, fhir_base, max_steps, job_dir, cleanup, _partial=_partial,
                           workspace=_ws_path)
    except (Exception, environments.BrowserPageCrashed) as _exc:   # the latter is a BaseException (environments.py)
        import crash_result as _CR
        _env = _partial.get("env")
        if _env is not None:
            try:
                _env.teardown()
            except Exception:
                pass
        result = _CR.crash_result(bench, task_id, _exc, agent_name=agent_name, max_steps=max_steps,
                                  trajectory=_partial.get("trajectory"), git_sha=_resolve_git_sha())
        result["_job_dir"] = _partial.get("job_dir")
        print("HARNESS_ERROR %s: %s: %s" % (task_id, type(_exc).__name__, str(_exc)[:300]), flush=True)
    else:
        import aggregate_report as _AR
        _d = _AR.native_task_outcome_detail(result, bench)
        result["outcome_status"] = _d["outcome_status"]
        if _d.get("reason"):
            result["outcome_status_reason"] = _d["reason"]
    result["run_config"] = dict(result.get("run_config") or {}, judges=_J.run_record(_binding, result),
                                workspace_digest=(_ws or {}).get("digest"),
                                scored=os.environ.get(UNSCORED_VAR) != "1")
    result["workspace"] = _ws
    result["resume_key"] = _key
    return result


UNSCORED_VAR = "MH_UNSCORED_RUN"
# Set to "1" by a caller that runs an episode without needing its score; the checkpoints are then recorded as
# skipped instead of judged (see _run_task), and run_config.scored records it.


def score_checkpoints(checkpoints, ctx, env_type):
    """Every checkpoint's result, in checkpoint order. MedCTA's (tool_sandbox) and PhysicianBench's (fhir) run at once:
    MedCTA's are independent LLM judges that only read ctx, and run one after another they take a noticeable share of
    a task's wall time; PhysicianBench's native pytest checkpoints are separate processes that only READ the FHIR store
    and the job dir (the upstream tests make GET requests; pytest's cache is off, each has its own judge log).
    HealthAdminBench (gui) keeps one at a time."""
    if env_type not in ("tool_sandbox", "fhir") or len(checkpoints) < 2:
        return [scoring.run_checkpoint(cp, ctx) for cp in checkpoints]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(8, len(checkpoints))) as ex:
        return list(ex.map(lambda cp: scoring.run_checkpoint(cp, ctx), checkpoints))


def pb_judge_env():
    """PB content checkpoints (upstream eval_helpers.llm_judge) need an OpenAI-compatible judge: set its key and base
    from the environment (native_pytest propagates os.environ to the pytest subprocess)."""
    if os.environ.get("LLM_JUDGE_MODEL"):
        # The run's judge route (MH_JUDGE_BASE + MH_JUDGE_KEY) is the PB content judge's too: upstream reads
        # LLM_JUDGE_API_KEY / LLM_JUDGE_BASE_URL first. Without this, the upstream judge would pair OPENAI_API_KEY (which may
        # be a key for a different gateway) with the judge base and fail authentication on every PB content checkpoint.
        if os.environ.get("MH_JUDGE_KEY") and os.environ.get("MH_JUDGE_BASE"):
            _jb2 = os.environ["MH_JUDGE_BASE"].rstrip("/")
            if _jb2.endswith("/v1"): _jb2 = _jb2[:-3].rstrip("/")
            os.environ.setdefault("LLM_JUDGE_API_KEY", os.environ["MH_JUDGE_KEY"])
            os.environ.setdefault("LLM_JUDGE_BASE_URL", _jb2 + "/v1")
        if not os.environ.get("OPENAI_API_KEY"):
            _kf = os.path.expanduser("~/.mh_openai_key")
            if os.path.exists(_kf):
                os.environ["OPENAI_API_KEY"] = open(_kf).read().strip()
        # base URL must default to the gateway WHENEVER a judge key is present -- a caller who supplies
        # OPENAI_API_KEY themselves (but not OPENAI_BASE_URL) would otherwise hit api.openai.com with a
        # gateway key and every LLM-graded checkpoint fake-fails as agent_failure (silent invalid run).
        if os.environ.get("OPENAI_API_KEY") and not os.environ.get("OPENAI_BASE_URL"):
            _jb = (os.environ.get("MH_JUDGE_BASE") or os.environ.get("MH_OPENAI_BASE", "https://api.openai.com")).rstrip("/")
            if _jb.endswith("/v1"): _jb = _jb[:-3].rstrip("/")
            os.environ["OPENAI_BASE_URL"] = _jb + "/v1"


def _run_task(bench, task_id, agent_name="stub", fhir_base=None, max_steps=12, job_dir=None, cleanup=True,
              _partial=None, *, workspace):
    # NOTE: in PhysicianBench the FHIR Patient.id == MRN, so context.patient_ref (an MRN) is also a
    # valid resource id and Patient/{patient_ref} resolves. If a future bench uses non-MRN ids,
    # split into patient_ref (resource id) + patient_identifier (MRN) before reusing this.
    task = load_task(bench, task_id)
    from harness.engines.semantic import sem_cache_reset as _sem_cache_reset
    _sem_cache_reset(task_id)   # the semantic judge cache belongs to this task (keyed task + judge model + prompt)
    env_type = (task.get("environment") or {}).get("type")  # needed inside the step loop (canonical_action), so it is set before the loop
    # Is the GRADED artifact the answer text, or state the agent already persisted in the environment?
    # Declared per task. Only substrates whose deliverable is state are listed; anything else (including an
    # unknown future value) keeps the conservative answer-withholding path.
    _ANSWER_KIND_IS_ENV_STATE = ("portal_state_and_notes",)
    _deliverable_is_env_state = ((task.get("expected_outcome") or {}).get("answer_kind")
                                 in _ANSWER_KIND_IS_ENV_STATE)
    _ds_adapter = get_adapter(bench=bench, env_type=env_type) or HarnessAdapter()   # bench-first (a 2nd GUI benchmark must NOT auto-get HABAdapter), env_type fallback
    aug = os.path.join(ROOT, "benchmark_dataprocess", "PhysicianBench", "augmentation")
    pb_repo = os.path.join(ROOT, "benchmark", "PhysicianBench", "PhysicianBench")
    # one directory per process, so two runs of one task on one node (two rollouts at once, two jobs sharing a node)
    # never share /tmp/mh_jobs/<bench>/<task> or remove each other's
    job_dir = job_dir or os.path.join("/tmp/mh_jobs", bench, "%s.%d" % (task_id, os.getpid()))
    shutil.rmtree(job_dir, ignore_errors=True)
    os.makedirs(os.path.join(job_dir, "workspace", "output"), exist_ok=True)
    os.makedirs(os.path.join(job_dir, "logs", "agent"), exist_ok=True)

    env = environments.make_env(task, fhir_base=fhir_base, aug_dir=aug,
                                workspace=os.path.join(job_dir, "workspace", "output"))
    if _partial is not None:
        _partial.update(env=env, job_dir=job_dir)
    env.reset()
    if getattr(env, "type", None) == "fhir":  # expose upstream granular FHIR tools (precise retrieval + native cp tool-name match)
        # a PB-native env supplies the FAITHFUL upstream schemas (full params, correct arg names so
        # agent calls match the upstream fn signatures); otherwise the degraded granular signatures.
        task["available_tools"] = getattr(env, "_native_specs", None) or task.get("available_tools") or []
    agent = agents.make_agent(agent_name, task, workspace)
    from instrumentation import Instrumentation, record_components
    _instr = Instrumentation()   # what ran from the workspace; written beside the trajectory, not into it

    # ---- Clinical Process Harness (Part 2). It always runs, in its one mode (enforce). A kernel that fails to
    #      build fails the run closed: no agent action is taken (_harness_infra_error below).
    _harness = None
    _harness_runtime_errors = []        # build/hook errors — surfaced in result["harness"]["runtime_errors"]
    _harness_build_failed = False
    try:
        import harness as _Harness
        # substrate from env_type; a specific environment ADAPTER may be named per-task (two datasets of
        # the same substrate with different tool names) — task.environment.adapter, else the default.
        _adapter = (task.get("environment") or {}).get("adapter") if isinstance(task, dict) else None
        _tool_model = None
        if env_type == "tool_sandbox":   # perception tool backend (VLM) -> the ACTUAL model the tools call
            try:
                import adapters.medcta.vlm_backend as _vb
                _bk = _vb.get_backend()
                _tool_model = getattr(_bk, "model", None) or getattr(_bk, "model_path", None)
            except Exception:
                _tool_model = None
        _harness = _Harness.build_kernel(task, env_type=env_type, adapter=_adapter, workspace=workspace,
                                         agent_model=getattr(agent, "model", None), tool_model=_tool_model,
                                         instrumentation=_instr)
    except Exception as _he:
        _harness = None
        _harness_build_failed = True
        _harness_runtime_errors.append("build_kernel: %r" % _he)
    # A GUI run records the portal state a replay cannot rebuild, per env call (observation only, size-bounded;
    # runner/portal_state_record.py). Attached before the first judge call so every judge response is in it.
    import portal_state_record as _PSR
    _psr = _PSR.attach(env, task=task, workspace=workspace, kernel=_harness,
                       path=os.path.join(job_dir, "logs", "portal_state_record.jsonl"))
    if _harness is not None and getattr(_harness, "contract", None) is not None and _harness.contract.meta is not None:
        try:
            from harness.engines.semantic import compile_goal_spec as _cgs
            if "goal_spec" not in _harness.contract.meta:
                _gspec = _cgs(_harness.contract.meta.get("goal"), _harness.contract.meta.get("public_context"),
                              (_harness.ctx.budgeted_judge("goal_spec") if _harness is not None else None))
                if _gspec:
                    _harness.contract.meta["goal_spec"] = _gspec
                _harness.contract.meta.setdefault("available_tools", task.get("available_tools"))
                _harness.contract.meta.setdefault("task_id", task.get("task_id"))
                # propagate the substrate manifest's repair_targets vocabulary (writable-path
                # equivalences) so the effect verifier can resolve a finding the judge mis-located.
                _harness.contract.meta.setdefault("repair_targets",
                    (getattr(_harness.ctx, "manifest", {}) or {}).get("repair_targets") or [])
        except Exception as _ge:
            _harness_runtime_errors.append("goal_spec: %r" % _ge)

    trajectory, last_obs, last_res, finished = [], None, None, False
    if _psr is not None:
        _psr.bind_trajectory(trajectory)
    if hasattr(_harness, "bind_trajectory"):   # the kernel's shell events (shell_hook, shell_refusal) go here
        _harness.bind_trajectory(trajectory)
    if _partial is not None:
        _partial["trajectory"] = trajectory
    # every input for the agent's next turn -- its tool result, harness read results, feedback -- is one typed item
    # in arrival order (runner/turn_inputs.py); none overwrites another
    from turn_inputs import TurnInputs as _TurnInputs
    _turn = _TurnInputs()
    max_repair_turns = MAX_REPAIR_TURNS   # cap on REVISE/BLOCK turns
    _repair_turns = 0

    def _repair_result(stage, source, capability=None, draws_budget=True):
        """The one record of a repair-type result -- a host Nudge / Revision / PostRevision, an
        on_answer Repair, a kernel REVISE or BLOCK at any hook: one record_revision naming its source. Every one but a
        Nudge also draws one repair turn from the task's single repair budget and unlocks repair (ctx.in_repair). A Nudge
        (draws_budget False) is workspace advice bounded by its own cap (MAX_NUDGES_PER_EPISODE): recorded with no repair
        turn (repair_turn None, repair_turn_started False), it never brings the budget closer to repair_budget_exhausted
        and leaves in_repair as it was. -> whether the result is within the budget. What an exhausted budget does is the
        caller's: the kernel paths and the shell's refusals end or flag the episode; a host result is still
        applied (bounded by its own cap)."""
        nonlocal _repair_turns
        if not draws_budget:
            _act.record_revision(getattr(_harness, "ledger", None), stage, step, capability=capability, delivered=True,
                                 repair_turn_started=False, repair_turn=None, suppressed_reason=None, source=source)
            return True
        _repair_turns += 1
        _in_repair[0] = True
        within = _repair_turns <= max_repair_turns
        _act.record_revision(getattr(_harness, "ledger", None), stage, step, capability=capability, delivered=True,
                             repair_turn_started=within, repair_turn=_repair_turns,
                             suppressed_reason=(None if within else "repair_budget_exhausted"), source=source)
        return within
    _env_actions = 0   # the AGENT's executed environment actions; repair turns and harness actions do NOT count
    # Every harness-origin env action (recovery, a forced read, next_read, the forced deliverable write, a content-guard
    # restore) goes through the harness action channel and is counted there against the harness ceilings, never here.
    from harness.action_service import HarnessActionService, RECOVERY_RESERVE_DIVISOR as _RECOVERY_RESERVE_DIVISOR
    _answer_attempts = 0   # count EVERY final answer the agent submits (not just the one that lands)
    _first_answer = None   # the agent EARLIEST submitted answer: the grading text when no final is committed
    # the one UNKNOWN commit under restricted read-back recovery, the agent's or the harness's;
    # it restricts the writes of both (runner/harness/reconcile.py)
    from harness.reconcile import ReconcileState as _ReconcileState
    _reconcile = _ReconcileState()
    _forced_action = None   # harness-dispatched read-only acquisition, run through the NORMAL tool pipeline
    _deliv_writes_used = 0   # at most ONE reserved over-budget deliverable write (off/enforce budget parity)
    # The workspace's middleware (empty where a substrate has no workspace yet). It decides; the shell applies.
    from middleware_host import load_middleware, MiddlewareContext, forced_deliverable_write
    _mw = load_middleware(workspace, task, instrumentation=_instr)
    try:
        record_components(_instr, agent, _mw, _harness if hasattr(_harness, "capabilities") else None, workspace)
    except Exception as _ie:   # measurement must never break a run
        _harness_runtime_errors.append("instrumentation: %r" % (_ie,))

    import middleware_host as _MH
    _observed_history, _answers, _env_budget_spent = [], [], False   # shell state hooks read through the context
    _budget_warned = [False]           # the deliverable budget warning record, at most once per episode

    def _budget_warning_once():
        if _budget_warned[0]:
            return False
        _budget_warned[0] = True
        return True

    def _mw_ctx(at_step, forced=None, answer=False, ended=False, final=False):
        # Workspace code reads the environment only through the context's views (middleware_host.environment_views over
        # env; the control model is the harness action service's one reader).
        _shell = {"controls": lambda: _action_service.list_controls(), "budget_warning_once": _budget_warning_once,
                  "observed_history": lambda: _observed_history, "last_obs": lambda: last_obs,
                  "budget_spent_at_turn_start": lambda: _env_budget_spent,
                  "agent_env_actions": lambda: _env_actions,
                  "harness_env_actions": lambda: _action_service.harness_actions(),
                  "recovery_env_actions": lambda: _action_service.recovery_actions(),
                  "recovery_reserve": lambda: _action_service.reserve(), "in_repair": lambda: _in_repair[0],
                  "forced_pending": lambda: _forced_action is not None, "answers": lambda: _answers,
                  "ledger": lambda: getattr(_harness, "ledger", None),
                  "contract_meta": lambda: getattr(getattr(_harness, "contract", None), "meta", None),
                  "obligations": lambda: (_MH.obligations_of(task, env, trajectory, workspace) if _oblig_status else None),
                  "verdict_feedback": _next_feedback}
        if answer:
            _shell.update(_MH.answer_services(_harness, at_step, trajectory, _instr))
        if final:        # the harness action service: a before_final middleware may open an invocation here
            _shell.update(_MH.harness_action_services(_action_service, at_step=at_step))
        if ended:
            _end = _MH.episode_end_reason(trajectory, finished, _aborted)
            _shell["end_reason"] = (lambda _r=_end: _r)
            if forced is not None:
                forced = (lambda fb, _f=forced, _r=_end: _f(fb, _r))
            _shell.update(_MH.harness_action_services(_action_service, trajectory=trajectory, end_reason=_end))   # the harness action services at episode end
        return MiddlewareContext(env=env, task=task, trajectory=trajectory, step=at_step, max_steps=max_steps,
                                 used_actions=_env_actions + _action_service.harness_actions(),
                                 judge_factory=(lambda consumer: _harness.ctx.budgeted_judge(consumer)
                                                if _harness is not None else None),
                                 errors=_harness_runtime_errors, forced_deliverable_write=forced, shell=_shell)
    # is a public-obligation model available for THIS bench at all? The terminal lock and ctx.obligations() read it
    _oblig_status = _ds_adapter.applies("obligation_gate")
    from harness.obligation_gate import lock_inputs   # (None, None) without an obligation model
    if _oblig_status:
        from harness.obligation_gate import obligation_snapshot
    _ob_before = None                  # conjunction truth BEFORE the action about to run
    _observed_text = []   # running record of what the agent OBSERVES, so the collection nudge can check code coverage
    _coll_on = _ds_adapter.applies("collection_nudge")  # remind, never fill
    # On a substrate whose adapter grants harness actions a reserve (HAB), the harness action service may act past the
    # agent's budget by an ADDITIVE slice (max_steps // RECOVERY_RESERVE_DIVISOR), so recovery can actually ACT after the
    # agent spends its share; 0 elsewhere (PB/MedCTA). The agent keeps the full native budget it has without the harness,
    # and recovery is granted a slice ON TOP: carving recovery's slice out of the agent's ceiling would measure a shorter
    # agent rather than the harness. A run with the harness therefore has a LARGER total action ceiling than the same run
    # without it; that is recorded in provenance (harness_extra_action_budget) and must be reported with the numbers.
    _reserve_granted = bool(_ds_adapter.applies("harness_action_reserve"))
    _action_reserve = (max_steps // _RECOVERY_RESERVE_DIVISOR) if _reserve_granted else 0
    _nudges = [0]                         # Nudges the shell applied this episode (capped by middleware_host.MAX_NUDGES_PER_EPISODE)
    _revisions = [0]                      # Revisions / Repairs the shell applied (capped by middleware_host.MAX_REVISIONS_PER_EPISODE)
    _in_repair = [False]                  # once repair starts, the adapter stops dispatching read-acquisitions
    # (read-acquire per-task state lives in HABAdapter.forced_action)
    _fail_sig = None; _fail_n = 0; _aborted = False  # circuit breaker: abort on repeated identical failing call
    # FAIL-CLOSED: if the harness FAILED to build, do NOT run the agent unprotected and admit it afterward —
    # abort BEFORE the first action (the task is an infrastructure error).
    _harness_infra_error = _harness_build_failed
    if _harness_infra_error:
        trajectory.append({"step": 0, "event_type": "harness_infrastructure_error", "status": "error",
                           "requested_mode": "enforce", "errors": list(_harness_runtime_errors)})
        _aborted = True
    if hasattr(env, "initial_observation"):
        _io = env.initial_observation()
        if _io is not None:
            last_res = _io; last_obs = json.dumps(_io, ensure_ascii=False)[:OBS_MAX_LEN]
            _turn.observation(last_res, last_obs, "initial_observation")
    from harness.executor import ActionExecutor
    # checkpoint credit: after each executed action, which state checkpoint holds and who acted (scorers/credit.py)
    _credit_tracker = _credit.CheckpointCreditTracker(task, lambda: _state_snapshot(env))
    _executor = ActionExecutor(_canon, env_type, _state_hash, _state_snapshot, observer=_credit_tracker.observe)
    # The harness action channel: the one path every harness-origin env action takes -- scope, ceilings, read-only
    # classification, kernel before_action with origin and the authorization ceremony, executor, after_action, event,
    # content guard. Workspace middleware reaches it through its context (middleware_host.harness_action_services; HAB's
    # recovery strategy is workspaces/hab/h0/middleware/recovery.py); the loop dispatches the reads a next_read, a kernel
    # ACQUIRE or an on_answer Read ask for through dispatch_read, and the deliverable enforcement writes through
    # forced_write.
    _action_service = HarnessActionService(
        _harness, _executor, env, task, trajectory, max_steps,
        agent_actions=lambda: _env_actions, reserve=lambda: _action_reserve, errors=_harness_runtime_errors,
        obligations=((lambda: obligation_snapshot(task, env, workspace, actions=trajectory)) if _oblig_status else None),
        observation=lambda: _harness_observation(env), workspace=workspace, reserve_granted=_reserve_granted,
        reconcile=_reconcile)
    def _harness_read_delivered(action, res, obs, _err):
        """An executed harness read (the harness action channel ran it in place of the agent's turn): what the agent is
        handed next turn, and the acquisition accounting of the kernel ledger. -> (harness_acquired, runtime error or
        None)."""
        if _harness is None:
            return None, None
        # hand the FORCED tool result to the agent next turn (the text agent programs render it unconditionally)
        acquired = {"tool": action.get("tool"), "args": action.get("args") or {},
                    "output": (res.get("output") if isinstance(res, dict) and "output" in res else obs)}
        # count a SUCCESSFUL acquisition iff this read raised the VALIDATED-evidence high-water mark.
        try:
            # SUCCESS = this acquisition raised the ledger's validated_evidence_VERSION (a NEW validated,
            # non-foreign evidence SIGNATURE). validated_evidence_version bumps only on a novel signature, so
            # a repeated identical observation (same output) does NOT count as success even though it appends
            # another VALIDATED record. Baseline is this acquisition's own dispatch snapshot.
            _valid_now = getattr(_harness.ledger, "validated_evidence_version", 0)
            _valid_before = action.get("_valid_before")
            if _valid_before is None:
                _valid_before = getattr(_harness.ledger, "_valid_seen", 0)
            _produced_novel = _valid_now > _valid_before
            # effective != executed: close the telemetry record this acquisition opened at plan time.
            _act.record_execution(_harness.ledger, action.get("_acquire_id"), executed=True,
                                  succeeded=(not _err), evidence_added=bool(_produced_novel))
            # NOVEL is not RESOLVING. An uncertain / localization-failed VLM output is still
            # a novel non-empty string, but it does NOT resolve the requirement -> it must not fill the
            # success (resolving) cap and stop acquisition after two hedged reads. successful_acquisitions
            # (the >=2 hard-stop) counts only CONFIDENT novel observations; attempts are still bounded at 3.
            _aa_args = action.get("args") or {}
            _resolving = False
            if _produced_novel:
                try:
                    from harness.observation import observation_reliability, RELIABILITY_CONFIDENT
                    _obs = getattr(_harness.ledger, "observations", []) or []
                    _last = _obs[-1] if _obs else {}
                    _rel = observation_reliability(_last.get("content"), _aa_args.get("attribute"),
                                                   _last.get("localization_resolved"))
                    _resolving = (_rel == RELIABILITY_CONFIDENT)
                except Exception:
                    _resolving = True
            if _resolving:
                _harness.ledger.successful_acquisitions = getattr(_harness.ledger, "successful_acquisitions", 0) + 1
            _harness.ledger._valid_seen = max(_valid_now, getattr(_harness.ledger, "_valid_seen", 0))
            # record the ATTEMPT with the REQUESTED tool/region/attribute (observations store
            # observed_attributes=[] and so lose the requested attribute; without this, diversification could not tell two
            # attribute requests apart). Failed/empty attempts are recorded too (they must block a repeat).
            try:
                _harness.ledger.acquisition_attempts.append(
                    {"tool": action.get("tool"), "region": _aa_args.get("region"),
                     "attribute": _aa_args.get("attribute"),
                     "status": action.get("status") or ("ok" if res is not None else "error"),
                     "produced_novel": bool(_produced_novel), "resolving": bool(_resolving)})
            except Exception:
                pass
        except Exception as _exc:
            return acquired, "acq_success_count: %r" % (_exc,)
        return acquired, None

    # The loop's iteration bound is the agent's: an iteration in which the harness action channel ran a read in place of
    # the agent's turn does not count against it; harness reads are bounded by the channel's ceilings. Every other
    # iteration counts, so the loop always ends.
    _iteration_cap = 0 if _harness_infra_error else (max_steps + max_repair_turns + 2)
    _iterations = 0
    step = -1
    while _iterations < _iteration_cap:
        step += 1
        _env_budget_spent = _env_actions >= max_steps   # the agent's own tool budget; still allow a final-only turn
        _bw = _mw.before_model(_mw_ctx(step))
        if _bw is not None:
            last_res, last_obs = _bw.last_res, _bw.last_obs
            _turn.replace_result(last_res, last_obs, "before_model")
        if _coll_on and last_obs:   # the collection nudge reads this buffer
            _observed_text.append(str(last_obs)[:4000])   # accumulate observed salient facts (codes / IDs)
        if last_obs:
            _observed_history.append(str(last_obs)[:4000])   # ctx.observed_history, on every substrate
        if _forced_action is None:   # a workspace's turn-start read (HAB: read_acquire.py)
            _forced_action = _MH.queued_read(_mw.next_read(_mw_ctx(step)))
        if _forced_action is not None:
            # A harness read (next_read, a kernel ACQUIRE, an on_answer Read) runs through the harness action channel in
            # place of the agent's turn: scope, the channel's ceilings, the manifest's read-only classification (a read that
            # is not read-only is refused; an irreversible commit falls under the irreversible-commit rule), the kernel with
            # origin harness, the executor, the record (origin harness, audience both) and the content guard. It is never
            # charged to the agent's budget, its repair turns or its circuit breaker.
            action = _forced_action; _forced_action = None
            _hd = _action_service.dispatch_read(action, step)
            if _hd.executed:
                _hpost, obs, res, _err = _hd.after, _hd.obs, _hd.outcome.res, _hd.outcome.err
                last_obs = obs; last_res = res
                _read_item = _turn.harness_read_result(action, res, obs, _origin.queued_requester(action)[0] or "next_read",
                                                        _hd.event)
                if (_hpost is not None and _hpost.type == "ACQUIRE" and _harness is not None
                        and not _reconcile.active):   # a kernel ACQUIRE after it: the next harness read, as for any action
                    _acqa = _act.plan(_hpost, "after_action", step, ledger=_harness.ledger,
                                      vocabulary=env.available_tools())
                    trajectory.extend(_acqa.events)
                    if _acqa.dispatchable:
                        _forced_action = _acqa.forced_action
                        _turn.feedback({"decision": "ACQUIRE", "stage": "after_action",
                            "reason": _MH.decision_text(_hpost) or _next_feedback(_hpost, "after_action")["reason"],
                            "missing_obligations": []}, "kernel:after_action")
                _read_item["acquired"], _hacc_err = _harness_read_delivered(action, res, obs, _err)
                if _hacc_err:
                    _harness_runtime_errors.append(_hacc_err)
                continue
            if getattr(_hd.verdict, "type", None) == "ACQUIRE" and _harness is not None:
                # the kernel asked for another read before this one: that read is the next harness read (the refused
                # read never ran), as a before_action ACQUIRE defers any action. Nothing ran in this iteration, so it
                # counts as a loop iteration (a kernel that keeps deferring reads cannot hold the loop open)
                _acqb = _act.plan(_hd.verdict, "before_action", step, ledger=_harness.ledger,
                                  vocabulary=env.available_tools())
                trajectory.extend(_acqb.events)
                if _acqb.dispatchable:
                    _forced_action = _acqb.forced_action
                    _turn.feedback({"decision": "ACQUIRE", "stage": "before_action",
                        "reason": _MH.decision_text(_hd.verdict) or _next_feedback(_hd.verdict, "before_action")["reason"],
                        "missing_obligations": []}, "kernel:before_action")
                    _iterations += 1
                    continue
            # refused: nothing ran; the agent takes this turn
        _iterations += 1
        _state = {"goal": task.get("goal"), "context": task.get("context"), "tools": env.available_tools()}
        _state.update(_turn.state(last_res, last_obs))
        action = agent.act(_state)
        _turn.consumed()   # marked consumed only once the agent's act returned; the next turn's inputs start empty
        if isinstance(action, dict):       # the shell's carrier keys (_harness_acquire, _requester, _mutation_authorization_id,
            action = {k: v for k, v in action.items() if not str(k).startswith("_")}   # ...) are never the agent's
        if not isinstance(action, dict) or "type" not in action:  # agent contract violation
            trajectory.append({"step": step, "event_type": "agent_error", "error": "invalid_action", "raw": str(action)[:200], "status": "error"})
            finished = True; break
        if action["type"] == "tool_call" and action.get("tool") in getattr(env, "EPISODE_END_TOOLS", ()) \
                and not action.get("_harness_acquire"):
            # environment fidelity: the substrate's episode-end tool (HealthAdminBench done()) ends the episode as
            # upstream does; the shell's one episode-end path is the agent's final, with no answer text
            trajectory.append({"step": step, "event_type": "shell_end_signal", "tool": action.get("tool"), "status": "ok"})
            action = {"type": "final", "answer": ""}
        if action["type"] == "final":
            # host before_final (HAB, in manifest order: recovery -- which may act through the harness action service and
            # hand the final back as a Revision(stage=reentry) --, obligation_nudge, grounding_audit)
            _action_service.episode_id = _MH.hook_call_id("before_final", step)   # recovery_episode_id
            _bf_ctx = _mw_ctx(step, final=True)
            try:
                _mf = _mw.before_final(_bf_ctx, action.get("answer", ""))
            finally:
                _MH.close_scope(_bf_ctx)   # harness actions only inside the hook call that opened them
            if isinstance(_mf, _MH.Nudge) and not _MH.nudge_admitted(_nudges, _mf, "before_final", step, trajectory):
                _mf = None                   # past the episode's Nudge cap the final goes on
            if (_mf is not None and not isinstance(_mf, _MH.Nudge)
                    and not _MH.revision_admitted(_revisions, _mf, "before_final", step, trajectory)):
                _mf = None                   # past the episode's revision cap the final goes on
            if isinstance(_mf, _MH.Nudge):   # refused; recorded, no repair turn, repair stays locked
                _repair_result("before_final", "host:Nudge", draws_budget=False)
                _fb, last_res, last_obs = _emit(_mf.label, _mf.stage, _mf.feedback, _mf.obs)
                _turn.feedback(_fb, "nudge:before_final", last_res, last_obs)
                continue
            if _mf is not None:
                _repair_result("before_final", "host:Revision")   # closure re-entry may draw the reserved repair budget
                last_res, last_obs = {"feedback": _mf.feedback}, _mf.obs
                _turn.feedback({"decision": "REVISE", "stage": _mf.stage, "reason": _mf.feedback},
                               "revision:before_final", last_res, last_obs)
                continue
            _answer_attempts += 1
            if _answer_attempts == 1:
                _first_answer = action.get("answer", "") or ""   # protect the agent EARLIEST answer from later harness-driven drift
            _ahash = hashlib.sha1((action.get("answer", "") or "").encode("utf-8")).hexdigest()[:12]
            trajectory.append({"step": step, "event_type": "answer_attempt", "attempt_index": _answer_attempts,
                               "answer_hash": _ahash, "answer": (action.get("answer", "") or "")[:500], "status": "ok",
                               "origin": "agent"})
            _answers.append((_answer_attempts, _ahash, action.get("answer", "") or ""))
            _oa_ctx = _mw_ctx(step, answer=True)
            _oa = _mw.on_answer(_oa_ctx, action.get("answer", ""))
            if isinstance(_oa, _MH.Repair) and not _MH.revision_admitted(_revisions, _oa, "on_answer", step, trajectory):
                _oa = None                           # past the episode's revision cap the final goes on
            if isinstance(_oa, _MH.Accept):         # the workspace delivers: the shell records it and ends the episode
                _av = None
                if _harness is not None:
                    # the delivered text's own verdict decides, whatever the strategy chose. A candidate finalize
                    # is repair exhaustion: a must_resolve violation is withheld exactly as on the path below.
                    try:
                        _av = _MH.delivered_verdict(_oa_ctx, _oa.text, flagged=_oa.flag is not None)   # recorded
                    except Exception as _he:
                        _harness_runtime_errors.append("before_final: %r" % _he)
                        trajectory.append({"step": step, "event_type": "harness_escalation", "stage": "before_final",
                                           "reason": "hook_error", "status": "error"}); _aborted = True; break
                    if _MH.must_resolve_violation(_av):
                        _MH.withhold_unresolved_violation(
                            _harness, trajectory, _harness_runtime_errors, _oa.text, _av.feedback, step,
                            _answer_attempts, _canon.canonical_action({"type": "final", "answer": _oa.text}, env_type),
                            _deliverable_is_env_state)
                        finished = True; break
                _oa_flag = _MH.delivery_flag(_oa, _av)
                if _oa_flag is not None and _harness is not None:
                    try: _harness.record_flagged_final(_oa.text, flag=_oa_flag, step=step)
                    except Exception as _re: _harness_runtime_errors.append("record_flagged_final: %r" % _re)
                trajectory.append(_MH.accept_event(_oa, step, _answer_attempts, _canon, env_type,
                                                  [a[2] for a in _answers]))   # author by text identity
                finished = True; break
            if isinstance(_oa, _MH.Read):           # another harness read, planned at before_final_readapt
                _forced_action = _MH.plan_answer_read(_oa, step, getattr(_harness, "ledger", None),
                                                      trajectory).forced_action
            elif isinstance(_oa, _MH.Repair):       # a re-answer round, accounted as every repair-type result
                _repair_result("on_answer", "host:Repair")
            elif _oa is not None:
                raise _MH.unexpected("on_answer", _oa)
            if _oa is not None:
                if _oa.feedback is not None:
                    _turn.feedback(_oa.feedback, "on_answer")
                continue
            if _harness is not None:                    # final answer is a commit point
                try:
                    _hf = _harness.before_final(action.get("answer", ""), step=step)
                except Exception as _he:
                    _hf = None
                    _harness_runtime_errors.append("before_final: %r" % _he)
                    trajectory.append({"step": step, "event_type": "harness_escalation", "stage": "before_final",
                                       "reason": "hook_error", "status": "error"}); _aborted = True; break
                if _hf is not None:
                    trajectory.extend(_hf.events)
                    _afv = (_MH.agent_text(_mw.after_final_verdict(_mw_ctx(step), action.get("answer", ""), _hf))
                            if _hf.type != "ALLOW" else None)   # a workspace's agent text for this verdict
                    if _hf.type == "ACQUIRE":   # ACTIVE read-only evidence acquisition: dispatch through the
                        # NORMAL tool pipeline -> a real tool_call event (scorers count it), error +
                        # localization normalization (failed/fallback obs is NOT marked valid), after_action
                        # observation + provenance, and last_obs update -> the agent re-reasons on a NORMAL
                        # observation, not a truncated authority message. Read-only tools only.
                        _acqf = _act.plan(_hf, "before_final", step, ledger=_harness.ledger,
                                              vocabulary=env.available_tools())
                        trajectory.extend(_acqf.events)
                        if _acqf.dispatchable:
                            _forced_action = _acqf.forced_action
                            # the workspace's text (after_final_verdict; in the MedCTA default workspace, middleware/answer_candidate.py), else
                            # the raising middleware's; a decision with neither is told its reason, like any other
                            _turn.feedback({"decision": "ACQUIRE", "stage": "before_final",
                                "reason": _afv or _MH.decision_text(_hf) or _next_feedback(_hf, "before_final")["reason"],
                                "missing_obligations": []}, "kernel:before_final")
                            continue   # the acquisition runs via the normal pipeline next iteration, then re-answer
                    if _hf.type == "DEFER_FINAL":
                        # CONTROL-ONLY gate: this final is not accepted yet and the agent keeps working. It
                        # demands no particular repair -- that is exactly what separates it from REVISE,
                        # which carries a direction, so the two do not share a branch. A final is deferred
                        # only here: before_final is the one hook where there is a final to defer.
                        _df_cap = _act.capability_of(_hf)
                        _df_extra = ((getattr(_hf, "raw", None) and getattr(_hf.raw, "extra", None)) or {})
                        _repair_turns += 1
                        _df_out = _repair_turns > max_repair_turns
                        _act.record_defer(_harness.ledger, "before_final", step, capability=_df_cap,
                                          deferred=True, repair_turn_started=(not _df_out),
                                          repair_turn=_repair_turns,
                                          control_only=bool(_df_extra.get("control_only")),
                                          instruction_supplied=_df_extra.get("instruction_supplied"),
                                          exhausted=_df_out)
                        trajectory.append({"step": step, "event_type": "final_deferred", "stage": "before_final",
                                           "capability": _df_cap, "final_deferred": True,
                                           "repair_turn_started": (not _df_out), "repair_turn": _repair_turns,
                                           "budget_exhausted": _df_out, "feedback": _hf.feedback, "status": "ok"})
                        if _df_out:
                            # A gate cannot hold a final forever. On exhaustion the answer is DELIVERED and
                            # flagged as deferred-without-resolution. Withholding it is reserved for the
                            # kernel's evidence-contradiction path, which is a safety claim; "not yet" is
                            # not one, and abstaining on it would forfeit runs the harness never disputed.
                            trajectory.append({"step": step, "event_type": "final_answer",
                                "thought": action.get("answer", ""), "status": "ok",
                                "verification_flag": "deferred_unresolved",
                                "final_disposition": "flagged_delivery_defer_budget_exhausted",
                                "origin": "agent", "final_answer_author": "agent",
                                "answer_attempt_index": _answer_attempts, "harness_feedback": _hf.feedback,
                                "canonical_action": _canon.canonical_action(action, env_type)})
                            try:
                                _harness.record_flagged_final(action.get("answer", ""),
                                                              flag="deferred_unresolved", step=step)
                            except Exception as _re:
                                _harness_runtime_errors.append("record_flagged_final: %r" % _re)
                            finished = True; break
                        _turn.feedback(_next_feedback(_hf, "before_final"), "kernel:before_final")
                        _in_repair[0] = True   # the deferral is worth nothing without budget to act on it
                        continue
                    if _hf.type in ("REVISE", "BLOCK"):
                        _extra_hf = ((getattr(_hf, "raw", None) and getattr(_hf.raw, "extra", None)) or {})
                        if _extra_hf.get("candidate"):       # candidate review: a workspace holds A (after_final_verdict)
                            _repair_result("before_final", "kernel:" + _hf.type, _act.capability_of(_hf))
                            _turn.feedback(_next_feedback(_hf, "before_final"), "kernel:before_final")
                            if _repair_turns > max_repair_turns:   # no candidate produced in budget -> keep A
                                trajectory.append({"step": step, "event_type": "final_answer", "thought": action.get("answer", ""),
                                    "status": "ok", "verification_flag": "unverified_grounding",
                                    "final_disposition": "kept_original_no_candidate",
                                    "origin": "agent", "final_answer_author": "agent",
                                    "answer_attempt_index": _answer_attempts,
                                    "harness_feedback": _hf.feedback, "canonical_action": _canon.canonical_action(action, env_type)})
                                try: _harness.record_flagged_final(action.get("answer", ""), flag="unverified_grounding", step=step)
                                except Exception as _re: _harness_runtime_errors.append("record_flagged_final: %r" % _re)
                                finished = True; break
                            continue
                        _repair_result("before_final", "kernel:" + _hf.type, _act.capability_of(_hf))
                        _turn.feedback(_next_feedback(_hf, "before_final"), "kernel:before_final")
                        _mr_viol = _MH.must_resolve_violation(_hf)
                        _budget_out = _repair_turns > max_repair_turns
                        # MUST-RESOLVE (high-confidence, LOCALIZED evidence contradiction): a confirmed conflict
                        # is NOT eligible for unverified_grounding flagged delivery. Bounded repair; on
                        # exhaustion the original (refuted) answer is WITHHELD -> safe abstention, never the
                        # violating answer, or delivered flagged when the deliverable is environment state
                        # (withhold_unresolved_violation, shared with an on_answer Accept). Every other
                        # REVISE keeps the flagged-delivery path below.
                        if _mr_viol:
                            if _budget_out:
                                _MH.withhold_unresolved_violation(
                                    _harness, trajectory, _harness_runtime_errors, action.get("answer", ""),
                                    _hf.feedback, step, _answer_attempts, _canon.canonical_action(action, env_type),
                                    _deliverable_is_env_state)
                                finished = True; break
                            continue                     # within budget -> agent gets the targeted REVISE
                        # any other REVISE: repair within budget, flagged delivery on exhaustion (advisory).
                        if _budget_out:
                            trajectory.append({"step": step, "event_type": "final_answer", "thought": action.get("answer", ""),
                                               "status": "ok", "verification_flag": "unverified_grounding",
                                               "final_disposition": "flagged_delivery",
                                               "origin": "agent", "final_answer_author": "agent",
                                               "answer_attempt_index": _answer_attempts,
                                               "harness_feedback": _hf.feedback,
                                               "canonical_action": _canon.canonical_action(action, env_type)})
                            try: _harness.record_flagged_final(action.get("answer", ""), flag="unverified_grounding", step=step)
                            except Exception as _re: _harness_runtime_errors.append("record_flagged_final: %r" % _re)
                            finished = True; break
                        continue
                    if _hf.type == "ESCALATE":
                        _xe = (getattr(_hf.raw, "extra", None) or {}) if getattr(_hf, "raw", None) is not None else {}
                        if not _xe.get("side_effecting"):   # epistemic answer (no env side effect) -> DELIVER with flag
                            trajectory.append({"step": step, "event_type": "final_answer", "thought": action.get("answer", ""),
                                               "status": "ok", "verification_flag": _xe.get("verification_flag", "unresolved_risk"),
                                               "final_disposition": "flagged_delivery_escalate",
                                               "origin": "agent", "final_answer_author": "agent",
                                               "answer_attempt_index": _answer_attempts,
                                               "harness_feedback": _hf.feedback,
                                               "canonical_action": _canon.canonical_action(action, env_type)})
                            try: _harness.record_flagged_final(action.get("answer", ""), flag=_xe.get("verification_flag", "unresolved_risk"), step=step)
                            except Exception as _re: _harness_runtime_errors.append("record_flagged_final: %r" % _re)
                            finished = True; break
                        trajectory.append({"step": step, "event_type": "harness_escalation",   # operational write -> fail-closed
                                           "stage": "before_final", "final_disposition": "escalated_side_effecting_final",
                                           "answer_attempt_index": _answer_attempts,
                                           "feedback": _hf.feedback, "status": "error"})
                        _aborted = True; break
            trajectory.append({"step": step, "event_type": "final_answer", "thought": action.get("answer", ""), "status": "ok",
                               "final_disposition": "clean_commit",
                               "origin": "agent", "final_answer_author": "agent",
                               "answer_attempt_index": _answer_attempts,
                               "canonical_action": _canon.canonical_action(action, env_type)})
            finished = True; break
        if action["type"] == "tool_call_truncated":  # cut-off tool call -> ask to re-issue the INTENDED call (any tool)
            _raw = action.get("raw", "") or ""
            _m = re.search(r'"(?:tool|name)"\s*:\s*"([A-Za-z0-9_]+)"', _raw)
            _tool = _m.group(1) if _m else None
            trajectory.append({"step": step, "event_type": "agent_error", "error": "truncated_tool_call",
                               "intended_tool": _tool, "raw": _raw[:200], "status": "error"})
            if _tool:
                _fb = ("Your `%s` tool call was CUT OFF before the JSON closed. Re-issue the COMPLETE `%s` call now "
                       "-- do NOT drop the action you intended. If the payload was long (e.g. write_file), keep it "
                       "focused so the JSON finishes." % (_tool, _tool))
            else:
                _fb = ("Your previous tool call was CUT OFF before the JSON closed. Re-issue the COMPLETE tool call "
                       "you intended -- do NOT drop it; keep any long content focused so the JSON finishes.")
            last_res = {"feedback": _fb}
            last_obs = "truncated_tool_call"
            _turn.feedback(None, "truncated_tool_call", last_res, last_obs)
            continue
        if action["type"] != "tool_call" or not action.get("tool"):
            trajectory.append({"step": step, "event_type": "agent_error", "error": "bad_action_type", "raw": str(action)[:200], "status": "error"})
            finished = True; break
        if _reconcile.active:                      # RESTRICTED recovery: only read/inspect until the commit is reconciled
            # A write is what the kernel classifies as one: the substrate manifest's semantic_type through
            # harness.semantics.canonicalize_on_page with the page's controls, as kernel before_action classifies it.
            from harness.semantics import canonicalize_on_page as _sem_canon
            from harness.authorization import MUTATION_TYPES as _MUTATION_TYPES
            _hobs = _harness_observation(env)
            _rst = _sem_canon(action, getattr(_harness, "manifest", None),
                              _hobs.get("controls") if isinstance(_hobs, dict) else None).semantic_type
            if _rst in _MUTATION_TYPES:
                # the refusal takes a turn from the agent: one repair turn, as the shell's other refusals
                _repair_turns += 1
                trajectory.append({"step": step, "event_type": "reconcile_block", "tool": action.get("tool"),
                                   "reason": "unconfirmed commit -- read back before any new write",
                                   "repair_turn": _repair_turns, "status": "ok",
                                   **({} if _reconcile.origin == "agent" else {"commit_origin": _reconcile.origin})})
                if _reconcile.origin == "agent":
                    last_res = {"feedback": "You have an UNCONFIRMED prior commit (it timed out). Use a READ/inspect "
                                "tool to read back whether it landed BEFORE any new write."}
                else:
                    last_res = {"feedback": "The harness has an UNCONFIRMED prior commit (it timed out). Use a "
                                "READ/inspect tool to read back whether it landed BEFORE any new write."}
                last_obs = "reconcile_pending"
                _turn.feedback(None, "reconcile_block", last_res, last_obs)
                if _repair_turns > max_repair_turns:
                    trajectory.append({"step": step, "event_type": "repair_budget_exhausted", "stage": "reconcile",
                                       "status": "escalated", "reason": "repair_budget_exhausted"})
                    _aborted = True; break
                continue
        _deliv_write = (_mw.exempt_from_budget(_mw_ctx(step), action) and _deliv_writes_used < 1)
        if _env_budget_spent and _deliv_write:
            _deliv_writes_used += 1   # consume the single reserved deliverable slot (success or fail)
        if _env_budget_spent and not _deliv_write:   # past the agent's tool budget: allow ONLY the one exact deliverable write
            _repair_turns += 1
            trajectory.append({"step": step, "event_type": "shell_refusal", "kind": "tool_allowance",
                               "tool": action.get("tool"), "repair_turn": _repair_turns, "status": "refused"})
            _turn.feedback({"decision": "REVISE", "stage": "runtime_budget", "missing_obligations": [],
                "reason": "Your exploration budget is used up. Do NOT call another tool; give your best "
                          "final answer from the evidence already gathered."}, "runtime_budget")
            if _repair_turns > max_repair_turns:
                trajectory.append({"step": step, "event_type": "tool_budget_exhausted", "status": "escalated"})
                # the refusal and the abort are the shell's; a workspace may still act once, at after_episode, through
                # the post-stop allowance (ctx.post_stop_allowance, the reserve re-granted at its ceiling)
                _aborted = True; break
            continue
        if _harness is not None:
            try:
                _ob_before = obligation_snapshot(task, env, workspace, actions=trajectory, pending=action) if _oblig_status else None
                _hb = _harness.before_action(action, _harness_observation(env), step=step)   # the agent's action
            except Exception as _he:
                _hb = None
                _harness_runtime_errors.append("before_action: %r" % _he)
                trajectory.append({"step": step, "event_type": "harness_escalation", "stage": "before_action",
                                   "reason": "hook_error", "status": "error"}); _aborted = True; break
            _rr = getattr(_harness.ledger, "_reference_repairs_pending", None)
            if _rr:                       # reference_integrity auto-stripped ungrounded refs from the payload
                for _r in _rr:
                    trajectory.append({"step": step, "event_type": "reference_repaired", "stage": "before_action",
                                       "path": _r.get("path"), "stripped_ref": _r.get("value"), "status": "ok"})
                _harness.ledger._reference_repairs_pending = []
            if _hb is not None:
                trajectory.extend(_hb.events)
                if _hb.type == "ACQUIRE":   # AMPLIFICATION: acquire a required read-only record BEFORE this
                    # commit, then defer the commit so the agent re-decides WITH the evidence. Runs via the NORMAL
                    # tool pipeline (real tool_call, provenance, last_obs update); read-only tools only.
                    _acqb = _act.plan(_hb, "before_action", step, ledger=_harness.ledger,
                                          vocabulary=env.available_tools())
                    trajectory.extend(_acqb.events)
                    if _acqb.dispatchable:
                        _forced_action = _acqb.forced_action
                        # the raising middleware's text; a decision without one is told its reason, like any other
                        _turn.feedback({"decision": "ACQUIRE", "stage": "before_action",
                            "reason": _MH.decision_text(_hb) or _next_feedback(_hb, "before_action")["reason"],
                            "missing_obligations": []}, "kernel:before_action")
                        continue   # the acquisition runs next iteration; the original commit is deferred
                if _hb.type in ("REVISE", "BLOCK"):     # do NOT execute the tool; ADDITIVE feedback (env obs intact)
                    _repair_result("before_action", "kernel:" + _hb.type, _act.capability_of(_hb))
                    _turn.feedback(_next_feedback(_hb, "before_action"), "kernel:before_action")
                    if _repair_turns > max_repair_turns:
                        trajectory.append({"step": step, "event_type": "repair_budget_exhausted", "stage": "before_action", "status": "escalated", "reason": "repair_budget_exhausted"}); _aborted = True; break
                    continue
                if _hb.type == "ESCALATE":
                    # a before_action ESCALATE is CORRECTIVE, not fatal. The tool is NOT executed;
                    # feed the concern back and let the agent choose a DIFFERENT action toward the objective
                    # with remaining budget, instead of killing the run mid-episode. Bounded by the repair
                    # budget (identical to REVISE/BLOCK); only an exhausted repair loop aborts.
                    _repair_turns += 1
                    trajectory.append({"step": step, "event_type": "harness_escalation", "stage": "before_action",
                                       "feedback": _hb.feedback, "status": "corrective"})
                    if _repair_turns > max_repair_turns:
                        trajectory.append({"step": step, "event_type": "repair_budget_exhausted", "stage": "before_action",
                                           "status": "escalated", "reason": "escalation_unresolved"}); _aborted = True; break
                    _turn.feedback(_next_feedback(_hb, "before_action"), "kernel:before_action")
                    continue
        if _harness is not None and not _env_budget_spent:
            # host before_tool runs BEFORE the authorization reserve: a tool it refuses leaves the matched
            # authorization AVAILABLE, never RESERVED for an action that does not run
            _mt = _mw.before_tool(_mw_ctx(step), action)
            if isinstance(_mt, _MH.Nudge) and not _MH.nudge_admitted(_nudges, _mt, "before_tool", step, trajectory):
                _mt = None                   # past the episode's Nudge cap the action goes on
            if (_mt is not None and not isinstance(_mt, _MH.Nudge)
                    and not _MH.revision_admitted(_revisions, _mt, "before_tool", step, trajectory)):
                _mt = None                   # past the episode's revision cap the action goes on
            if _mt is not None and _harness.ledger is not None:
                _harness.ledger.pending_authorization = None   # the refused action's match is not reserved
            if isinstance(_mt, _MH.Nudge):   # refused; recorded, no repair turn, repair stays locked
                _repair_result("before_tool", "host:Nudge", draws_budget=False)
                _fb, last_res, last_obs = _emit(_mt.label, _mt.stage, _mt.feedback, _mt.obs)
                _turn.feedback(_fb, "nudge:before_tool", last_res, last_obs)
                continue
            if _mt is not None:
                _repair_result("before_tool", "host:Revision")
                last_res, last_obs = {"feedback": _mt.feedback}, _mt.obs
                _turn.feedback({"decision": "REVISE", "stage": _mt.stage, "reason": _mt.feedback},
                               "revision:before_tool", last_res, last_obs)
                continue
        _pending_auth = None   # the mutation authorization the executor will dispatch for THIS action
        if (_harness is not None and _hb is not None and _hb.type == "ALLOW"
                and getattr(_harness.ledger, "pending_authorization", None) is not None):
            _cand_auth = _harness.ledger.pending_authorization
            if _harness.ledger.reserve_authorization(_cand_auth):   # AVAILABLE -> RESERVED (combined winner is ALLOW)
                _pending_auth = _cand_auth
            else:   # fail-closed: a matched auth that is NOT AVAILABLE cannot be reserved -> do NOT dispatch/execute
                trajectory.append({"step": step, "event_type": "authorization_reserve_failed",
                                   "authorization_id": getattr(_cand_auth, "authorization_id", None),
                                   "auth_status": getattr(_cand_auth, "status", None), "status": "ok"})
                _turn.feedback({"decision": "BLOCK", "stage": "authorization",
                                "reason": "authorization could not be reserved (not AVAILABLE)", "missing_obligations": []},
                               "authorization")
                continue
        _outcome = _executor.execute_and_normalize(action, env, ledger=(_harness.ledger if _harness is not None else None), auth=_pending_auth)   # dispatches the reserved authorization (if any) with the action
        res = _outcome.res; _err = _outcome.err; _recon = _outcome.recon
        _state_before = _outcome.state_before; _state_after = _outcome.state_after
        _snap_before = _outcome.snap_before; _snap_after = _outcome.snap_after
        _result_status = _outcome.result_status
        _env_actions += 1   # the environment call HAPPENED (count before any post-action escalate / circuit-break)
        _tool_item = _turn.tool_result(action, res)   # the action's result is this turn's input before any verdict on it
        if _reconcile.active:                        # a read executed under recovery -> try to resolve the pending commit
            _rp = _reconcile.pending
            try:
                _rr = env.reconcile_write(_rp["action"]["tool"], _rp["action"].get("args", {}), _rp["res"])
            except Exception as _rrx:
                _rr = {"confirmed": None, "detail": "reconcile_error:%r" % (_rrx,)}
            if _rr and _rr.get("confirmed") is True:
                trajectory.append({"step": step, "event_type": "reconcile_resolved", "outcome": "committed", "detail": _rr.get("detail"), "status": "ok",
                                   **({} if _rp["origin"] == "agent" else {"commit_origin": _rp["origin"]})}); _reconcile.clear()
            elif _rr and _rr.get("confirmed") is False:
                trajectory.append({"step": step, "event_type": "reconcile_resolved", "outcome": "failed", "detail": _rr.get("detail"), "status": "ok",
                                   **({} if _rp["origin"] == "agent" else {"commit_origin": _rp["origin"]})})
                # To the agent next turn, after this read's own result (the turn inputs keep both, and any later
                # decision in this iteration after them).
                _turn.feedback({"decision": "RECONCILE", "stage": "reconcile",
                                "reason": ("Read-back confirms the prior commit did NOT land -- you may re-attempt it."
                                           if _rp["origin"] == "agent" else
                                           "Read-back confirms the harness's prior commit did NOT land; new writes are "
                                           "no longer restricted.")}, "reconcile")
                _reconcile.clear()
            else:
                _rp["budget"] -= 1
                if _rp["budget"] <= 0:
                    trajectory.append({"step": step, "event_type": "harness_escalation", "stage": "reconcile", "reason": "reconcile_budget_exhausted", "status": "error"}); _aborted = True; break
        if _recon and _recon.get("confirmed") is not None:
            trajectory.append({"step": step, "event_type": "reconciliation", "tool": action["tool"],
                               "confirmed": _recon.get("confirmed"), "detail": _recon.get("detail"), "status": "ok"})
        _hpost = None
        if _harness is not None:
            # OBSERVED commit + obligation status for the terminal lock. Both None when this bench
            # declares no obligation spec -> the kernel keeps its original behaviour.
            try:
                # the conjunction with this action's own decision in it (`pending`), before and after the action: a
                # submission conjunct newly true across THIS action == it committed (obligation_gate.lock_inputs)
                _ob_after = obligation_snapshot(task, env, workspace, actions=trajectory, pending=action) if _oblig_status else None
                (_harness.ledger.public_obligations_met,
                 _harness.ledger.observed_submission_realized) = lock_inputs(_ob_before, _ob_after)
            except Exception as _oe:
                _harness.ledger.public_obligations_met = None
                _harness.ledger.observed_submission_realized = None
                _harness_runtime_errors.append("obligations_met: %r" % _oe)
            try:
                _hpost = _executor.run_after_action(_harness, action, _outcome, step)   # the agent's action
            except Exception as _he:
                _hpost = None
                _harness_runtime_errors.append("after_action: %r" % _he)
                trajectory.append({"step": step, "event_type": "harness_escalation", "stage": "after_action",
                                   "reason": "hook_error", "status": "error"}); _aborted = True; break
        if _pending_auth is not None and _harness is not None:   # finalize the DISPATCHED auth AFTER after_action + commit verification (never VERIFIED on a read-back alone)
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
        # ACTOR PROVENANCE: this is the agent's own action (a harness read is recorded by the harness action
        # channel, origin harness, audience both).
        # AUDIT: this agent action arrived in near-miss markup and was recovered by the text agent program's
        # parser (its protocol_variant_last). Recorded as a KEY on the tool_call event rather than a new
        # event type, so no scorer's event counts shift.
        _apv = getattr(agent, "_protocol_variant_last", None)
        try:
            agent._protocol_variant_last = None
        except Exception:
            pass
        _ev, obs = _executor.build_event(action, _outcome, step, origin="agent", audience="agent")   # the agent's tool_call event + the observation handed back
        if _apv:
            _ev["protocol_variant"] = _apv
        trajectory.append(_ev)
        _turn.set_event(_tool_item, _ev)
        _tool_item["observation"] = obs
        if _hpost is not None:
            trajectory.extend(_hpost.events)
            if _hpost.type == "RECONCILE":         # UNKNOWN commit -> RECOVER (read back), do NOT terminate
                _reconcile.enter(action, res, "agent", step=step, budget=RECONCILE_BUDGET)
                trajectory.append({"step": step, "event_type": "reconcile_enter", "stage": "after_action",
                                   "feedback": _hpost.feedback, "status": "ok"})
                _turn.feedback(_next_feedback(_hpost, "after_action"), "kernel:after_action")
            elif _hpost.type == "ESCALATE":        # retrospective escalation TERMINATES the run
                trajectory.append({"step": step, "event_type": "harness_escalation",
                                   "stage": "after_action", "feedback": _hpost.feedback, "status": "error"})
                _aborted = True; break
            if _hpost.type in ("REVISE", "BLOCK") and _hpost.feedback:   # fold into next obs (RESERVE room)
                _max = OBS_MAX_LEN
                _htxt = "\n[HARNESS] " + json.dumps(_hpost.feedback, ensure_ascii=False)
                obs = (_htxt[:_max] if len(_htxt) >= _max else obs[:_max - len(_htxt)] + _htxt)
                _tool_item["observation"] = obs
                _turn.feedback(_next_feedback(_hpost, "after_action"), "kernel:after_action")
                # A REVISE is a REPAIR TURN at every hook, or the primitive is misnamed at that hook.
                # The two lines above alone would only splice the instruction into an observation string: it would
                # not count against the repair budget and would not guarantee a turn in which the agent could act on
                # it (a REVISE issued on the last permitted step would be delivered to nobody yet recorded as
                # delivered). The action here has already run and cannot be undone, so "repair" means the agent gets
                # a turn to change what it does NEXT. Same single repair budget as the other two hooks: this is one
                # harness deciding how many times it may send the agent back, not three independent allowances.
                _rv_cap = _act.capability_of(_hpost)
                if _repair_result("after_action", "kernel:" + _hpost.type, _rv_cap):
                    trajectory.append({"step": step, "event_type": "harness_revision", "stage": "after_action",
                                       "capability": _rv_cap, "revision_delivered": True,
                                       "repair_turn_started": True, "repair_turn": _repair_turns, "status": "ok"})
                else:
                    # Bounded, but NOT fatal. before_action aborts on an exhausted repair loop because the
                    # agent is stuck proposing; here the action already succeeded, so killing the episode
                    # would forfeit a run over the harness's own talkativeness. The instruction is still
                    # delivered; it just stops claiming a turn, and says so.
                    trajectory.append({"step": step, "event_type": "harness_revision", "stage": "after_action",
                                       "capability": _rv_cap, "revision_delivered": True,
                                       "repair_turn_started": False,
                                       "suppressed_reason": "repair_budget_exhausted", "status": "ok"})
            if _hpost.type == "ACQUIRE" and _forced_action is None and not _reconcile.active:
                # The third hook. ACQUIRE is hook-INDEPENDENT ("run a read-only tool for me"), so it goes
                # through the SAME actuator as the other two; only the loop control differs -- the action
                # here has already run, so there is nothing to defer, the acquisition simply runs next
                # iteration and its result reaches the agent as a normal observation.
                _acqa = _act.plan(_hpost, "after_action", step, ledger=_harness.ledger,
                                      vocabulary=env.available_tools())
                trajectory.extend(_acqa.events)
                if _acqa.dispatchable:
                    _forced_action = _acqa.forced_action
                    # the raising middleware's text; a decision without one is told its reason, like any other
                    _turn.feedback({"decision": "ACQUIRE", "stage": "after_action",
                        "reason": _MH.decision_text(_hpost) or _next_feedback(_hpost, "after_action")["reason"],
                        "missing_obligations": []}, "kernel:after_action")
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
        _pt = _mw.after_tool(_mw_ctx(step), action, res)   # the action ran; a workspace may overwrite what it left
        if _pt is not None:
            _fb, last_res, last_obs = _emit("REVISE", _pt.stage, _pt.feedback, _pt.obs)
            _turn.feedback(_fb, "post_revision:after_tool", last_res, last_obs)
            _repair_result("after_tool", "host:PostRevision")
    if not finished and not _aborted:  # ran out of steps without a final answer (circuit-breaker abort logs its own event)
        trajectory.append({"step": max_steps, "event_type": "agent_error", "error": "max_steps_exceeded", "status": "error"})
    _action_service.episode_id = _MH.hook_call_id("after_episode", max_steps)   # recovery_episode_id
    _ae_ctx = _mw_ctx(max_steps, ended=True, forced=lambda fb, end_reason: forced_deliverable_write(
        fb, agent, env, task, trajectory, _action_service, max_steps, end_reason, turn=_turn))
    try:
        _mw.after_episode(_ae_ctx)
    finally:
        _MH.close_scope(_ae_ctx)   # no harness action after the hook that opened it
    try:
        _caps = env.capabilities()   # four-state manifest captured while env is alive (pre-teardown)
    except Exception:
        _caps = {}
    env.teardown()

    # Scoring inputs, split by origin (runner/scorers/credit.py scoring_views).
    # The scorers read the agent-credited view, built from agent-origin events only; the harness view is recorded
    # beside it (logs/harness/harness_tool_calls.jsonl, result["credit"]).
    _views = _credit.scoring_views(trajectory, first_answer=_first_answer,
                                   outdir=os.path.join(job_dir, "workspace", "output"))
    # upstream-format trajectory.log for native_pytest: the agent's own tool calls
    _credit.write_trajectory_logs(job_dir, _views, _canon_fhir_tool)
    # text sources for recommendation/documentation safety: the agent's own final (the loop's no-final fallback when
    # the delivered final is not the agent's) and notes (agent write_file contents + the output files, end state)
    final_texts = _views["final_texts"]
    note_texts = _views["note_texts"]
    # tool-call sequences for MedCTA ToolAcc/ArgAcc — normalize args (str|dict) on BOTH sides; agent calls only
    agent_tool_calls = [(ev["tool"], scoring.parse_args(ev.get("args", {}))) for ev in _views["agent_tool_calls"]]
    ref_tool_calls = []
    for ev in (task.get("reference") or {}).get("reference_trace", []) or []:
        if ev.get("role") == "assistant" and ev.get("tool_calls"):
            for tc in ev["tool_calls"]:
                fn = tc.get("function", {}); ref_tool_calls.append((fn.get("name"), scoring.parse_args(fn.get("arguments"))))
    # lazy: only load PB drug-safety verifiers if a checkpoint needs them
    needs_pb = any((cp.get("check") or {}).get("verifier", "").startswith(("augmentation/drug_safety_check", "drug_safety_check"))
                   for cp in task.get("checkpoints", []))
    _mm_on = os.environ.get("MH_MM_JUDGE", "").lower() in ("1", "on", "true") or bool(os.environ.get("MH_MM_JUDGE_MODEL"))
    _mm_fn = None
    if _mm_on:
        import adapters.medcta.mm_judge_backend as mm_judge_backend; _mm_fn = mm_judge_backend.judge_grounding
    _gacc_on = os.environ.get("MH_GACC", "").lower() in ("1", "on", "true") or bool(os.environ.get("MH_GACC_MODEL"))
    _gacc_fn = None
    if _gacc_on:
        import adapters.medcta.gacc_judge as gacc_judge; _gacc_fn = gacc_judge.score
    # PB content checkpoints (upstream eval_helpers.llm_judge) need an OpenAI-compatible judge. Opt-in by
    # setting LLM_JUDGE_MODEL; pb_judge_env fills key/base from the judge route, a local key file and the default
    # gateway base, so only the model is chosen. (native_pytest propagates os.environ to the pytest subprocess.)
    # The judge actually used is recorded per checkpoint (judge_backend) and in provenance.
    pb_judge_env()
    # Prompt-source provenance -- record which AGENT-VISIBLE segments fed the agent and whether the
    # hidden reference (gold) leaked into any of them (system/user prompt path), so Clinical Conduct Boundaries G1 reads a real
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
           "trajectory": trajectory, "agent_trajectory": _views["agent_trajectory"],
           "created_meds": scoring.created_meds(_views["agent_trajectory"]),
           "final_texts": final_texts, "note_texts": note_texts,
           "full_state": (env.eval_state() if hasattr(env, "eval_state") else getattr(env, "full_state", None)),
           "reference": task.get("reference", {}), "agent_tool_calls": agent_tool_calls, "ref_tool_calls": ref_tool_calls,
           "verifiers": scoring._load_verifiers() if needs_pb else None, "pb_repo": pb_repo, "job_dir": job_dir,
           "mm_judge": _mm_fn, "medcta_img": getattr(env, "image_path", None),
           "medcta_question": (task.get("context") or {}).get("text"), "gacc": _gacc_fn,
           "available_tools": [t.get("name") for t in (task.get("available_tools") or []) if t.get("name")],
           "prompt_provenance": _prompt_prov,
           "task": task, "source_benchmark": task.get("source_benchmark")}
    _PSR.finish(_psr, ctx.get("full_state"))     # the final eval_state, as the scorer reads it
    if os.environ.get(UNSCORED_VAR) == "1":
        # a run whose caller never reads its score (e.g. a search that reads only the legal-action rate from the
        # trajectory): judging it would spend judge calls for nothing. Its checkpoints are recorded as skipped
        # (skip_reason unscored_by_caller).
        results = [{"id": cp["id"], "dimension": cp["dimension"], "subdimension": cp.get("subdimension"),
                    "weight": float(cp.get("weight", 1.0)), "checkpoint_status": "skipped", "failure_mode": None,
                    "skip_reason": "unscored_by_caller"} for cp in task.get("checkpoints", [])]
    else:
        results = score_checkpoints(task.get("checkpoints", []), ctx, (task.get("environment") or {}).get("type"))
    env_type = (task.get("environment") or {}).get("type")
    env_cls = type(env).__name__
    vlm = os.environ.get("MH_VLM_API_MODEL", "gpt-5.5")   # the ONE perception backend (ApiVLM)
    # ---- role-separated provenance: brain != tool-backend != judge. Record EACH even if same model,
    #      so nobody mistakes "agent saw the image" for "an image tool (also a VLM) saw it". ----
    if env_type == "fhir":
        tool_backend = {"fhir": "live_hapi", "rxnorm": "frozen"}; tool_backend_model = None
    elif env_type == "gui":
        tool_backend = {"gui": "real_playwright_portal"}   # always the real portal
        tool_backend_model = None  # DOM actions, no model
    elif env_type == "tool_sandbox":
        tool_backend = {"tool_sandbox": "real_vlm_tools"}
        # tool_backend_model = the ACTUAL VLM perception backend at runtime (NOT a hardcoded default).
        # MedCTA perception = an API VLM via the gateway (vlm_backend.get_backend() -> ApiVLM, the only
        # backend). Resolve from the live backend; fall back to MH_VLM_API_MODEL.
        try:
            import adapters.medcta.vlm_backend as _vb
            _be = _vb.get_backend()
            _bm = getattr(_be, "model", None) or getattr(_be, "model_path", None)
            if _bm:
                _bm = os.path.basename(str(_bm))
            tool_backend_model = ("%s:%s" % (getattr(_be, "name", "vlm"), _bm)) if _bm else getattr(_be, "name", vlm)
        except Exception:
            tool_backend_model = vlm  # last resort: the configured MH_VLM_API_MODEL
    else:
        tool_backend = {env_type: "unknown"}; tool_backend_model = None
    aname = getattr(agent, "name", "") or agent_name
    if aname == "gpt5":
        # MH_API_MODEL is the string actually put on the wire. MH_MODEL_LABEL overrides only how the
        # run is LABELLED, for the case where the gateway maps an alias onto a different backend tier and
        # the operator wants the report to name the backend. Both are recorded (see api_model_string in
        # provenance below) so a reader can always reconcile the label with what was actually called.
        _api_model = os.environ.get("MH_API_MODEL") or os.environ.get("MH_OPENAI_MODEL", "gpt-5.5")
        agent_model = "%s (api brain)" % (os.environ.get("MH_MODEL_LABEL") or _api_model)
        uses_hidden_ref = False; validation_only = False
    elif aname == "replay":
        agent_model = "gold_replay:reference_trace"; uses_hidden_ref = True; validation_only = True
    elif aname == "scripted":
        agent_model = "scripted:gold_path"; uses_hidden_ref = True; validation_only = True
    else:
        agent_model = "stub:%s" % aname; uses_hidden_ref = False; validation_only = True
    # Role-specific judges: outcome via Gacc (MH_GACC) | offline proxy; grounding via multimodal judge
    # (MH_MM_JUDGE). Record EACH truthfully — result.json must say who judged the answer and who judged
    # grounding (reproducibility / integrity gate).
    def _ind_str(jm):
        # EXACT model identity, NOT substring: a judge whose model name is a prefix of the agent's (model-x judging
        # model-x-mini) is a different model and so independent; a substring test would wrongly flag it.
        am = str(agent_model or "").split(" (")[0].strip()          # "<model> (api brain)" -> "<model>"
        tb = str(tool_backend_model or "").split(":")[-1].strip()   # "api:gateway:<model>" -> "<model>"
        return "independent" if (jm and jm != am and jm != tb) else "shared_model_with_agent_or_tool"
    _TIER = {"gacc_judge": "gacc_semantic", "multimodal_judge": "multimodal_judge",
             "llm_judge": "local_model_judge", "proxy": "offline_whitelist_proxy"}
    # judges are derived from what ACTUALLY ran (per-checkpoint evaluator_kind), NOT from env flags, so a
    # stray MH_GACC_MODEL during a native_pytest (PB) run cannot falsely claim the Gacc judge judged the answer.
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
    # Judge independence is RECORDED, never enforced. A judge sharing the agent's or the tool
    # backend's model scores as usual; every checkpoint it scored is marked, and run_config says the run was
    # self-judged, so a report can state it instead of the score silently disappearing.
    _shared_subs = {sub for sub, j in (("result_verification", judges.get("outcome")),
                                       ("clinical_task_success", judges.get("outcome")),
                                       ("context_grounding", judges.get("grounding")))
                    if j and j.get("independence") == "shared_model_with_agent_or_tool"}
    if _shared_subs:
        for _r in results:
            if _r.get("subdimension") in _shared_subs:
                _r["judge_independence"] = "shared_model_with_agent_or_tool"
    _outc = judges.get("outcome", {})
    judge_model = _outc.get("model", "none")
    judge_decoding = ({"max_tokens": 1024} if _outc.get("tier") == "gacc_semantic" else  # matches gacc_judge actual (no temperature)
                      ({"temperature": 0, "do_sample": False, "max_new_tokens": 220} if _outc.get("tier") == "local_model_judge" else None))
    # ---- per-run fidelity provenance: native-fidelity is a RECORDED field of THIS run, not a separate
    #      evaluation system. Derived from source benchmark + env type + the agent's protocol. ----
    _bench_name = {"PhysicianBench": "PhysicianBench", "MedCTA": "MedCTA",
                   "HealthAdminBench": "HealthAdminBench"}.get(bench, bench)
    # prompt fidelity: PB replays the upstream verbatim SYSTEM_PROMPT; MedCTA/HAB are semantically aligned.
    _prompt_fidelity = "verbatim" if env_type == "fhir" else "semantic_aligned"
    # protocol fidelity: native function calling when the agent ran its model's function-calling program
    # (runner/models.yaml), else the canonical <tool_call>/<answer> text protocol.
    _protocol_fidelity = ("native_function_calling" if getattr(agent, "protocol", None) == "function_calling"
                          else "canonicalized")
    # environment fidelity: real substrate (live FHIR / real playwright portal / real VLM tool sandbox).
    _env_fidelity = "full" if env_type in ("fhir", "gui", "tool_sandbox") else "unknown"
    # MedCTA single-system config (which of the 3 modes this run used) -> self-describing provenance.
    _medcta_cfg = None
    if env_type == "tool_sandbox":
        _medcta_cfg = {"image_visible": False, "tools_enabled": True, "mode": "tool_mediated",
                       "is_default_faithful": True}
    fidelity = {"source_benchmark": _bench_name,
                "prompt_fidelity": _prompt_fidelity,
                "protocol_fidelity": _protocol_fidelity,
                "environment_fidelity": _env_fidelity,
                "metric_definition": "native_plus_harness"}
    provenance = {"agent_model": agent_model,
                  # the literal model string sent to the gateway; equals agent_model unless
                  # MH_MODEL_LABEL was set to name a different backend the alias maps to.
                  "api_model_string": os.environ.get("MH_API_MODEL") or os.environ.get("MH_OPENAI_MODEL"),
                  "wire_model_string": os.environ.get("MH_AGENT_WIRE_MODEL"),   # transport.chat_fc
                  "model_label_override": os.environ.get("MH_MODEL_LABEL") or None,
                  "tool_backend": tool_backend,
                  "tool_backend_model": tool_backend_model, "judge_model": judge_model,
                  "judge_tier": _outc.get("tier", "none"),
                  "judge_independence": _outc.get("independence", "n/a"),
                  "judge_independence_policy": "record_only",
                  "judge_decoding": judge_decoding, "judges": judges,
                  "uses_hidden_reference": uses_hidden_ref, "scorer_validation_only": validation_only,
                  "fidelity": fidelity, "medcta_config": _medcta_cfg, "capabilities": _caps,
                  "prompt_provenance": _prompt_prov, "git_sha": _resolve_git_sha()}
    result = scoring.build_result(task, trajectory, results, provenance)
    # agent vs harness credit beside the native score: per-origin counts, the final's author, harness-written files and
    # persisted fields, per-checkpoint credit, native and agent-credited outcome (runner/scorers/credit.py). A
    # measurement failure is recorded in the block, never raised out of the run.
    try:
        result["credit"] = _credit.task_credit(task, bench, result, _views, _credit_tracker,
                                               final_root=ctx.get("full_state"), pb_repo=pb_repo)
    except Exception as _ce:
        result["credit"] = {"version": _credit.CREDIT_VERSION, "error": repr(_ce)}
    result["final_answer_author"] = result["credit"].get("final_answer_author")
    # Budget accounting: the agent's and the harness's (recovery) env actions are exposed separately, with whether
    # the agent's cap was hit. The agent's budget counts the agent's own env actions; harness env actions are counted
    # by the harness action channel against its ceilings, by requester, and never charged to the agent.
    _hc = _action_service.counts()
    from harness import action_service as _AS
    result["budget"] = {"max_steps": max_steps, "agent_env_actions": _env_actions,
                        "harness_env_actions": _hc["harness_env_actions"],
                        "harness_env_actions_by_requester": _hc["harness_env_actions_by_requester"],
                        "recovery_env_actions": _hc["recovery_env_actions"],
                        "total_env_actions": _env_actions + _hc["harness_env_actions"],
                        "budget_exhausted": _env_actions >= max_steps,
                        "agent_loop_iterations": _iterations,
                        "harness_ceilings": {"harness_total_cap": _AS.HARNESS_TOTAL_CAP,
                                             "recovery_total_cap": _AS.RECOVERY_TOTAL_CAP,
                                             "completion_allowance": _AS.RECOVERY_COMPLETION_ALLOWANCE,
                                             "reserve_granted": _reserve_granted, "reserve": _action_reserve,
                                             "post_stop_granted": bool(_action_service._post_stop_granted),
                                             "nudge_cap": _MH.MAX_NUDGES_PER_EPISODE, "nudges_applied": _nudges[0],
                                             "revision_cap": _MH.MAX_REVISIONS_PER_EPISODE,
                                             "revisions_applied": _revisions[0],
                                             "repair_turn_cap": max_repair_turns, "repair_turns": _repair_turns}}
    if getattr(env, "GUARDED_SEARCH_TOOLS", ()):   # a web search: what the search guard refused / withheld, per task
        import search_guard
        result["search_blocked"] = search_guard.task_counts(trajectory)
    # Harness RUNTIME STATUS — always honest about what the harness actually did this run:
    #   requested_mode : 'enforce', the harness's one mode (recorded so every result carries the field).
    #   effective_mode : the kernel's mode, or 'off' when the kernel failed to build.
    #   status         : 'active' / 'degraded' (a capability error occurred) / 'failed' (the kernel failed
    #                    to build -> fail closed, do NOT pretend active).
    #   runtime_errors : every build/hook error string collected during the run.
    # Drain the middleware's judge-outage records so a swallowed judge failure (403 / timeout) surfaces as a
    # runtime_error instead of masquerading as "no opportunity" -> status goes degraded.
    try:
        _je = list(_mw.deferred_errors())
        if _je:
            _harness_runtime_errors.extend("judge_unavailable: " + s for s in _je)
    except Exception:
        pass

    if _harness is not None:
        try:
            from harness.ledger import governance as _hgov
            _h_events = [ev for ev in trajectory if str(ev.get("event_type", "")).startswith("harness_")]
            _h_audit = _harness.audit()
            _h_status = _h_audit.get("status", "active")
            result["harness"] = {"mode": _harness.mode,
                                 "requested_mode": "enforce",
                                 "effective_mode": _harness.mode,
                                 "status": "degraded" if (_h_status == "degraded" or _harness_runtime_errors) else "active",
                                 "runtime_errors": _harness_runtime_errors,
                                 "audit": _h_audit,
                                 "recovery_env_actions": _hc["recovery_env_actions"],
                                 "metrics": _hgov.summarize(_harness.ledger, _h_events, mode=_harness.mode)}
        except Exception as _he:
            _harness_runtime_errors.append("audit/summarize: %r" % _he)
            result["harness"] = {"mode": getattr(_harness, "mode", "enforce"),
                                 "requested_mode": "enforce",
                                 "effective_mode": getattr(_harness, "mode", "enforce"),
                                 "status": "degraded", "runtime_errors": _harness_runtime_errors}
    else:
        # the kernel failed to build -> report it, never silently 'off'.
        result["harness"] = {"mode": "off", "requested_mode": "enforce",
                             "effective_mode": "off",
                             "status": "failed",
                             "runtime_errors": _harness_runtime_errors}
    if _harness_infra_error:
        # the harness could not be built -> the task ran NO agent actions; it is an infrastructure error
        # (kept apart from agent results).
        result["evaluation_status"] = "infrastructure_error"
    result["_trajectory"] = trajectory
    result["_instrumentation"] = _instr.records()
    # (schema validation is deferred to the END of run_task -- see below -- so its verdict covers EVERY
    # result mutation, not just the prefix that existed at this point.)
    try:   # shadow VERIFY: order self-consistency advisory (never enforces)
        from harness.order_integrity import scan_orders
        result["order_integrity_advisory"] = scan_orders(
            [e.get("args") or {} for e in trajectory
             if e.get("event_type") == "tool_call" and "service_request_create" in (e.get("tool") or "")])
    except Exception as _oie:
        result["_order_integrity_error"] = repr(_oie)
    result["_job_dir"] = job_dir
    # ---- qualification: downgrade ONLY for mock / replay / proxy / hidden-reference — NOT by substrate.
    #      A real Playwright GUI run or real VLM-tool run is NOT auto-"stub". ----
    quals = []
    if uses_hidden_ref: quals.append("uses_hidden_reference")
    if validation_only: quals.append("scorer_validation_only")
    if any(c.get("evaluator_kind") == "proxy" and c.get("subdimension") in ("clinical_task_success", "result_verification")
           for c in result.get("checkpoints", [])): quals.append("outcome_proxy")
    if any(j.get("independence") == "shared_model_with_agent_or_tool" for j in judges.values()):
        quals.append("non_independent_judge")
    if any(c.get("score_eligible") is False for c in result.get("checkpoints", [])): quals.append("proxy_scored_checkpoints")
    if any("API_BRAIN_ERROR" in str(ev.get("thought", "")) for ev in trajectory): quals.append("api_backend_error")
    quals = sorted(set(quals))
    if any(not (c or {}).get("healthy", True) for c in (_caps or {}).values()):
        quals.append("degraded_tool_health")   # a down backing service is NOT agent incompetence
    quals = sorted(set(quals))
    result["qualification"] = quals  # non-underscore -> survives --out / run_batch result.json
    if quals:
        result["_warning"] = "qualified: " + ", ".join(quals)
    # light per-task cleanup of agent-created (stub-tagged) resources to avoid FHIR pollution
    if cleanup and env_type == "fhir" and ctx.get("base"):
        cl = cleanup_stub(ctx["base"])
        result["_cleanup_deleted"] = cl["deleted"]
        if cl["error"]:
            result["_cleanup_error"] = cl["error"]
            result["failure_tags"] = sorted(set(result.get("failure_tags", [])) | {"environment_error"})
    # validate LAST -- order_integrity_advisory, qualification, failure_tags and cleanup are all done
    # now, so schema_validation honestly covers the object that gets written (not a prefix). Only
    # schema_validation/_schema (the verdict itself) are added after -- they cannot validate self-referentially.
    _sv = validate_result(result)
    _sv = _sv if isinstance(_sv, dict) else {"valid": False, "errors": ["validator returned non-dict: " + str(_sv)]}
    result["schema_validation"] = _sv                 # NON-underscore -> survives --out / run_batch
    result["_schema"] = "OK" if _sv.get("valid") else "INVALID: " + "; ".join(_sv.get("errors", []))
    if not _sv.get("valid"):
        print("SCHEMA INVALID:", result["_schema"])
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
    ap.add_argument("--workspace", default=None, help="the workspace directory to run (default: the substrate's H0)")
    args = ap.parse_args()
    reset_fhir(args.reset_mode)
    result = run_task(args.bench, args.task, args.agent, args.fhir_base, args.max_steps, args.job_dir,
                      cleanup=not args.unsafe_no_reset, workspace=args.workspace)
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
        try:
            import os as _os
            _tp = _os.path.join(_os.path.dirname(_os.path.abspath(args.out)), "trajectory.jsonl")
            with open(_tp, "w") as _tf:
                for _ev in (result.get("_trajectory") or []):
                    _tf.write(json.dumps(_ev, ensure_ascii=False, default=str) + "\n")
            print("wrote", _tp)
            with open(_os.path.join(_os.path.dirname(_tp), "instrumentation.jsonl"), "w") as _inf:
                for _rec in (result.get("_instrumentation") or []):
                    _inf.write(json.dumps(_rec, ensure_ascii=False) + "\n")
        except Exception as _te:
            print("trajectory dump failed:", repr(_te))

if __name__ == "__main__":
    main()
