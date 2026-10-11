#!/usr/bin/env python3
"""Batch runner: select tasks, run each with the chosen agent (default: stub), write a per-task RESULT BUNDLE + summary.json.

Bundle layout (per task):
  results/<agent>/<task_id>/{task.json, trajectory.jsonl, instrumentation.jsonl, result.json, verifier_logs/,
                             workspace/output/}
  results/<agent>/summary.json (and report.json unless --no-report)

Filters: --task-id, --task-ids-file, --has-dimension, --has-subdimension, --governance-only, --offset/--limit.
Usage: python3 runner/run_batch.py --bench PhysicianBench --governance-only --limit 26 --out results/

--workspace DIR runs that workspace (default: the benchmark's default workspace; runner/run.py resolve_workspace). A
workspace whose manifest does not declare the batch's benchmark and env_type is refused before anything runs.

Resume: a task whose bundle already holds a result.json is reused only when the resume_key that result records equals
the key of the run that would replace it (task id, workspace digest, bound judge configuration, agent model, protocol,
max_steps; runner/run.py resume_key). Otherwise the task runs again and its bundle is rewritten. A reused task counts
in summary.json and report.json like one run now (rows mark it "reused").
"""
import os, sys, json, argparse, collections, shutil, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scorers"))   # scoring machinery lives in runner/scorers/
import run as R

MODULES = ["Care Action Fulfillment", "Clinical Resource Competence", "Patient–Evidence Alignment", "Clinical Workflow Continuity", "Clinical Auditability", "Closed-Loop Verification", "Clinical Conduct Boundaries"]

def select_tasks(bench, args):
    p = os.path.join(ROOT, "benchmark_dataprocess", bench, "tasks_unified.jsonl")
    out = []
    for line in open(p):
        t = json.loads(line)
        dims = {c.get("dimension") for c in t.get("checkpoints", [])}
        subs = {c.get("subdimension") for c in t.get("checkpoints", [])}
        if args.task_id and t["task_id"] != args.task_id: continue
        if args.has_dimension and args.has_dimension not in dims: continue
        if args.has_subdimension and args.has_subdimension not in subs: continue
        if args.governance_only and "Clinical Conduct Boundaries" not in dims: continue
        out.append(t["task_id"])
    if args.task_ids_file:
        want = set(l.strip() for l in open(args.task_ids_file) if l.strip())
        return [t for t in out if t in want]
    return out[args.offset:args.offset + args.limit]

def write_bundle(agent_out, tid, res, task):
    """One task's bundle: task.json, trajectory.jsonl, instrumentation.jsonl, result.json, verifier_logs/, and
    the deliverables. Written for EVERY task in the batch -- a crashed or excluded task included."""
    bdir = os.path.join(agent_out, tid); os.makedirs(os.path.join(bdir, "workspace"), exist_ok=True)
    json.dump(task or {}, open(os.path.join(bdir, "task.json"), "w"), indent=1, ensure_ascii=False)
    with open(os.path.join(bdir, "trajectory.jsonl"), "w") as f:
        for ev in res.get("_trajectory", []): f.write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")
    with open(os.path.join(bdir, "instrumentation.jsonl"), "w") as f:   # which workspace components ran
        for rec in res.get("_instrumentation", []): f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    clean = {k: v for k, v in res.items() if not k.startswith("_")}
    json.dump(clean, open(os.path.join(bdir, "result.json"), "w"), indent=1, ensure_ascii=False, default=str)
    # full raw checkpoint results (detail/note/judge_backend) for debugging
    vdir = os.path.join(bdir, "verifier_logs"); os.makedirs(vdir, exist_ok=True)
    json.dump(res.get("_checkpoints_full", []), open(os.path.join(vdir, "checkpoints_full.json"), "w"),
              indent=1, ensure_ascii=False, default=str)
    jd = res.get("_job_dir")
    if jd and os.path.isfile(os.path.join(jd, "logs", "portal_state_record.jsonl")):    # runner/portal_state_record.py
        shutil.copyfile(os.path.join(jd, "logs", "portal_state_record.jsonl"),
                        os.path.join(bdir, "portal_state_record.jsonl"))
    if jd and os.path.isdir(os.path.join(jd, "workspace", "output")):
        shutil.copytree(os.path.join(jd, "workspace", "output"),
                        os.path.join(bdir, "workspace", "output"), dirs_exist_ok=True)
    # run.py's default job dir is per process (/tmp/mh_jobs/<bench>/<task>.<pid>); once its outputs are in the bundle
    # it is removed, or every batch would leave one per task behind
    if jd and os.path.dirname(os.path.dirname(os.path.abspath(jd))) == "/tmp/mh_jobs" and \
            os.path.basename(jd).rsplit(".", 1)[-1].isdigit():
        shutil.rmtree(jd, ignore_errors=True)
    return bdir


def stored_result(agent_out, tid):
    """The result.json a previous batch left in this task's bundle, or None (absent, or not a JSON object: a file cut
    off mid-write counts as no result)."""
    p = os.path.join(agent_out, tid, "result.json")
    try:
        with open(p, encoding="utf-8") as f:
            res = json.load(f)
    except (OSError, ValueError):
        return None
    return res if isinstance(res, dict) else None


def resume_plan(bench, tids, tasks_by_id, agent, workspace, binding, max_steps_of, agent_out):
    """-> ({task id: (max_steps, resume key)}, {task id: stored result reused}). Resolves the workspace for every task
    first, so a workspace the run would refuse stops the batch before anything runs (SystemExit). A stored result is
    reused only when the key it records equals the key of the run that would replace it."""
    import workspace as W
    plan, reuse = {}, {}
    for tid in tids:
        ms = max_steps_of(tid)
        try:
            _path, ident = R.resolve_workspace(bench, tasks_by_id.get(tid) or {}, workspace)
        except W.WorkspaceError as e:
            raise SystemExit("REFUSING: %s" % e)
        key = R.resume_key(tid, (ident or {}).get("digest"), binding, agent, ms)
        plan[tid] = (ms, key)
        stored = stored_result(agent_out, tid)
        if stored is not None and stored.get("resume_key") == key:
            reuse[tid] = stored
    return plan, reuse


def _wire_fhir_port(fhir_base):
    """The base URL the AGENT is handed is the only source of the FHIR port.

    restore_pristine_h2.sh boots and health-checks ${MH_FHIR_PORT:-38080}. If that variable and --fhir-base
    disagree, the reset succeeds on one port while every agent call goes to the other, with no error anywhere.
    So MH_FHIR_PORT is derived from --fhir-base, and a contradiction is refused rather than silently resolved
    in favour of either. Returns a short note describing where the port came from."""
    if not fhir_base:
        return "no --fhir-base"
    try:
        from urllib.parse import urlparse
        port = urlparse(fhir_base).port
    except (ImportError, ValueError):
        port = None
    if port is None:
        return "no port in --fhir-base %s" % fhir_base
    have = (os.environ.get("MH_FHIR_PORT") or "").strip()
    if have and have != str(port):
        raise SystemExit("REFUSING: MH_FHIR_PORT=%s contradicts --fhir-base %s. One port, one source -- "
                         "unset MH_FHIR_PORT and let the base URL decide." % (have, fhir_base))
    os.environ["MH_FHIR_PORT"] = str(port)
    return "MH_FHIR_PORT=%d (derived from --fhir-base)" % port


def _require_fhir_reachable(fhir_base, note, attempts=20, delay=3):
    """Probe THE AGENT'S OWN URL before any task runs, and refuse the batch if nothing answers.

    A PhysicianBench task with no reachable FHIR still completes: the agent gets error strings, writes a
    deliverable from nothing, and the run produces checkpoints and a score. There is no downstream signal
    that distinguishes that from a real result, so it has to be caught here."""
    import time
    import urllib.request
    url = fhir_base.rstrip("/") + "/metadata"
    last = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/fhir+json"})
            urllib.request.urlopen(req, timeout=10).read(1)
            print("[fhir] reachable at %s (%s) after %ds" % (fhir_base, note, i * delay), flush=True)
            return True
        except Exception as ex:                       # noqa: BLE001 -- any failure to answer is the same failure
            last = repr(ex)
            time.sleep(delay)
    raise SystemExit("REFUSING: nothing answered %s after %ds (%s). Last error: %s\n"
                     "Every task would run with zero retrievable evidence and still produce scores."
                     % (url, attempts * delay, note, last))


def fhir_slots(fhir_base, workers, h2_parent=None):
    """One FHIR instance per worker, for PhysicianBench run with --workers: [{"port", "base", "env"}].

    Slot i serves on the --fhir-base port + i, from its own H2 copy under <MH_H2_PARENT>/w<i>, and logs there. The
    reset and start scripts (benchmark_dataprocess/PhysicianBench/augmentation/restore_pristine_h2.sh, run_fhir.sh)
    already take MH_FHIR_PORT, MH_H2_PARENT and MH_FHIR_LOG, and kill only the server on their own port, so a slot is
    nothing more than those three values. Before each task its process restores ITS slot to the pristine H2 and starts
    it (_run_one): every task starts from the same store whatever worker it lands on -- which the serial
    restore_pristine batch does not give, since there a task reads what the tasks before it wrote."""
    from urllib.parse import urlparse
    u = urlparse(fhir_base)
    parent = h2_parent or os.environ.get("MH_H2_PARENT") or os.path.join(ROOT, "benchmark_dataprocess",
                                                                            "PhysicianBench", "h2data")
    out = []
    for i in range(workers):
        port = u.port + i
        d = os.path.join(parent, "w%d" % i)
        out.append({"port": port, "base": "%s://%s:%d%s" % (u.scheme, u.hostname, port, u.path),
                    "env": {"MH_FHIR_PORT": str(port), "MH_H2_PARENT": d, "MH_FHIR_LOG": os.path.join(d, "fhir.log")}})
    return out


def stop_fhir(port):
    """Kill the FHIR server on `port` (the scripts' own way: by PID, matched on server.port)."""
    import subprocess
    subprocess.run("ps -u \"$USER\" -o pid,cmd | grep '[m]ain.war' | grep 'server.port=%d' | awk '{print $1}' "
                   "| xargs -r kill -9" % int(port), shell=True, check=False)


def _run_one(bench, tid, agent, fhir_base, max_steps, workspace, agent_out, task, slot=None):
    """One episode in its own process: run it and write its bundle. Returns the result the parent aggregates.

    Module level, because a worker process must be able to call it. It touches nothing the other workers touch: one
    task, one bundle -- and, with a `slot` (fhir_slots), one FHIR instance, restored to pristine before the task."""
    if bench == "MedCTA":                            # the OCR model loads while the task starts (tools_medcta)
        try:
            from adapters.medcta import tools_medcta as _tm
            _tm.preload_ocr()
        except Exception:
            pass
    try:
        if slot is not None:
            os.environ.update(slot["env"])
            os.makedirs(slot["env"]["MH_H2_PARENT"], exist_ok=True)
            fhir_base = slot["base"]
            R.reset_fhir("per_task")
        res = R.run_task(bench, tid, agent, fhir_base, max_steps=max_steps, workspace=workspace)
        if slot is not None and isinstance(res, dict):
            res.setdefault("run_config", {})["fhir_slot"] = {"port": slot["port"], "reset": "pristine before this task"}
    except Exception as e:
        import crash_result as _CR
        res = _CR.crash_result(bench, tid, e, agent_name=agent, max_steps=max_steps)
    write_bundle(agent_out, tid, res, task)
    return res


def _worker(results, bench, tid, agent, fhir_base, max_steps, workspace, agent_out, task, slot=None):
    """A worker process: one episode, its bundle, then the result back to the parent through the queue."""
    results.put((tid, _run_one(bench, tid, agent, fhir_base, max_steps, workspace, agent_out, task, slot)))


# A process that exited may still have its result in flight on the queue, so a task is only called crashed this
# long after its process is gone.
WORKER_EXIT_GRACE = 5.0


def run_concurrently(bench, todo, agent, fhir_base, plan, workspace, agent_out, tasks_by_id, workers, slots=None):
    """Run these tasks at once, at most `workers` of them, each in its own process. Returns {task_id: result}.

    A process, not a thread: run.py keeps module-level state per episode, so two episodes in one interpreter would
    read each other's. One process per TASK rather than a pool of reused ones, because a pool is poisoned by a
    worker that dies abruptly -- ProcessPoolExecutor fails every pending future with BrokenProcessPool, so one
    segfaulting episode would void the whole batch. Here an episode that kills its process costs that task only,
    and the parent turns it into the same crash_result the serial loop would have produced. The caller aggregates
    in TASK ORDER, so results arriving out of order changes nothing downstream.

    `slots` (fhir_slots): a task is started only on a free slot and holds it until its process is gone, so no two
    running tasks ever share a FHIR instance."""
    import multiprocessing as mp
    import queue as _q
    import crash_result as CR
    done = {}
    if not todo:
        return done
    ctx = mp.get_context("fork")
    results = ctx.Queue()
    waiting, running, exited = list(todo), {}, {}
    free = list(range(len(slots))) if slots else None
    held = {}
    print("[parallel] %d task(s) over %d worker process(es)%s" % (len(todo), workers,
          " with %d FHIR instance(s), ports %s" % (len(slots), [s["port"] for s in slots]) if slots else ""), flush=True)
    while waiting or running or exited:
        while waiting and len(running) < workers and (free is None or free):
            t = waiting.pop(0)
            si = free.pop(0) if free is not None else None
            proc = ctx.Process(target=_worker, args=(results, bench, t, agent, fhir_base, plan[t][0], workspace,
                                                     agent_out, tasks_by_id.get(t, {}),
                                                     slots[si] if si is not None else None))
            proc.start()
            running[t] = proc
            if si is not None:
                held[t] = si
        try:
            tid, res = results.get(timeout=0.2)
        except _q.Empty:
            pass
        else:
            done[tid] = res
            exited.pop(tid, None)
            proc = running.pop(tid, None)
            if proc is not None:
                proc.join()
            if tid in held and tid not in exited:
                free.append(held.pop(tid))
            print("[parallel] %s done (%d/%d)" % (tid, len(done), len(todo)), flush=True)
        now = time.time()
        for t, proc in list(running.items()):
            if not proc.is_alive():
                proc.join()
                running.pop(t)
                exited[t] = now                     # its result may still be on the queue
        for t, at in list(exited.items()):
            if now - at < WORKER_EXIT_GRACE:
                continue
            exited.pop(t)
            if t in held:
                free.append(held.pop(t))
            done[t] = CR.crash_result(bench, t, RuntimeError("worker process died without a result"),
                                      agent_name=agent, max_steps=plan[t][0])
            write_bundle(agent_out, t, done[t], tasks_by_id.get(t, {}))
            print("HARNESS_ERROR %s: worker process died without a result" % t, flush=True)
            print("[parallel] %s done (%d/%d)" % (t, len(done), len(todo)), flush=True)
    results.close()
    for s in slots or []:
        stop_fhir(s["port"])
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--agent", default="stub")
    ap.add_argument("--fhir-base", default=None)
    ap.add_argument("--reset-mode", default="none", choices=["none", "restore_pristine", "per_task"])
    ap.add_argument("--task-id", default=None)
    ap.add_argument("--offset", type=int, default=0)   # shard: skip first OFFSET tasks
    ap.add_argument("--task-ids-file", default=None)   # run EXACTLY these task_ids (one per line)
    ap.add_argument("--has-dimension", default=None)
    ap.add_argument("--has-subdimension", default=None)
    ap.add_argument("--governance-only", action="store_true")
    ap.add_argument("--out", default="results/")
    ap.add_argument("--max-steps", type=int, default=0)  # 0 = use NATIVE_MAX_STEPS per bench
    ap.add_argument("--workspace", default=None,
                    help="the workspace directory every task runs (default: the substrate's H0)")
    ap.add_argument("--no-report", action="store_true",
                    help="skip the enhanced report.json")
    ap.add_argument("--workers", type=int, default=1,
                    help="tasks run at once, in separate PROCESSES (default 1: the serial loop). Only for a "
                         "benchmark with no exclusive stateful service -- see PARALLEL_SAFE.")
    # Which benchmarks may run tasks at once -- decided by what their tasks SHARE, not by which of them have a
    # service running.
    #   MedCTA            a per-task tool sandbox; nothing is shared.
    #   HealthAdminBench  the Next.js portal has no API route and keeps nothing: every page's data is seeded into
    #                     the BROWSER's localStorage (`portals_state`), and runner/environments.py reset() launches
    #                     a fresh Chromium with a fresh page per task. Two tasks at once are two browser contexts
    #                     against a server that stores nothing, so they cannot read each other's state.
    #   PhysicianBench    its tasks write to a FHIR store, so each worker gets its OWN instance (fhir_slots: port
    #                     + i, its own H2 copy), restored to pristine before every task. The comparison with runs made
    #                     serially: those restore once per batch and a task reads what earlier tasks wrote; here
    #                     every task starts pristine. Each result records its slot (run_config.fhir_slot).
    PARALLEL_SAFE = ("MedCTA", "HealthAdminBench", "PhysicianBench")

    # Default per-bench step caps (upstream values). HealthAdminBench tasks take their budget from
    # _hab_task_max_steps below; its entry here is upstream's generic limits.max_steps default.
    NATIVE_MAX_STEPS = {"PhysicianBench": 100, "MedCTA": 10, "HealthAdminBench": 100}

    def _hab_task_max_steps(task_id):
        """HAB step budget is PER TASK, not flat. Ported from the vendored upstream
        benchmark/HealthAdminBench/harness/config/settings.py get_task_max_steps() (axtree observation mode);
        upstream's `limits.max_steps = 100` is only the fallback for task ids no branch matches. The branch
        ORDER matches upstream exactly, since an id can contain more than one keyword (e.g. "fax-easy" and "easy")."""
        t = (task_id or "").lower()
        if "fax-easy" in t:      n = 35
        elif "fax-medium" in t:  n = 50
        elif "fax-hard" in t:    n = 60
        elif "hard" in t:        n = 100
        elif "emr-medium" in t:  n = 60
        elif "medium" in t:      n = 75
        elif "easy" in t:        n = 20
        else:                    n = 100
        return n
    args = ap.parse_args()
    if args.workers > 1 and args.bench not in PARALLEL_SAFE:
        raise SystemExit("--workers %d: %s tasks share a stateful service (FHIR store / portal); only %s may run "
                         "tasks at once" % (args.workers, args.bench, ", ".join(PARALLEL_SAFE)))
    if args.workers > 1 and args.reset_mode != "none" and args.bench != "PhysicianBench":
        raise SystemExit("--workers %d with --reset-mode %s: a reset between tasks and tasks at once are exclusive"
                         % (args.workers, args.reset_mode))
    if args.bench == "PhysicianBench" and args.workers > 1 and not args.fhir_base:
        raise SystemExit("--workers %d for PhysicianBench needs --fhir-base: each worker's FHIR instance is placed "
                         "at its port + i" % args.workers)
    if args.agent in R.agents.API_AGENTS:
        import models
        models.protocol(models.agent_model())   # unlisted and undeclared (MH_MODEL_PROTOCOL) stops the batch here
    # Every judge this benchmark needs must be configured BEFORE the first task: a batch that would score
    # nothing (or score with a silent default) is refused here, not discovered after the agent has run.
    import judges as _J
    try:
        _binding = _J.require(args.bench)
    except _J.JudgeConfigError as _je:
        raise SystemExit(str(_je))
    # PhysicianBench parity: agent-created ServiceRequest/Appointment/Communication persist across
    # tasks unless the FHIR store is reset (upstream resets per task).
    if args.bench == "PhysicianBench" and args.reset_mode == "none":
        _pbmsg = "[PB PARITY] --reset-mode=none leaks agent-created resources across tasks; use --reset-mode restore_pristine for a faithful PhysicianBench run."
        print("WARNING: " + _pbmsg)

    base_out = os.path.join(ROOT, args.out) if not os.path.isabs(args.out) else args.out
    agent_out = os.path.join(base_out, args.agent)
    # APPEND (keep prior bundles). A re-run or fill into an existing agent dir never wipes
    # already-produced results; only the tasks that run again have their bundles rewritten.
    os.makedirs(agent_out, exist_ok=True)
    # The substrate the AGENT will use: the FHIR port is single-sourced from --fhir-base, then probed at that
    # address (see _wire_fhir_port / _require_fhir_reachable).
    _fhir_note = _wire_fhir_port(args.fhir_base)

    tids = select_tasks(args.bench, args)
    status_hist = collections.Counter(); tag_hist = collections.Counter()
    buckets = collections.Counter()  # complete_success/partial_success/failed/error/task_error
    dim_acc = {m: [] for m in MODULES}; proxy_dim_acc = {m: [] for m in MODULES}; schema_ok = 0; rows = []
    proxy_cp_total = 0
    tasks_by_id = {json.loads(l)["task_id"]: json.loads(l)
                   for l in open(os.path.join(ROOT, "benchmark_dataprocess", args.bench, "tasks_unified.jsonl"))}

    def _max_steps_of(tid):
        return args.max_steps or (_hab_task_max_steps(tid) if args.bench == "HealthAdminBench"
                                  else NATIVE_MAX_STEPS.get(args.bench, 12))
    _plan, _reuse = resume_plan(args.bench, tids, tasks_by_id, args.agent, args.workspace, _binding, _max_steps_of,
                                agent_out)
    if _reuse:
        print("[resume] %d/%d tasks reuse a stored result whose resume key matches; the rest run"
              % (len(_reuse), len(tids)), flush=True)
    _pb_slots = (fhir_slots(args.fhir_base, args.workers)
                 if args.bench == "PhysicianBench" and args.workers > 1 and args.fhir_base else None)
    if len(_reuse) < len(tids) and not _pb_slots:   # reset and probed only when some task will run; slots reset their own
        if args.reset_mode == "restore_pristine":
            R.reset_fhir("restore_pristine")
        if args.bench == "PhysicianBench" and args.fhir_base:
            _require_fhir_reachable(args.fhir_base, _fhir_note)

    try:
        from capability_manifest import precheck as _cap_precheck
    except Exception:
        _cap_precheck = lambda t, b: {"ok": True}
    # With --workers > 1 the episodes run first, each in its own process, and the loop below then reads their
    # bundles in TASK ORDER: the aggregation, the printing and the summary stay exactly as they are serially. A
    # worker writes only its own task's bundle, so nothing is shared but the output directory.
    _done = {}
    if args.workers > 1:
        todo = [t for t in tids if t not in _reuse and _cap_precheck(tasks_by_id.get(t, {}), args.bench).get("ok")]
        _done = run_concurrently(args.bench, todo, args.agent, args.fhir_base, _plan, args.workspace, agent_out,
                                 tasks_by_id, args.workers, slots=_pb_slots)
    for tid in tids:
        # Skip tasks whose required capabilities the env lacks (e.g. HAB file upload/download)
        # -> not_exercised_due_to_missing_capability, NOT an agent failure.
        _pc = _cap_precheck(tasks_by_id.get(tid, {}), args.bench)
        if not _pc.get("ok"):
            buckets["capability_excluded"] += 1
            rows.append({"task": tid, "evaluation_status": "not_exercised_due_to_missing_capability",
                         "qualification": _pc.get("qualification"), "missing_capabilities": _pc.get("missing_capabilities")})
            write_bundle(agent_out, tid, {"task_id": tid, "success": False, "evaluation_status": "not_evaluated",
                                          "end_reason": "capability_excluded", "outcome_status": "capability_excluded",
                                          "checkpoints": [], "dimension_scores": {m: None for m in MODULES},
                                          "failure_tags": [], "missing_capabilities": _pc.get("missing_capabilities"),
                                          "qualification": _pc.get("qualification"),
                                          "provenance": {"agent_model": args.agent, "task_id": tid}},
                         tasks_by_id.get(tid, {}))
            continue
        _ms = _plan[tid][0]
        if tid in _reuse:
            # a stored result made under this run's resume key: reused, its bundle left as it is
            res = dict(_reuse[tid])
            res["_schema"] = "OK" if (res.get("schema_validation") or {}).get("valid") else "INVALID"
            print("[resume] %s: reused (resume key matches)" % tid, flush=True)
        elif tid in _done:
            res = _done[tid]                       # already run by a worker process; its bundle is written
        else:
            if args.reset_mode == "per_task": R.reset_fhir("per_task")
            try:
                res = R.run_task(args.bench, tid, args.agent, args.fhir_base, max_steps=_ms, workspace=args.workspace)
            except Exception as e:
                # run_task returns a crash result for an exception inside the episode; what reaches here failed
                # around it. Still a result, still in the denominator. It records no resume key: it runs again.
                import crash_result as _CR
                res = _CR.crash_result(args.bench, tid, e, agent_name=args.agent, max_steps=_ms)
                print("HARNESS_ERROR %s (outside the episode): %r" % (tid, e), flush=True)
            # --- per-task bundle ---
            write_bundle(agent_out, tid, res, tasks_by_id.get(tid, {}))
        _reused = {"reused": True} if tid in _reuse else {}
        if res.get("end_reason") == "harness_error":
            buckets["harness_error"] += 1
            rows.append(dict({"task": tid, "end_reason": "harness_error",
                              "harness_error": {k: (res.get("harness_error") or {}).get(k)
                                                for k in ("type", "message", "traceback_sha256", "origin")}}, **_reused))
            continue
        # --- aggregate ---
        if res.get("_schema") == "OK": schema_ok += 1
        for c in res["checkpoints"]: status_hist[c["checkpoint_status"]] += 1
        for t in res["failure_tags"]: tag_hist[t] += 1
        for m in MODULES:
            if res["dimension_scores"].get(m) is not None: dim_acc[m].append(res["dimension_scores"][m])
            if res.get("proxy_dimension_scores", {}).get(m) is not None: proxy_dim_acc[m].append(res["proxy_dimension_scores"][m])
        proxy_cp_total += res.get("proxy_evaluated_checkpoints", 0)
        if res["success"]:
            buckets["complete_success" if res["evaluation_status"] == "complete" else "partial_success"] += 1
        elif res["evaluation_status"] == "error" or any(c["checkpoint_status"] == "error" for c in res["checkpoints"]):
            buckets["error"] += 1
        elif res["evaluation_status"] in ("proxy_only", "not_evaluated"):
            buckets[res["evaluation_status"]] += 1
        else:
            buckets["failed"] += 1
        rows.append(dict({"task": tid, "success": res["success"], "evaluation_status": res["evaluation_status"],
                          "dimension_scores": res["dimension_scores"]}, **_reused))

    summary = {"bench": args.bench, "agent": args.agent, "n_tasks": len(tids),
               "n_results_written": len(rows), "n_reused": len(_reuse), "workspace": args.workspace,
               "filters": {"has_dimension": args.has_dimension, "has_subdimension": args.has_subdimension,
                           "governance_only": args.governance_only},
               "schema_valid": f"{schema_ok}/{len(tids)}",
               "success_buckets": dict(buckets),
               "dimension_means": {m: (round(sum(v) / len(v), 3) if v else None) for m, v in dim_acc.items()},
               "dimension_coverage_tasks": {m: len(v) for m, v in dim_acc.items()},
               "proxy_dimension_means": {m: (round(sum(v) / len(v), 3) if v else None) for m, v in proxy_dim_acc.items()},
               "proxy_evaluated_checkpoints": proxy_cp_total,
               "checkpoint_status_histogram": dict(status_hist),
               "failure_tag_counts": dict(tag_hist), "tasks": rows}
    json.dump(summary, open(os.path.join(agent_out, "summary.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: summary[k] for k in ("bench", "n_tasks", "schema_valid", "success_buckets",
                      "dimension_means", "checkpoint_status_histogram", "failure_tag_counts")},
                     indent=1, ensure_ascii=False))
    # --- enhanced report (native_metrics + 2-category dims w/ honest coverage + integrity + taxonomy) ---
    try:
        if args.no_report:
            raise RuntimeError("--no-report")
        from aggregate_report import build as _build_report
        _rep = _build_report(agent_out, args.bench, task_ids=tids)   # the task list is the denominator
        json.dump(_rep, open(os.path.join(agent_out, "report.json"), "w"), indent=1, ensure_ascii=False)
        print("enhanced report -> report.json :", json.dumps(_rep["native_metrics"], ensure_ascii=False))
    except Exception as _e:
        print("report.json skipped:", repr(_e))
    print("bundles + summary.json ->", agent_out)

if __name__ == "__main__":
    main()
