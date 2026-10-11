"""Bounded Clinical Recovery - one recovery invocation (workspace strategy, HealthAdminBench H0).

middleware/recovery.py opens a harness-action invocation through its middleware context and calls run_recovery_v3 with
a HarnessDriver over that context. Everything below reaches the environment only through the context's shell services
(runner/middleware_host.py): ctx.request_action and ctx.read (the shell's harness action service: ceilings, the
irreversible-commit checks, write provenance, the kernel with the authorization ceremony, the executor, the recorded
origin=recovery event, the content guard) and the read-only views ctx.recovery_state(), ctx.controls() and
ctx.read_back(). No raw environment, no judge, no budget counters.

The package (all loaded through workspace.load_relative):
    entry.py           this file: wires the stack to the invocation and records one recovery_v3 event per episode
    kernel.py          the episode state machine (one episode per committed goal, the closed observe/decide/act loop)
    registry.py        workflow registry and the gui recovery stack
    gui_substrate.py   substrate adapter: strategy steps -> harness actions and read-backs
    gui_completion.py  the generic GUI completion workflow (what the next action is)
    hab.py             the HealthAdminBench benchmark adapter (task normalization, committed goals)
    contracts.py       the vocabulary the layers share
    metrics.py         episode outcome buckets

Oracle-blind: it reads only the task, the root-agent deliverable content and the live trajectory; never
gold/checkpoint/reference material. Python 3.8 compatible.
"""
import workspace_api as _W

_registry = _W.load_relative(__file__, "registry.py")
_kernel = _W.load_relative(__file__, "kernel.py")
get_recovery_stack = _registry.get_recovery_stack
RecoveryKernel = _kernel.RecoveryKernel


class HarnessDriver(object):
    """One invocation: the middleware context whose harness-action services the strategy acts through, and the step the
    shell opened the invocation at."""

    def __init__(self, ctx, step):
        self._ctx = ctx
        self.step = step

    def can_execute_recovery_action(self):
        """The shell's budget ceiling admits a harness action now (a read-only view)."""
        return bool(self._ctx.harness_actions_open())

    def emit(self, event):
        self._ctx.emit(event)


# --- GUI backend (wiring boundary) ------------------------------------------------------------------------------
# GuiSubstrateAdapter calls backend.call_tool(name, args) / .read(name, args) / .read_recovery_state() /
# .list_controls() / .reconcile_write(); each is a shell service of the invocation's context. Without a driver every
# action is refused and the views are empty.
class _GuiBackend(object):
    def __init__(self, driver):
        self._ctx = getattr(driver, "_ctx", None)

    def call_tool(self, name, args):
        if self._ctx is None:
            return {"error": "no_harness_action_service"}
        return self._ctx.request_action(name, args)

    def read(self, name, args):
        """An evidence-reveal READ: the shell's READ actuator path."""
        if self._ctx is None:
            return {"error": "no_harness_action_service"}
        return self._ctx.read(name, args)

    def reconcile_write(self, name, args, r):
        if self._ctx is None:
            return None
        return self._ctx.read_back(name, args or {}, r)

    def read_recovery_state(self):
        # A pure state read is never a billed action, and the shell settles before it reads (the portal writes in a
        # post-render effect). Draft DOM values come back separately as draft_ui_state, never merged into agentActions.
        if self._ctx is None:
            return {"full_state": {}, "payer_a_state": {}, "payer_b_state": {}, "fax": {}, "draft_ui_state": {}}
        return self._ctx.recovery_state()

    def list_controls(self):
        """The live control model of the current page: [{ref, role, label, name, required, value, options, commit}]."""
        return self._ctx.controls() if self._ctx is not None else []


def _build_gui_backend(driver):
    return _GuiBackend(driver)


def _env_type(task):
    env = (task or {}).get("environment") or {} if isinstance(task, dict) else {}
    if isinstance(env, dict):
        return env.get("type")
    return env


def _augment_task(task, root_content):
    """Non-mutating: surface the root-agent deliverable where a benchmark adapter looks for it, without
    touching the caller's task dict and without reading any oracle field."""
    if not isinstance(task, dict) or root_content is None:
        return task
    t = dict(task)
    for k in ("deliverable", "answer"):
        t.setdefault(k, root_content)
    return t


def run_recovery_v3(task, root_content, lifecycle, trajectory=None, driver=None, judge=None, goal=None, step=None):
    """One recovery invocation. Returns a summary dict and records one recovery_v3 event per episode.

    `trajectory` is the run's trajectory, read (the agent's own actions are what commitments are resolved from). Events
    are recorded through the driver (ctx.emit); without a driver (offline use) they are appended to `trajectory`."""
    record = driver.emit if driver is not None else (trajectory.append if isinstance(trajectory, list) else None)
    summary = {"dispatched": False, "triggered": False, "env_type": _env_type(task), "episodes": []}

    stack = get_recovery_stack(summary["env_type"])
    if stack is None:
        summary["reason"] = "no_stack_for_env"
        return summary
    substrate, wf_registry, bench = stack

    # lifecycle gate lives in the benchmark adapter (layer 4), never in the kernel/substrate.
    try:
        triggered = bool(bench.should_trigger(lifecycle))
    except Exception:
        triggered = False
    summary["triggered"] = triggered
    if not triggered:
        summary["reason"] = "lifecycle_not_triggered"
        return summary

    if driver is not None:
        try:
            substrate.driver = _build_gui_backend(driver)
        except Exception as _e:
            summary["reason"] = "gui_backend_wire_error:%r" % _e
            return summary
    run_task = _augment_task(task, root_content)
    # surface the LIVE portal state (agentActions/appealActions) into authoritative_state so the benchmark adapter can
    # derive a landed-decision documentation gap oracle-blind (never reads gold).
    if getattr(substrate, "driver", None) is not None:
        try:
            live = substrate.driver.read_recovery_state() or {}
            fs = live.get("full_state") or {}
            if isinstance(run_task, dict) and isinstance(fs, dict):
                run_task = dict(run_task)
                astate = dict(run_task.get("authoritative_state") or {})
                astate["agentActions"] = fs.get("agentActions") or {}
                astate["appealActions"] = fs.get("appealActions") or {}
                # draft (selected-but-unsubmitted) UI values are surfaced SEPARATELY -- the
                # commitment resolver may read them to detect an operational commitment, but they are NOT part
                # of the persisted agentActions that verify_effect/_submit_realized score.
                astate["draft_ui_state"] = live.get("draft_ui_state") or {}
                astate["_live_full_state"] = fs
                astate["payer_a_state"] = live.get("payer_a_state") or {}
                astate["payer_b_state"] = live.get("payer_b_state") or {}
                run_task["authoritative_state"] = astate
                run_task["recovery_readback"] = live          # the obligation view is built from it (hab.py context)
                try:
                    run_task["gui_controls"] = substrate.driver.list_controls()
                except Exception:
                    run_task["gui_controls"] = []
                try:
                    snap = substrate.driver.call_tool("snapshot", {})
                    run_task["observation"] = (snap or {}).get("observation") if isinstance(snap, dict) else None
                except Exception:
                    pass
        except Exception as _le:
            summary["gui_live_state_error"] = "%r" % _le

    results = RecoveryKernel().run_all_episodes(
        bench, wf_registry, substrate, driver, run_task, trajectory, goal, judge)
    summary["dispatched"] = True

    for res in results:
        created = getattr(res, "created_ids", None) or []
        rec = {
            "step": step,
            "event_type": "recovery_v3",
            "surface": "state",
            "goal_id": getattr(res, "goal_id", None),
            "episode_state": getattr(res, "state", None),
            "verified_level": getattr(res, "verified_level", None),
            "verified_as": ("VERIFIED_PAYLOAD" if getattr(res, "verified_level", None) == "payload"
                            else "VERIFIED_EFFECT" if getattr(res, "verified_level", None) == "effect"
                            else getattr(res, "state", None)),
            "path": getattr(res, "path", None),
            "metrics_bucket": getattr(res, "metrics_bucket", None),
            "created_id": (created[0] if created else None),
            "reason": getattr(res, "reason", None),
            "status": "ok",
        }
        if record is not None:
            record(rec)
        summary["episodes"].append(
            {**{k: rec[k] for k in ("goal_id", "episode_state", "path", "metrics_bucket", "reason")},
             "blocked_argument": getattr(res, "blocked_argument", None),
             "re_entry_requested": bool("re_entry_requested" in str(getattr(res, "reason", "") or "")
                                        or "needs_new_content" in str(getattr(res, "reason", "") or ""))})
    return summary
