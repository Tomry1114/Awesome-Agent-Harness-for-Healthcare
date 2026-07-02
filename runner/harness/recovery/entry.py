"""Bounded Clinical Recovery v3 - feature-flag entry point (MH_RECOVERY_V3==1).

run.py's existing effect-completion trigger calls run_recovery_v3(...) INSTEAD of the legacy
recovery_adapter path when os.environ.get('MH_RECOVERY_V3')=='1'. With the flag unset (default) the
legacy path runs EXACTLY as before, so the current PB-cp3 recovery keeps working untouched.

This module is ADDITIVE. It wires get_recovery_stack(env_type, manifest) to the RecoveryKernel: it lets
the benchmark ADAPTER decide whether the lifecycle event triggers (layer 4), injects the live driver into
the (otherwise unwired) substrate, runs one episode per committed goal, and appends the episode results
to the run trajectory.

Oracle-blind: it reads only the task, the root-agent deliverable content and the live trajectory; never
gold/checkpoint/reference material. Python 3.8 compatible.
"""
from .registry import get_recovery_stack
from .kernel import RecoveryKernel


# --- LIVE FHIR backend shim (wiring boundary; may import legacy, keeps substrate/kernel pure) -----------
# FhirSubstrateAdapter calls backend.search(rt,subj) / backend.create(res). A raw RunDriver has neither
# method, so injecting it directly made every call AttributeError -> UNKNOWN/FAILED (recovery_env_actions=0,
# created_id=None). This shim satisfies that contract by driving the battle-tested legacy ceremony:
#   .search -> RunDriver.execute_recovery_read (budget gate + strict reader + provenance + _count_env)
#   .create -> RecoveryOrchestrator.realize    (mint/evaluate ALLOW/enforce/reserve/execute/server-readback)
class _RunDriverFhirBackend(object):
    def __init__(self, driver, manifest=None):
        self.d = driver
        self.manifest = manifest or {}
        self._orch = None

    def _orchestrator(self):
        if self._orch is None:
            from ..recovery_orchestrator import RecoveryOrchestrator
            self._orch = RecoveryOrchestrator(self.d)
        return self._orch

    def search(self, resource_type, subject):
        rd = {"type": "tool_call", "tool": "fhir_search",
              "args": {"resourceType": resource_type, "subject": subject}}
        state, outcome = self.d.execute_recovery_read(
            rd, expected_subject=subject, expected_evidence_unit=resource_type)
        if outcome is None:                       # never issued (budget/before_action) -> UNKNOWN shape
            return {"error": "read_not_issued", "status": "unknown"}
        res = getattr(outcome, "res", None)
        return _to_entries_shape(res)

    def read(self, resource_type, rid):
        rd = {"type": "tool_call", "tool": "fhir_read",
              "args": {"resourceType": resource_type, "id": rid}}
        state, outcome = self.d.execute_recovery_read(
            rd, expected_subject=None, expected_evidence_unit=resource_type)
        res = getattr(outcome, "res", None) if outcome is not None else None
        return res if isinstance(res, dict) else {}

    def create(self, resource):
        from ..semantics import canonicalize
        from ..authorization import action_target_path
        mact = {"type": "tool_call", "tool": "fhir_create", "args": {"resource": resource}}
        fsem = canonicalize(mact, self.manifest or {})
        rtb = resource.get("resourceType") if isinstance(resource, dict) else None
        scope = {"allowed_semantic_type": fsem.semantic_type, "allowed_tool": "fhir_create",
                 "allowed_effect": fsem.effect, "target_path": action_target_path(fsem, mact),
                 "expected_postcondition": {"resource": rtb, "status": "active", "verify": "server_readback"}}
        result = self._orchestrator().realize(mact, scope)
        st = getattr(result, "state", None)
        cid = getattr(result, "created_id", None)
        if st in ("VERIFIED", "ALREADY_REALIZED") and cid:
            return {"id": cid}
        # RECONCILING / BLOCKED / no-id -> may or may not have landed -> UNKNOWN (v3 never re-creates)
        raise RuntimeError("create_not_verified:%s" % st)


def _build_fhir_backend(driver, manifest):
    return _RunDriverFhirBackend(driver, manifest)


def _to_entries_shape(res):
    """The harness's canonical search shape is {'entries':[{'resource':..}, ..]} (see _FHIR_SEM). A raw HAPI
    FHIR Bundle uses 'entry'. If the live search came back as a raw Bundle, project it onto the canonical
    'entries' shape so classify_evidence_state / classify_effect_inspection see the records (else empty ->
    absence_when_empty -> a spurious ABSENT). Pass-through when already canonical."""
    if not isinstance(res, dict):
        return {"entries": []}
    if "entries" in res:
        return res
    if "entry" in res:
        ents = []
        for e in (res.get("entry") or []):
            r = e.get("resource", e) if isinstance(e, dict) else e
            ents.append({"resource": r if isinstance(r, dict) else {}})
        out = dict(res)
        out["entries"] = ents
        return out
    return res



def _env_type(task):
    env = (task or {}).get("environment") or {} if isinstance(task, dict) else {}
    if isinstance(env, dict):
        return env.get("type")
    return env


def _inject_driver(substrate, driver):
    """Wire the LIVE driver/backend into a stack substrate that get_recovery_stack built unwired. The
    substrate exposes exactly one of {driver, backend, _env}; only a currently-None slot is filled."""
    if driver is None or substrate is None:
        return
    for attr in ("driver", "backend", "_env"):
        if getattr(substrate, attr, "_unset") is None:
            try:
                setattr(substrate, attr, driver)
            except Exception:
                pass


def _augment_task(task, root_content):
    """Non-mutating: surface the root-agent deliverable where a benchmark adapter looks for it, without
    touching the caller's task dict and without reading any oracle field."""
    if not isinstance(task, dict) or root_content is None:
        return task
    t = dict(task)
    for k in ("deliverable", "answer"):
        t.setdefault(k, root_content)
    return t


def run_recovery_v3(task, root_content, lifecycle, trajectory=None,
                    driver=None, judge=None, goal=None, manifest=None, step=None):
    """Feature-flag v3 recovery entry. Returns a summary dict and appends per-episode events to trajectory.

    Call-site compatible with run.py's _run_effect_completion(_root_content, _lifecycle): the caller passes
    the task, the root deliverable content, the lifecycle event, the live trajectory list, a RunDriver-style
    driver, the judge_fn, the goal text and the substrate manifest.
    """
    events = trajectory if isinstance(trajectory, list) else []
    summary = {"dispatched": False, "triggered": False, "env_type": _env_type(task), "episodes": []}

    stack = get_recovery_stack(summary["env_type"], manifest)
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

    if summary["env_type"] == "fhir" and driver is not None:
        try:
            substrate.backend = _build_fhir_backend(driver, manifest)
        except Exception as _e:
            summary["reason"] = "fhir_backend_wire_error:%r" % _e
            return summary
    else:
        _inject_driver(substrate, driver)
    run_task = _augment_task(task, root_content)

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
            "path": getattr(res, "path", None),
            "metrics_bucket": getattr(res, "metrics_bucket", None),
            "created_id": (created[0] if created else None),
            "reason": getattr(res, "reason", None),
            "status": "ok",
        }
        events.append(rec)
        summary["episodes"].append(
            {k: rec[k] for k in ("goal_id", "episode_state", "path", "metrics_bucket")})
    return summary
