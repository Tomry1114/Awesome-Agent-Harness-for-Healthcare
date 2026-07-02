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


# --- LIVE GUI backend shim (wiring boundary) -----------------------------------------------------------
# GuiSubstrateAdapter calls backend.call_tool(name,args) / .read_recovery_state() / .reconcile_write() /
# .page / .full_state. A raw RunDriver exposes none of these (its .env=GuiEnvReal does). Injecting the raw
# driver would AttributeError on every call. This shim routes each GUI primitive through the SAME counted
# executor path recovery FHIR reads used (budget gate + execute_and_normalize + _count_env + provenance
# event) so recovery GUI actions are FAIR against the tool budget and attributable -- never a free action.
class _RunDriverGuiBackend(object):
    def __init__(self, driver):
        self.d = driver
        self.env = getattr(driver, "env", None)

    def call_tool(self, name, args):
        # HARD budget gate BEFORE the real env call (parity with OFF; recovery cannot exceed the tool budget)
        if hasattr(self.d, "can_execute_recovery_action") and not self.d.can_execute_recovery_action():
            return {"error": "recovery_budget_exhausted"}
        rd = {"type": "tool_call", "tool": name, "args": args or {}}
        out = self.d.ex.execute_and_normalize(rd, self.env)
        try:
            self.d._count_env(rd)                              # #6 provenance + budget accounting
        except Exception:
            pass
        try:
            ev, _ = self.d.ex.build_event(rd, out, getattr(self.d, "step", 0),
                                          origin="recovery", audience="harness")
            self.d._emit_event(ev)
        except Exception:
            pass
        res = getattr(out, "res", None)
        return res if isinstance(res, dict) else {"result": res}

    def reconcile_write(self, name, args, r):
        fn = getattr(self.env, "reconcile_write", None)
        if not callable(fn):
            return None
        try:
            return fn(name, args or {}, r)
        except Exception:
            return None

    def read_recovery_state(self):
        # counted snapshot refreshes the emr digest through the executor; then read the FULL portals_state
        # (emr + payer portals + fax) from localStorage so payer-landed submissions are also visible.
        try:
            self.call_tool("snapshot", {})
        except Exception:
            pass
        import json as _json
        fs, pa, pb, fax = {}, {}, {}, {}
        try:
            raw = self.env.page.evaluate(
                "() => { try { return localStorage.getItem('portals_state'); } catch(e){ return null; } }")
            st = _json.loads(raw) if raw else None
            if isinstance(st, dict):
                fs = st.get("emr") or {}
                pa = st.get("payerA") or {}
                pb = st.get("payerB") or {}
                fax = st.get("fax") or {}
        except Exception:
            fs = getattr(self.env, "full_state", None) or {}
        return {"full_state": fs, "payer_a_state": pa, "payer_b_state": pb, "fax": fax}

    def list_controls(self):
        """Live unified control model for the current page: [{ref, role, label, name, required, value,
        options, commit}]. Benchmark-name-free; the generic GUI workflow decides fills from this."""
        js = ("() => {"
              " const SEL='button,a,[role=button],input,select,textarea';"
              " const els=Array.from(document.querySelectorAll(SEL));"
              " return els.map((el,i)=>{"
              "  let ref=el.getAttribute('data-mh-ref'); ref=ref!==null?parseInt(ref):i;"
              "  const tag=el.tagName.toLowerCase();"
              "  const type=(el.getAttribute('type')||el.getAttribute('role')||'').toLowerCase();"
              "  let label=(el.innerText||'').trim();"
              "  if(!label) label=el.getAttribute('aria-label')||el.getAttribute('placeholder')||el.getAttribute('name')||el.getAttribute('title')||'';"
              "  const name=el.getAttribute('name')||el.getAttribute('id')||'';"
              "  const required=!!(el.required||el.getAttribute('aria-required')==='true');"
              "  const value=String(el.value||'').slice(0,80);"
              "  let options=[]; if(tag==='select'){try{options=Array.from(el.options).map(o=>o.value||o.text);}catch(e){}}"
              "  const isSubmit=(type==='submit')||(tag==='button' && /submit|save|send|file appeal|confirm/i.test(label));"
              "  const role=tag==='select'?'select':(tag==='textarea'?'textbox':(tag==='input'?(type||'input'):(tag==='button'?'button':tag)));"
              "  return {ref, role, label:String(label).slice(0,60), name, required, value, options, commit:!!isSubmit};"
              " }).filter(c=>c.label||c.role==='select'||c.name);"
              "}")
        try:
            ctrls = self.env.page.evaluate(js)
            return ctrls if isinstance(ctrls, list) else []
        except Exception:
            return []

    @property
    def page(self):
        return getattr(self.env, "page", None)

    @property
    def full_state(self):
        return getattr(self.env, "full_state", None) or {}


def _build_gui_backend(driver):
    return _RunDriverGuiBackend(driver)



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
    elif summary["env_type"] == "gui" and driver is not None:
        try:
            substrate.driver = _build_gui_backend(driver)
        except Exception as _e:
            summary["reason"] = "gui_backend_wire_error:%r" % _e
            return summary
    else:
        _inject_driver(substrate, driver)
    run_task = _augment_task(task, root_content)
    # GUI: surface the LIVE portal state (agentActions/appealActions) into authoritative_state so the
    # benchmark adapter can derive a landed-decision documentation gap oracle-blind (never reads gold).
    if summary["env_type"] == "gui" and getattr(substrate, "driver", None) is not None:
        try:
            live = substrate.driver.read_recovery_state() or {}
            fs = live.get("full_state") or {}
            if isinstance(run_task, dict) and isinstance(fs, dict):
                run_task = dict(run_task)
                astate = dict(run_task.get("authoritative_state") or {})
                astate["agentActions"] = fs.get("agentActions") or {}
                astate["appealActions"] = fs.get("appealActions") or {}
                astate["_live_full_state"] = fs
                astate["payer_a_state"] = live.get("payer_a_state") or {}
                astate["payer_b_state"] = live.get("payer_b_state") or {}
                run_task["authoritative_state"] = astate
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
