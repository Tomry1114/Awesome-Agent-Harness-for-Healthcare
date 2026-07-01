"""ActionExecutor (Commit B) -- the SINGLE per-action pipeline stage: execute one action, normalize its result,
active read-back, run the harness after_action, build the canonical tool_call event.

This is a faithful extraction of run.py's inline per-action logic (equivalence refactor -- the statements are
copied verbatim, not rewritten). Agent actions, ACQUIRE reads, and (Commit C) COMPLETE-effect mutations all go
through the SAME three stages here, so there is exactly one execution/normalization/verification path.

The executor holds only environment-facing dependencies (state hash/snapshot fns, the canonical-schema module,
env_type); the LOOP keeps loop-control (reconcile recovery, budgets, circuit-breaker, feedback folding, and
after-action decision processing). No benchmark names.
"""
import json
import os


class ActionOutcome:
    __slots__ = ("res", "err", "result_ok", "result_status", "recon",
                 "state_before", "state_after", "snap_before", "snap_after", "created_id")

    def __init__(self, res, err, result_ok, result_status, recon,
                 state_before, state_after, snap_before, snap_after):
        self.res = res
        self.err = err
        self.result_ok = result_ok
        self.result_status = result_status
        self.recon = recon
        self.state_before = state_before
        self.state_after = state_after
        self.snap_before = snap_before
        self.snap_after = snap_after
        self.created_id = None


class ActionExecutor:
    def __init__(self, canon_module, env_type, state_hash, state_snapshot):
        self._canon = canon_module          # canonical_schema
        self.env_type = env_type
        self._state_hash = state_hash        # callable(env) -> hash
        self._state_snapshot = state_snapshot  # callable(env) -> structured snapshot | None

    # ---- stage 1: execute + normalize + active read-back (verbatim from run.py) ----------------------
    def execute_and_normalize(self, action, env, ledger=None, auth=None):
        state_before = self._state_hash(env); snap_before = self._state_snapshot(env)
        if ledger is not None and auth is not None:
            if not ledger.dispatch_authorization(auth):   # C3.1: dispatch REQUIRES status==RESERVED. If not, REFUSE to execute (fail-closed) -- never run a mutation whose authorization is not in a dispatchable state
                return ActionOutcome({"error": "authorization_not_dispatchable"}, "authorization_not_dispatchable",
                                     False, "failed", None, state_before, state_before, snap_before, snap_before)
            # DISPATCHED from here: the mutation may have landed even on a transport error, so this auth can never re-authorize another mutation
        try:
            res = env.call_tool(action["tool"], action.get("args", {}))
        except Exception as _e:
            res = {"error": repr(_e)}
        state_after = self._state_hash(env); snap_after = self._state_snapshot(env)
        # UNIFIED tool-result status, computed BEFORE the harness sees the result.
        err = res.get("error") if isinstance(res, dict) else None
        if not err and isinstance(res, dict):
            if res.get("success") is False or res.get("ok") is False or \
               str(res.get("status", "")).lower() in ("failed", "error", "failure"):
                err = "result_status_failure"
            _out = res.get("output")
            if not err and isinstance(_out, str) and _out.lstrip().startswith("["):
                _marker = (_out[_out.find("[") + 1:_out.find("]")] if "]" in _out else _out[:40]).lower()
                if any(w in _marker for w in ("error", "unknown", "invalid", "fail")):
                    err = _out[:120]
        # ACTIVE READ-BACK: a write that CLAIMED success -> re-read to confirm it landed; a CLAIMED-FAILED
        # write -> read back, it may have landed despite a transport error.
        recon = None
        if not err:
            try:
                recon = env.reconcile_write(action["tool"], action.get("args", {}), res)
            except Exception as _rex:
                recon = {"confirmed": None, "detail": "reconcile_error:%r" % (_rex,)}
            if recon and recon.get("confirmed") is False:
                err = "readback_unconfirmed"
        elif err:
            try:
                recon = env.reconcile_write(action["tool"], action.get("args", {}), res)
            except Exception as _rex:
                recon = {"confirmed": None, "detail": "reconcile_error:%r" % (_rex,)}
            if recon and recon.get("confirmed") is True:
                err = None
        _cres = self._canon.canonical_result(res) or {}
        _estr = ("%s %s" % (_cres.get("error_type"), err)).lower()
        result_status = ("ok" if not err
                         else ("unknown" if ("timeout" in _estr or "readback" in _estr
                                             or any(c in _estr for c in ("500", "502", "503", "429")))
                               else "failed"))
        return ActionOutcome(res, err, (not err), result_status, recon,
                             state_before, state_after, snap_before, snap_after)

    # ---- stage 2: harness after_action (verbatim call; caller keeps try/except + enforce handling) ----
    def run_after_action(self, harness, action, outcome, step):
        hb_before = outcome.snap_before if outcome.snap_before is not None else outcome.state_before
        hb_after = outcome.snap_after if outcome.snap_after is not None else outcome.state_after
        return harness.after_action(action, outcome.res, hb_before, hb_after, step=step,
                                    canonical_observation=self._canon.canonical_observation(outcome.res, self.env_type),
                                    result_ok=outcome.result_ok, raw_observation=outcome.res,
                                    result_status=outcome.result_status)

    # ---- stage 3: the canonical tool_call event (verbatim) -------------------------------------------
    def build_event(self, action, outcome, step, origin="agent", audience="agent"):
        res, err = outcome.res, outcome.err
        src_full = json.dumps(res, ensure_ascii=False)
        obs = src_full[:int(os.environ.get("MH_OBS_MAX_LEN", "10000"))]
        ev = {"step": step, "event_type": "tool_call", "tool": action["tool"],
              "args": action.get("args", {}), "result": res, "observation": obs, "ts": str(step),
              "status": "error" if err else "ok",
              "canonical_action": self._canon.canonical_action(action, self.env_type),
              "canonical_result": self._canon.canonical_result(res),
              "agent_visible_text": obs,  # EXACT string fed into the agent context (Observability truth)
              "canonical_observation": self._canon.canonical_observation(res, self.env_type),
              "origin": origin,        # agent | recovery  (C4: who initiated this action)
              "audience": audience,    # agent | harness | both  (C4: who the observation is delivered to)
              "delivery_record": {"produced": bool(res), "rendered_to_agent": (audience in ("agent", "both")), "consumed_by_agent": False,
                                  "source_hash": __import__("hashlib").sha256(src_full.encode("utf-8", "replace")).hexdigest()[:12],
                                  "rendered_hash": __import__("hashlib").sha256(obs.encode("utf-8", "replace")).hexdigest()[:12],
                                  "truncated": len(src_full) > len(obs),
                                  "error_state_rendered": bool(err) and any(w in obs.lower() for w in ("error", "fail", "invalid", "exception", "operationoutcome"))},
              "state_record": {"state_before_hash": outcome.state_before, "state_after_hash": outcome.state_after,
                               "state_changed": (outcome.state_before != outcome.state_after)
                               if (outcome.state_before is not None and outcome.state_after is not None) else None}}
        if err:
            _es = str(err)
            ev["error_type"] = next(("http_" + c for c in ("400", "401", "403", "404", "409", "422", "500", "502", "503")
                                     if ("HTTP " + c) in _es),
                                    "exception" if any(k in _es for k in ("Error", "Exception", "Traceback")) else "tool_error")
        return ev, obs
