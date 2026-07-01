"""RunDriver (Commit C5b, hardened C5c) -- the real-environment implementation of the RecoveryOrchestrator's
driver, over the live harness kernel + ActionExecutor + env. The ONLY glue between the orchestrator's abstract
primitives and run.py's actual harness/executor/env. No benchmark names.

C5c hardening (reviewer round 5):
  #3  the prerequisite READ goes through the SAME before_action -> execute -> after_action pipeline as any
      action (a harness-vetoed read is NOT executed and does NOT resolve the prerequisite).
  #4  RESOLVED is proven from the LEDGER (a new evidence record for the ACTIVE subject, scope matched, in a
      resolving state PRESENT|ABSENT) -- NOT guessed from the raw env result. No hard-coded result semantics.
  #5  after_action exception -> auth UNKNOWN (never VERIFIED); ctx.verification is reset BEFORE after_action so
      a stale True cannot leak into a VERIFIED verdict; a read whose after_action crashed is NOT resolved.
  #6  every real env.call_tool made by recovery is reported through on_env_action so it counts against the
      tool budget and is attributable (origin=recovery).
"""

_RESOLVING = ("PRESENT", "ABSENT")


class RunDriver:
    def __init__(self, harness, executor, env, task, state_snapshot, on_env_action=None):
        self.h = harness
        self.ex = executor
        self.env = env
        self.task = task
        self.snap = state_snapshot
        self.on_env = on_env_action   # #6: callback(action, origin) invoked on every real env.call_tool
        self.step = 0                 # set by run.py before each realize()
        self.trajectory = []          # set by run.py before each realize()
        self.episode_id = None        # #9: set by run.py per effect episode (recovery-<n>)
        self._action_seq = 0          # #9: monotonic per-recovery-action id, unique WITHIN a step

    # -- authorization --
    def mint(self, scope):
        return self.h.ledger.mint_authorization(
            source="deterministic_gap",
            allowed_semantic_type=scope.get("allowed_semantic_type"),
            allowed_tool=scope.get("allowed_tool"),
            allowed_effect=scope.get("allowed_effect"),
            target_path=scope.get("target_path"),
            expected_postcondition=scope.get("expected_postcondition"))

    def auth_id(self, auth):
        return auth.authorization_id

    def set_hold(self):
        self.h.ledger.set_mutation_hold(capability="recovery")

    def cancel(self, auth):
        self.h.ledger.cancel_authorization(auth)

    def reserve(self, auth):
        return self.h.ledger.reserve_authorization(auth)

    def auth_status(self, auth):
        return auth.status

    def mode(self):
        # the harness enforcement mode ("enforce" | "assist" | "observe"); gates whether recovery may write.
        return getattr(self.h, "mode", None)

    # -- internal helpers ------------------------------------------------------
    def _count_env(self, action):
        if self.on_env:
            try:
                self.on_env(action, "recovery")   # #6
            except Exception:
                pass

    def _runtime_error(self, detail):
        self.trajectory.append({"step": self.step, "event_type": "harness_runtime_error",
                                "origin": "recovery", "detail": detail, "status": "error"})

    def _emit_event(self, ev):
        # #9: stamp a unique action_id + episode id + parent step so metric dedup keys on action_id, not step
        # (multiple recovery reads/creates can share one agent step and must NOT collapse into one opportunity).
        self._action_seq += 1
        ev["action_id"] = "rec-%d-%d" % (self.step, self._action_seq)
        ev["recovery_episode_id"] = self.episode_id
        ev["parent_agent_step"] = self.step
        self.trajectory.append(ev)

    def _reset_verification(self):
        try:
            setattr(self.h.ctx, "verification", None)   # #5: no stale True may leak into a verdict
        except Exception:
            pass

    # -- full before_action for the MUTATION (returns raw + effective + the ACQUIRE next_action) --
    def evaluate(self, action):
        eff = self.h.before_action(action, self.snap(self.env), step=self.step)
        if eff is not None and getattr(eff, "events", None):
            self.trajectory.extend(eff.events)
        raw = getattr(self.h.ctx, "last_raw_decision", None)
        raw_t = raw.type if raw is not None else (eff.type if eff is not None else "ALLOW")
        eff_t = eff.type if eff is not None else "ALLOW"
        nxt = None
        if eff is not None and eff.type == "ACQUIRE":
            ex = (getattr(eff, "extra", None)
                  or (getattr(eff, "raw", None) and getattr(eff.raw, "extra", None)) or {})
            nxt = ex.get("next_action")
        return (raw_t, eff_t, nxt)

    # -- #3: the prerequisite READ through the SAME full pipeline; #4: LEDGER-proven resolved --
    def acquire(self, next_action):
        if not next_action or not next_action.get("tool"):
            return False
        rd = dict(next_action); rd.setdefault("type", "tool_call")
        active = self.h.ledger.subject_id()
        before_idx = len(self.h.ledger.evidence)
        # #3: before_action -- a read the harness vetoes (BLOCK/ESCALATE) is NOT executed and does NOT resolve.
        try:
            hb = self.h.before_action(rd, self.snap(self.env), step=self.step)
        except Exception as _e:
            hb = None
            self._runtime_error("recovery_read_before_action:%r" % _e)
        if hb is not None and getattr(hb, "events", None):
            self.trajectory.extend(hb.events)
        if hb is not None and getattr(hb, "type", "ALLOW") in ("BLOCK", "ESCALATE"):
            return False
        outcome = self.ex.execute_and_normalize(rd, self.env)
        self._count_env(rd)                                       # #6
        self._reset_verification()
        after_ok = True
        try:
            self.ex.run_after_action(self.h, rd, outcome, self.step)   # ScopeEvidenceBinding records EvidenceState
        except Exception as _e:
            after_ok = False                                      # #5: crashed read did NOT bind -> unresolved
            self._runtime_error("recovery_read_after_action:%r" % _e)
        try:
            self.h.ledger.acquire_count = getattr(self.h.ledger, "acquire_count", 0) + 1
        except Exception:
            pass
        ev, _ = self.ex.build_event(rd, outcome, self.step, origin="recovery", audience="harness")
        self._emit_event(ev)
        if not after_ok:
            return False
        # #4: RESOLVED proven from the LEDGER (not the raw result): a NEW record for the ACTIVE subject, scope
        # matched, in a resolving state. Mirrors RequiredContext._resolved_units so acquire and gate AGREE.
        for rec in self.h.ledger.evidence[before_idx:]:
            if (rec.get("scope_relation") == "matched"
                    and rec.get("subject_id") == active
                    and rec.get("evidence_state") in _RESOLVING):
                return True
        return False

    # -- #7: existing-effect probe as a RECOVERY READ through the SAME executor (counted #6, canonical event,
    #    after_action binds evidence) instead of a private env.call_tool. Returns {state, texts, matched_ids}. --
    def inspect_effect(self, resource_type, subject_ref):
        from .effect_completion import classify_effect_inspection
        rd = {"type": "tool_call", "tool": "fhir_search",
              "args": {"resourceType": resource_type, "subject": subject_ref}}
        outcome = self.ex.execute_and_normalize(rd, self.env)
        self._count_env(rd)                                       # #6
        self._reset_verification()
        try:
            self.ex.run_after_action(self.h, rd, outcome, self.step)
        except Exception as _e:
            self._runtime_error("recovery_inspect_after_action:%r" % _e)
        ev, _ = self.ex.build_event(rd, outcome, self.step, origin="recovery", audience="harness")
        self._emit_event(ev)
        if outcome.err:                                          # a failed probe is UNKNOWN, never ABSENT
            return {"state": "UNKNOWN", "texts": [], "matched_ids": [], "reason": "probe_failed"}
        return classify_effect_inspection(outcome.res)

    # -- authorized CREATE through the same executor (dispatch just before the env call) + finalize the auth --
    def execute(self, action, auth):
        outcome = self.ex.execute_and_normalize(action, self.env, ledger=self.h.ledger, auth=auth)
        L = self.h.ledger
        if outcome.err == "authorization_not_dispatchable":
            # env was NOT called (auth not in a dispatchable state) -> never landed -> FAILED.
            L.fail_authorization(auth)
            ev, _ = self.ex.build_event(action, outcome, self.step, origin="recovery", audience="harness")
            self._emit_event(ev)
            outcome.created_id = None
            return outcome
        self._count_env(action)                                   # #6: the create hit the env
        self._reset_verification()                                # #5: before after_action
        hp = None; after_ok = True
        try:
            hp = self.ex.run_after_action(self.h, action, outcome, self.step)
        except Exception as _e:
            after_ok = False
            self._runtime_error("recovery_create_after_action:%r" % _e)
        ver = getattr(self.h.ctx, "verification", None)
        aeff = (hp.type if hp is not None else "ALLOW") if after_ok else None
        cf = outcome.recon.get("confirmed") if outcome.recon else None
        if not after_ok:
            L.unknown_authorization(auth)                         # #5: crashed after_action -> may have landed -> UNKNOWN, NEVER VERIFIED
        elif outcome.result_status == "failed" and cf is False:
            L.fail_authorization(auth)
        elif outcome.result_status == "unknown" or aeff == "RECONCILE" or cf is None:
            L.unknown_authorization(auth)
        elif cf is True and ver is True and aeff == "ALLOW":
            L.verify_authorization(auth)
        elif ver is False:
            L.fail_authorization(auth)
        else:
            L.unknown_authorization(auth)
        ev, _ = self.ex.build_event(action, outcome, self.step, origin="recovery", audience="harness")
        self._emit_event(ev)
        outcome.created_id = (outcome.res.get("id") if isinstance(outcome.res, dict) else None)
        return outcome
