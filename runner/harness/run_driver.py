"""RunDriver (Commit C5b, hardened C5c/C5d) -- the real-environment implementation of the RecoveryOrchestrator's
driver, over the live harness kernel + ActionExecutor + env. The ONLY glue between the orchestrator's abstract
primitives and run.py's actual harness/executor/env. No benchmark names.

C5d hardening (reviewer round 6): ONE strict internal read executor (execute_recovery_read) backs BOTH the
prerequisite ACQUIRE and the existing-effect INSPECT, so they cannot diverge:
  P0-1  before_action must be ALLOW: an exception OR any non-ALLOW decision (ACQUIRE/REVISE/RECONCILE/BLOCK/
        ESCALATE) aborts the read WITHOUT executing it (harness-internal actions are strict, never fail-open).
  P0-2  after_action must succeed: any exception -> the read yields UNKNOWN (never a raw-result ABSENT).
  P0-3  RESOLVED requires an EXACT ledger delta from THIS action: a NEW record (since before_idx) whose
        resource == the requested evidence_unit, subject_id == active subject, scope_relation == "matched",
        evidence_state in {PRESENT, ABSENT}. The index slice IS the "came from this action" provenance the
        acquisition_key was meant to give -- a stray PRESENT/ABSENT record can no longer close a prerequisite.
  P1    can_execute_recovery_action(): a HARD budget gate checked BEFORE every real env.call_tool (read AND
        create), so recovery cannot exceed the tool budget (not merely be counted after the fact).
"""

_RESOLVING = ("PRESENT", "ABSENT")


class RunDriver:
    def __init__(self, harness, executor, env, task, state_snapshot, on_env_action=None, budget_check=None):
        self.h = harness
        self.ex = executor
        self.env = env
        self.task = task
        self.snap = state_snapshot
        self.on_env = on_env_action     # #6: callback(action, origin) on every real env.call_tool
        self.budget_check = budget_check  # P1: () -> bool, True while recovery tool budget remains
        self.step = 0                   # set by run.py before each realize()
        self.trajectory = []            # set by run.py before each realize()
        self.episode_id = None          # #9: set by run.py per effect episode (recovery-<n>)
        self._action_seq = 0            # #9: monotonic per-recovery-action id, unique WITHIN a step

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

    # -- P1 HARD budget gate: refuse a recovery env call when no tool budget remains --
    def can_execute_recovery_action(self):
        if self.budget_check is None:
            return True
        try:
            return bool(self.budget_check())
        except Exception:
            return False   # R7 fix1: admission infra unknown -> deny (a harness-internal action never fails open)

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

    # == THE single strict internal read executor (P0-1/P0-2/P0-3) ==============
    def execute_recovery_read(self, action, expected_subject=None, expected_evidence_unit=None):
        """Run ONE harness-internal read to a TRUSTWORTHY state. Returns (state, outcome) with state in
        {"PRESENT","ABSENT","UNKNOWN"}. UNKNOWN on ANY deviation (no budget / before_action not ALLOW /
        after_action crash / tool error / no exact ledger delta). Never returns ABSENT off a raw result."""
        rd = dict(action); rd.setdefault("type", "tool_call")
        active = expected_subject or self.h.ledger.subject_id()
        unit = expected_evidence_unit or ((rd.get("args") or {}).get("resourceType"))
        # P1: hard budget gate BEFORE the env call
        if not self.can_execute_recovery_action():
            self._runtime_error("recovery_read_budget_exhausted")
            return ("UNKNOWN", None)
        # P0-1: before_action MUST be ALLOW; exception or any non-ALLOW aborts WITHOUT executing.
        try:
            hb = self.h.before_action(rd, self.snap(self.env), step=self.step)
        except Exception as _e:
            self._runtime_error("recovery_read_before_action:%r" % _e)
            return ("UNKNOWN", None)
        if hb is not None and getattr(hb, "events", None):
            self.trajectory.extend(hb.events)
        if hb is not None and getattr(hb, "type", "ALLOW") != "ALLOW":
            return ("UNKNOWN", None)
        before_idx = len(self.h.ledger.evidence)
        outcome = self.ex.execute_and_normalize(rd, self.env)
        self._count_env(rd)                                       # #6
        self._reset_verification()
        # P0-2: after_action MUST succeed.
        after_ok = True
        try:
            self.ex.run_after_action(self.h, rd, outcome, self.step)   # ScopeEvidenceBinding records EvidenceState
        except Exception as _e:
            after_ok = False
            self._runtime_error("recovery_read_after_action:%r" % _e)
        ev, _ = self.ex.build_event(rd, outcome, self.step, origin="recovery", audience="harness")
        self._emit_event(ev)
        if outcome.err or not after_ok:
            return ("UNKNOWN", outcome)
        # P0-3: EXACT ledger delta from THIS action (index slice = provenance).
        for rec in self.h.ledger.evidence[before_idx:]:
            if (rec.get("scope_relation") == "matched"
                    and rec.get("subject_id") == active
                    and rec.get("evidence_state") in _RESOLVING
                    and (unit is None or rec.get("resource") == unit)):
                return (rec.get("evidence_state"), outcome)
        return ("UNKNOWN", outcome)

    # -- prerequisite ACQUIRE: resolved iff the strict reader confirms PRESENT/ABSENT --
    def acquire(self, next_action):
        if not next_action or not next_action.get("tool"):
            return False
        state, outcome = self.execute_recovery_read(next_action)
        if outcome is not None:   # R7 fix3: count ONLY a query that was actually issued -- an admission failure
            try:                  # (budget/before_action) never sent a query, so it must not spend the acquisition budget
                self.h.ledger.acquire_count = getattr(self.h.ledger, "acquire_count", 0) + 1
            except Exception:
                pass
        return state in _RESOLVING

    # -- #7: existing-effect probe via the SAME strict reader; PRESENT still fail-closes on no comparable text --
    def inspect_effect(self, resource_type, subject_ref):
        from .effect_completion import classify_effect_inspection
        rd = {"type": "tool_call", "tool": "fhir_search",
              "args": {"resourceType": resource_type, "subject": subject_ref}}
        state, outcome = self.execute_recovery_read(rd, expected_subject=subject_ref,
                                                    expected_evidence_unit=resource_type)
        if state == "ABSENT":
            return {"state": "ABSENT", "texts": [], "matched_ids": []}
        if state == "PRESENT":
            # trust the ledger's PRESENT, but still extract texts + apply the "present-but-no-comparable-
            # representation -> UNKNOWN" fail-closed so a create decision has something to compare against.
            return classify_effect_inspection(outcome.res if outcome is not None else None)
        return {"state": "UNKNOWN", "texts": [], "matched_ids": []}

    # -- authorized CREATE through the same executor (dispatch just before the env call) + finalize the auth --
    # ===== GUI substrate: authoritative snapshot (H1) + affordance resolution (H2) + marker verify (H3) =====
    def snapshot_gui_state(self):
        """H1: read the portal full_state THROUGH the executor (counted, canonical event) -- never a private
        env read. Returns the full_state/emr dict, or None (budget/read failure)."""
        if not self.can_execute_recovery_action():
            self._runtime_error("recovery_snapshot_budget_exhausted")
            return None
        rd = {"type": "tool_call", "tool": "snapshot", "args": {}}
        outcome = self.ex.execute_and_normalize(rd, self.env)
        self._count_env(rd)
        ev, _ = self.ex.build_event(rd, outcome, self.step, origin="recovery", audience="harness")
        self._emit_event(ev)
        res = outcome.res if isinstance(outcome.res, dict) else {}
        fs = res.get("full_state")
        fs = fs if isinstance(fs, dict) else (res if isinstance(res, dict) else None)
        import copy as _copy
        return _copy.deepcopy(fs) if isinstance(fs, dict) else None   # point-in-time snapshot (never a live env ref)

    def resolve_document_affordance(self, affordance, state_view):
        """H2: resolve the commit affordance to ONE concrete action. Mock portal -> click(target=target_key).
        A real portal would resolve a unique DOM ref by label/role. None if it cannot be uniquely resolved."""
        aff = affordance or {}
        tk = aff.get("target_key")
        if not tk:
            return None
        return {"type": "tool_call", "tool": aff.get("tool", "click"), "args": {"target": tk}}

    def _marker_true(self, state_view, marker):
        from .recovery_adapter import _get_state_path
        v = _get_state_path(state_view or {}, marker)
        return v is True or (isinstance(v, str) and v.strip().lower() == "true")

    def _execute_gui_marker(self, action, auth):
        """H3: authorized GUI commit + EXACT verify the marker went False->True. Reuses ActionExecutor +
        MutationAuthorization (the orchestrator already reserved this auth)."""
        L = self.h.ledger
        marker = action.get("_verify_marker")
        if not self.can_execute_recovery_action():
            self._runtime_error("recovery_gui_budget_exhausted")
            L.cancel_authorization(auth)
            class _O:
                def __init__(s): s.res = {"error": "recovery_budget_exhausted"}; s.err = "recovery_budget_exhausted"; s.result_status = "failed"; s.recon = None; s.created_id = None
            return _O()
        pre = self.snapshot_gui_state()
        outcome = self.ex.execute_and_normalize(action, self.env, ledger=L, auth=auth)   # dispatches auth + runs the click
        if outcome.err == "authorization_not_dispatchable":
            L.fail_authorization(auth)
            ev, _ = self.ex.build_event(action, outcome, self.step, origin="recovery", audience="harness")
            self._emit_event(ev); outcome.created_id = None
            return outcome
        self._count_env(action)
        post = self.snapshot_gui_state()
        ev, _ = self.ex.build_event(action, outcome, self.step, origin="recovery", audience="harness")
        self._emit_event(ev)
        pre_t, post_t = self._marker_true(pre, marker), self._marker_true(post, marker)
        if post_t and not pre_t:
            L.verify_authorization(auth)          # EXACT False->True: the mechanical effect landed
        elif post_t:
            L.unknown_authorization(auth)         # already true before -> ambiguous (should not happen; ABSENT was checked)
        else:
            L.fail_authorization(auth)            # marker did not flip -> the action did not take
        outcome.created_id = None
        return outcome

    def execute(self, action, auth):
        if action.get("_verify_marker"):          # GUI state-marker effect (documentedAppealInEpic False->True)
            return self._execute_gui_marker(action, auth)
        L = self.h.ledger
        # P1: hard budget gate BEFORE the create's env call.
        if not self.can_execute_recovery_action():
            self._runtime_error("recovery_create_budget_exhausted")
            L.cancel_authorization(auth)   # R7 fix2: never dispatched -> CANCELLED (RESERVED->CANCELLED valid); fail() would be an illegal RESERVED->FAILED and leave it stuck
            class _O:
                def __init__(s): s.res = {"error": "recovery_budget_exhausted"}; s.err = "recovery_budget_exhausted"; s.result_status = "failed"; s.recon = None; s.created_id = None
            return _O()
        outcome = self.ex.execute_and_normalize(action, self.env, ledger=L, auth=auth)
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
