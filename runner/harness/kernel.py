"""HarnessKernel — orchestrates the capabilities, enforces budgets, and emits harness events. This is the object
run.py drives via before_action / after_action / before_final.

The harness has one mode, enforce: the effective decision is the capabilities' decision (ALLOW / REVISE / BLOCK /
ESCALATE / ...), bounded by the budgets. `mode` ("enforce") is recorded on every event and in the audit.

Budgets (must exist, to bound revision loops): max_revisions_per_action, max_interventions_per_task,
max_semantic_checks. They bound RESOURCES, not safety:

    safety verdict      BLOCK, ESCALATE, RECONCILE, and any decision carrying extra.must_resolve (a confirmed
                        contradiction). A budget never changes its type: it is applied with its own meaning at every
                        hook, exhausted or not.
    advisory verdict    REVISE, ACQUIRE, DEFER_FINAL without must_resolve. An exhausted intervention budget, or an
                        identical REVISE past max_revisions_per_action, stops the harness issuing it: the effective
                        type is ALLOW, the harness_decision event and the ledger intervention carry the note
                        (budget_exhausted_interventions / max_revisions_exceeded), and the episode goes on.

So a budget limit never turns an advisory verdict into ESCALATE (which after_action applies by ending the episode).

Final verdicts. before_final is the hook the shell applies (it records the proposal, findings, interventions, the
revision and intervention counters, the repair lifecycle, the mutation hold and the final-answer commit).
evaluate_final is the same capabilities' verdict as a QUERY: none of those records, no counter moves; what it
returns can later be recorded as the real verdict with record_final. Both reset the hook's working context (step, sem,
risk, verification, final_audit, hook_serial) as every hook call does; capabilities keep their own state.

Trace. Every _ev event carries the step. Every hook call the shell applies (before_action, after_action,
before_final, record_final) adds one shell_hook event {layer kernel, hook, step, origin, requester, result} and every
semantic unit the meter refuses adds one shell_refusal event, through the trajectory bound with bind_trajectory (held
until one is bound).
"""
from . import decision as D
from .capability import HarnessContext

# max_semantic_checks is the shell's total semantic ceiling. How it is split across goal-spec compilation, the final
# verification, kernel middleware and host middleware is the workspace's manifest section semantic_budget
# (harness/semantic_budget.py); the one meter enforcing both is HarnessContext.semantic_meter.
_DEFAULT_BUDGET = {"max_revisions_per_action": 2, "max_interventions_per_task": 8, "max_semantic_checks": 5}


class Effective:
    """What run.py acts on: the effective decision + feedback + the raw capability verdict (for audit).
    `note` names the resource limit that stopped an advisory verdict (None otherwise). `evaluation` is the
    _FinalEvaluation a before_final verdict came from (None at the other hooks): record_final records it."""
    __slots__ = ("type", "feedback", "raw", "events", "note", "evaluation")

    def __init__(self, type, feedback=None, raw=None, events=None, note=None):
        self.type = type
        self.feedback = feedback
        self.raw = raw
        self.events = events or []
        self.note = note
        self.evaluation = None


class _FinalEvaluation(object):
    """The capabilities' verdict on one final text, as evaluate_final / before_final computed it: what record_final needs
    to record it later without asking the capabilities again."""
    __slots__ = ("answer", "step", "sem", "risk", "is_commit", "final_is_commit", "verification", "final_audit",
                 "decisions", "calls", "raw_winner", "winner")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


def is_safety_verdict(decision):
    """A verdict no resource limit may change (module docstring): BLOCK, ESCALATE, RECONCILE, or must_resolve."""
    if decision is None:
        return False
    if decision.type in (D.BLOCK, D.ESCALATE, D.RECONCILE):
        return True
    return bool((getattr(decision, "extra", None) or {}).get("must_resolve"))


class HarnessKernel:
    mode = "enforce"   # the harness's one mode: recorded on events, in the audit and in result["harness"]

    def __init__(self, contract, capabilities, policy=None, env_type=None,
                 risk_of=None, budget=None, judge_fn=None, judge_model=None, instrumentation=None,
                 semantic_allocation=None):
        from .state import Ledger
        self.instrumentation = instrumentation   # records workspace kernel middleware calls (runner/instrumentation.py)
        self.workspace = None                    # the workspace the capabilities and policy came from (build_kernel)
        self.contract = contract
        self.capabilities = list(capabilities or [])
        self.policy = policy or {}
        self.env_type = env_type
        self.risk_of = risk_of
        self.manifest = (policy or {}).get("manifest") or {}   # substrate adapter manifest
        self.budget = dict(_DEFAULT_BUDGET); self.budget.update(budget or {})
        self.ledger = Ledger()
        if contract is not None and contract.subject:
            self.ledger.set_subject(contract.subject)
        self.ctx = HarnessContext(self.ledger, contract, self.policy, env_type, risk_of,
                                  judge_fn=judge_fn, judge_model=judge_model,
                                  semantic_budget=self.budget["max_semantic_checks"],
                                  semantic_allocation=semantic_allocation,
                                  manifest=self.manifest)
        # The deliverable the agent has written, kept because nothing else retains it: ledger.commit_history holds no
        # write_file entries (it records commits like a FHIR create), proposed_actions carries no args, and the final answer
        # is typically a short POINTER ("the note at /workspace/output/... now includes ..."), not the artifact. At runtime the
        # content exists only in the write actions' own arguments (see _note_deliverable).
        self.ctx.last_write = None
        self._deliverable_decl = False    # not yet looked up; None after lookup = the substrate declares none
        self._tool_params = False         # not yet looked up; None = the substrate publishes no parameters
        if self._deliverables() is None:  # from the first hook on, not from the first executed action
            from .unknown import UNKNOWN
            self.ctx.last_write = UNKNOWN
        self._n_interventions = 0
        self._n_semantic = 0
        self._rev_for_action = {}
        self._capability_errors = []     # capability-hook exceptions (fail-closed: see _cap_error)
        self._last_obs = None            # most recent canonical_observation -> prospective GUI scope
        self._last_displayed_subject = None   # last known displayed subject (sticky: empty obs won't clear it)
        self._issued = None              # (action, SemanticAction) classified at before_action on the page it was issued on
        self._open_repairs = {}          # commit identity -> REVISE event id (missing-prereq, not yet repaired)
        self._gate_passed = {}           # commit identity -> event id (prereqs met + gate passed, awaiting verify)
        self._evk = 0
        self._qk = 0                     # ids of final-query events (evaluate_final), apart from the decision ids
        self._querying = False           # evaluate_final is running (instrumentation marks its calls)
        self._cap_ctx = {}
        self._workspace_state = {}       # the workspace kernel middleware's own per-task state (WorkspaceContextView)
        for cap in self.capabilities:
            cap.on_contract(self._ctx_for(cap))
            if instrumentation is not None:
                instrumentation.call(cap, "on_contract", None, None)

    # ---- event helpers -------------------------------------------------------
    def bind_trajectory(self, trajectory):
        """Where the kernel's shell events go (run.py binds the run's trajectory); events recorded before are appended
        first, in order."""
        self.ctx.bind_trace(trajectory.append)

    def _hook_event(self, hook, eff, raw_winner, origin=None, requester=None):
        """The one shell_hook event of a hook call the shell applies (module docstring, "Trace")."""
        winner = eff.raw
        dec = next((e.get("id") for e in eff.events if e.get("event_type") == "harness_decision"), None)
        self.ctx.shell_event({
            "event_type": "shell_hook", "layer": "kernel", "hook": hook, "step": self.ctx.step,
            "origin": origin or "agent", "requester": requester if origin is not None else None,
            "result": {"effective": eff.type, "raw": getattr(raw_winner, "type", D.ALLOW),
                       "capability": getattr(winner, "capability", None) if eff.type != D.ALLOW or eff.note else None,
                       "rule_id": getattr(winner, "rule_id", None) if eff.type != D.ALLOW or eff.note else None,
                       "note": eff.note, "decision_event": dec}})

    def _ev(self, event_type, stage, decision, mode_applied=None, extra=None, event_id=None):
        if event_id is None:
            self._evk += 1
            event_id = "hdec-%d" % self._evk
        e = {"event_type": event_type, "id": event_id, "step": self.ctx.step, "stage": stage, "mode": self.mode,
             "capability": decision.capability if decision else None,
             "decision": decision.type if decision else None,
             "effective": mode_applied, "rule_id": getattr(decision, "rule_id", None),
             "missing_obligations": getattr(decision, "missing_obligations", []),
             "deterministic": getattr(decision, "deterministic", None)}
        de = getattr(decision, "extra", None) if decision else None
        if de:                      # surface a capability's decision.extra (e.g. subject_unspecified candidate) into the event
            e.update((k, v) for k, v in de.items() if k != D.AGENT_TEXT)   # the agent's text is not the verdict
        if extra:
            e.update(extra)
        return e

    # ---- effective decision --------------------------------------------------
    def _decide_effective(self, decision, stage, origin=None):
        """Map a raw capability decision -> (effective_type, feedback, event). The raw verdict is ALWAYS recorded
        to the ledger for metrics; the EFFECTIVE verdict is the raw one bounded by the budgets, and a non-ALLOW
        effective verdict consumes the per-task intervention budget.
        `origin` is None for the agent's own action. A harness-origin action ("harness": a run.py forced read,
        action_service.py, the forced deliverable write) gets the same mapping, is recorded with action_origin, and draws neither the
        intervention budget nor the per-action revision counter: both bound the agent's repair loop."""
        raw = decision.type
        if raw == D.ALLOW:
            return Effective(D.ALLOW, raw=decision)
        eff, note = self._bounded_type(decision, origin, count=True)
        ev = self._ev("harness_decision", stage, decision, mode_applied=eff,
                      extra=({"note": note} if note else None))
        self._stamp_origin(ev, origin)
        # record the intervention (raw) with its effective outcome
        d = decision.to_dict(); d["effective"] = eff; d["event_id"] = ev["id"]
        # surface action_key to the TOP LEVEL so governance can dedup before/after violations of one action.
        d["action_key"] = (decision.extra or {}).get("action_key") or getattr(self.ctx, "action_key", None)
        self._stamp_origin(d, origin)
        self.ledger.record_intervention(d)
        if eff != D.ALLOW and origin is None:
            self._n_interventions += 1
        return Effective(eff, feedback=decision.feedback, raw=decision, events=[ev], note=note)

    def _bounded_type(self, decision, origin, count):
        """(effective type, note) of a non-ALLOW verdict under the budgets (module docstring). `count`: move the
        per-action revision counter (the hook the shell applies); False computes the same answer without moving it
        (evaluate_final).

        A harness-origin action ("harness": a run.py forced read, action_service.py, the forced deliverable write)
        draws neither budget: both bound the agent's repair loop. MAX_REVISIONS_PER_ACTION: a REVISE that keeps
        re-firing on the SAME proposal is a stuck loop. Revision identity = (semantic_type, resource,
        target_entity, payload_hash, validated_evidence_version, reason_code) [+ capability]. Because payload_hash
        and evidence_version are IN the key, the counter resets the moment the agent revises its answer/args or new
        evidence is added; only a truly identical repeated rejection accumulates toward the cap."""
        raw = decision.type
        if origin is not None:
            return raw, None
        note = None
        if self._n_interventions >= self.budget["max_interventions_per_task"]:
            note = "budget_exhausted_interventions"
        elif raw == D.REVISE:
            sem = self.ctx.sem
            ev_ver = self.ledger.validated_evidence_version   # only genuine (VALIDATED, new) progress resets it
            if sem is not None:
                rkey = (sem.semantic_type, sem.resource, sem.target_entity,
                        _payload_fingerprint(sem), ev_ver, decision.reason_code, decision.capability)
            else:
                rkey = (None, None, None, None, ev_ver, decision.reason_code, decision.capability)
            n = self._rev_for_action.get(rkey, 0) + 1
            if count:
                self._rev_for_action[rkey] = n
            if n > self.budget["max_revisions_per_action"]:
                note = "max_revisions_exceeded"
        if note is None:
            return raw, None
        if is_safety_verdict(decision):
            return raw, None          # a budget never changes a safety verdict (not even its note: it was applied)
        return D.ALLOW, note          # resource exhaustion: the harness stops issuing it; the episode goes on

    def _ctx_for(self, cap):
        """The shell's own capabilities get the context; kernel middleware from a workspace gets the read-only view
        without the raw judge (capability.WorkspaceContextView), sharing the workspace's per-task state."""
        view = self._cap_ctx.get(id(cap))
        if view is None:
            if self._from_workspace(cap):
                from .capability import WorkspaceContextView
                view = WorkspaceContextView(self.ctx, self._workspace_state,
                                            consumer="kernel_middleware:%s" % type(cap).__name__)
            else:
                view = self.ctx
            self._cap_ctx[id(cap)] = view
        return view

    @staticmethod
    def _from_workspace(cap):
        try:
            from workspace import component_path
            return component_path(cap) is not None
        except ImportError:
            return False

    def _args_for(self, cap, args, copied):
        """The hook arguments `cap` receives: the shell's own capabilities get the kernel's objects; workspace kernel
        middleware gets one copy per hook call (copied: a one-slot list holding it once made), so no workspace code
        changes the action, result or states the shell goes on to use."""
        if self._ctx_for(cap) is self.ctx:
            return args
        if not copied:
            from middleware_host import readonly_copy
            copied.append(tuple(readonly_copy(a) for a in args))
        return copied[0]

    def _note_call(self, calls, cap, hook, d, raised):
        """Instrumentation of one capability call (recorded only for workspace kernel middleware)."""
        if self.instrumentation is None:
            return
        extra = {}
        if d is not None and d.type != D.ALLOW:
            extra["reason_code"] = d.reason_code
        if raised is not None:
            extra["raised"] = repr(raised)[:300]
        if d is not None:
            from instrumentation import summarize
            extra["summary"] = summarize(d)
        if self._querying:
            extra["query"] = True        # a call made by evaluate_final (a query), not by a hook the shell applied
        rec = self.instrumentation.call(cap, hook, self.ctx.step, d.type if d is not None else None, **extra)
        if rec is not None:
            calls.append((d, rec))

    @staticmethod
    def _note_outcome(calls, raw_winner, eff):
        for d, rec in calls:
            if d is not None and d is raw_winner:
                rec["won"] = True
                rec["effective"] = eff.type

    def _cap_error(self, cap, stage, ex):
        """A capability hook RAISED. Fail closed: record the error AND emit an ESCALATE (reason_code
        'capability_error'), which the kernel applies like any other ESCALATE — a buggy capability is never
        silently treated as ALLOW."""
        err = "%s.%s: capability_error:%r" % (getattr(cap, "name", "?"), stage, ex)
        self._capability_errors.append(err)
        return D.HarnessDecision(D.ESCALATE, capability=getattr(cap, "name", None),
                                 rule_id="capability_error", reason_code="capability_error",
                                 reason="capability_error:%r" % ex, deterministic=True)

    def _repair_key(self):
        sem = self.ctx.sem
        # key on semantic_type + resource + SUBJECT, so a different patient's commit cannot close another's
        # repair (a coarse (type,resource) key would mis-attribute repairs across subjects).
        return (sem.semantic_type, sem.resource, self.ledger.subject_id())

    def _track_repair(self, winner, eff):
        """Repair lifecycle (PRE-commit half): a missing-prerequisite REVISE on a commit OPENS a repair
        opportunity; when the SAME commit (semantic_type+resource+subject) later passes the gate, that is a
        `precondition_repaired` resolution. The commit is NOT yet `repaired` — that needs execution+verify
        (see _close_verified_repair). Keys on the RAW verdict so observe measures would-be repairs too."""
        sem = self.ctx.sem
        if not (sem and sem.is_commit()):
            return
        key = self._repair_key()
        if winner.type != D.ALLOW and getattr(winner, "reason_code", None) == "missing_prerequisite":
            if key not in self._open_repairs and key not in self._gate_passed:
                self.ledger.bump_opportunity("repair")           # an action that COULD be repaired
            self._open_repairs[key] = (eff.events[0]["id"] if eff.events else None)
        elif winner.type == D.ALLOW and key in self._open_repairs:
            ev = self._open_repairs.pop(key)
            self._gate_passed[key] = ev
            self.ledger.resolutions.append(self.resolution_event(ev, "precondition_repaired"))

    def _close_verified_repair(self):
        """Repair lifecycle (POST-commit half): the gate-passed commit ACTUALLY executed and its
        postcondition VERIFIED -> `repaired`. A failed/unverifiable execution does NOT close it."""
        sem = self.ctx.sem
        if not (sem and sem.is_commit()):
            return
        key = self._repair_key()
        if key in self._gate_passed and self.ctx.verification is True:
            self.ledger.resolutions.append(self.resolution_event(self._gate_passed.pop(key), "repaired"))

    def _stamp_origin(self, rec, origin):
        """A record of a harness-origin action names its origin and requester (runner/harness/origin.py)."""
        if origin is not None:
            rec["action_origin"] = origin
            if getattr(self.ledger, "action_requester", None) is not None:
                rec["action_requester"] = self.ledger.action_requester

    def _record_findings(self, decisions, stage, origin=None):
        """Record EVERY non-ALLOW finding (not just the hook winner) so a lower-priority finding survives
        for metrics — e.g. a commit that is BOTH wrong-subject (BLOCK) and missing-prerequisite (REVISE)
        contributes to BOTH rates, not only the higher-priority one."""
        for d in decisions:
            if d.type != D.ALLOW:
                f = {"action_key": getattr(self.ctx, "action_key", None) or ("act%d" % self.ctx.step),
                     "reason_code": d.reason_code, "capability": d.capability,
                     "decision": d.type, "rule_id": d.rule_id, "stage": stage}
                self._stamp_origin(f, origin)
                self.ledger.record_finding(f)

    # ---- public hooks --------------------------------------------------------
    def _canon(self, action, observation=None, controls=None, issued=None):
        """controls: the live control model of the page the action is issued on (before_action), the evidence a manifest
        classifier rule needs (semantics.canonicalize_on_page). issued: the SemanticAction before_action classified for
        this same action; after_action keeps its classification (the page after a commit no longer shows the control),
        with its own subject projection from the observation."""
        from .semantics import canonicalize, canonicalize_on_page
        from .risk import classify_risk
        if controls:
            sem = canonicalize_on_page(action, self.manifest, controls)
        else:
            sem = canonicalize(action, self.manifest, observation=observation)
        if issued is not None:
            for k in ("semantic_type", "effect", "source_class", "modality", "resource", "mapped", "subject_binding"):
                setattr(sem, k, getattr(issued, k))
        self.ctx.sem = sem
        self.ctx.risk = classify_risk(sem, self.contract)
        self.ctx.verification = None
        self.ctx.final_audit = None   # an audit of a final lives for the before_final hook call that ran it
        return sem

    def before_action(self, action, env_state=None, step=0, origin=None, requester=None):
        """origin: None for the agent's action, "harness" for a harness-origin action, with its requester
        (runner/harness/origin.py; see _decide_effective)."""
        self.ledger.action_origin = origin
        self.ledger.action_requester = requester if origin is not None else None
        self.ctx.step = step
        self.ctx.hook_serial += 1
        self.ctx.last_observation = self._last_obs      # the page the agent is currently looking at
        self.ctx.displayed_subject = self._last_displayed_subject   # for the prospective commit guard
        self.ctx.observed_subject = None
        self.ctx.current_state = env_state   # structured pre-commit state for goal/field checks
        sem = self._canon(action, controls=(env_state.get("controls") if isinstance(env_state, dict) else None))
        self._issued = (action, sem)
        risk = self.ctx.risk
        pid = self.ledger.record_proposed(sem.capability, risk, step, origin=origin)
        self.ctx.action_key = pid   # ONE canonical key for this action across before/after hooks (dedup)
        from .risk import at_least, R2
        if at_least(risk, R2):
            self.ledger.bump_opportunity("commit_proposal", step)   # denom for missing_prerequisite_rate (per action)
        decisions, calls, copied = [], [], []
        for cap in self.capabilities:
            raised = None
            try:
                d = cap.before_action(*self._args_for(cap, (action,), copied), self._ctx_for(cap))
            except Exception as ex:                       # a buggy capability must never crash the run
                d, raised = self._cap_error(cap, "before_action", ex), ex
            self._note_call(calls, cap, "before_action", d, raised)
            if d is not None:
                d.stage = "before_action"
                d.extra.setdefault("action_key", pid)
                decisions.append(d)
        self._record_findings(decisions, "before_action", origin)
        raw_winner = D.resolve(decisions, stage="before_action").winner
        winner = D.combine(decisions, stage="before_action")
        self.ctx.last_raw_decision = raw_winner   # the genuinely RAW winner (before budgets)
        eff = self._decide_effective(winner, "before_action", origin)
        eff.feedback = _feedback(winner) if eff.type != D.ALLOW else None
        if origin is None:
            self._track_repair(raw_winner, eff)   # repair lifecycle keys on the RAW finding (the agent's repairs)
        self._note_outcome(calls, raw_winner, eff)
        self._hook_event("before_action", eff, raw_winner, origin, requester)
        return eff

    def after_action(self, action, result, before_state, after_state, step=0, canonical_observation=None,
                     result_ok=None, raw_observation=None, result_status=None, origin=None, requester=None):
        self.ledger.action_origin = origin
        self.ledger.action_requester = requester if origin is not None else None
        self.ctx.step = step
        self.ctx.hook_serial += 1
        self.ctx.observation = canonical_observation
        self._last_obs = canonical_observation        # carried into the NEXT before_action (prospective scope)
        # the displayed subject is projected from the RAW observation via the manifest (the canonical
        # observation may not carry portal-specific fields). Sticky: an empty/error observation that yields
        # no subject does NOT erase the last known one.
        from .semantics import observed_subject as _obs_subj
        _cur = _obs_subj(self.manifest, raw_observation if raw_observation is not None else canonical_observation)
        self.ctx.observed_subject = _cur
        if _cur is not None:
            self._last_displayed_subject = _cur
        _issued = self._issued[1] if (self._issued is not None and self._issued[0] is action) else None
        sem = self._canon(action, observation=canonical_observation, issued=_issued)
        self.ctx.result_ok = result_ok      # whether the tool result succeeded (adapter signal)
        if origin is None:                    # a harness write is not the agent's deliverable
            self._note_deliverable(action, result_ok)
        try:   # record a normalized observation for perception/read tools (evidence_coverage input)
            from .affordance import is_perception_tool
            _tn = action.get("tool") if isinstance(action, dict) else None
            if _tn and is_perception_tool(_tn) and getattr(sem, "semantic_type", None) not in ("create", "update", "submit"):
                _ar = (action.get("args") or {}) if isinstance(action, dict) else {}
                # SAVE only the real tool OUTPUT, never the whole envelope {tool, args, mode,
                # output}. Otherwise the REQUESTED attribute in args pollutes the observed content (a request
                # for "SMV identity" makes an unrelated output look like it discussed the SMV), and the nested
                # output.localization.resolved is missed by a top-level lookup. Unwrap output first.
                _payload = result.get("output", result) if isinstance(result, dict) else result
                if isinstance(_payload, dict):
                    _content = str(_payload.get("text") or _payload.get("content") or _payload.get("description") or "")
                    _lz = _payload.get("localization")
                    _loc = _lz.get("resolved") if isinstance(_lz, dict) and "resolved" in _lz else None
                else:
                    _content = str(_payload) if _payload is not None else ""
                    _loc = None
                self.ledger.record_observation(tool_capability=_tn,
                    subject=_ar.get("image") or _ar.get("subject") or _ar.get("image_id"),
                    region=_ar.get("region"), modality=_ar.get("modality"),
                    attributes_observed=[],  # the agent's REQUESTED attribute is not OBSERVED; the content judge decides
                    result_status=("invalid" if result_ok is False else "valid"), content=_content,
                    localization_resolved=_loc)
        except Exception:
            pass
        self.ctx.result_status = result_status   # ok|failed|unknown (the kernel passes run.py's tri-state)
        self.ctx.evidence_version_before = self.ledger.validated_evidence_version  # progress baseline (pre-bind)
        decisions, calls, copied = [], [], []
        for cap in self.capabilities:
            raised = None
            try:
                d = cap.after_action(*self._args_for(cap, (action, result, before_state, after_state), copied),
                                     self._ctx_for(cap))
            except Exception as ex:
                d, raised = self._cap_error(cap, "after_action", ex), ex
            self._note_call(calls, cap, "after_action", d, raised)
            if d is not None:
                d.stage = "after_action"
                decisions.append(d)
        self._record_findings(decisions, "after_action", origin)
        raw_winner = D.resolve(decisions, stage="after_action").winner
        winner = D.combine(decisions, stage="after_action")
        self.ctx.last_raw_decision = raw_winner
        self._maybe_hold(raw_winner)   # mutation hold follows the RAW finding (safety-side)
        from .risk import at_least, R2
        if at_least(self.ctx.risk, R2):
            # verified is the EXPLICIT tri-state from verify_commit (True/False/None), NOT inferred from
            # the combined decision — an unverifiable commit must record verified=None, never True.
            self.ledger.record_commit(sem.capability, step, verified=self.ctx.verification,
                                      detail=raw_winner.reason, semantic_type=getattr(sem, "semantic_type", None),
                                      descriptor=action, effect=getattr(sem, "effect", None), origin=origin)
            if self.ctx.verification is True and getattr(sem, "effect", None) == "irreversible":
                from .capabilities.verify_commit import commit_identity
                self.ledger.completed_commits.add(commit_identity(sem, self.ledger))
                if getattr(sem, "semantic_type", None) == "submit":   # graded terminal commit verified -> lock
                    # ...but only if it really is TERMINAL. A task whose declared public obligations are not
                    # yet observably realized is a conjunction still in progress (HAB: submit AND document
                    # AND clear the worklist); locking there blocks the remaining subgoals and forfeits them.
                    # None = this bench has no obligation model -> the verified submit locks.
                    _pom = getattr(self.ledger, "public_obligations_met", None)
                    if _pom is not False:
                        self.ledger.terminal_locked = True
            if origin is None:
                self._close_verified_repair()    # a gate-passed commit that executed + verified -> repaired
        # OBSERVED terminal commit: on a substrate where the manifest cannot tell a commit from a page
        # look by the action alone (HAB's button labels are actively misleading), the declared outward-
        # submission obligation flipping to satisfied IS the commit, and it is visible right here in
        # after_action. Independent of the risk path (outside the R2 branch above) so an action the
        # manifest did not type as a commit still reaches it -- and it still only LOCKS once the whole
        # declared conjunction holds, so a task that still owes documentation keeps its remaining subgoals.
        if getattr(self.ledger, "observed_submission_realized", None) is True:
            from .capabilities.verify_commit import commit_identity
            try:
                self.ledger.completed_commits.add(commit_identity(sem, self.ledger))
            except Exception:
                pass
            if getattr(self.ledger, "public_obligations_met", None) is True:
                self.ledger.terminal_locked = True
        eff = self._decide_effective(winner, "after_action", origin)
        eff.feedback = _feedback(winner) if eff.type != D.ALLOW else None
        self._note_outcome(calls, raw_winner, eff)
        self._hook_event("after_action", eff, raw_winner, origin, requester)
        return eff

    def _maybe_hold(self, winner):
        """OPERATIONAL NON-DEGRADATION: a NON-deterministic semantic finding (REVISE/ACQUIRE, any channel) puts
        state writes under a mutation_hold -- subsequent mutations then need an explicit scoped authorization.
        A DETERMINISTIC finding does NOT hold (it mints an authorization instead)."""
        from .authorization import should_set_mutation_hold
        if winner is not None and should_set_mutation_hold(getattr(winner, "type", None),
                                                           getattr(winner, "deterministic", None)):
            self.ledger.set_mutation_hold(finding_id=getattr(winner, "rule_id", None),
                                          capability=getattr(winner, "capability", None))

    def _enter_final(self, answer, step):
        """The start of a before_final hook call, query or not: the hook's working context for `answer`."""
        self.ledger.action_origin = None
        self.ledger.action_requester = None
        self.ctx.step = step
        self.ctx.hook_serial += 1
        sem = self._canon({"type": "final", "answer": answer})
        self.ctx.final_is_commit = bool(sem and sem.is_commit())  # manifest commit status of the final
        from .risk import at_least, R2
        return sem, at_least(self.ctx.risk, R2)

    def _propose_final(self, sem, is_commit, step):
        # the final answer is a commit -> it enters the SAME commit lifecycle (proposal + opportunity).
        self.ctx.action_key = self.ledger.record_proposed(sem.capability, self.ctx.risk, step)
        if is_commit:
            self.ledger.bump_opportunity("commit_proposal", step)

    def _evaluate_capabilities_final(self, answer, step, sem, is_commit):
        decisions, calls = [], []
        for cap in self.capabilities:
            raised = None
            try:
                d = cap.before_final(answer, self._ctx_for(cap))
            except Exception as ex:
                d, raised = self._cap_error(cap, "before_final", ex), ex
            self._note_call(calls, cap, "before_final", d, raised)
            if d is not None:
                d.stage = "before_final"
                decisions.append(d)
        raw_winner = D.resolve(decisions, stage="before_final").winner
        winner = D.combine(decisions, stage="before_final")
        self.ctx.last_raw_decision = raw_winner
        return _FinalEvaluation(answer=answer, step=step, sem=sem, risk=self.ctx.risk, is_commit=is_commit,
                                final_is_commit=self.ctx.final_is_commit, verification=self.ctx.verification,
                                final_audit=self.ctx.final_audit, decisions=decisions, calls=calls,
                                raw_winner=raw_winner, winner=winner)

    def before_final(self, answer, step=0):
        """The before_final hook the shell applies: the capabilities' verdict on `answer`, recorded (module docstring)."""
        sem, is_commit = self._enter_final(answer, step)
        self._propose_final(sem, is_commit, step)
        fe = self._evaluate_capabilities_final(answer, step, sem, is_commit)
        return self._record_final(fe)

    def evaluate_final(self, answer, step=0):
        """The same verdict as a QUERY: no proposal, finding, intervention, counter, repair, hold or commit is
        recorded. The effective type is the one before_final would return now. A non-ALLOW verdict carries one
        shell_final_query event (id hqry-N; a query is not a harness_decision). Pass the result to record_final to
        record it as the real verdict."""
        sem, is_commit = self._enter_final(answer, step)
        self._querying = True
        try:
            fe = self._evaluate_capabilities_final(answer, step, sem, is_commit)
        finally:
            self._querying = False
        winner = fe.winner
        events, eff_type, note = [], D.ALLOW, None
        if winner.type != D.ALLOW:
            eff_type, note = self._bounded_type(winner, None, count=False)
            self._qk += 1
            events.append(self._ev("shell_final_query", "before_final", winner, mode_applied=eff_type,
                                   extra=({"note": note} if note else None), event_id="hqry-%d" % self._qk))
        eff = Effective(eff_type, feedback=(_feedback(winner) if eff_type != D.ALLOW else None), raw=winner,
                        events=events, note=note)
        eff.evaluation = fe
        self._note_outcome(fe.calls, fe.raw_winner, eff)
        return eff

    def record_final(self, verdict, step=None, flagged=False):
        """Record a verdict evaluate_final returned as the real before_final verdict on its text (an on_answer Accept:
        middleware_host.delivered_verdict), without asking the capabilities again. `flagged`: the shell delivers the
        text with a verification flag (record_flagged_final records that commit), so an ALLOW records no second
        final-answer commit. -> the recorded Effective."""
        fe = verdict.evaluation
        self.ledger.action_origin = None
        self.ledger.action_requester = None
        self.ctx.step = fe.step if step is None else step
        self.ctx.hook_serial += 1
        self.ctx.sem, self.ctx.risk, self.ctx.verification = fe.sem, fe.risk, fe.verification
        self.ctx.final_audit, self.ctx.final_is_commit = fe.final_audit, fe.final_is_commit
        self._propose_final(fe.sem, fe.is_commit, self.ctx.step)
        return self._record_final(fe, flagged=flagged)

    def _record_final(self, fe, flagged=False):
        self._record_findings(fe.decisions, "before_final")
        self.ctx.last_raw_decision = fe.raw_winner
        self._maybe_hold(fe.raw_winner)   # hold follows the RAW finding
        eff = self._decide_effective(fe.winner, "before_final")
        eff.feedback = _feedback(fe.winner) if eff.type != D.ALLOW else None
        eff.evaluation = fe
        self._track_repair(fe.raw_winner, eff)
        self._note_outcome(fe.calls, fe.raw_winner, eff)
        # an ACCEPTED final answer (effective ALLOW) is a committed action.
        if fe.is_commit and eff.type == D.ALLOW and not flagged:
            self.ledger.record_commit(fe.sem.capability, self.ctx.step, verified=self.ctx.verification,
                                      detail="final_answer", semantic_type="answer")
            self._close_verified_repair()    # a final-answer commit verifies + closes its repair too
        self._hook_event("before_final", eff, fe.raw_winner)
        return eff

    def record_flagged_final(self, answer, flag="unresolved_risk", step=0):
        """run.py delivered a no-side-effect terminal answer WITH a verification flag (graceful degradation)
        instead of via effective-ALLOW. Record it as a final-answer commit with verified=None
        (delivered but NOT verified) so answer_delivered / outcome_preservation / unknown_verification see it."""
        sem = self._canon({"type": "final", "answer": answer})
        self.ledger.record_commit(sem.capability, step, verified=None, detail="final_answer", semantic_type="answer")
        if self.ledger.commit_history:
            self.ledger.commit_history[-1]["verification_flag"] = flag

    def build_observation(self, tool_result, post_eff):
        """Fold any retrospective harness feedback into what the agent sees next."""
        if post_eff is None or post_eff.type == D.ALLOW or not post_eff.feedback:
            return tool_result
        return {"tool_result": tool_result, "harness_feedback": post_eff.feedback,
                "harness_decision": post_eff.type}

    def resolution_event(self, original_decision_id, resolution, satisfying_event_ids=None):
        self._evk += 1
        return {"event_type": "harness_resolution", "id": "hres-%d" % self._evk,
                "original_decision_id": original_decision_id, "resolution": resolution,
                "satisfying_event_ids": list(satisfying_event_ids or [])}

    def _bind_acquire_subject(self, d):
        """Name the task's assigned subject in an ACQUIRE's forced read. -> problem, or None.

        Shell READ subject binding, for a READ decision that does not fill the patient itself (the base ACQUIRE
        capability does). Dispatched with `args: {}`, such a read is REVISEd by the scope gate as
        subject_unspecified and never runs. The argument and its value form come from manifest.subject.to_args
        against the tool's own parameters; a decision that names a subject itself is refused, because the
        subject is the task's, not the decision's."""
        from .semantics import bind_subject, subject_arg_names
        na = (getattr(d, "extra", None) or {}).get("next_action")
        if not isinstance(na, dict):
            return None
        args = dict(na.get("args") or {})
        named = sorted(set(args) & subject_arg_names(self.manifest))
        if named:
            return "spec_names_subject:%s" % ",".join(named)
        if self._tool_params is False:
            try:
                from .substrate_catalog import tool_parameters
                self._tool_params = tool_parameters(self.env_type)
            except Exception:
                self._tool_params = None
        params = None if self._tool_params is None else self._tool_params.get(na.get("tool"))
        ref = self.ledger.subject_ref() or {}
        bare = str(ref.get("id") or "").rsplit("/", 1)[-1] or None
        patch, problem = bind_subject(self.manifest, params, bare)
        if problem:
            return problem
        args.update(patch)
        na["args"] = args
        return None

    def _deliverables(self):
        """The adapter's DELIVERABLE_WRITES for this kernel's substrate, looked up once. None when it declares
        none -- and also when the kernel was built without a resolvable env_type, since nothing then says
        which write is the deliverable."""
        if self._deliverable_decl is False:
            try:
                from .substrate_catalog import deliverable_writes
                self._deliverable_decl = deliverable_writes(self.env_type)
            except Exception:
                self._deliverable_decl = None
        return self._deliverable_decl

    def _note_deliverable(self, action, result_ok):
        """Remember the deliverable an EXECUTED write produced.

        WHICH writes are the deliverable, and which argument holds its path and its text, is declared by the
        adapter that performs the write (environments.EnvironmentAdapter.DELIVERABLE_WRITES), not inferred from
        the action's shape: a heuristic such as "any create/update/submit carrying a long string" would take, e.g.,
        a ServiceRequest note_text for the deliverable, and a commit-without-deliverable check would then read the
        wrong artifact.

        AFTER the action ran, not before: a write the harness blocked, or one that failed, produced no
        deliverable. A substrate that declares no deliverable leaves the artifact UNKNOWN, so a reader of it
        abstains there instead of reading absence as a fact."""
        from .unknown import UNKNOWN
        decl = self._deliverables()
        if decl is None:
            self.ctx.last_write = UNKNOWN
            return
        tool = action.get("tool") if isinstance(action, dict) else None
        m = decl.get(tool)
        if m is None or result_ok is False:
            return
        args = action.get("args") if isinstance(action.get("args"), dict) else {}
        path, content = args.get(m["path"]), args.get(m["content"])
        content = content if isinstance(content, str) else ""
        prev = self.ctx.last_write if isinstance(self.ctx.last_write, dict) else None
        # an append (the adapter declares the mode argument and the value that overwrites; any other value appends, as
        # the upstream writer does) extends the same file's content
        if m.get("mode") and args.get(m["mode"], m.get("overwrite")) != m.get("overwrite") and prev \
                and prev.get("path") == path:
            content = (prev.get("content") or "") + content
        seen = ({prev["path"]} | set(prev.get("paths") or ())) if prev else set()
        paths = sorted({str(p) for p in seen | {path} if p is not None})
        self.ctx.last_write = {"path": path, "content": content,
                               "tool": tool, "content_from_arg": m["content"], "path_from_arg": m["path"],
                               "paths": paths, "n_paths": len(paths)}

    def audit(self):
        _meter = self.ctx.semantic_meter
        _audit = {"mode": self.mode, "contract": self.contract.to_dict() if self.contract else None,
                "ledger": self.ledger.to_dict(), "budget": self.budget,
                "n_interventions": self._n_interventions,
                "n_semantic_checks": _meter.spent_total(),
                "status": ("degraded" if (self._capability_errors or self.policy.get("_errors")) else "active"),
                "capability_errors": self._capability_errors,
                "policy_errors": self.policy.get("_errors", []),
                # the harness's semantic judge model + how many times it was called — so the report can
                # verify INDEPENDENCE (judge != agent brain != tool backend) instead of asserting it.
                "runtime_judge_model": self.ctx.judge_model,
                "runtime_judge_calls": _meter.spent_total(),   # units spent, despite the key name (judge calls per consumer are in semantic_budget)
                # Spent / cap per pool: a combined total cannot show that the reserve pool was actually kept for the final
                # verification (if the general pool consumed the whole budget, the total would still be <= the cap).
                # The pools named general / reserve (the allocation of the workspaces/*/h0 manifests); None for an allocation
                # without that pool.
                "n_general_spent": _meter.pool_spent("general"),
                "n_general_budget": _meter.pool_cap("general"),
                "n_reserve_spent": _meter.pool_spent("reserve"),
                "n_reserve_budget": _meter.pool_cap("reserve"),
                # Number of semantic checks refused because the budget was exhausted -- these are abstain / not-checked,
                # never "verified". Counted once per refusal at the meter, whichever consumer was refused.
                "n_semantic_refused": _meter.refused_total(),
                # the total, the workspace's allocation, and units / judge calls / refusals per consumer.
                "semantic_budget": _meter.record()}
        # RUN CONFIG -> every result.json: the conditions (judge models, agent model, mode, budgets) that
        # decide what a run IS, so one result.json can say how it was produced.
        try:
            from . import run_config as _rc
            _audit["run_config"] = _rc.snapshot()
        except Exception as _e:
            _audit["run_config"] = {"error": repr(_e)[:160]}
        # ACTUATION TELEMETRY -> the persisted artifact: the READ actuator's rungs (requested / attempted /
        # executed / succeeded / evidence_added) and the REVISE / DEFER_FINAL delivery records. Emitted
        # unconditionally, so `0` is a measurement and not a missing field.
        try:
            from . import actuation as _act_tel
            _audit["actuation"] = _act_tel.episode_summary(self.ledger)
        except Exception as _e:                       # never fail a run over instrumentation
            _audit["actuation"] = {"error": repr(_e)[:160]}
        return _audit


def _payload_fingerprint(sem):
    """Stable, leak-safe hash of the agent's OWN proposed payload — the answer TEXT on a final, the
    normalized ARGS on a tool action. Lets the revision-identity key tell an IDENTICAL repeated rejection
    (accumulates toward the ESCALATE cap) from a revised one (resets the counter). Never reads gold."""
    import hashlib, json
    raw = sem.raw if (sem is not None and isinstance(sem.raw, dict)) else {}
    payload = raw.get("answer") if raw.get("type") == "final" else raw.get("args")
    try:
        s = json.dumps(payload, sort_keys=True, default=str)
    except Exception:
        s = repr(payload)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _feedback(decision):
    """The leak-safe structured message handed to the agent (never gold/reference)."""
    fb = {"decision": decision.type, "rule_id": decision.rule_id, "reason": decision.reason}
    if decision.missing_obligations:
        fb["missing_obligations"] = decision.missing_obligations
    if decision.suggested_capabilities:
        fb["suggested_capabilities"] = decision.suggested_capabilities
    if getattr(decision, "avoid_capabilities", None):
        fb["avoid_capabilities"] = decision.avoid_capabilities
    if decision.feedback:
        fb["message"] = decision.feedback
    _rf = (decision.extra or {}).get("repair_findings")
    if _rf:
        fb["repair_findings"] = _rf   # Scoped Repair: localized patch spec for the renderer
    return fb
