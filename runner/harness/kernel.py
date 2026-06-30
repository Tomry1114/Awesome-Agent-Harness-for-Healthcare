"""HarnessKernel — orchestrates the capabilities, applies the run MODE, enforces budgets, and emits
harness events. This is the object run.py drives via before_action / after_action / before_final.

MODE semantics (the experiment's three settings):
  off      kernel inert — always ALLOW, no events (run.py normally skips creating it).
  observe  compute + RECORD decisions (events + ledger interventions) but the EFFECTIVE decision is
           always ALLOW — measures what the harness WOULD do without changing the run (baseline).
  assist   surface problems as structured feedback to the agent (effective <= REVISE), never hard-block
           or terminate — tests whether feedback alone is enough.
  enforce  full ALLOW / REVISE / BLOCK / ESCALATE — the complete runtime control.

Budgets (must exist, to bound revision loops): max_revisions_per_action, max_interventions_per_task,
max_semantic_checks. When a budget is exhausted the kernel stops intervening (effective ALLOW) and
records a budget_exhausted note.
"""
from . import decision as D
from .capability import HarnessContext

MODES = ("off", "observe", "assist", "enforce")
_DEFAULT_BUDGET = {"max_revisions_per_action": 2, "max_interventions_per_task": 8, "max_semantic_checks": 5}


class Effective:
    """What run.py acts on: the post-mode decision + feedback + the raw capability verdict (for audit)."""
    __slots__ = ("type", "feedback", "raw", "events")

    def __init__(self, type, feedback=None, raw=None, events=None):
        self.type = type
        self.feedback = feedback
        self.raw = raw
        self.events = events or []


class HarnessKernel:
    def __init__(self, contract, capabilities, mode="observe", policy=None, env_type=None,
                 risk_of=None, budget=None, judge_fn=None, judge_model=None):
        if mode not in MODES:
            raise ValueError("unknown harness mode %r" % (mode,))
        from .state import Ledger
        self.mode = mode
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
        self.ctx = HarnessContext(self.ledger, contract, self.policy, mode, env_type, risk_of,
                                  judge_fn=judge_fn, judge_model=judge_model,
                                  semantic_budget=self.budget["max_semantic_checks"],
                                  manifest=self.manifest)
        self._n_interventions = 0
        self._n_semantic = 0
        self._rev_for_action = {}
        self._capability_errors = []     # capability-hook exceptions (fail-closed: see _cap_error)
        self._last_obs = None            # most recent canonical_observation -> prospective GUI scope
        self._last_displayed_subject = None   # last known displayed subject (sticky: empty obs won't clear it)
        self._open_repairs = {}          # commit identity -> REVISE event id (missing-prereq, not yet repaired)
        self._gate_passed = {}           # commit identity -> event id (prereqs met + gate passed, awaiting verify)
        self._evk = 0
        for cap in self.capabilities:
            cap.on_contract(self.ctx)

    # ---- event helpers -------------------------------------------------------
    def _ev(self, event_type, stage, decision, mode_applied=None, extra=None):
        self._evk += 1
        e = {"event_type": event_type, "id": "hdec-%d" % self._evk, "stage": stage, "mode": self.mode,
             "capability": decision.capability if decision else None,
             "decision": decision.type if decision else None,
             "effective": mode_applied, "rule_id": getattr(decision, "rule_id", None),
             "missing_obligations": getattr(decision, "missing_obligations", []),
             "deterministic": getattr(decision, "deterministic", None)}
        if extra:
            e.update(extra)
        return e

    # ---- mode application ----------------------------------------------------
    def _apply_mode(self, decision, stage):
        """Map a raw capability decision -> (effective_type, feedback, event). The raw (would-be) verdict
        is ALWAYS recorded to the ledger for metrics (so 'observe' measures what the harness WOULD do);
        the EFFECTIVE verdict is mode-downgraded and only it consumes the per-task intervention budget."""
        raw = decision.type
        if raw == D.ALLOW:
            return Effective(D.ALLOW, raw=decision)
        # effective decision per mode
        if self.mode == "observe":
            eff = D.ALLOW                                    # never changes the run; record-only
        elif self.mode == "assist":
            eff = D.REVISE if raw in (D.REVISE, D.BLOCK, D.ESCALATE) else raw   # feedback, never hard-block
        else:  # enforce
            eff = raw
        note = None
        # budget is a RESOURCE constraint, not a safety override. On exhaustion: enforce ESCALATEs
        # (terminate safely, never silently ALLOW a would-be BLOCK); assist STOPS giving feedback (ALLOW)
        # but must NOT terminate (assist is feedback-only by contract); observe never reaches here.
        # Only an EFFECTIVE (run-affecting) intervention consumes budget.
        if eff != D.ALLOW and self._n_interventions >= self.budget["max_interventions_per_task"]:
            note = "budget_exhausted_interventions"
            eff = D.ESCALATE if self.mode == "enforce" else D.ALLOW
        # MAX_REVISIONS_PER_ACTION: a REVISE that keeps re-firing on the SAME proposal is a stuck loop.
        # CONTRACT(3) identity = (semantic_type, resource, target_entity, payload_hash, validated_evidence_version,
        # reason_code) [+ capability, kept from the prior key so two capabilities don't share a counter].
        # Because payload_hash and evidence_version are IN the key, the counter RESETS automatically the
        # moment the agent revises its answer/args OR new evidence is added (genuine progress) — only a
        # TRULY identical repeated rejection accumulates toward the cap. Applies to EVERY feedback stage
        # (P0-7: a before_action loop is now bounded by the same per-FINGERPRINT key, not a global one).
        elif eff == D.REVISE:
            sem = self.ctx.sem
            ev_ver = self.ledger.validated_evidence_version   # only genuine (VALIDATED, new) progress resets the per-action revision counter
            if sem is not None:
                rkey = (sem.semantic_type, sem.resource, sem.target_entity,
                        _payload_fingerprint(sem), ev_ver, decision.reason_code, decision.capability)
            else:
                rkey = (None, None, None, None, ev_ver, decision.reason_code, decision.capability)
            n = self._rev_for_action.get(rkey, 0) + 1
            self._rev_for_action[rkey] = n
            if n > self.budget["max_revisions_per_action"]:
                note = "max_revisions_exceeded"
                # a stuck revision loop is a RESOURCE limit, mode-aware like the intervention budget:
                # enforce terminates safely (ESCALATE); assist is feedback-only so it must NOT terminate.
                eff = D.ESCALATE if self.mode == "enforce" else D.ALLOW
        ev = self._ev("harness_decision", stage, decision, mode_applied=eff,
                      extra=({"note": note} if note else None))
        # record the WOULD-BE intervention (raw) with its effective outcome, in every mode
        d = decision.to_dict(); d["effective"] = eff; d["event_id"] = ev["id"]
        # surface action_key to the TOP LEVEL so governance can dedup before/after violations of one action.
        d["action_key"] = (decision.extra or {}).get("action_key") or getattr(self.ctx, "action_key", None)
        self.ledger.record_intervention(d)
        if eff != D.ALLOW:
            self._n_interventions += 1
        return Effective(eff, feedback=decision.feedback, raw=decision, events=[ev])

    def _cap_error(self, cap, stage, ex):
        """A capability hook RAISED. Fail closed: record the error AND emit an ESCALATE (reason_code
        'capability_error'). The mode logic then yields ALLOW+record under observe but a real ESCALATE
        under enforce — a buggy capability is never silently treated as ALLOW."""
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

    def _record_findings(self, decisions, stage):
        """Record EVERY non-ALLOW finding (not just the hook winner) so a lower-priority finding survives
        for metrics — e.g. a commit that is BOTH wrong-subject (BLOCK) and missing-prerequisite (REVISE)
        contributes to BOTH rates, not only the higher-priority one."""
        for d in decisions:
            if d.type != D.ALLOW:
                self.ledger.record_finding({"action_key": getattr(self.ctx, "action_key", None) or ("act%d" % self.ctx.step),
                                            "reason_code": d.reason_code, "capability": d.capability,
                                            "decision": d.type, "rule_id": d.rule_id, "stage": stage})

    # ---- public hooks --------------------------------------------------------
    def _canon(self, action, observation=None):
        from .semantics import canonicalize
        from .risk import classify_risk
        sem = canonicalize(action, self.manifest, observation=observation)
        self.ctx.sem = sem
        self.ctx.risk = classify_risk(sem, self.contract)
        self.ctx.verification = None
        return sem

    def before_action(self, action, env_state=None, step=0):
        self.ctx.step = step
        self.ctx.last_observation = self._last_obs      # the page the agent is currently looking at
        self.ctx.displayed_subject = self._last_displayed_subject   # for the prospective commit guard
        self.ctx.observed_subject = None
        self.ctx.current_state = env_state   # structured pre-commit state for goal/field checks
        sem = self._canon(action)
        risk = self.ctx.risk
        pid = self.ledger.record_proposed(sem.capability, risk, step)
        self.ctx.action_key = pid   # ONE canonical key for this action across before/after hooks (dedup)
        from .risk import at_least, R2
        if at_least(risk, R2):
            self.ledger.bump_opportunity("commit_proposal", step)   # denom for missing_prerequisite_rate (per action)
        decisions = []
        for cap in self.capabilities:
            try:
                d = cap.before_action(action, self.ctx)
            except Exception as ex:                       # a buggy capability must never crash the run
                d = self._cap_error(cap, "before_action", ex)
            if d is not None:
                d.stage = "before_action"
                d.extra.setdefault("action_key", pid)
                decisions.append(d)
        self._record_findings(decisions, "before_action")
        winner = D.combine(decisions, stage="before_action")
        eff = self._apply_mode(winner, "before_action")
        eff.feedback = _feedback(winner) if eff.type != D.ALLOW else None
        self._track_repair(winner, eff)
        return eff

    def after_action(self, action, result, before_state, after_state, step=0, canonical_observation=None,
                     result_ok=None, raw_observation=None, result_status=None):
        self.ctx.step = step
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
        sem = self._canon(action, observation=canonical_observation)
        self.ctx.result_ok = result_ok      # whether the tool result succeeded (adapter signal)
        try:   # record a normalized observation for perception/read tools (evidence_coverage input)
            from .affordance import is_perception_tool
            _tn = action.get("tool") if isinstance(action, dict) else None
            if _tn and is_perception_tool(_tn) and getattr(sem, "semantic_type", None) not in ("create", "update", "submit"):
                _ar = (action.get("args") or {}) if isinstance(action, dict) else {}
                _content = result if isinstance(result, str) else (str(result) if result is not None else "")
                self.ledger.record_observation(tool_capability=_tn,
                    subject=_ar.get("image") or _ar.get("subject") or _ar.get("image_id"),
                    region=_ar.get("region"), modality=_ar.get("modality"),
                    attributes_observed=[],  # P0-7: the agent's REQUESTED attribute is not OBSERVED; the content judge decides
                    result_status=("invalid" if result_ok is False else "valid"), content=_content)
        except Exception:
            pass
        self.ctx.result_status = result_status   # ok|failed|unknown (the kernel passes run.py's tri-state)
        self.ctx.evidence_version_before = self.ledger.validated_evidence_version  # progress baseline (pre-bind)
        decisions = []
        for cap in self.capabilities:
            try:
                d = cap.after_action(action, result, before_state, after_state, self.ctx)
            except Exception as ex:
                d = self._cap_error(cap, "after_action", ex)
            if d is not None:
                d.stage = "after_action"
                decisions.append(d)
        self._record_findings(decisions, "after_action")
        winner = D.combine(decisions, stage="after_action")
        from .risk import at_least, R2
        if at_least(self.ctx.risk, R2):
            # verified is the EXPLICIT tri-state from verify_commit (True/False/None), NOT inferred from
            # the combined decision — an unverifiable commit must record verified=None, never True.
            self.ledger.record_commit(sem.capability, step, verified=self.ctx.verification,
                                      detail=winner.reason, semantic_type=getattr(sem, "semantic_type", None))
            if self.ctx.verification is True and getattr(sem, "effect", None) == "irreversible":
                from .capabilities.verify_commit import commit_identity
                self.ledger.completed_commits.add(commit_identity(sem, self.ledger))
            self._close_verified_repair()    # a gate-passed commit that executed + verified -> repaired
        eff = self._apply_mode(winner, "after_action")
        eff.feedback = _feedback(winner) if eff.type != D.ALLOW else None
        return eff

    def before_final(self, answer, step=0):
        self.ctx.step = step
        sem = self._canon({"type": "final", "answer": answer})
        self.ctx.final_is_commit = bool(sem and sem.is_commit())  # P0-A: manifest commit status of the final
        from .risk import at_least, R2
        # the final answer is a commit -> it enters the SAME commit lifecycle (proposal + opportunity).
        self.ctx.action_key = self.ledger.record_proposed(sem.capability, self.ctx.risk, step)
        is_commit = at_least(self.ctx.risk, R2)
        if is_commit:
            self.ledger.bump_opportunity("commit_proposal", step)
        decisions = []
        for cap in self.capabilities:
            try:
                d = cap.before_final(answer, self.ctx)
            except Exception as ex:
                d = self._cap_error(cap, "before_final", ex)
            if d is not None:
                d.stage = "before_final"
                decisions.append(d)
        self._record_findings(decisions, "before_final")
        winner = D.combine(decisions, stage="before_final")
        eff = self._apply_mode(winner, "before_final")
        eff.feedback = _feedback(winner) if eff.type != D.ALLOW else None
        self._track_repair(winner, eff)
        # an ACCEPTED final answer (effective ALLOW) is a committed action.
        if is_commit and eff.type == D.ALLOW:
            self.ledger.record_commit(sem.capability, step, verified=self.ctx.verification,
                                      detail="final_answer", semantic_type="answer")
            self._close_verified_repair()    # a final-answer commit verifies + closes its repair too
        return eff

    def record_flagged_final(self, answer, flag="unresolved_risk", step=0):
        """run.py delivered a no-side-effect terminal answer WITH a verification flag (graceful degradation,
        CONTRACT(5)) instead of via effective-ALLOW. Record it as a final-answer commit with verified=None
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

    def audit(self):
        return {"mode": self.mode, "contract": self.contract.to_dict() if self.contract else None,
                "ledger": self.ledger.to_dict(), "budget": self.budget,
                "n_interventions": self._n_interventions,
                "n_semantic_checks": self.budget["max_semantic_checks"] - self.ctx.semantic_remaining,
                "status": ("degraded" if (self._capability_errors or self.policy.get("_errors")) else "active"),
                "capability_errors": self._capability_errors,
                "policy_errors": self.policy.get("_errors", []),
                # the harness's semantic judge model + how many times it was called — so the report can
                # verify INDEPENDENCE (judge != agent brain != tool backend) instead of asserting it.
                "runtime_judge_model": self.ctx.judge_model,
                "runtime_judge_calls": self.budget["max_semantic_checks"] - self.ctx.semantic_remaining}


def _action_name(action):
    if not isinstance(action, dict):
        return ""
    if action.get("type") == "final":
        return "final"
    return action.get("tool") or action.get("action") or action.get("type") or ""


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
