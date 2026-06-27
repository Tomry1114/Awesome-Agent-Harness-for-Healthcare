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
        # MAX_REVISIONS_PER_ACTION: a commit-stage REVISE that keeps re-firing is a stuck revision loop.
        # Count effective REVISEs per (reason_code, capability) for the commit stages; once the count
        # exceeds the budget, escalate that decision to ESCALATE instead of looping REVISE forever.
        elif eff == D.REVISE and stage in ("after_action", "before_final"):
            rkey = (decision.reason_code, decision.capability)
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

    def _record_findings(self, decisions, stage):
        """Record EVERY non-ALLOW finding (not just the hook winner) so a lower-priority finding survives
        for metrics — e.g. a commit that is BOTH wrong-subject (BLOCK) and missing-prerequisite (REVISE)
        contributes to BOTH rates, not only the higher-priority one."""
        for d in decisions:
            if d.type != D.ALLOW:
                self.ledger.record_finding({"action_key": "%s-step%d" % (stage, self.ctx.step),
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
        sem = self._canon(action)
        risk = self.ctx.risk
        pid = self.ledger.record_proposed(sem.capability, risk, step)
        from .risk import at_least, R2
        if at_least(risk, R2):
            self.ledger.bump_opportunity("commit_proposal")   # denominator for missing_prerequisite_rate
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
        return eff

    def after_action(self, action, result, before_state, after_state, step=0, canonical_observation=None,
                     result_ok=None):
        self.ctx.step = step
        self.ctx.observation = canonical_observation
        self._last_obs = canonical_observation        # carried into the NEXT before_action (prospective scope)
        sem = self._canon(action, observation=canonical_observation)
        self.ctx.result_ok = result_ok      # whether the tool result succeeded (adapter signal)
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
                                      detail=winner.reason)
        eff = self._apply_mode(winner, "after_action")
        eff.feedback = _feedback(winner) if eff.type != D.ALLOW else None
        return eff

    def before_final(self, answer, step=0):
        self.ctx.step = step
        sem = self._canon({"type": "final", "answer": answer})
        from .risk import at_least, R2
        # the final answer is a commit -> it enters the SAME commit lifecycle (proposal + opportunity).
        self.ledger.record_proposed(sem.capability, self.ctx.risk, step)
        is_commit = at_least(self.ctx.risk, R2)
        if is_commit:
            self.ledger.bump_opportunity("commit_proposal")
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
        # an ACCEPTED final answer (effective ALLOW) is a committed action.
        if is_commit and eff.type == D.ALLOW:
            self.ledger.record_commit(sem.capability, step, verified=self.ctx.verification,
                                      detail="final_answer")
        return eff

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
                "policy_errors": self.policy.get("_errors", [])}


def _action_name(action):
    if not isinstance(action, dict):
        return ""
    if action.get("type") == "final":
        return "final"
    return action.get("tool") or action.get("action") or action.get("type") or ""


def _feedback(decision):
    """The leak-safe structured message handed to the agent (never gold/reference)."""
    fb = {"decision": decision.type, "rule_id": decision.rule_id, "reason": decision.reason}
    if decision.missing_obligations:
        fb["missing_obligations"] = decision.missing_obligations
    if decision.suggested_capabilities:
        fb["suggested_capabilities"] = decision.suggested_capabilities
    if decision.feedback:
        fb["message"] = decision.feedback
    return fb
