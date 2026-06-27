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
        self.budget = dict(_DEFAULT_BUDGET); self.budget.update(budget or {})
        self.ledger = Ledger()
        if contract is not None and contract.subject:
            self.ledger.set_subject(contract.subject)
        self.ctx = HarnessContext(self.ledger, contract, self.policy, mode, env_type, risk_of,
                                  judge_fn=judge_fn, judge_model=judge_model,
                                  semantic_budget=self.budget["max_semantic_checks"])
        self._n_interventions = 0
        self._n_semantic = 0
        self._rev_for_action = {}
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
        # budget is a RESOURCE constraint, NOT a safety override: an exhausted budget must NEVER silently
        # turn a would-be BLOCK/REVISE/ESCALATE into ALLOW. Instead it ESCALATEs (terminate safely). Only
        # an EFFECTIVE (run-affecting) intervention consumes budget.
        if eff != D.ALLOW and self._n_interventions >= self.budget["max_interventions_per_task"]:
            eff, note = D.ESCALATE, "budget_exhausted_interventions"
        ev = self._ev("harness_decision", stage, decision, mode_applied=eff,
                      extra=({"note": note} if note else None))
        # record the WOULD-BE intervention (raw) with its effective outcome, in every mode
        d = decision.to_dict(); d["effective"] = eff; d["event_id"] = ev["id"]
        self.ledger.record_intervention(d)
        if eff != D.ALLOW:
            self._n_interventions += 1
        return Effective(eff, feedback=decision.feedback, raw=decision, events=[ev])

    # ---- public hooks --------------------------------------------------------
    def before_action(self, action, env_state=None, step=0):
        self.ctx.step = step
        risk = self.risk_of(action) if self.risk_of else None
        pid = self.ledger.record_proposed(_action_name(action), risk, step)
        from .risk import at_least, R2
        if risk and at_least(risk, R2):
            self.ledger.bump_opportunity("commit_proposal")   # denominator for missing_prerequisite_rate
        decisions = []
        for cap in self.capabilities:
            try:
                d = cap.before_action(action, self.ctx)
            except Exception as ex:                       # a buggy capability must never crash the run
                d = D.HarnessDecision(D.ALLOW, capability=cap.name, reason="capability_error:%r" % ex)
            if d is not None:
                d.stage = "before_action"
                d.extra.setdefault("action_key", pid)
                decisions.append(d)
        winner = D.combine(decisions, stage="before_action")
        eff = self._apply_mode(winner, "before_action")
        eff.feedback = _feedback(winner) if eff.type != D.ALLOW else None
        return eff

    def after_action(self, action, result, before_state, after_state, step=0, canonical_observation=None):
        self.ctx.step = step
        self.ctx.observation = canonical_observation
        decisions = []
        for cap in self.capabilities:
            try:
                d = cap.after_action(action, result, before_state, after_state, self.ctx)
            except Exception as ex:
                d = D.HarnessDecision(D.ALLOW, capability=cap.name, reason="capability_error:%r" % ex)
            if d is not None:
                d.stage = "after_action"
                decisions.append(d)
        winner = D.combine(decisions, stage="after_action")
        risk = self.risk_of(action) if self.risk_of else None
        from .risk import at_least, R2
        if risk and at_least(risk, R2):
            self.ledger.record_commit(_action_name(action), step,
                                      verified=(winner.type == D.ALLOW),
                                      detail=winner.reason)
        eff = self._apply_mode(winner, "after_action")
        eff.feedback = _feedback(winner) if eff.type != D.ALLOW else None
        return eff

    def before_final(self, answer, step=0):
        decisions = []
        for cap in self.capabilities:
            try:
                d = cap.before_final(answer, self.ctx)
            except Exception as ex:
                d = D.HarnessDecision(D.ALLOW, capability=cap.name, reason="capability_error:%r" % ex)
            if d is not None:
                d.stage = "before_final"
                decisions.append(d)
        winner = D.combine(decisions, stage="before_final")
        eff = self._apply_mode(winner, "before_final")
        eff.feedback = _feedback(winner) if eff.type != D.ALLOW else None
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
                "n_interventions": self._n_interventions, "n_semantic_checks": self._n_semantic}


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
