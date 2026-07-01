"""RunDriver (Commit C5b) -- the real-environment implementation of the RecoveryOrchestrator's driver, over
the live harness kernel + ActionExecutor + env. This is the ONLY glue between the orchestrator's abstract
primitives and run.py's actual harness/executor/env. No benchmark names.

Every recovery action (the prerequisite READ and the authorized CREATE) goes through the SAME ActionExecutor
as agent actions, so it gets: canonical semantics, before/after_action, evidence binding, active read-back,
and a real tool_call event (tagged origin=recovery, audience=harness so it does not misreport Observability).
"""
from .evidence_state import classify_evidence_state, is_resolved

_FHIR_SEM = {"collection_paths": ["entries"], "absence_when_empty": True}


class RunDriver:
    def __init__(self, harness, executor, env, task, state_snapshot):
        self.h = harness
        self.ex = executor
        self.env = env
        self.task = task
        self.snap = state_snapshot
        self.step = 0             # set by run.py before each realize()
        self.trajectory = []      # set by run.py before each realize()

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

    # -- full before_action (returns raw + effective + the ACQUIRE next_action) --
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

    # -- prerequisite READ through the same executor -> binds evidence to the ledger --
    def acquire(self, next_action):
        if not next_action or not next_action.get("tool"):
            return False
        rd = dict(next_action); rd.setdefault("type", "tool_call")
        outcome = self.ex.execute_and_normalize(rd, self.env)
        try:
            self.ex.run_after_action(self.h, rd, outcome, self.step)   # ScopeEvidenceBinding records EvidenceState
        except Exception:
            pass
        try:
            self.h.ledger.acquire_count = getattr(self.h.ledger, "acquire_count", 0) + 1
        except Exception:
            pass
        ev, _ = self.ex.build_event(rd, outcome, self.step, origin="recovery", audience="harness")
        self.trajectory.append(ev)
        # a FAILED/UNKNOWN read did NOT resolve the prerequisite -> stop the episode (never loop on a broken read)
        return is_resolved(classify_evidence_state(outcome.res, _FHIR_SEM))

    # -- authorized CREATE through the same executor (dispatch just before the env call) + finalize the auth --
    def execute(self, action, auth):
        outcome = self.ex.execute_and_normalize(action, self.env, ledger=self.h.ledger, auth=auth)
        try:
            hp = self.ex.run_after_action(self.h, action, outcome, self.step)
        except Exception:
            hp = None
        ver = getattr(self.h.ctx, "verification", None)
        aeff = hp.type if hp is not None else "ALLOW"
        cf = outcome.recon.get("confirmed") if outcome.recon else None
        L = self.h.ledger
        if outcome.result_status == "failed" and cf is False:
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
        self.trajectory.append(ev)
        outcome.created_id = (outcome.res.get("id") if isinstance(outcome.res, dict) else None)
        return outcome
