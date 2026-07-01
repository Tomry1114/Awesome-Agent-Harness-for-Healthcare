"""RecoveryOrchestrator (Commit C4) -- the episode state machine that safely realizes ONE agent-committed
effect through the SINGLE action pipeline, handling ACQUIRE prerequisites as a LOOP (never recursion).

It does NOT execute the environment itself and holds no substrate knowledge. It drives an injected `driver`
(the run.py adapter in C5; a stub in tests) whose primitive operations are the ONLY way it touches the world:

  driver.mint(scope)                 -> a fresh deterministic_gap authorization at the CURRENT evidence version
  driver.set_hold()                  -> put writes under a mutation_hold (so verify_commit requires the auth)
  driver.evaluate(action)            -> (raw_decision_type, effective_decision_type[, next_action])  [before_action]
  driver.acquire(next_action)        -> run the prerequisite READ through the SAME executor (binds evidence to
                                        the ledger, re-runs ScopeEvidenceBinding); returns an EvidenceState-ish
                                        resolution flag (True == resolved PRESENT/ABSENT, False == unresolved)
  driver.cancel(auth)                -> cancel a now-stale authorization (AVAILABLE|RESERVED -> CANCELLED)
  driver.reserve(auth)              -> AVAILABLE -> RESERVED (bool)
  driver.execute(action, auth)       -> dispatch+execute+after_action+finalize; returns the ActionOutcome
  driver.auth_status(auth)           -> the authorization's terminal status after execute

Integrity guarantees enforced HERE (not left to the driver):
  - ACQUIRE loop: a prerequisite is fetched, the STALE authorization is CANCELLED, and the mutation is
    RE-EVALUATED from scratch (a resolved obligation is NOT auto-approval -- point 11).
  - raw AND effective must both be ALLOW before any mutation (point 10): observe/assist (effective ALLOW from a
    raw BLOCK) never mutates.
  - bounded prerequisite rounds + bounded create retries; UNKNOWN is reconcile-only (never re-created).
No benchmark names.
"""
from dataclasses import dataclass, field

# episode states
NOT_STARTED = "NOT_STARTED"
WAITING_PREREQUISITE = "WAITING_PREREQUISITE"
READY = "READY"
DISPATCHED = "DISPATCHED"
RECONCILING = "RECONCILING"
VERIFIED = "VERIFIED"
ALREADY_REALIZED = "ALREADY_REALIZED"
RETRYABLE_FAILURE = "RETRYABLE_FAILURE"
BLOCKED_TERMINAL = "BLOCKED_TERMINAL"
_TERMINAL = (VERIFIED, ALREADY_REALIZED, BLOCKED_TERMINAL, RECONCILING)


@dataclass(frozen=True)
class EffectCompletionKey:
    """Identity of ONE committed effect -- so several orders in one deliverable complete / block / retry
    INDEPENDENTLY (point 12). Substrate-agnostic."""
    subject: str
    artifact_hash: str
    commitment_signature: str
    effect_type: str


@dataclass
class EpisodeResult:
    state: str
    created_id: str = None
    auth_status: str = None
    reason: str = None
    prereq_rounds: int = 0
    events: list = field(default_factory=list)

    @property
    def realized(self):
        return self.state in (VERIFIED, ALREADY_REALIZED)


class RecoveryOrchestrator:
    def __init__(self, driver, max_prereq=3, max_create_retry=1):
        self.d = driver
        self.max_prereq = max_prereq
        self.max_create_retry = max_create_retry

    def realize(self, mutation_action, scope, key=None):
        """Drive ONE effect-completion episode to a terminal state. Returns EpisodeResult."""
        ev = []
        prereq_rounds = 0
        create_attempts = 0
        while True:
            auth = self.d.mint(scope)                       # fresh, AVAILABLE, at the CURRENT evidence version
            self.d.set_hold()                               # writes now require the authorization
            raw, eff, nxt = _norm(self.d.evaluate(mutation_action))
            ev.append({"step": "evaluate", "raw": raw, "effective": eff})

            if eff == "ACQUIRE":
                self.d.cancel(auth)                         # the just-minted auth is stale once evidence changes
                if prereq_rounds >= self.max_prereq:
                    return EpisodeResult(BLOCKED_TERMINAL, auth_status="CANCELLED",
                                         reason="max_prerequisite_rounds", prereq_rounds=prereq_rounds, events=ev)
                prereq_rounds += 1
                resolved = self.d.acquire(nxt)              # READ through the same executor -> binds evidence
                ev.append({"step": "acquire", "resolved": bool(resolved), "next": nxt})
                if not resolved:
                    return EpisodeResult(BLOCKED_TERMINAL, reason="prerequisite_unresolved",
                                         prereq_rounds=prereq_rounds, events=ev)
                continue                                    # RE-EVALUATE the mutation from scratch (point 11)

            # point 10: recovery mutation requires BOTH raw and effective ALLOW (observe/assist never mutate)
            if not (raw == "ALLOW" and eff == "ALLOW"):
                self.d.cancel(auth)
                return EpisodeResult(BLOCKED_TERMINAL, reason="not_allowed:raw=%s,eff=%s" % (raw, eff),
                                     prereq_rounds=prereq_rounds, events=ev)

            if not self.d.reserve(auth):                    # AVAILABLE -> RESERVED
                return EpisodeResult(BLOCKED_TERMINAL, reason="reserve_failed",
                                     prereq_rounds=prereq_rounds, events=ev)
            outcome = self.d.execute(mutation_action, auth)  # dispatch(RESERVED->DISPATCHED)+execute+after+finalize
            st = self.d.auth_status(auth)
            created = getattr(outcome, "created_id", None) if outcome is not None else None
            ev.append({"step": "execute", "auth_status": st, "created_id": created})

            if st == "VERIFIED":
                return EpisodeResult(VERIFIED, created_id=created, auth_status=st,
                                     reason="landed_and_verified", prereq_rounds=prereq_rounds, events=ev)
            if st == "UNKNOWN":                             # may have landed -> reconcile only, NEVER re-create
                return EpisodeResult(RECONCILING, auth_status=st, reason="ambiguous_outcome_reconcile_only",
                                     prereq_rounds=prereq_rounds, events=ev)
            # FAILED (definitely not landed) -> bounded retry
            create_attempts += 1
            if create_attempts > self.max_create_retry:
                return EpisodeResult(RETRYABLE_FAILURE, auth_status=st, reason="create_failed_retries_exhausted",
                                     prereq_rounds=prereq_rounds, events=ev)
            # loop: re-mint + re-evaluate + re-execute (bounded)


def _norm(evald):
    """Accept (raw, eff) or (raw, eff, next_action)."""
    if isinstance(evald, (list, tuple)):
        if len(evald) >= 3:
            return evald[0], evald[1], evald[2]
        if len(evald) == 2:
            return evald[0], evald[1], None
    return evald, evald, None
