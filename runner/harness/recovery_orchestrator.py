"""RecoveryOrchestrator (Commit C4, hardened C4.1) -- the episode state machine that safely realizes ONE
agent-committed effect through the SINGLE action pipeline, handling ACQUIRE prerequisites as a LOOP.

Driven by an injected `driver` (the run.py adapter in C5b; a stub in tests):
  driver.mint(scope) -> auth ; driver.auth_id(auth) -> str ; driver.set_hold()
  driver.evaluate(bound_action) -> (raw_type, effective_type[, next_action])   [full before_action]
  driver.acquire(next_action) -> bool (True == prerequisite RESOLVED PRESENT/ABSENT, binds evidence to ledger)
  driver.cancel(auth) ; driver.reserve(auth) -> bool ; driver.execute(bound_action, auth) -> ActionOutcome
  driver.auth_status(auth) -> terminal status string

C4.1 hardening:
  - EXHAUSTIVE post-execute switch: ONLY FAILED retries; DISPATCHED/UNKNOWN -> RECONCILING (may have landed,
    never re-create); any other/None status -> BLOCKED_TERMINAL (never silently retried).
  - EXACT authorization: the auth id is BOUND into the action, so before_action审 and executor dispatch see the
    SAME authorization (driver + verify_commit enforce exact-id match).
  - EPISODE REGISTRY keyed by EffectCompletionKey: a VERIFIED/RECONCILING/BLOCKED key is NOT re-run (create once).
  - raw AND effective must both be ALLOW (point 10); bounded prerequisite rounds + bounded create retries.
No benchmark names.
"""
from dataclasses import dataclass, field

NOT_STARTED = "NOT_STARTED"
WAITING_PREREQUISITE = "WAITING_PREREQUISITE"
READY = "READY"
DISPATCHED = "DISPATCHED"
RECONCILING = "RECONCILING"
VERIFIED = "VERIFIED"
ALREADY_REALIZED = "ALREADY_REALIZED"
RETRYABLE_FAILURE = "RETRYABLE_FAILURE"
BLOCKED_TERMINAL = "BLOCKED_TERMINAL"


@dataclass(frozen=True)
class EffectCompletionKey:
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
        self.episodes = {}                      # EffectCompletionKey -> EpisodeResult (C4.1 registry)

    def realize(self, mutation_action, scope, key=None):
        """Drive ONE effect-completion episode to a terminal state, deduped by `key`."""
        if key is not None:
            rec = self.episodes.get(key)
            if rec is not None:
                if rec.state in (VERIFIED, ALREADY_REALIZED):
                    return EpisodeResult(ALREADY_REALIZED, created_id=rec.created_id, auth_status=rec.auth_status,
                                         reason="already_realized_this_episode", events=rec.events)
                if rec.state in (RECONCILING, BLOCKED_TERMINAL):
                    return rec                  # reconcile-only / terminal -> never re-run
                # RETRYABLE_FAILURE -> fall through and retry
        result = self._run(mutation_action, scope)
        if key is not None:
            self.episodes[key] = result
        return result

    def _run(self, mutation_action, scope):
        ev = []
        prereq_rounds = 0
        create_attempts = 0
        while True:
            auth = self.d.mint(scope)                       # fresh, AVAILABLE, at the CURRENT evidence version
            bound = dict(mutation_action)                   # C4.1: BIND the auth id so审 == dispatch is the SAME auth
            bound["_mutation_authorization_id"] = self.d.auth_id(auth)
            bound["_recovery"] = "COMPLETE_EFFECT"
            self.d.set_hold()
            raw, eff, nxt = _norm(self.d.evaluate(bound))
            ev.append({"step": "evaluate", "raw": raw, "effective": eff})

            if eff == "ACQUIRE":
                self.d.cancel(auth)                         # stale once evidence changes
                if prereq_rounds >= self.max_prereq:
                    return EpisodeResult(BLOCKED_TERMINAL, auth_status="CANCELLED",
                                         reason="max_prerequisite_rounds", prereq_rounds=prereq_rounds, events=ev)
                prereq_rounds += 1
                resolved = self.d.acquire(nxt)
                ev.append({"step": "acquire", "resolved": bool(resolved)})
                if not resolved:
                    return EpisodeResult(BLOCKED_TERMINAL, reason="prerequisite_unresolved",
                                         prereq_rounds=prereq_rounds, events=ev)
                continue                                    # RE-EVALUATE from scratch (point 11)

            if not (raw == "ALLOW" and eff == "ALLOW"):     # point 10: raw AND effective ALLOW
                self.d.cancel(auth)
                return EpisodeResult(BLOCKED_TERMINAL, reason="not_allowed:raw=%s,eff=%s" % (raw, eff),
                                     prereq_rounds=prereq_rounds, events=ev)

            # #2: EXPERIMENT INVARIANT -- a recovery mutation writes state ONLY in enforce mode. observe = record
            # only, assist = feedback only; neither may create a resource. This is independent of raw/effective
            # ALLOW (a raw-ALLOW in observe still must NOT write).
            _mode = self.d.mode() if hasattr(self.d, "mode") else "enforce"
            if _mode != "enforce":
                self.d.cancel(auth)
                return EpisodeResult(BLOCKED_TERMINAL, reason="mode_not_enforce:%s" % _mode,
                                     prereq_rounds=prereq_rounds, events=ev)

            if not self.d.reserve(auth):
                return EpisodeResult(BLOCKED_TERMINAL, reason="reserve_failed",
                                     prereq_rounds=prereq_rounds, events=ev)
            outcome = self.d.execute(bound, auth)
            st = self.d.auth_status(auth)
            created = getattr(outcome, "created_id", None) if outcome is not None else None
            ev.append({"step": "execute", "auth_status": st, "created_id": created})

            # C4.1 EXHAUSTIVE switch -- only an explicit FAILED may retry-create
            if st == "VERIFIED":
                return EpisodeResult(VERIFIED, created_id=created, auth_status=st,
                                     reason="landed_and_verified", prereq_rounds=prereq_rounds, events=ev)
            if st in ("UNKNOWN", "DISPATCHED"):             # may have landed -> reconcile only, NEVER re-create
                return EpisodeResult(RECONCILING, auth_status=st,
                                     reason="ambiguous_or_undispatched_terminal_reconcile_only",
                                     prereq_rounds=prereq_rounds, events=ev)
            if st != "FAILED":                              # AVAILABLE/RESERVED/CANCELLED/None -> a driver bug; NEVER retry
                return EpisodeResult(BLOCKED_TERMINAL, auth_status=st,
                                     reason="invalid_auth_terminal_state:%s" % st,
                                     prereq_rounds=prereq_rounds, events=ev)
            # FAILED (definitely not landed) -> bounded retry
            create_attempts += 1
            if create_attempts > self.max_create_retry:
                return EpisodeResult(RETRYABLE_FAILURE, auth_status=st, reason="create_failed_retries_exhausted",
                                     prereq_rounds=prereq_rounds, events=ev)


def _norm(evald):
    if isinstance(evald, (list, tuple)):
        if len(evald) >= 3:
            return evald[0], evald[1], evald[2]
        if len(evald) == 2:
            return evald[0], evald[1], None
    return evald, evald, None
