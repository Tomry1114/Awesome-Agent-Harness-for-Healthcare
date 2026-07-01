"""C4/C4.1 RecoveryOrchestrator episode tests via a scripted stub driver."""
import sys
sys.path.insert(0, "runner")
from harness.recovery_orchestrator import (RecoveryOrchestrator, EffectCompletionKey,
    VERIFIED, RECONCILING, BLOCKED_TERMINAL, RETRYABLE_FAILURE, ALREADY_REALIZED)

R = []
def ck(n, c): R.append((n, bool(c))); print(("OK  " if c else "FAIL") + " " + n)

class Outcome:
    def __init__(s, cid=None): s.created_id = cid

class StubDriver:
    def __init__(s, evals, acquire_results=None, auth_status_seq=None, created="sr-1"):
        s.evals = list(evals); s.acquire_results = list(acquire_results or []); s.auth_status_seq = list(auth_status_seq or [])
        s.created = created
        s.minted = 0; s.cancelled = 0; s.reserved = 0; s.executed = 0; s.acquired = 0
        s.bound_ids = []
    def mint(s, scope): s.minted += 1; return {"id": "auth-%d" % s.minted, "status": "AVAILABLE"}
    def auth_id(s, auth): return auth["id"]
    def set_hold(s): pass
    def evaluate(s, action): s.bound_ids.append(action.get("_mutation_authorization_id")); return s.evals.pop(0)
    def acquire(s, nxt): s.acquired += 1; return s.acquire_results.pop(0) if s.acquire_results else True
    def cancel(s, auth): s.cancelled += 1; auth["status"] = "CANCELLED"
    def reserve(s, auth):
        if auth["status"] != "AVAILABLE": return False
        auth["status"] = "RESERVED"; s.reserved += 1; return True
    def execute(s, action, auth): s.executed += 1; auth["status"] = "DISPATCHED"; return Outcome(s.created)
    def auth_status(s, auth): return s.auth_status_seq.pop(0) if s.auth_status_seq else "VERIFIED"

ACT = {"tool": "fhir_create", "args": {}}
SCOPE = {"target_path": "ServiceRequest/Patient/1"}

# 1) happy: real kernel combination raw=ACQUIRE/eff=ACQUIRE -> acquire -> re-eval ALLOW/ALLOW -> VERIFIED
d = StubDriver(evals=[("ACQUIRE", "ACQUIRE", {"tool": "fhir_search"}), ("ALLOW", "ALLOW")],
               acquire_results=[True], auth_status_seq=["VERIFIED"])
r = RecoveryOrchestrator(d).realize(ACT, SCOPE)
ck("happy_raw_acquire_verified", r.state == VERIFIED and d.acquired == 1 and d.executed == 1 and d.cancelled == 1)
ck("happy_auth_id_bound", all(b is not None for b in d.bound_ids))   # every evaluate saw a bound auth id

# 2) failed acquire -> blocked, no execute
d = StubDriver(evals=[("ACQUIRE", "ACQUIRE", {})], acquire_results=[False])
r = RecoveryOrchestrator(d).realize(ACT, SCOPE)
ck("failed_acquire_blocked", r.state == BLOCKED_TERMINAL and r.reason == "prerequisite_unresolved" and d.executed == 0)

# 3) contradictory: acquire resolves, re-eval REVISE -> no create
d = StubDriver(evals=[("ACQUIRE", "ACQUIRE", {}), ("REVISE", "REVISE")], acquire_results=[True])
r = RecoveryOrchestrator(d).realize(ACT, SCOPE)
ck("contradictory_no_create", r.state == BLOCKED_TERMINAL and d.executed == 0)

# 4) observe raw BLOCK / eff ALLOW -> no mutate
d = StubDriver(evals=[("BLOCK", "ALLOW")])
r = RecoveryOrchestrator(d).realize(ACT, SCOPE)
ck("observe_raw_block_no_mutate", r.state == BLOCKED_TERMINAL and "raw=BLOCK" in r.reason and d.executed == 0)

# 5) direct allow -> VERIFIED
d = StubDriver(evals=[("ALLOW", "ALLOW")], auth_status_seq=["VERIFIED"])
r = RecoveryOrchestrator(d).realize(ACT, SCOPE)
ck("direct_allow_verified", r.state == VERIFIED and d.executed == 1)

# 6) UNKNOWN -> reconcile only
d = StubDriver(evals=[("ALLOW", "ALLOW")], auth_status_seq=["UNKNOWN"])
ck("unknown_reconcile", RecoveryOrchestrator(d).realize(ACT, SCOPE).state == RECONCILING)

# 7) C4.1: DISPATCHED terminal status -> RECONCILING (may have landed), NOT retry
d = StubDriver(evals=[("ALLOW", "ALLOW")], auth_status_seq=["DISPATCHED"])
r = RecoveryOrchestrator(d).realize(ACT, SCOPE)
ck("dispatched_reconcile_no_retry", r.state == RECONCILING and d.executed == 1)

# 8) C4.1: invalid/None terminal status -> BLOCKED_TERMINAL, NEVER retried
for bad in ("AVAILABLE", "RESERVED", "CANCELLED", None):
    d = StubDriver(evals=[("ALLOW", "ALLOW")] * 5, auth_status_seq=[bad] * 5)
    r = RecoveryOrchestrator(d).realize(ACT, SCOPE)
    ck("invalid_status_%s_blocked" % bad, r.state == BLOCKED_TERMINAL and "invalid_auth_terminal_state" in r.reason and d.executed == 1)

# 9) C4.1: FAILED -> bounded retry then RETRYABLE_FAILURE
d = StubDriver(evals=[("ALLOW", "ALLOW")] * 3, auth_status_seq=["FAILED", "FAILED", "FAILED"])
r = RecoveryOrchestrator(d, max_create_retry=1).realize(ACT, SCOPE)
ck("failed_bounded_retry", r.state == RETRYABLE_FAILURE and d.executed == 2)

# 10) C4.1 REGISTRY: same key twice -> create once (second returns ALREADY_REALIZED)
orch = RecoveryOrchestrator(StubDriver(evals=[("ALLOW", "ALLOW")], auth_status_seq=["VERIFIED"]))
key = EffectCompletionKey("P/1", "hashA", "pelvic ultrasound", "ServiceRequest")
r1 = orch.realize(ACT, SCOPE, key=key)
execed_after_first = orch.d.executed
r2 = orch.realize(ACT, SCOPE, key=key)   # driver has no more evals; if it re-ran it would IndexError -> so registry MUST short-circuit
ck("same_key_create_once", r1.state == VERIFIED and r2.state == ALREADY_REALIZED and orch.d.executed == execed_after_first == 1)

# 11) registry: a BLOCKED key is not re-run
orch = RecoveryOrchestrator(StubDriver(evals=[("BLOCK", "ALLOW")]))
k2 = EffectCompletionKey("P/1", "hashB", "cbc", "ServiceRequest")
b1 = orch.realize(ACT, SCOPE, key=k2)
b2 = orch.realize(ACT, SCOPE, key=k2)   # no more evals; registry must short-circuit
ck("blocked_key_not_rerun", b1.state == BLOCKED_TERMINAL and b2.state == BLOCKED_TERMINAL and orch.d.executed == 0)

n = sum(1 for _, c in R if c)
print("\n%d/%d recovery_orchestrator(v2) tests passed" % (n, len(R)))
assert n == len(R), "FAILURES"
