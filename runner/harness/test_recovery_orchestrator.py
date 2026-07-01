"""C4 RecoveryOrchestrator episode tests via a scripted stub driver (point-11 cross-component matrix)."""
import sys
sys.path.insert(0, "runner")
from harness.recovery_orchestrator import (RecoveryOrchestrator, EffectCompletionKey,
    VERIFIED, RECONCILING, BLOCKED_TERMINAL, RETRYABLE_FAILURE)

R = []
def ck(n, c): R.append((n, bool(c))); print(("OK  " if c else "FAIL") + " " + n)

class Outcome:
    def __init__(s, cid=None): s.created_id = cid

class StubDriver:
    """Scripted driver. `evals` is a list of (raw, eff[, next]) returned by successive evaluate() calls.
    `acquire_results` scripts each acquire() return. `auth_status_seq` scripts auth_status() after execute()."""
    def __init__(s, evals, acquire_results=None, auth_status_seq=None, created="sr-1"):
        s.evals = list(evals); s.acquire_results = list(acquire_results or []); s.auth_status_seq = list(auth_status_seq or [])
        s.created = created
        s.minted = 0; s.cancelled = 0; s.reserved = 0; s.executed = 0; s.acquired = 0; s.holds = 0
        s._auths = []
    def mint(s, scope): s.minted += 1; a = {"id": "auth-%d" % s.minted, "status": "AVAILABLE"}; s._auths.append(a); return a
    def set_hold(s): s.holds += 1
    def evaluate(s, action): return s.evals.pop(0)
    def acquire(s, nxt): s.acquired += 1; return s.acquire_results.pop(0) if s.acquire_results else True
    def cancel(s, auth): s.cancelled += 1; auth["status"] = "CANCELLED"
    def reserve(s, auth):
        if auth["status"] != "AVAILABLE": return False
        auth["status"] = "RESERVED"; s.reserved += 1; return True
    def execute(s, action, auth): s.executed += 1; auth["status"] = "DISPATCHED"; return Outcome(s.created)
    def auth_status(s, auth): return s.auth_status_seq.pop(0) if s.auth_status_seq else "VERIFIED"

ACT = {"tool": "fhir_create", "args": {}}
SCOPE = {"target_path": "ServiceRequest/Patient/1"}

# 1) HAPPY PATH: first evaluate ACQUIRE (prereq), acquire resolves, re-evaluate ALLOW/ALLOW -> execute -> VERIFIED
d = StubDriver(evals=[("ALLOW", "ACQUIRE", {"tool": "fhir_search"}), ("ALLOW", "ALLOW")],
               acquire_results=[True], auth_status_seq=["VERIFIED"])
r = RecoveryOrchestrator(d).realize(ACT, SCOPE)
ck("happy_verified", r.state == VERIFIED and r.created_id == "sr-1" and r.realized)
ck("happy_acquire_then_execute", d.acquired == 1 and d.executed == 1)
ck("happy_stale_auth_cancelled_on_acquire", d.cancelled == 1 and d.minted == 2)   # minted round1 (cancelled) + round2 (used)

# 2) FAILED ACQUIRE -> BLOCKED, never executes (governance prerequisite unresolved)
d = StubDriver(evals=[("ALLOW", "ACQUIRE", {"tool": "fhir_search"})], acquire_results=[False])
r = RecoveryOrchestrator(d).realize(ACT, SCOPE)
ck("failed_acquire_blocked", r.state == BLOCKED_TERMINAL and r.reason == "prerequisite_unresolved")
ck("failed_acquire_no_execute", d.executed == 0)

# 3) CONTRADICTORY EVIDENCE: acquire resolves, re-evaluate returns REVISE (new evidence refutes) -> no create
d = StubDriver(evals=[("ALLOW", "ACQUIRE", {"tool": "fhir_search"}), ("REVISE", "REVISE")], acquire_results=[True])
r = RecoveryOrchestrator(d).realize(ACT, SCOPE)
ck("contradictory_revise_blocked", r.state == BLOCKED_TERMINAL and "not_allowed" in r.reason)
ck("contradictory_no_execute", d.executed == 0)

# 4) OBSERVE MODE: raw BLOCK but effective ALLOW (mode downgrade) -> recovery must NOT mutate (point 10)
d = StubDriver(evals=[("BLOCK", "ALLOW")])
r = RecoveryOrchestrator(d).realize(ACT, SCOPE)
ck("observe_raw_block_no_mutate", r.state == BLOCKED_TERMINAL and "raw=BLOCK" in r.reason and d.executed == 0)

# 5) DIRECT ALLOW (no prereq) -> execute -> VERIFIED
d = StubDriver(evals=[("ALLOW", "ALLOW")], auth_status_seq=["VERIFIED"])
r = RecoveryOrchestrator(d).realize(ACT, SCOPE)
ck("direct_allow_verified", r.state == VERIFIED and d.acquired == 0 and d.executed == 1)

# 6) UNKNOWN after execute -> RECONCILING (reconcile only, never re-create)
d = StubDriver(evals=[("ALLOW", "ALLOW")], auth_status_seq=["UNKNOWN"])
r = RecoveryOrchestrator(d).realize(ACT, SCOPE)
ck("unknown_reconcile_only", r.state == RECONCILING and d.executed == 1)

# 7) FAILED create -> bounded retry -> RETRYABLE_FAILURE after retries exhausted
d = StubDriver(evals=[("ALLOW", "ALLOW"), ("ALLOW", "ALLOW"), ("ALLOW", "ALLOW")],
               auth_status_seq=["FAILED", "FAILED", "FAILED"])
r = RecoveryOrchestrator(d, max_create_retry=1).realize(ACT, SCOPE)
ck("failed_create_retryable", r.state == RETRYABLE_FAILURE and d.executed == 2)   # initial + 1 retry

# 8) reserve failure -> BLOCKED, no execute
class NoReserve(StubDriver):
    def reserve(s, auth): return False
d = NoReserve(evals=[("ALLOW", "ALLOW")])
r = RecoveryOrchestrator(d).realize(ACT, SCOPE)
ck("reserve_failed_blocked", r.state == BLOCKED_TERMINAL and r.reason == "reserve_failed" and d.executed == 0)

# 9) max prerequisite rounds bound
d = StubDriver(evals=[("ALLOW", "ACQUIRE", {}), ("ALLOW", "ACQUIRE", {}), ("ALLOW", "ACQUIRE", {}), ("ALLOW", "ACQUIRE", {})],
               acquire_results=[True, True, True, True])
r = RecoveryOrchestrator(d, max_prereq=2).realize(ACT, SCOPE)
ck("max_prereq_bound", r.state == BLOCKED_TERMINAL and r.reason == "max_prerequisite_rounds")

# EffectCompletionKey identity (per-order independence)
k1 = EffectCompletionKey("P/1", "hashA", "pelvic ultrasound", "ServiceRequest")
k2 = EffectCompletionKey("P/1", "hashA", "cbc", "ServiceRequest")
ck("effect_key_distinct_orders", k1 != k2 and hash(k1) != hash(k2))

n = sum(1 for _, c in R if c)
print("\n%d/%d recovery_orchestrator tests passed" % (n, len(R)))
assert n == len(R), "FAILURES"
