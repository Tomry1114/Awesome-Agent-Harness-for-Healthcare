"""C5c/C5d RunDriver hardening. C5d unifies acquire+inspect into ONE strict internal reader:
 P0-1 before_action must be ALLOW (exception or any non-ALLOW -> abort, no env call)
 P0-2 after_action must succeed (crash -> UNKNOWN, never raw ABSENT)
 P0-3 resolved requires an EXACT ledger delta (resource==unit, subject==active, matched, PRESENT/ABSENT)
 P1   can_execute_recovery_action() hard-gates every env call
Plus the preserved create VERIFIED path (#5) and env counting (#6). Stub harness/executor/env."""
import sys
sys.path.insert(0, "runner")
from harness.run_driver import RunDriver

R = []
def ck(n, c): R.append((n, bool(c))); print(("OK  " if c else "FAIL") + " " + n)

class Dec:
    def __init__(s, t="ALLOW"): s.type = t; s.events = []
class Outcome:
    def __init__(s, res=None, err=None, status="ok", recon=None):
        s.res = res if res is not None else {"id": "x1"}; s.err = err
        s.result_status = status; s.recon = recon; s.created_id = None
class Ctx:
    def __init__(s): s.verification = None; s.last_raw_decision = None
class Ledger:
    def __init__(s, subject="Patient/1"):
        s._subj = subject; s.evidence = []; s.acquire_count = 0; s.calls = []
    def subject_id(s): return s._subj
    def dispatch_authorization(s, a): a["status"] = "DISPATCHED"; return True
    def fail_authorization(s, a): a["status"] = "FAILED"; s.calls.append("fail")
    def unknown_authorization(s, a): a["status"] = "UNKNOWN"; s.calls.append("unknown")
    def verify_authorization(s, a): a["status"] = "VERIFIED"; s.calls.append("verify")
class Harness:
    def __init__(s, before="ALLOW", raise_before=False, mode="enforce"):
        s.ledger = Ledger(); s.ctx = Ctx(); s._before = before; s._raise = raise_before; s.mode = mode
    def before_action(s, action, snap, step=0):
        if s._raise: raise RuntimeError("before boom")
        return Dec(s._before)
class Ex:
    def __init__(s, outcome, after_fn=None): s.o = outcome; s.after_fn = after_fn; s.env_calls = 0
    def execute_and_normalize(s, action, env, ledger=None, auth=None):
        if ledger is not None and auth is not None:
            if not ledger.dispatch_authorization(auth):
                s.o.err = "authorization_not_dispatchable"; return s.o
        s.env_calls += 1
        return s.o
    def run_after_action(s, h, action, outcome, step):
        return s.after_fn(h, action, outcome, step) if s.after_fn else None
    def build_event(s, action, outcome, step, origin="agent", audience="agent"): return ({"origin": origin}, "")

def driver(h, ex, on_env=None, budget=None):
    return RunDriver(h, ex, env={}, task={}, state_snapshot=lambda e: {}, on_env_action=on_env, budget_check=budget)

def bind(state, resource="ServiceRequest", subject="Patient/1", rel="matched"):
    def _f(h, a, o, s):
        h.ledger.evidence.append({"scope_relation": rel, "subject_id": subject,
                                  "evidence_state": state, "resource": resource})
        return None
    return _f

READ = {"tool": "fhir_search", "args": {"resourceType": "ServiceRequest"}}

# ===== P0-1: before_action must be ALLOW -- exception + every non-ALLOW aborts WITHOUT executing =====
for _t in ("BLOCK", "ESCALATE", "ACQUIRE", "REVISE", "RECONCILE"):
    h = Harness(before=_t); ex = Ex(Outcome(), after_fn=bind("ABSENT"))
    st, _ = driver(h, ex).execute_recovery_read(READ)
    ck("before_%s_aborts_no_exec" % _t, st == "UNKNOWN" and ex.env_calls == 0)
h = Harness(raise_before=True); ex = Ex(Outcome(), after_fn=bind("ABSENT"))
st, _ = driver(h, ex).execute_recovery_read(READ)
ck("before_action_exception_unknown_no_exec", st == "UNKNOWN" and ex.env_calls == 0)

# ===== P0-3: EXACT ledger delta -- resource must match the requested unit; stray records don't resolve =====
h = Harness(); ex = Ex(Outcome(res={"entries": []}), after_fn=bind("ABSENT", resource="ServiceRequest"))
st, _ = driver(h, ex).execute_recovery_read(READ, expected_evidence_unit="ServiceRequest")
ck("exact_unit_absent_resolves", st == "ABSENT")

h = Harness(); ex = Ex(Outcome(), after_fn=bind("PRESENT", resource="Condition"))   # wrong resource
st, _ = driver(h, ex).execute_recovery_read(READ, expected_evidence_unit="ServiceRequest")
ck("wrong_resource_unit_not_resolved", st == "UNKNOWN")

h = Harness(); ex = Ex(Outcome(), after_fn=bind("PRESENT", subject="Patient/999"))  # wrong subject
st, _ = driver(h, ex).execute_recovery_read(READ, expected_evidence_unit="ServiceRequest")
ck("wrong_subject_not_resolved", st == "UNKNOWN")

h = Harness(); ex = Ex(Outcome(), after_fn=bind("PRESENT", rel="foreign"))          # foreign scope
st, _ = driver(h, ex).execute_recovery_read(READ, expected_evidence_unit="ServiceRequest")
ck("foreign_scope_not_resolved", st == "UNKNOWN")

# ===== P0-2: after_action crash -> UNKNOWN (never a raw ABSENT) =====
def crash(h, a, o, s): raise RuntimeError("after boom")
h = Harness(); ex = Ex(Outcome(res={"entries": []}), after_fn=crash)
st, _ = driver(h, ex).execute_recovery_read(READ, expected_evidence_unit="ServiceRequest")
ck("after_action_crash_unknown", st == "UNKNOWN")

# tool-error envelope -> UNKNOWN
h = Harness(); ex = Ex(Outcome(res={"error": "boom"}, err="boom", status="failed"), after_fn=bind("ABSENT"))
st, _ = driver(h, ex).execute_recovery_read(READ, expected_evidence_unit="ServiceRequest")
ck("tool_error_unknown", st == "UNKNOWN")

# ===== P1: hard budget gate -- no budget -> no env call at all =====
h = Harness(); ex = Ex(Outcome(), after_fn=bind("ABSENT"))
st, _ = driver(h, ex, budget=lambda: False).execute_recovery_read(READ)
ck("budget_exhausted_read_no_exec", st == "UNKNOWN" and ex.env_calls == 0)

h = Harness(); ex = Ex(Outcome())
auth = {"status": "RESERVED"}
d = driver(h, ex, budget=lambda: False); d.execute({"tool": "fhir_service_request_create", "args": {}}, auth)
ck("budget_exhausted_create_no_exec", ex.env_calls == 0 and auth["status"] == "FAILED")

# ===== acquire / inspect delegate correctly =====
h = Harness(); ex = Ex(Outcome(res={"entries": []}), after_fn=bind("ABSENT"))
envc = []
d = driver(h, ex, on_env=lambda a, o: envc.append(o))
ck("acquire_resolves_and_counts", d.acquire(READ) is True and envc == ["recovery"] and h.ledger.acquire_count == 1)

h = Harness(); ex = Ex(Outcome(res={"entries": []}), after_fn=bind("ABSENT"))
d = driver(h, ex); d.episode_id = "recovery-s5-0"
ins = d.inspect_effect("ServiceRequest", "Patient/1")
ck("inspect_absent_evented_ids", ins["state"] == "ABSENT"
   and d.trajectory[-1].get("action_id") and d.trajectory[-1].get("recovery_episode_id") == "recovery-s5-0")

h = Harness(); ex = Ex(Outcome(res={"entries": [{"resource": {"id": "1", "code": {"text": "pelvic ultrasound"}}}]}),
                       after_fn=bind("PRESENT"))
ins = driver(h, ex).inspect_effect("ServiceRequest", "Patient/1")
ck("inspect_present_texts", ins["state"] == "PRESENT" and "pelvic ultrasound" in ins["texts"])

# inspect: ledger says UNKNOWN (no matching delta) -> UNKNOWN even if raw looked empty (fail-closed)
h = Harness(); ex = Ex(Outcome(res={"entries": []}), after_fn=None)   # after binds nothing
ins = driver(h, ex).inspect_effect("ServiceRequest", "Patient/1")
ck("inspect_no_ledger_delta_unknown", ins["state"] == "UNKNOWN")

# ===== create VERIFIED path preserved (#5) + crash safety =====
def after_ok_verify(h, a, o, s): h.ctx.verification = True; return None
h = Harness(); ex = Ex(Outcome(recon={"confirmed": True}), after_fn=after_ok_verify)
auth = {"status": "RESERVED"}
out = driver(h, ex).execute({"tool": "fhir_service_request_create", "args": {}}, auth)
ck("create_verified_path_preserved", auth["status"] == "VERIFIED" and out.created_id == "x1")

def crash2(h, a, o, s): raise RuntimeError("boom")
h = Harness(); ex = Ex(Outcome(recon={"confirmed": True}), after_fn=crash2)
h.ctx.verification = True
auth = {"status": "RESERVED"}
driver(h, ex).execute({"tool": "fhir_service_request_create", "args": {}}, auth)
ck("create_after_crash_unknown_not_verified", auth["status"] == "UNKNOWN")

n = sum(1 for _, c in R if c)
print("\n%d/%d run_driver C5d tests passed" % (n, len(R)))
assert n == len(R), "FAILURES"
