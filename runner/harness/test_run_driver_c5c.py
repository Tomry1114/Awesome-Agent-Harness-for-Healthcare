"""C5c RunDriver hardening: #3 read goes through before_action, #4 ledger-proven resolved, #5 after_action
safety (crash -> UNKNOWN never VERIFIED), #6 recovery env calls counted. Stub harness/executor/env."""
import sys
sys.path.insert(0, "runner")
from harness.run_driver import RunDriver

R = []
def ck(n, c): R.append((n, bool(c))); print(("OK  " if c else "FAIL") + " " + n)

class Dec:
    def __init__(s, t="ALLOW", events=None): s.type = t; s.events = events or []
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
    def __init__(s, before=Dec("ALLOW"), mode="enforce"):
        s.ledger = Ledger(); s.ctx = Ctx(); s._before = before; s.mode = mode
    def before_action(s, action, snap, step=0): return s._before
class Ex:
    """Stub executor. after_action_fn(h, action, outcome, step) drives evidence binding / verification / crash."""
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

def driver(h, ex, on_env=None): return RunDriver(h, ex, env={}, task={}, state_snapshot=lambda e: {}, on_env_action=on_env)

# ---- #3: a read the harness VETOES (BLOCK) is not executed and does not resolve ----
h = Harness(before=Dec("BLOCK")); ex = Ex(Outcome())
d = driver(h, ex)
res = d.acquire({"tool": "fhir_search", "args": {}})
ck("read_vetoed_not_executed", res is False and ex.env_calls == 0)

# ---- #4: resolved ONLY when a NEW matched record for the ACTIVE subject in PRESENT/ABSENT lands in ledger ----
def bind_present(h, a, o, s):
    h.ledger.evidence.append({"scope_relation": "matched", "subject_id": "Patient/1", "evidence_state": "ABSENT"})
    return None
h = Harness(); ex = Ex(Outcome(), after_fn=bind_present)
envc = []
d = driver(h, ex, on_env=lambda a, o: envc.append(o))
ck("resolved_when_ledger_matched_absent", d.acquire({"tool": "fhir_search", "args": {}}) is True)
ck("acquire_counts_env_6", envc == ["recovery"])   # #6

# foreign / wrong-subject record does NOT resolve
def bind_foreign(h, a, o, s):
    h.ledger.evidence.append({"scope_relation": "foreign", "subject_id": "Patient/1", "evidence_state": "PRESENT"})
    h.ledger.evidence.append({"scope_relation": "matched", "subject_id": "Patient/999", "evidence_state": "PRESENT"})
    return None
h = Harness(); ex = Ex(Outcome(), after_fn=bind_foreign)
ck("unmatched_or_wrong_subject_not_resolved", driver(h, ex).acquire({"tool": "fhir_search", "args": {}}) is False)

# read whose after_action CRASHES -> not resolved (#5)
def after_crash(h, a, o, s): raise RuntimeError("boom")
h = Harness(); ex = Ex(Outcome(), after_fn=after_crash)
ck("read_after_action_crash_unresolved", driver(h, ex).acquire({"tool": "fhir_search", "args": {}}) is False)

# ---- create path ----
# normal success: after_action returns None(=ALLOW) + verification True + readback confirmed -> VERIFIED (preserves cp3) ----
def after_ok_verify(h, a, o, s): h.ctx.verification = True; return None
h = Harness(); ex = Ex(Outcome(recon={"confirmed": True}), after_fn=after_ok_verify)
envc = []
d = driver(h, ex, on_env=lambda a, o: envc.append(o))
auth = {"status": "RESERVED"}
out = d.execute({"tool": "fhir_service_request_create", "args": {}}, auth)
ck("create_verified_path_preserved", auth["status"] == "VERIFIED" and out.created_id == "x1" and envc == ["recovery"])

# after_action CRASH on create -> UNKNOWN, NEVER VERIFIED (#5), even if a stale verification=True lingered
def after_crash2(h, a, o, s): raise RuntimeError("boom")
h = Harness(); ex = Ex(Outcome(recon={"confirmed": True}), after_fn=after_crash2)
h.ctx.verification = True    # stale value that MUST NOT leak into VERIFIED
auth = {"status": "RESERVED"}
d = driver(h, ex); d.execute({"tool": "fhir_service_request_create", "args": {}}, auth)
ck("create_after_crash_unknown_not_verified", auth["status"] == "UNKNOWN")

# verification reset before after_action: after_fn that does NOT set verification -> stays None -> not VERIFIED
def after_no_verify(h, a, o, s): return None
h = Harness(); ex = Ex(Outcome(recon={"confirmed": True}), after_fn=after_no_verify)
h.ctx.verification = True    # stale True from a prior action
auth = {"status": "RESERVED"}
driver(h, ex).execute({"tool": "fhir_service_request_create", "args": {}}, auth)
ck("stale_verification_reset_no_false_verify", auth["status"] != "VERIFIED")

# ---- #7: inspect_effect routes the existing-effect probe through the executor (counted #6), classifies,
#      and stamps a #9 action_id. ----
h = Harness(); ex = Ex(Outcome(res={"entries": []}))
envc = []
d = driver(h, ex, on_env=lambda a, o: envc.append(o))
d.episode_id = "recovery-s5-0"
ins = d.inspect_effect("ServiceRequest", "Patient/1")
ck("inspect_absent_counted_evented", ins["state"] == "ABSENT" and envc == ["recovery"]
   and d.trajectory[-1].get("action_id") and d.trajectory[-1].get("recovery_episode_id") == "recovery-s5-0")

h = Harness(); ex = Ex(Outcome(res={"entries": [{"resource": {"id": "1", "code": {"text": "pelvic ultrasound"}}}]}))
ins = driver(h, ex).inspect_effect("ServiceRequest", "Patient/1")
ck("inspect_present_texts", ins["state"] == "PRESENT" and "pelvic ultrasound" in ins["texts"])

h = Harness(); ex = Ex(Outcome(res={"error": "boom"}, err="boom", status="failed"))
ins = driver(h, ex).inspect_effect("ServiceRequest", "Patient/1")
ck("inspect_failed_is_unknown_never_absent", ins["state"] == "UNKNOWN")

n = sum(1 for _, c in R if c)
print("\n%d/%d run_driver C5c tests passed" % (n, len(R)))
assert n == len(R), "FAILURES"
