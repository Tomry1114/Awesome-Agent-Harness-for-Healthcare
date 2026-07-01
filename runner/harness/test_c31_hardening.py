"""C3.1 hardening: strict auth transitions, evidence-version match, resolved-units subject strictness."""
import sys
sys.path.insert(0, "runner")
from harness.state import Ledger
from harness.authorization import (AUTH_AVAILABLE, AUTH_RESERVED, AUTH_DISPATCHED, AUTH_VERIFIED,
                                   AUTH_FAILED, AUTH_UNKNOWN, AUTH_CANCELLED)
from harness.capabilities.required_context import RequiredContext

R = []
def ck(n, c): R.append((n, bool(c))); print(("OK  " if c else "FAIL") + " " + n)

def L():
    l = Ledger.__new__(Ledger); Ledger.__init__(l); return l

class Sem:
    def __init__(s): s.semantic_type="create"; s.effect="irreversible"; s.resource="ServiceRequest"; s.target_entity="Patient/1"
act = {"tool": "fhir_create", "args": {}}
def mint(l):
    return l.mint_authorization(source="deterministic_gap", allowed_semantic_type="create", allowed_tool="fhir_create",
                                allowed_effect="irreversible", target_path="ServiceRequest/Patient/1",
                                expected_postcondition={"r": 1})

# --- Fix 1: illegal transitions return False and DON'T change state ---
l = L(); a = mint(l)
ck("verify_from_available_refused", l.verify_authorization(a) is False and a.status == AUTH_AVAILABLE)
ck("dispatch_from_available_refused", l.dispatch_authorization(a) is False and a.status == AUTH_AVAILABLE)
ck("reserve_ok", l.reserve_authorization(a) is True and a.status == AUTH_RESERVED)
ck("double_reserve_refused", l.reserve_authorization(a) is False and a.status == AUTH_RESERVED)
ck("verify_from_reserved_refused", l.verify_authorization(a) is False and a.status == AUTH_RESERVED)
ck("dispatch_ok", l.dispatch_authorization(a) is True and a.status == AUTH_DISPATCHED)
ck("release_from_dispatched_refused", l.release_authorization(a) is False and a.status == AUTH_DISPATCHED)
ck("verify_ok", l.verify_authorization(a) is True and a.status == AUTH_VERIFIED)
ck("terminal_is_terminal", l.verify_authorization(a) is False and l.fail_authorization(a) is False and a.status == AUTH_VERIFIED)

# fail/unknown only from DISPATCHED
l = L(); b = mint(l); l.reserve_authorization(b)
ck("fail_from_reserved_refused", l.fail_authorization(b) is False and b.status == AUTH_RESERVED)
l.dispatch_authorization(b)
ck("fail_from_dispatched_ok", l.fail_authorization(b) is True and b.status == AUTH_FAILED)

# cancel from AVAILABLE/RESERVED
l = L(); c = mint(l)
ck("cancel_available_ok", l.cancel_authorization(c) is True and c.status == AUTH_CANCELLED)

# --- Fix 7: stale evidence_version -> find_matching returns None ---
l = L(); d = mint(l)
ck("match_when_version_current", l.find_matching_authorization(Sem(), act) is d)
l.validated_evidence_version += 1                      # new validated evidence landed AFTER the auth was minted
ck("no_match_when_version_stale", l.find_matching_authorization(Sem(), act) is None)

# --- Fix 5: resolved-units requires matched scope + exact active subject ---
class Led2:
    def __init__(self, ev): self.evidence = ev; self.acquire_count = 0
    def subject_id(self): return "MRN1"
rc = RequiredContext()
matched = [{"resource": "AllergyIntolerance", "evidence_state": "ABSENT", "scope_relation": "matched", "subject_id": "MRN1"}]
ck("matched_active_resolves", rc._resolved_units(Led2(matched)) == {"AllergyIntolerance"})
unknown_scope = [{"resource": "AllergyIntolerance", "evidence_state": "ABSENT", "scope_relation": "unknown", "subject_id": "MRN1"}]
ck("unknown_scope_not_resolved", rc._resolved_units(Led2(unknown_scope)) == set())
wrong_subj = [{"resource": "AllergyIntolerance", "evidence_state": "ABSENT", "scope_relation": "matched", "subject_id": "MRN9"}]
ck("wrong_subject_not_resolved", rc._resolved_units(Led2(wrong_subj)) == set())

# --- Fix 2b: executor refuses to execute a non-RESERVED auth ---
from harness.executor import ActionExecutor
class Env:
    def __init__(s): s.called = False
    def call_tool(s, n, a): s.called = True; return {"ok": True}
    def reconcile_write(s, *a): return {"confirmed": True}
ex = ActionExecutor(__import__("canonical_schema"), "fhir", lambda e: "h", lambda e: None)
l = L(); au = mint(l)   # AVAILABLE (never reserved)
env = Env()
out = ex.execute_and_normalize(act, env, ledger=l, auth=au)
ck("executor_refuses_unreserved", env.called is False and out.err == "authorization_not_dispatchable")
ck("unreserved_auth_untouched", au.status == AUTH_AVAILABLE)

n = sum(1 for _, c in R if c)
print("\n%d/%d C3.1 hardening tests passed" % (n, len(R)))
assert n == len(R), "FAILURES"
