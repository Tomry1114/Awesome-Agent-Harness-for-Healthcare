"""Commit C1: MutationAuthorization lifecycle state machine."""
import sys
sys.path.insert(0, "runner")
from harness.state import Ledger
from harness.authorization import (exact_scope_match, AUTH_AVAILABLE, AUTH_RESERVED, AUTH_DISPATCHED,
                                   AUTH_VERIFIED, AUTH_FAILED, AUTH_UNKNOWN, AUTH_CANCELLED)

R = []
def ck(n, c): R.append((n, bool(c))); print(("OK  " if c else "FAIL") + " " + n)

class Sem:
    def __init__(self, st="create", eff="irreversible", res="ServiceRequest", ent="Patient/1"):
        self.semantic_type = st; self.effect = eff; self.resource = res; self.target_entity = ent

led = Ledger() if hasattr(Ledger, "__init__") else None
# Ledger may need args; construct minimally
try:
    led = Ledger()
except Exception:
    import inspect
    led = Ledger.__new__(Ledger); Ledger.__init__(led)

action = {"tool": "fhir_create", "args": {"resource": {"resourceType": "ServiceRequest"}}}
sem = Sem()

auth = led.mint_authorization(source="deterministic_gap", allowed_semantic_type="create",
                              allowed_tool="fhir_create", allowed_effect="irreversible",
                              target_path="ServiceRequest/Patient/1",
                              expected_postcondition={"resource": "ServiceRequest"})
ck("mint_available", auth.status == AUTH_AVAILABLE and auth.matchable and not auth.consumed)

# exact_scope_match is PURE scope (no status); matched (same scope)
ck("scope_match_pure", exact_scope_match(auth, sem, action) is True)

# find_matching returns it while AVAILABLE
ck("find_when_available", led.find_matching_authorization(sem, action) is auth)

# reserve -> RESERVED -> no longer matchable/found
ck("reserve_ok", led.reserve_authorization(auth) is True and auth.status == AUTH_RESERVED)
ck("not_matchable_reserved", auth.matchable is False and auth.consumed is True)
ck("not_found_reserved", led.find_matching_authorization(sem, action) is None)
ck("scope_still_pure", exact_scope_match(auth, sem, action) is True)   # scope unchanged; only matchability gates

# release -> back AVAILABLE (combine was NOT ALLOW)
led.release_authorization(auth)
ck("release_back_available", auth.status == AUTH_AVAILABLE and led.find_matching_authorization(sem, action) is auth)

# reserve again -> dispatch -> DISPATCHED, spent forever
led.reserve_authorization(auth); led.dispatch_authorization(auth)
ck("dispatched", auth.status == AUTH_DISPATCHED and not auth.matchable)
led.release_authorization(auth)   # release must NOT rescue a DISPATCHED auth
ck("dispatch_irreversible", auth.status == AUTH_DISPATCHED)

# terminal transitions
led.verify_authorization(auth); ck("verify", auth.status == AUTH_VERIFIED)
auth2 = led.mint_authorization(source="deterministic_gap", allowed_semantic_type="create", allowed_tool="fhir_create",
                               allowed_effect="irreversible", target_path="X", expected_postcondition={"x": 1})
led.reserve_authorization(auth2); led.dispatch_authorization(auth2); led.unknown_authorization(auth2)
ck("unknown", auth2.status == AUTH_UNKNOWN and not auth2.matchable)

auth3 = led.mint_authorization(source="deterministic_gap", allowed_semantic_type="create", allowed_tool="fhir_create",
                               allowed_effect="irreversible", target_path="Y", expected_postcondition={"y": 1})
led.reserve_authorization(auth3); led.dispatch_authorization(auth3); led.fail_authorization(auth3)
ck("failed", auth3.status == AUTH_FAILED)

# consume_authorization back-compat == DISPATCHED (spent)
auth4 = led.mint_authorization(source="deterministic_gap", allowed_semantic_type="create", allowed_tool="fhir_create",
                               allowed_effect="irreversible", target_path="Z", expected_postcondition={"z": 1})
led.consume_authorization(auth4)
ck("consume_compat_dispatched", auth4.status == AUTH_DISPATCHED and auth4.consumed is True)

n = sum(1 for _, c in R if c)
print("\n%d/%d auth state-machine tests passed" % (n, len(R)))
assert n == len(R), "FAILURES"
