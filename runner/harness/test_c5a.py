"""C5a: kernel exposes raw winner; verify_commit denies user_goal fallback to recovery-marked mutations."""
import sys
sys.path.insert(0, "runner")
from harness.state import Ledger
from harness.capabilities.verify_commit import VerifyAndCommit
from harness import decision as D

R = []
def ck(n, c): R.append((n, bool(c))); print(("OK  " if c else "FAIL") + " " + n)

# --- verify_commit: recovery-marked mutation with NO matching auth is BLOCKED, even if a commit point covers it ---
class Sem:
    def __init__(s): s.semantic_type="create"; s.effect="irreversible"; s.resource="ServiceRequest"; s.target_entity="Patient/1"; s.mapped=True; s.capability="fhir_create"
    def is_commit(s): return True
class Contract:
    meta = {}
    def matching_commit_points(s, sem): return [{"id": "cp_order"}]   # a commit point DOES cover it
class Ctx:
    def __init__(s, led): s.ledger=led; s.sem=Sem(); s.contract=Contract(); s.risk="R2"; s.manifest={}; s.verification=None
    def __getattr__(s, k): return None

vc = VerifyAndCommit()
led = Ledger.__new__(Ledger); Ledger.__init__(led); led.set_mutation_hold()

# recovery-marked action, no matching auth -> must NOT auto-mint user_goal -> BLOCK
rec_action = {"tool": "fhir_create", "args": {"resource": {"resourceType": "ServiceRequest"}}, "_recovery": "COMPLETE_EFFECT"}
d_rec = vc.before_action(rec_action, Ctx(led))
ck("recovery_no_user_goal_fallback", d_rec is not None and d_rec.type == "BLOCK")
ck("recovery_no_auth_minted", len(led.mutation_authorizations) == 0)

# a NON-recovery agent mutation covered by a commit point STILL gets the user_goal pass-through (unchanged)
led2 = Ledger.__new__(Ledger); Ledger.__init__(led2); led2.set_mutation_hold()
agent_action = {"tool": "fhir_create", "args": {"resource": {"resourceType": "ServiceRequest"}}}
d_agent = vc.before_action(agent_action, Ctx(led2))
ck("agent_user_goal_pass_through", d_agent is None or d_agent.type == "ALLOW")
ck("agent_auth_minted", len(led2.mutation_authorizations) == 1 and led2.mutation_authorizations[0].source == "user_goal")

n = sum(1 for _, c in R if c)
print("\n%d/%d C5a tests passed" % (n, len(R)))
assert n == len(R), "FAILURES"
