import sys
sys.path.insert(0, "runner")
from harness.state import Ledger
from harness.capabilities.verify_commit import VerifyAndCommit
from harness.authorization import (should_set_mutation_hold, authorize_evidence_supported_plan,
                                   authorize_deterministic_gap, exact_scope_match)
from harness.risk import R2

class Sem:
    def __init__(self, st, effect="reversible", mapped=True, resource="form", target_entity="phone"):
        self.semantic_type=st; self.effect=effect; self.mapped=mapped
        self.resource=resource; self.target_entity=target_entity; self.capability="t"
class Contract:
    def __init__(self, covered=False): self._cov=covered
    def matching_commit_points(self, sem): return [object()] if self._cov else []
class Ctx:
    def __init__(self, sem, ledger, manifest=None, covered=False):
        self.sem=sem; self.ledger=ledger; self.manifest=manifest or {}; self.contract=Contract(covered)
        self.risk=R2; self.step=1

VC = VerifyAndCommit()
def ba(sem, ledger, action, manifest=None, covered=False):
    d = VC.before_action(action, Ctx(sem, ledger, manifest, covered))
    return (None, None) if d is None else (d.type, getattr(d, "rule_id", None))

def scope(tp="form/phone", tool="update_field", st="update"):
    return {"allowed_semantic_type": st, "allowed_tool": tool, "target_path": tp,
            "allowed_effect": "reversible", "expected_postcondition": {"field": tp, "is": "set"}}

results=[]
def check(name, cond):
    results.append((name, bool(cond))); print(("OK  " if cond else "FAIL")+" "+name)

# 1
L=Ledger(); L.set_mutation_hold(intervention_id="iv1")
t,r = ba(Sem("update"), L, {"tool":"update_field","args":{"target_path":"form/phone"}})
check("01 semantic_feedback_then_unscoped_mutation_is_blocked", t=="BLOCK" and r=="semantic_feedback_no_write_auth")
# 2
L=Ledger(); L.set_mutation_hold()
t,r = ba(Sem("read", effect="none"), L, {"tool":"get","args":{}})
check("02 semantic_feedback_then_read_only_action_is_allowed", t is None)
# 3
L=Ledger()
a = authorize_deterministic_gap(L, scope())
check("03 deterministic_gap_mints_scoped_authorization", a is not None and a.source=="deterministic_gap" and a.target_path=="form/phone" and len(L.mutation_authorizations)==1)
# 4
L=Ledger(); L.set_mutation_hold(); authorize_deterministic_gap(L, scope(tp="form/phone"))
t1,_ = ba(Sem("update"), L, {"tool":"update_field","args":{"target_path":"form/disposition"}})  # wrong target
t2,_ = ba(Sem("update"), L, {"tool":"update_field","args":{"target_path":"form/phone"}})         # exact target
check("04 authorization_allows_only_exact_target_path", t1=="BLOCK" and t2 is None)
# 5
L=Ledger(); L.set_mutation_hold(); authorize_deterministic_gap(L, scope(tp="form/phone"))
act={"tool":"update_field","args":{"target_path":"form/phone"}}
first,_ = ba(Sem("update"), L, act)    # verify_commit AUTHORIZES (records pending_authorization); does NOT consume (C2)
a5=L.pending_authorization; L.reserve_authorization(a5); L.dispatch_authorization(a5)   # the executor SPENDS it via strict reserve->dispatch (C3.1)
second,_ = ba(Sem("update"), L, act)   # auth now DISPATCHED (spent) -> single-use -> BLOCK
check("05 authorization_is_single_use", first is None and second=="BLOCK")
# 6 failure doesn't expand scope: after consume, no new auth minted -> next mutation blocked
L=Ledger(); L.set_mutation_hold(); authorize_deterministic_gap(L, scope(tp="form/phone"))
ba(Sem("update"), L, act); a6=L.pending_authorization; L.reserve_authorization(a6); L.dispatch_authorization(a6)   # authorize + executor spends it via reserve->dispatch
# failure path mints nothing; a broader retry on a different field stays blocked
t,_ = ba(Sem("update"), L, {"tool":"update_field","args":{"target_path":"form/other"}})
check("06 mutation_failure_does_not_expand_repair_scope", t=="BLOCK")
# 7
L=Ledger(); L.terminal_locked=True
t,r = ba(Sem("submit", effect="irreversible"), L, {"tool":"submit","args":{}})
check("07 verified_commit_locks_future_mutations", t=="BLOCK" and r=="post_verified_mutation")
# 8
L=Ledger()
a = authorize_evidence_supported_plan(L, scope(), new_evidence=[], support_passed=True)
check("08 acquire_without_new_validated_evidence_cannot_authorize_write", a is None and len(L.mutation_authorizations)==0)
# 9
L=Ledger()
a = authorize_evidence_supported_plan(L, scope(tp="Patient/123/medication"), new_evidence=[{"evidence_id":"ev-9","status":"VALIDATED"}], support_passed=True)
check("09 acquire_with_supported_plan_can_authorize_scoped_write", a is not None and a.source=="evidence_supported_plan")
# 10
L=Ledger()   # no hold
t,_ = ba(Sem("update"), L, {"tool":"update_field","args":{"target_path":"form/phone"}})
check("10 normal_agent_write_without_harness_feedback_is_unchanged", t is None)
# 11 channel-independent
check("11 external_and_inline_feedback_both_activate_mutation_hold",
      should_set_mutation_hold("REVISE", False) is True and should_set_mutation_hold("ACQUIRE", False) is True
      and should_set_mutation_hold("REVISE", True) is False)
# 12
L=Ledger(); L.set_mutation_hold()
t,r = ba(Sem("update", mapped=False), L, {"tool":"mystery","args":{}})
check("12 unknown_mutation_semantics_fail_closed", t=="ESCALATE" and r=="unmapped_action")

# 13 user_goal pass-through: covered commit point under hold -> auto-mint user_goal, allowed
L=Ledger(); L.set_mutation_hold()
t,_ = ba(Sem("submit", effect="reversible"), L, {"tool":"submit","args":{}}, covered=True)
check("13 covered_commit_under_hold_is_user_goal_authorized", t is None and any(a.source=="user_goal" for a in L.mutation_authorizations))

ok=sum(1 for _,c in results if c)
print("\n%d/%d passed"%(ok,len(results)))
sys.exit(0 if ok==len(results) else 1)
