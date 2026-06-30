import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness.repair import RepairFinding, RepairOperation, make_finding_id
from harness.repair_delta import validate_repair
def _f(path):
    return RepairFinding(make_finding_id("t","r","field",path,"missing"),"r","field",path,"missing",RepairOperation.ADD,"x")
def test_add_resolves_by_effect_when_agent_wrote_elsewhere():
    # judge guessed a phantom/unreachable path; agent persisted to its real schema slot instead
    f=_f("emr.denialsWorklist[0].triageDisposition")
    before={"target":None,"protected":{},"root":{"emr":{"denialsWorklist":[{"id":"D1"}],"agentActions":{}}}}
    after={"target":None,"protected":{},"root":{"emr":{"denialsWorklist":[{"id":"D1"}],"agentActions":{"selectedDisposition":"Route to Clinical Appeals"}}}}
    v=validate_repair(f, before, after)
    assert v.accepted, v   # resolved by effect -> no infinite churn on the unreachable path
def test_add_not_resolved_when_no_new_content():
    f=_f("emr.x.y")
    before={"target":None,"protected":{},"root":{"emr":{"a":1}}}
    after={"target":None,"protected":{},"root":{"emr":{"a":1}}}
    v=validate_repair(f, before, after)
    assert (not v.accepted) and v.reason=="target_not_resolved", v
if __name__=="__main__":
    import traceback; fails=0
    for n,fn in sorted(globals().items()):
        if n.startswith("test_") and callable(fn):
            try: fn(); print("PASS",n)
            except Exception: fails+=1; print("FAIL",n); traceback.print_exc()
    print("p1c:","ALL PASS" if not fails else "FAIL"); sys.exit(1 if fails else 0)
