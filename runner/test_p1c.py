import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness.repair import RepairFinding, RepairOperation, make_finding_id
from harness.repair_delta import validate_repair
def _f(path):
    return RepairFinding(make_finding_id("t","r","field",path,"missing"),"r","field",path,"missing",RepairOperation.ADD,"x")
def test_add_resolves_by_effect_at_declared_equivalent_path():
    # judge guessed a phantom path; agent persisted to the substrate's REAL slot (a declared effect path)
    f=_f("emr.denialsWorklist[0].triageDisposition")
    before={"target":None,"protected":{},"root":{"emr":{"denialsWorklist":[{"id":"D1"}],"agentActions":{}}}}
    after={"target":None,"protected":{},"root":{"emr":{"denialsWorklist":[{"id":"D1"}],"agentActions":{"selectedDisposition":"Route to Clinical Appeals"}}}}
    eps=["emr.denialsWorklist[0].triageDisposition","emr.agentActions.selectedDisposition"]
    assert validate_repair(f, before, after, effect_paths=eps).accepted
def test_add_not_resolved_by_unrelated_write():
    # P0-2: an UNRELATED write must NOT resolve the finding (the old global leaf-count bug)
    f=_f("emr.x.y")
    before={"target":None,"protected":{},"root":{"emr":{"a":1}}}
    after={"target":None,"protected":{},"root":{"emr":{"a":1,"unrelated":"junk"}}}
    v=validate_repair(f, before, after, effect_paths=["emr.x.y"])
    assert (not v.accepted) and v.reason=="target_not_resolved", v
if __name__=="__main__":
    import traceback; fails=0
    for n,fn in sorted(globals().items()):
        if n.startswith("test_") and callable(fn):
            try: fn(); print("PASS",n)
            except Exception: fails+=1; print("FAIL",n); traceback.print_exc()
    sys.exit(1 if fails else 0)
