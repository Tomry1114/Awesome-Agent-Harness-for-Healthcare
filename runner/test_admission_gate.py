import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness.repair import enforceable, RepairFinding, RepairOperation, make_finding_id
def _f(defect):
    return RepairFinding(make_finding_id("t","r","claim","p",defect),"r","claim","p",defect,RepairOperation.ADD,"x")
def test_deterministic_defects_enforced():
    assert enforceable(_f("missing")) is True
    assert enforceable(_f("unobserved_target")) is True
def test_semantic_defects_advisory():
    for d in ("unsupported_by_observation","insufficient_content","untraceable_claim","conflicting"):
        assert enforceable(_f(d)) is False, d
def test_enforceable_accepts_string():
    assert enforceable("missing") is True and enforceable("insufficient_content") is False
if __name__=="__main__":
    import traceback; f=0
    for n,fn in sorted(globals().items()):
        if n.startswith("test_") and callable(fn):
            try: fn(); print("PASS",n)
            except Exception: f+=1; print("FAIL",n); traceback.print_exc()
    print("admission_gate:", "ALL PASS" if not f else "%d FAIL"%f); sys.exit(1 if f else 0)
