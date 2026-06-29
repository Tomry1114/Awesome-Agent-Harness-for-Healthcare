"""Regression tests for the churn fixes: admissibility invariant + path-space + target-scoped dedup."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness.repair import RepairFinding, RepairOperation, make_finding_id
from harness.repair_surface import surface_for, path_space, target_sig


def _f(path, op=RepairOperation.ADD, defect="missing"):
    return RepairFinding(make_finding_id("t", "scoped_repair", "field", path, defect),
                         "scoped_repair", "field", path, defect, op, "x")


STATE = {"emr": {"agentActions": {"selectedDisposition": None},
                 "denials": [{"id": "DEN-014", "status": "open"}]}}


def test_can_localize_drops_phantom_path():
    # the verified churn cause: a hallucinated path that resolves nowhere -> inadmissible
    assert surface_for("gui").can_localize(STATE, None, _f("totally.made.up.path")) is False


def test_can_localize_admits_real_leaf():
    assert surface_for("gui").can_localize(STATE, None, _f("emr.agentActions.selectedDisposition")) is True


def test_can_localize_admits_new_child_of_real_parent():
    # the CORRECT output target (triageNote not set yet, but its container exists)
    assert surface_for("gui").can_localize(STATE, None, _f("emr.agentActions.triageNote")) is True


def test_path_space_lists_real_paths_for_judge():
    ps = path_space(STATE)
    assert "emr.agentActions.selectedDisposition" in ps
    assert "emr.denials[0].id" in ps


def test_target_sig_ignores_root_churn():
    # same target+protected, different state root -> SAME signature (so dedup is not defeated every step)
    a = {"target": "X", "protected": {"p": 1}, "root": {"big": 1}}
    b = {"target": "X", "protected": {"p": 1}, "root": {"big": 2, "more": 3}}
    assert target_sig(a) == target_sig(b)
    c = {"target": "Y", "protected": {"p": 1}, "root": {"big": 1}}
    assert target_sig(a) != target_sig(c)


if __name__ == "__main__":
    import traceback
    fails = 0
    for nm, fn in sorted(globals().items()):
        if nm.startswith("test_") and callable(fn):
            try:
                fn(); print("PASS", nm)
            except Exception:
                fails += 1; print("FAIL", nm); traceback.print_exc()
    print("admissibility:", "ALL PASS" if not fails else "%d FAILED" % fails)
    sys.exit(1 if fails else 0)
