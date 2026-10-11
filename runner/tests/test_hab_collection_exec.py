"""HealthAdminBench collection EXECUTOR -- committed-set accumulation + type->Add->re-observe loop.
Safety boundary: values come STRICTLY from the agent's own trajectory; the harness never invents a code."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adapters.hab.collection import committed_collection, next_collection_action, collection_delta


def test_committed_collection_accumulates_not_overwrites():
    traj = [{"tool": "type", "args": {"field": "diagnosisCode", "text": "E11.9"}},
            {"tool": "click", "args": {"ref": "add"}},
            {"tool": "type", "args": {"field": "diagnosisCode", "text": "I10"}},
            {"tool": "type", "args": {"field": "diagnosisCode", "text": "N18.3"}}]
    assert committed_collection(traj, "diagnosis") == ["E11.9", "I10", "N18.3"]


def test_committed_collection_dedups_order_preserving():
    traj = [{"tool": "type", "args": {"field": "dx", "text": "E11.9"}},
            {"tool": "type", "args": {"field": "diagnosis", "text": "e11.9"}},   # dup (normalized)
            {"tool": "type", "args": {"field": "diagnosis", "text": "I10"}}]
    assert committed_collection(traj, "diagnosis") == ["E11.9", "I10"]


def test_free_text_mention_not_harvested():
    # a code buried in a free-text note field is NOT a collection commitment (content-boundary safety)
    traj = [{"tool": "type", "args": {"field": "note", "text": "patient stable, consider E11.9 followup"}}]
    assert committed_collection(traj, "diagnosis") == []


def test_bare_code_typed_into_unknown_field_is_harvested():
    # a value that is ITSELF exactly a concept code counts even if the field key is unknown
    traj = [{"tool": "type", "args": {"ref": "field-7", "text": "J2778"}}]   # HCPCS
    assert committed_collection(traj, "cpt") == ["J2778"]


def test_delta_is_committed_minus_present():
    assert collection_delta(["E11.9", "I10", "N18.3"], ["E11.9"]) == ["I10", "N18.3"]
    assert collection_delta(["E11.9"], ["e11.9"]) == []   # normalized match


def test_next_action_only_returns_committed_values():
    act = next_collection_action(["E11.9", "I10"], ["E11.9"], "dx", "add")
    assert act["value"] == "I10" and act["input_ref"] == "dx" and act["add_ref"] == "add"
    # nothing left -> None (never invents)
    assert next_collection_action(["E11.9"], ["E11.9"], "dx", "add") is None


def test_executor_drives_type_then_add_loop():
    from workspace import H0, Workspace
    W = Workspace(H0["gui"]).module("recovery/gui_completion.py").GenericGuiCompletionWorkflow
    w = W()

    class _G:
        committed_fields = {}; goal_type = "gui"
        raw = {"verify_spec": {"path": "x", "check": "nonempty"},
               "verify_all": [{"path": "x", "check": "nonempty"}], "operational_commitment": True,
               "committed_collections": {"diagnosis": ["E11.9", "I10"]}}

    ctrls = [{"ref": "dx", "role": "combobox", "label": "Diagnosis Code", "value": ""},
             {"ref": "add-dx", "role": "button", "label": "Add Diagnosis"}]
    state = {"payer_a_state": {"authForm": {"diagnoses": [{"code": "E11.9"}]}}}
    d = w.next_action(_G, {}, "", ctrls, state, [])
    assert d["kind"] == "step" and d["step"].name == "collection_fill"
    assert d["step"].action["value"] == "I10"           # types the missing one, not the present one
    # value already typed -> Add it
    ctrls2 = [{"ref": "dx", "role": "combobox", "label": "Diagnosis Code", "value": "I10"},
              {"ref": "add-dx", "role": "button", "label": "Add Diagnosis"}]
    d2 = w.next_action(_G, {}, "", ctrls2, state, ["coll_fill:diagnosis:i10"])
    assert d2["kind"] == "step" and d2["step"].name == "collection_add"
    assert d2["step"].action["ref"] == "add-dx"


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    ok = 0
    for fn in fns:
        try:
            fn(); ok += 1; print("PASS", fn.__name__)
        except Exception:
            print("FAIL", fn.__name__); traceback.print_exc()
    print("%d/%d passed" % (ok, len(fns)))
