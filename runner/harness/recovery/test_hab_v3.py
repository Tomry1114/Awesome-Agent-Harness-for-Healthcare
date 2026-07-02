"""BCR v3 - HAB generic GUI-completion unit tests. Validates GenericGuiCompletionWorkflow drives entirely
off the live control model + a verify_spec (no task-name workflows). Hard asserts (pytest)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from harness.recovery.workflows.gui_completion import GenericGuiCompletionWorkflow, GOAL_TYPE
from harness.recovery.benchmark.hab import HabBenchmarkAdapter, _VERIFY_BY_TASKTYPE
from harness.recovery import contracts as C


def _goal(verify_spec, committed=None):
    return C.CommittedGoal(goal_id="gui-0", goal_type=GOAL_TYPE,
                           committed_fields=committed or {},
                           raw={"verify_spec": verify_spec})


# a live control model: two required-empty fields (bindable) + one filled + one submit button.
_CONTROLS = [
    {"ref": 3, "role": "input", "label": "Diagnosis Code", "name": "diagnosisCodes",
     "required": True, "value": "", "options": [], "commit": False},
    {"ref": 4, "role": "input", "label": "CPT Code", "name": "cptCodes",
     "required": True, "value": "", "options": [], "commit": False},
    {"ref": 5, "role": "input", "label": "Patient DOB", "name": "patientDOB",
     "required": True, "value": "1965-01-01", "options": [], "commit": False},   # already filled
    {"ref": 9, "role": "button", "label": "Submit", "name": "", "required": False,
     "value": "", "options": [], "commit": True},
]


def test_generic_gui_fills_required_and_submits():
    wf = GenericGuiCompletionWorkflow()
    g = _goal({"path": "payer_a_state.differences.priorAuth.added", "check": "nonempty"})
    ctx = {
        "gui_controls": _CONTROLS,
        "authoritative_state": {"diagnosisCodes": "H35.32", "cptCodes": "67028"},
        "system_metadata": {}, "bound_evidence": {},
    }
    assert wf.match_goal(g, ctx) is True
    plan = wf.compile_plan(g, ctx)
    kinds = [s.kind for s in plan.steps]
    # read + 2 staged fills (diagnosis, cpt; DOB already filled -> skipped) + 1 commit
    assert kinds.count(C.STAGED_WRITE) == 2, kinds
    assert kinds.count(C.IRREVERSIBLE_COMMIT) == 1, kinds
    assert kinds[0] == C.READ, kinds
    # the fills carry the bound values directly (dynamic)
    vals = sorted(s.action.get("value") for s in plan.steps if s.kind == C.STAGED_WRITE)
    assert vals == ["67028", "H35.32"], vals


def test_generic_gui_blocks_on_unbindable_required():
    wf = GenericGuiCompletionWorkflow()
    g = _goal({"path": "payer_a_state.appealActions.submittedAppeal", "check": "truthy"})
    controls = [
        {"ref": 2, "role": "textbox", "label": "Appeal Rationale", "name": "rationale",
         "required": True, "value": "", "options": [], "commit": False},
        {"ref": 9, "role": "button", "label": "Submit", "name": "", "required": False,
         "value": "", "options": [], "commit": True},
    ]
    ctx = {"gui_controls": controls, "authoritative_state": {}, "system_metadata": {}, "bound_evidence": {}}
    plan = wf.compile_plan(g, ctx)
    # a REQUIRED field with no bindable value -> BLOCK (kernel BLOCKS on the unresolvable marker binding)
    assert plan.steps == [], plan.steps
    assert plan.required_bindings and plan.required_bindings[0].startswith("__gui_needs"), plan.required_bindings


def test_generic_gui_verify_effect():
    wf = GenericGuiCompletionWorkflow()
    g = _goal({"path": "payer_a_state.differences.priorAuth.added", "check": "nonempty"})
    # landed submission -> True
    sv_ok = {"payer_a_state": {"differences": {"priorAuth": {"added": [{"requestType": "x"}]}}}}
    assert wf.verify_effect(g, sv_ok) is True
    # empty -> False (we committed; the read-back shows nothing)
    sv_no = {"payer_a_state": {"differences": {"priorAuth": {"added": []}}}}
    assert wf.verify_effect(g, sv_no) is False
    # unreadable -> None
    assert wf.verify_effect(g, {}) is None


def test_adapter_emits_generic_goal_and_declines_unknown():
    ad = HabBenchmarkAdapter()
    # a known GUI-completion task_type + unrealized state -> one generic goal
    task = {"environment": {"config": {"task_type": "submit_auth_aetna"}},
            "authoritative_state": {}}
    ctx = ad.context(task) if hasattr(ad, "context") else {"authoritative_state": {}}
    goals = ad.resolve_commitments(task, [], "", None, {"authoritative_state": {}})
    assert len(goals) == 1 and goals[0].goal_type == GOAL_TYPE, goals
    assert goals[0].raw.get("verify_spec") == _VERIFY_BY_TASKTYPE["submit_auth_aetna"]
    # an unknown task_type -> decline (no known GUI surface)
    task2 = {"environment": {"config": {"task_type": "denial_triage"}}}
    assert ad.resolve_commitments(task2, [], "", None, {"authoritative_state": {}}) == []
    # already-realized -> decline
    live = {"authoritative_state": {"payer_a_state": {"differences": {"priorAuth": {"added": [{"x": 1}]}}}}}
    assert ad.resolve_commitments(task, [], "", None, live) == []
