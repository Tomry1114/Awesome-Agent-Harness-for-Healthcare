"""HealthAdminBench write verification: the shell's write receipt and the workspace's write-verify middleware.

The run.py consequences of write_misroute are pinned by runner/tests/test_inline_hab_gates.py. Here: the misroute
reconciler -- a value that LANDED in a control whose semantics do not match the agent's stated field_intent, while a
required-empty control that DOES match exists, is a misroute -> point the agent at the right control (no value, no
gold); and the middleware's own guards.
"""
import os
import sys

_RUNNER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RUNNER not in sys.path:
    sys.path.insert(0, _RUNNER)

import middleware_host as MH                                            # noqa: E402
from workspace import H0, Workspace                                      # noqa: E402

WriteVerify = Workspace(H0["gui"]).load("middleware/write_verify.py", "WriteVerify")
WV = sys.modules[WriteVerify.__module__]
find_misroute, semantic_field_match, _field_concept = WV.find_misroute, WV.semantic_field_match, WV._field_concept

CTRLS = [{"ref": "member-id-input", "label": "Member ID", "required": True, "value": "", "visible": True, "role": "input"},
         {"ref": "patient-search-input", "label": "Patient Search", "value": "AET1", "visible": True, "role": "input"},
         {"ref": "dob-input", "label": "Date of Birth", "required": True, "value": "", "visible": True, "role": "input"}]

def test_concept_and_match():
    assert _field_concept("Member ID") == "member_id"
    assert _field_concept("Patient Search") == "patient_search"
    assert semantic_field_match("Member ID", "Member Number") is True
    assert semantic_field_match("Member ID", "Patient Search") is False

def test_misroute_points_to_correct_empty_control():
    m = find_misroute("Member ID", "Patient Search", CTRLS)
    assert m and m["ref"] == "member-id-input"
    m2 = find_misroute("Date of Birth", "Patient Search", CTRLS)
    assert m2 and m2["ref"] == "dob-input"

def test_no_misroute_when_target_matches_intent():
    assert find_misroute("Member ID", "Member Number", CTRLS) is None

def test_no_misroute_when_intent_unknown_or_no_candidate():
    assert find_misroute("Some Freeform Field", "Patient Search", CTRLS) is None   # unknown intent -> silent
    filled = [{"ref": "member-id-input", "label": "Member ID", "required": True, "value": "X", "visible": True, "role": "input"}]
    assert find_misroute("Member ID", "Patient Search", filled) is None            # matching control already filled


def test_misroute_requires_required_or_unique_candidate():
    """A misroute must not steer the agent to an OPTIONAL same-name field when the match is ambiguous."""
    opt = [{"ref": "a", "label": "Member ID", "required": False, "value": "", "visible": True, "role": "input"},
           {"ref": "b", "label": "Member Number", "required": False, "value": "", "visible": True, "role": "input"}]
    assert find_misroute("Member ID", "Patient Search", opt) is None            # ambiguous optionals -> silent
    mix = [{"ref": "req", "label": "Member ID", "required": True, "value": "", "visible": True, "role": "input"},
           {"ref": "opt", "label": "Member Number", "required": False, "value": "", "visible": True, "role": "input"}]
    assert find_misroute("Member ID", "Patient Search", mix)["ref"] == "req"     # prefer required
    one = [{"ref": "only", "label": "Member ID", "required": False, "value": "", "visible": True, "role": "input"}]
    assert find_misroute("Member ID", "Patient Search", one)["ref"] == "only"    # unique optional ok


def test_section_aware_same_name_field_routing():
    """Member ID typed into the SEARCH section (a control legitimately labeled Member ID) while the
    authorization PAYLOAD member-id field (same label, different section) is empty -> misroute toward the
    payload field, even though the landed label matches the intent."""
    ctrls = [{"ref": "search-mid", "label": "Member ID", "section": "Patient Search", "required": False,
              "value": "AET1", "visible": True, "role": "input"},
             {"ref": "form-mid", "label": "Member ID", "section": "Authorization Request", "required": True,
              "value": "", "visible": True, "role": "input"}]
    m = find_misroute("Member ID", "Member ID", ctrls, target_section="Patient Search")
    assert m and m["ref"] == "form-mid"                       # points at the payload field
    # if the landed section is NOT a lookup section and label matches -> not a misroute (silent)
    assert find_misroute("Member ID", "Member ID", ctrls, target_section="Authorization Request") is None



def test_write_landed_custom_dropdown_from_trigger_text():
    """A custom dropdown (button) has no form .value; READBACK_JS reads the trigger's DISPLAYED text as the value.
    write_landed must then confirm a landed selection and reject an unchanged placeholder -- so write-verify does not
    false-flag a landed custom-dropdown pick."""
    from adapters.hab.gui_control_model import write_landed
    assert write_landed("Outpatient", {"found": True, "value": "Outpatient", "selected_text": None}) is True
    assert write_landed("Outpatient", {"found": True, "value": "All", "selected_text": None}) is False
    assert write_landed("Outpatient", {"found": True, "value": "", "selected_text": None}) is False


# ------------------------------------------------------------------------------------------------ the middleware
class _CtrlEnv(object):
    def list_controls(self):
        return [{"role": "button", "label": "Submit", "ref": "sub", "visible": True}]


def _ctx(traj, spent=False, step=3, env=None):
    return MH.MiddlewareContext(env=env if env is not None else _CtrlEnv(), task={}, trajectory=traj, step=step, max_steps=10, used_actions=0,
                                judge_factory=lambda: None, errors=[],
                                shell={"budget_spent_at_turn_start": lambda: spent})


NOT_LANDED = {"write_receipt": {"landed": False, "ref": "r1", "field": "Diagnosis"}}
TYPE = {"tool": "type", "args": {"field_intent": "dx"}}


def test_not_landed_emits_its_event_and_unlocks_repair():
    traj = []
    out = WriteVerify({}).after_tool(_ctx(traj), TYPE, NOT_LANDED)
    assert isinstance(out, MH.PostRevision) and (out.stage, out.obs) == ("write_verify", "write_not_landed")
    assert traj == [{"step": 3, "event_type": "write_not_landed", "round": 1, "ref": "r1", "field": "Diagnosis",
                     "attempted": None, "observed": None, "status": "ok"}]


def test_silent_when_budget_spent_no_receipt_or_a_harness_read():
    mw, traj = WriteVerify({}), []
    assert mw.after_tool(_ctx(traj, spent=True), TYPE, NOT_LANDED) is None
    assert mw.after_tool(_ctx(traj), TYPE, {}) is None
    assert mw.after_tool(_ctx(traj), dict(TYPE, _harness_acquire=True), NOT_LANDED) is None
    assert traj == [] and mw.rounds == 0


def test_not_landed_and_misroute_share_one_bound():
    mw, traj = WriteVerify({}), []
    misrouted = {"write_receipt": {"landed": True, "ref": "member-id-input", "control_label": "Member ID"}}

    class _Form(object):
        def list_controls(self):
            return [{"ref": "dx-input", "label": "Diagnosis Code", "required": True, "value": "", "visible": True,
                     "role": "input"}]
    ctx = _ctx(traj, env=_Form())
    fill = {"tool": "fill", "args": {"ref": "member-id-input", "field_intent": "diagnosis code"}}
    stages = [getattr(mw.after_tool(ctx, *call), "stage", None)
              for call in [(TYPE, NOT_LANDED), (fill, misrouted)] * 3]
    assert stages == ["write_verify", "misroute", "write_verify", "misroute", None, None]
    assert [e["round"] for e in traj] == [1, 2, 3, 4]
