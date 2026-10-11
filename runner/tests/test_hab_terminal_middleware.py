"""The HealthAdminBench workspace's middleware at a terminal commit: the pre-commit audit, the attachment check, the
collection nudge.

What the loop does with their results (the refused action never runs; which refusals unlock repair; bounds; order)
is pinned through run.run_task by runner/tests/test_inline_hab_gates.py. Here: each middleware's own rules.
"""
import os
import sys

_RUNNER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RUNNER not in sys.path:
    sys.path.insert(0, _RUNNER)

import middleware_host as MH                                            # noqa: E402
from workspace import H0, Workspace                                      # noqa: E402

_WS = Workspace(H0["gui"])
PrecommitAudit = _WS.load("middleware/precommit.py", "PrecommitAudit")
AttachmentBlock = _WS.load("middleware/attachment.py", "AttachmentBlock")
CollectionNudge = _WS.load("middleware/collection_nudge.py", "CollectionNudge")
ATT = sys.modules[AttachmentBlock.__module__]
COLL = sys.modules[CollectionNudge.__module__]

SUBMIT = {"tool": "submit", "args": {"ref": "sub"}}


def _ctx(env, traj, step=5, observed=()):
    return MH.MiddlewareContext(env=env, task={}, trajectory=traj, step=step, max_steps=10, used_actions=0,
                                judge_factory=lambda: None, errors=[],
                                shell={"observed_history": lambda: list(observed)})


class _Env(object):
    def __init__(self, controls=(), state=None, downloads=None):
        self._controls, self._state = list(controls), state
        if downloads is not None:
            self._download_registry = downloads

    def list_controls(self):
        return list(self._controls)

    def eval_state(self):
        return self._state if self._state is not None else {}


# ------------------------------------------------------------------------------------------------ pre-commit audit
FORM = [{"role": "button", "label": "Submit", "ref": "sub", "visible": True},
        {"label": "MRN*", "required": True},
        {"label": "Diagnosis", "value": "E11.9"},                 # has a value -> complete
        {"label": "Referral", "required": True, "value": ""}]     # required + empty


def test_precommit_names_only_required_empty_fields_then_refuses_without_a_bound():
    mw, traj = PrecommitAudit({}), []
    outs = [mw.before_tool(_ctx(_Env(FORM), traj, step=s), SUBMIT) for s in range(5)]
    assert [(o.stage, o.obs) for o in outs] == [("precommit", "precommit_block")] * 3 + \
        [("precommit_terminal", "precommit_terminal")] * 2
    assert all(isinstance(o, MH.Revision) for o in outs)
    assert "MRN*, Referral." in outs[0].feedback and "Diagnosis" not in outs[0].feedback
    assert traj[0] == {"step": 0, "event_type": "precommit_audit", "round": 1, "blocked_tool": "submit",
                       "required_empty": ["MRN*", "Referral"], "status": "ok"}
    assert [e["event_type"] for e in traj] == ["precommit_audit"] * 3 + ["precommit_terminal"] * 2
    assert "round" not in traj[3] and mw.rounds == 3


def test_precommit_silent_on_a_complete_form_a_non_terminal_action_or_a_harness_read():
    mw, traj = PrecommitAudit({}), []
    assert mw.before_tool(_ctx(_Env([{"label": "x", "value": "y"}]), traj), SUBMIT) is None
    assert mw.before_tool(_ctx(_Env(FORM), traj), {"tool": "type", "args": {"ref": "sub"}}) is None
    assert mw.before_tool(_ctx(_Env(FORM), traj), dict(SUBMIT, _harness_acquire=True)) is None
    assert traj == [] and mw.rounds == 0


# ------------------------------------------------------------------------------------------------ attachment check
def _fax(attached):
    return _Env([{"role": "button", "label": "Send Fax", "ref": "send-fax-button", "visible": True}],
                state={"full_state": {"faxPortal": {"attachmentNames": list(attached)}}},
                downloads={"1": {"path": "/tmp/referral.pdf"}})


SEND = {"tool": "click", "args": {"ref": "send-fax-button"}}


def test_send_with_downloads_but_no_attachments_is_blocked():
    assert ATT.send_readiness_gap(["Rx.pdf", "F2F.pdf"], []) is True       # downloaded, attached none -> block
    assert ATT.send_readiness_gap(["Rx.pdf"], ["Rx.pdf"]) is False         # something attached -> ready
    assert ATT.send_readiness_gap([], []) is False                         # nothing downloaded -> not our concern
    assert ATT.send_readiness_gap([], ["X.pdf"]) is False


def test_feedback_never_names_a_specific_document():
    fb = ATT.feedback()
    assert "attach" in fb.lower()
    for doc in ("Face_to_Face", "Prescription", "History_and_Physical"):
        assert doc not in fb                                               # never forces a specific / wrong doc


def test_attachment_nudges_twice_then_lets_the_send_through():
    mw, traj = AttachmentBlock({}), []
    outs = [mw.before_tool(_ctx(_fax([]), traj, step=s), SEND) for s in range(3)]
    assert [(o.stage, o.obs, o.label) for o in outs[:2]] == [("attachment", "attachment_block", "REVISE")] * 2
    assert all(isinstance(o, MH.Nudge) for o in outs[:2]) and outs[2] is None
    assert traj == [{"event_type": "attachment_block", "status": "ok", "step": 0, "round": 1},
                    {"event_type": "attachment_block", "status": "ok", "step": 1, "round": 2}]


def test_attachment_silent_with_something_attached_or_no_fax_portal():
    traj = []
    assert AttachmentBlock({}).before_tool(_ctx(_fax(["referral.pdf"]), traj), SEND) is None
    no_portal = _Env([{"role": "button", "label": "Submit", "ref": "sub", "visible": True}], state={},
                     downloads={"1": {"path": "/tmp/referral.pdf"}})
    assert AttachmentBlock({}).before_tool(_ctx(no_portal, traj), SUBMIT) is None
    assert traj == []


# ------------------------------------------------------------------------------------------------ collection nudge
def test_coverage_nudge_fires_when_observed_exceeds_form():
    obs = "Clinical note: diagnoses H35.32, H35.31, E11.9. Procedures 67028, J2778."
    assert COLL.observed_codes(obs, "diagnosis") == {"H35.32", "H35.31", "E11.9"}
    # form has 1 diagnosis, agent saw 3 -> nudge (no specific code named)
    fb = COLL.coverage_nudge(obs, ["H35.32"], "diagnosis")
    assert fb is not None and "go back" in fb.lower() and "do not invent" in fb.lower()
    for code in ("H35.31", "E11.9"):
        assert code not in fb                                          # never supplies the missing code
    # form covers what was seen -> no nudge
    assert COLL.coverage_nudge(obs, ["H35.32", "H35.31", "E11.9"], "diagnosis") is None
    # nothing observed -> no nudge
    assert COLL.coverage_nudge("no codes here", [], "diagnosis") is None


def _auth_form(dx_label):
    return _Env([{"role": "textbox", "label": dx_label, "ref": "dx-input", "visible": True, "value": "H35.32"},
                 {"role": "button", "label": "Submit Authorization", "ref": "submit-auth-button", "visible": True}],
                state={"full_state": {"priorAuthForm": {"diagnosisCodes": [{"code": "H35.32"}]}}})


AUTH = {"tool": "click", "args": {"ref": "submit-auth-button"}}
SEEN = ["Referral: diagnoses H35.32 and H35.31"]


def test_collection_nudges_twice_on_a_page_with_the_field():
    mw, traj = CollectionNudge({}), []
    outs = [mw.before_tool(_ctx(_auth_form("Diagnosis Code"), traj, step=s, observed=SEEN), AUTH) for s in range(3)]
    assert [(o.stage, o.obs, o.label) for o in outs[:2]] == [("collection", "collection_nudge", "REVISE")] * 2
    assert outs[0].feedback.startswith("The diagnosis section of the authorization form has 1 code(s), but you saw 2")
    assert outs[2] is None and [e["round"] for e in traj] == [1, 2]


def test_collection_silent_without_the_field_or_on_a_harness_read():
    traj = []
    assert CollectionNudge({}).before_tool(_ctx(_auth_form("Member ID"), traj, observed=SEEN), AUTH) is None
    assert CollectionNudge({}).before_tool(_ctx(_auth_form("Diagnosis Code"), traj, observed=SEEN),
                                           dict(AUTH, _harness_acquire=True)) is None
    assert traj == []
