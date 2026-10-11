"""HealthAdminBench completion conjunction: a completion conjunct comes from the public goal or from the agent's own
decision. The native-checkpoint family (tab clicks, the auth-letter view and download, the go-to-portal hop) is never
composed, the task type does not change the composition, an outward commit is composed only from the agent's own
attempt, and every conjunct is an existence check -- never a gold value, never note content."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))   # runner/
from harness.obligation_gate import completion_conjunction
from workspace import H0

_HAB = H0["gui"]   # the workspace whose completion content is composed

_EVERY_PHRASE = ("Open denial DEN-001, check the remittance image, submit a prior authorization request, record the "
                 "authorization number in EMR, download the required documents, fax them to the DME supplier, attach "
                 "them, add a progress note, document it in a triage note and clear the referral from the worklist")
_COMMITS = ["submit-disposition-button", "submit-appeal-button", "submit-auth-button", "send-fax-button",
            "clear-from-worklist-button"]
_ALL_DECISIONS = [{"event_type": "tool_call", "origin": "agent", "tool": "click", "args": {"ref": r}} for r in _COMMITS]
_TYPES = ("denial_triage", "dme_fax", "submit_auth_aetna", "electronic_appeals", "fax_appeal", "no_auth_medicare", "")


def _leaves(tt, goal=_EVERY_PHRASE, actions=_ALL_DECISIONS):
    conj = completion_conjunction({"goal": goal, "environment": {"type": "gui", "config": {"task_type": tt}}},
                                  _HAB, actions=actions)[1]
    return [sg["path"].split(".")[-1] for sg in conj]


def test_the_native_family_coverage_is_never_composed():
    for tt in _TYPES:
        leaves = _leaves(tt)
        for f in ("clicked_diagnoses_tab", "clicked_services_tab", "clicked_coverages_tab", "viewedAuthLetter",
                  "downloaded_auth_letter", "clicked_go_to_portal"):
            assert f not in leaves, (tt, f)
        assert len(leaves) == 12, leaves          # the 5 effects of the agent's decisions + the 7 goal requirements


def test_the_task_type_does_not_change_the_composition():
    for goal, actions in ((_EVERY_PHRASE, _ALL_DECISIONS), (_EVERY_PHRASE, []), ("review the chart", _ALL_DECISIONS)):
        assert len(set(tuple(_leaves(tt, goal, actions)) for tt in _TYPES)) == 1


def test_an_outward_commit_is_composed_only_from_the_agents_own_attempt():
    leaves = _leaves("dme_fax", actions=[])
    for commit in ("selectedDisposition", "submittedAppeal", "submissions", "faxesSent", "clearedReferrals",
                   "addedAuthNote", "attachmentNames", "faxRecipient"):     # the last three need the agent's commit too
        assert commit not in leaves, commit
    assert leaves == ["viewedDenialDetails", "viewedRemittanceImage", "documentedAppealInEpic", "addedProgressNote"]


def test_no_gold_value_in_any_subgoal():
    conj = completion_conjunction({"goal": _EVERY_PHRASE, "environment": {"config": {}}}, _HAB,
                                  actions=_ALL_DECISIONS)[1]
    for sg in conj:
        assert sg.get("check") in ("truthy", "nonempty"), sg          # existence only
        assert "expected" not in sg and "contains" not in sg and "equals" not in sg and "value" not in sg, sg
