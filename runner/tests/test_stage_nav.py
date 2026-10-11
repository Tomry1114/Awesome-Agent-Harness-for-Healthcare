"""The GENERIC stage navigator resolves the current stage from the observation and returns the transition control to
the NEXT stage -- driving a multi-step flow in order (not greedy keyword match)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from workspace import H0, Workspace
W = Workspace(H0["gui"]).module("recovery/gui_completion.py").GenericGuiCompletionWorkflow
STAGES = [
    {"name": "s1", "markers": ["worklist"], "advance": {"keywords": ["go to portal"]}},
    {"name": "s2", "markers": ["provider login", "password"], "advance": {"keywords": ["log in"]}},
    {"name": "s3", "markers": ["submit authorizations"], "advance": {"keywords": ["new authorization"]}},
]
CTRLS = [{"ref": "a", "role": "link", "label": "Go to Portal"},
         {"ref": "b", "role": "button", "label": "Log In"},
         {"ref": "c", "role": "button", "label": "New Authorization"}]

def test_resolves_current_stage_and_advances():
    # on the worklist -> advance via 'Go to Portal'
    assert W._stage_next(STAGES, "Patient worklist page", CTRLS, [])["ref"] == "a"
    # on the login page -> advance via 'Log In'
    assert W._stage_next(STAGES, "Provider login - enter password", CTRLS, [])["ref"] == "b"
    # most-advanced stage wins when several markers present
    assert W._stage_next(STAGES, "worklist ... submit authorizations dashboard", CTRLS, [])["ref"] == "c"

def test_silent_when_no_stage_or_no_control_or_repeated():
    assert W._stage_next(STAGES, "unrelated page", CTRLS, []) is None          # no stage marker
    assert W._stage_next(None, "worklist", CTRLS, []) is None                   # no map
    assert W._stage_next(STAGES, "worklist", CTRLS, ["stagenav:a"]) is None     # already tried this transition


def test_stage_nav_never_returns_a_commit_control():
    """SAFETY: even if a stage's advance keyword matches a terminal button, the engine must refuse to return a COMMIT
    control as a navigation click -- that would bypass the irreversible-commit authorization."""
    stages = [{"name": "form", "markers": ["auth form"], "advance": {"keywords": ["submit"]}}]
    assert W._stage_next(stages, "auth form open",
                         [{"ref": "s", "role": "button", "label": "Submit Request", "commit": True}], []) is None
    # commit flag unset -> the label regex still catches submit/send/transmit
    assert W._stage_next(stages, "auth form open",
                         [{"ref": "s2", "role": "button", "label": "Submit Authorization"}], []) is None
    assert W._stage_next([{"name": "fax", "markers": ["fax form"], "advance": {"keywords": ["send"]}}],
                         "fax form", [{"ref": "s3", "role": "button", "label": "Send Fax"}], []) is None

def test_route_match_defeats_sidebar_marker_pollution():
    # sidebar shows 'submit authorizations' (dashboard marker) but the ROUTE says we are on the referral page
    stages = [{"name": "referral", "route": "/emr/referral", "markers": ["referral"],
               "advance": {"keywords": ["go to portal"]}},
              {"name": "dashboard", "markers": ["submit authorizations"],
               "advance": {"keywords": ["new authorization"]}}]
    ctrls = [{"ref": "p", "role": "link", "label": "Go to Portal"},
             {"ref": "d", "role": "link", "label": "New Authorization"}]
    obs = "URL: http://x/emr/referral ... sidebar: Submit Authorizations | Worklist"
    assert W._stage_next(stages, obs, ctrls, [])["ref"] == "p"     # route pins referral -> advance via portal


def test_all_markers_requires_full_conjunction():
    stages = [{"name": "form", "all_markers": ["request type", "diagnosis code", "cpt"],
               "advance": {"keywords": ["add"]}}]
    ctrls = [{"ref": "a", "role": "button", "label": "Add"}]
    assert W._stage_next(stages, "request type only", ctrls, []) is None          # partial -> no match
    assert W._stage_next(stages, "request type / diagnosis code / cpt hcpcs", ctrls, [])["ref"] == "a"

def test_nav_submit_allowed_but_real_commit_still_refused():
    st = [{"name": "referral", "markers": ["referral detail"],
           "advance": {"keywords": ["submit to insurance", "submit to payer"]}}]
    nav = [{"ref": "submit-to-insurance-portal", "role": "button", "label": "Submit to Insurance Portal"}]
    assert W._stage_next(st, "referral detail", nav, [])["ref"] == "submit-to-insurance-portal"
    st2 = [{"name": "form", "markers": ["auth form"], "advance": {"keywords": ["submit"]}}]
    for lbl, flag in [("Submit Authorization Request", True), ("Submit Request", False), ("Send Fax", True)]:
        assert W._stage_next(st2, "auth form", [{"ref": "x", "role": "button", "label": lbl, "commit": flag}], []) is None
