"""HAB eval_state native parity: our GuiEnvReal.eval_state() must reproduce the native get_final_state()
shape so every native JMESPath query resolves -- signals.* (223 prior-auth checkpoints), payer_*_state.*,
aetna/anthem_state.*, full_state.faxPortal.*. Oracle-blind, no browser."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root, for `from runner import ...`

import jmespath
from runner import environments as E


def _mk_env(portals):
    env = E.GuiEnvReal.__new__(E.GuiEnvReal)
    env.full_state = portals.get("emr") or {}
    env._portals = portals
    env._read_state = lambda: env.full_state
    return env


PORTALS = {
    "emr": {"agentActions": {"clickedDiagnosesTab": True, "clickedServicesTab": True,
                             "clickedCoveragesTab": True, "downloadedAuthLetter": True,
                             "viewedAuthLetter": True, "clickedGoToPortal": True}},
    "payerA": {"submissions": [{"patientDOB": "1970-01-02"}]},
    "payerB": {},
    "fax": {"faxesSent": 2},
}


def test_eval_state_has_full_native_shape():
    es = _mk_env(PORTALS).eval_state()
    assert set(es) >= {"signals", "actions", "actions_history", "full_state",
                       "payer_a_state", "payer_b_state", "aetna_state", "anthem_state"}


def test_signals_match_native_semantics():
    es = _mk_env(PORTALS).eval_state()
    s = es["signals"]
    assert s["clicked_diagnoses_tab"] is True
    assert s["clicked_services_tab"] is True
    assert s["downloaded_auth_letter"] is True
    assert s["submitted"] is False           # no submitted/submittedAppeal flag set


def test_native_signals_and_payer_queries_resolve():
    es = _mk_env(PORTALS).eval_state()
    for q, exp in [
        ("signals.clicked_diagnoses_tab", True),
        ("signals.downloaded_auth_letter", True),
        ("signals.clicked_referral_tab", False),
        ("full_state.faxPortal.faxesSent", 2),
        ("aetna_state.differences.priorAuth.added[0].patientDOB", "1970-01-02"),
    ]:
        assert jmespath.search(q, es) == exp, (q, jmespath.search(q, es))
