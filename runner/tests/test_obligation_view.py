# -*- coding: utf-8 -*-
"""The terminal lock, ctx.obligations() and recovery evaluate a HealthAdminBench completion conjunction against ONE
state view (runner/harness/obligation_gate.py obligation_view), so they agree on truth values and not only on the
conjunct list.

The views must not diverge: on a portal where every public effect is done, recovery's raw read-back does not satisfy
the prior-auth clicked_* / downloaded_auth_letter conjuncts (the EMR stores camelCase flags; the snake_case names
exist only in eval_state's signals), and a view with no fax slice does not satisfy fax_sent / fax_attachment /
fax_recipient. The read-back's portal slices (readback_portals) give the same obligation view as the portals
themselves.
"""
import copy
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inline_hab_stub as S  # noqa: E402
from inline_hab_stub import click, control, final  # noqa: E402

import environments as E  # noqa: E402
import middleware_host as MH  # noqa: E402
import workspace as W  # noqa: E402
from harness import obligation_gate as OG  # noqa: E402

# a portal on which every public effect of every HAB task type is done
EMR = {"agentActions": {"addedAuthNote": True, "addedProgressNote": True, "documentedAppealInEpic": True,
                        "selectedDisposition": "x", "viewedDenialDetails": True, "viewedRemittanceImage": True,
                        "clickedDiagnosesTab": True, "clickedServicesTab": True, "clickedCoveragesTab": True,
                        "clickedGoToPortal": True, "viewedAuthLetter": True, "downloadedAuthLetter": True},
       "clearedReferrals": ["REF-1"]}
PAYER = {"submissions": [{"x": 1}], "appealActions": {"submittedAppeal": True}}
FAX = {"faxesSent": 1, "attachmentNames": ["a.pdf"], "faxRecipient": "555"}
PORTALS = {"emr": EMR, "payerA": PAYER, "payerB": PAYER, "fax": FAX}
READBACK = {"full_state": EMR, "payer_a_state": PAYER, "payer_b_state": PAYER, "fax": FAX, "draft_ui_state": {}}
# a goal naming every requirement and the agent's own attempt at every commit turn every conjunct on
FULL_GOAL = ("Open denial DEN-001, check the remittance image, record the authorization number in EMR, download the "
             "required documents, fax them to the supplier, add a progress note and document it in a triage note")
COMMIT_REFS = ["submit-disposition-button", "submit-appeal-button", "submit-auth-button", "send-fax-button",
               "clear-from-worklist-button"]
ALL_DECISIONS = [{"event_type": "tool_call", "origin": "agent", "tool": "click", "args": {"ref": r}} for r in COMMIT_REFS]

TASK_TYPES = [("denial_triage", FULL_GOAL), ("electronic_appeals", FULL_GOAL + " on the Payer B portal"),
              ("submit_auth_anthem", FULL_GOAL + " for Anthem"), ("dme_fax", FULL_GOAL), ("", FULL_GOAL)]


class _PortalEnv(E.GuiEnvReal):
    """GuiEnvReal over a fixed portals_state: its real eval_state, no browser."""

    def __init__(self, portals):
        self._portals = copy.deepcopy(portals)
        self.full_state = self._portals.get("emr")

    def _read_state(self):
        return self.full_state


def _recovery():
    ws = W.Workspace(W.H0["gui"])
    return ws.module("recovery/gui_completion.py"), ws.module("recovery/hab.py"), ws.module("recovery/contracts.py")


@pytest.mark.parametrize("task_type, goal", TASK_TYPES)
def test_every_reader_finds_every_conjunct_of_an_all_done_portal(task_type, goal):
    task = {"goal": goal, "environment": {"type": "gui", "config": {"task_type": task_type}}}
    dec, conj = OG.completion_conjunction(task, W.H0["gui"], actions=ALL_DECISIONS)
    assert len(conj) == 12
    done = (True,) * len(conj)
    assert OG.obligation_snapshot(task, _PortalEnv(PORTALS), W.H0["gui"], actions=ALL_DECISIONS) == done   # the terminal lock
    assert OG.gui_obligations(task, _PortalEnv(PORTALS), W.H0["gui"], actions=ALL_DECISIONS) == (True, dec, done)   # ctx.obligations()
    gui_completion, hab, C = _recovery()
    wf = gui_completion.GenericGuiCompletionWorkflow()
    assert tuple(bool(wf._check_one(c, READBACK)) for c in conj) == done                   # recovery's read-back
    goal_obj = C.CommittedGoal(goal_id="gui-0", goal_type=gui_completion.GOAL_TYPE,
                               raw={"verify_all": conj, "verify_spec": conj[0]})
    assert wf.is_realized(goal_obj, READBACK) is True
    assert wf._submit_subgoal_realized(goal_obj, READBACK) is True
    assert wf._verify_effect(goal_obj, READBACK)["unmet"] == []


def test_the_lock_view_is_the_one_obligation_view():
    assert _PortalEnv(PORTALS).eval_state() == OG.obligation_view(PORTALS)
    assert OG.obligation_view(OG.readback_portals(READBACK)) == OG.obligation_view(PORTALS)
    empty = OG.obligation_view({})
    assert empty["full_state"] == {} and not any(v for k, v in empty["signals"].items())


class _Readers(object):
    """Pass-through spies: every lock snapshot, ctx.obligations() read and recovery commitment resolution of a run."""

    def __init__(self, h):
        entry = S.recovery_entry()
        self.lock, self.nudge, self.recovery = [], [], []
        readers = self
        snapshot, obligations_of = OG.obligation_snapshot, MH.obligations_of

        def obligation_snapshot(task, env, *a, **k):
            out = snapshot(task, env, *a, **k)
            readers.lock.append(out)
            return out

        def obligations(task, env, *a, **k):
            out = obligations_of(task, env, *a, **k)
            readers.nudge.append(out)
            return out

        class ResolveOnly(entry.RecoveryKernel):
            def run_all_episodes(self, bench, wf_registry, substrate, driver, task, trajectory, goal, judge,
                                 events=None):
                ctx = self._build_ctx(bench, task, driver)
                readers.recovery.append(self._resolve(bench, task, trajectory, goal, judge, ctx))
                return []
        h.mp.setattr(OG, "obligation_snapshot", obligation_snapshot)
        h.mp.setattr(MH, "obligations_of", obligations)
        h.mp.setattr(entry, "RecoveryKernel", ResolveOnly)


@pytest.mark.parametrize("task_id", ["HAB-emr-medium-3", "HAB-emr-medium-10", "HAB-fax-easy-1", "HAB-denial-medium-8",
                                     "HAB-denial-medium-1", "HAB-denial-easy-1"])
def test_on_an_all_done_portal_the_lock_the_nudge_and_recovery_agree_it_is_complete(monkeypatch, tmp_path, task_id):
    """run.run_task on the scripted portal: the agent's own clicks on every commit control (none is on the page, so none
    is a classified commit; each is a decision), then a final. The lock sees every conjunct met, ctx.obligations() reports
    met (no nudge), and recovery, reading the same portal back, declines."""
    h = S.Harness(monkeypatch, tmp_path)
    h.use_task(S.load_hab_task(task_id))
    for slice_, value in PORTALS.items():
        h.portal.state[slice_] = copy.deepcopy(value)
    h.portal.controls = [control("tab-summary", label="Summary")]
    readers = _Readers(h)
    h.run([click(r) for r in COMMIT_REFS] + [final()], max_steps=12)
    assert h.net == [] and h.portal.unscripted == [] and h.runtime_errors() == []
    n = len(OG.completion_conjunction(h.task, W.H0["gui"], actions=ALL_DECISIONS)[1])
    assert readers.lock and all(len(s) and all(s) for s in readers.lock), readers.lock
    assert readers.lock[-1] == (True,) * n
    assert readers.nudge and all(o[0] is True and o[2] == (True,) * n for o in readers.nudge), readers.nudge
    assert h.events("obligation_nudge") == []
    assert readers.recovery and all(goals == [] for goals in readers.recovery), readers.recovery
