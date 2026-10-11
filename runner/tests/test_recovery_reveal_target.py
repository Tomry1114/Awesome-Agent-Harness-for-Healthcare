# -*- coding: utf-8 -*-
"""An evidence-reveal click is a READ on the control the recovery step bound, never a click without a target.

The completion workflow's structured-gap acquire (an unmet viewed*/clicked* conjunct, matched to a page control by
its hints) puts the control's ref in the step's affordance target and not in its action. If the substrate read the
ref only from the action, the READ would go out as click {} and GuiEnvReal would click the first element matching
the empty text. The substrate reads the step's ref, else the affordance the kernel bound, and issues nothing without
one; the shell's READ actuator refuses a READ that names no target.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inline_hab_stub as S  # noqa: E402
from inline_hab_stub import compact, control, final, set_action  # noqa: E402

import workspace as W  # noqa: E402

DENIAL_TASK = "HAB-denial-easy-1"


def _clean(h):
    assert h.net == [], h.net
    assert h.portal.unscripted == [], h.portal.unscripted
    assert h.runtime_errors() == [], h.runtime_errors()


def test_a_reveal_click_reads_the_control_the_step_bound(monkeypatch, tmp_path):
    """denial_triage with no disposition yet: the workflow's first move is the structured-gap acquire for the unmet
    viewed_denial_details conjunct, on the page's Denial Details button."""
    h = S.Harness(monkeypatch, tmp_path)
    h.use_task(S.load_hab_task(DENIAL_TASK))
    h.portal.controls = [control("denial-details-button", label="Denial Details")]
    h.portal.on_click = {"denial-details-button": set_action("viewedDenialDetails", text="Denial details open")}
    h.run([final("done")], max_steps=12)
    _clean(h)
    reads = [compact(e) for e in h.events("harness_acquire_predispatch") if e.get("stage") == "recovery"]
    assert reads and reads[0]["step"] == 0 and reads[0]["tool"] == "click"
    assert reads[0]["args"] == {"ref": "denial-details-button"}
    assert all(r["args"].get("ref") for r in reads)
    assert ("click", "denial-details-button") in h.portal.log
    assert [x for x in h.portal.log if x[0] == "click_text"] == []
    assert h.portal.emr()["agentActions"]["viewedDenialDetails"] is True
    assert not [e for e in h.events("tool_call") if e.get("origin") == "harness" and e.get("tool") == "click"
                and not (e.get("args") or {}).get("ref")]


def test_the_substrate_never_issues_a_reveal_click_without_a_target():
    ws = W.Workspace(W.H0["gui"])
    C = ws.module("recovery/contracts.py")
    sub = ws.module("recovery/gui_substrate.py").GuiSubstrateAdapter()
    calls = []

    class Backend(object):
        def read(self, name, args):
            calls.append(("read", name, args))
            return {"ok": True}

        def call_tool(self, name, args):
            calls.append(("call_tool", name, args))
            return {"ok": True}
    sub.driver = Backend()
    bound = C.AffordanceBinding(target_spec={"label": "Denial Details"}, ref="denial-details-button")

    out = sub.execute_primitive(C.ACQUIRE, {"action": {"op": "click"}, "affordance": None}, None)
    assert (out.status, out.reason, calls) == (C.RESULT_FAILED, "reveal_without_target", [])
    sub.execute_primitive(C.ACQUIRE, {"action": {"op": "click"}, "affordance": bound}, None)
    sub.execute_primitive(C.ACQUIRE, {"action": {"op": "click", "ref": "tab-remittance"}, "affordance": bound}, None)
    assert calls == [("read", "click", {"ref": "denial-details-button"}),      # the bound affordance
                     ("read", "click", {"ref": "tab-remittance"})]             # the step's own ref first


def test_the_read_actuator_refuses_a_read_that_names_no_target(monkeypatch, tmp_path):
    h = S.Harness(monkeypatch, tmp_path)
    h.use_task(S.load_hab_task(DENIAL_TASK))
    h.portal.controls = [control("denial-details-button", label="Denial Details")]
    entry = S.recovery_entry()
    results = []

    class ScriptedRecoveryKernel(object):
        def run_all_episodes(self, bench, wf_registry, substrate, driver, task, trajectory, goal, judge, events=None):
            if not results:
                results.append(substrate.driver.read("click", {}))
                results.append(substrate.driver.read("click", {"ref": "denial-details-button"}))
            return []
    monkeypatch.setattr(entry, "RecoveryKernel", ScriptedRecoveryKernel)
    h.run([final("done")], max_steps=12)
    _clean(h)
    assert results[0] == {"error": "read_without_target"} and "error" not in results[1]
    refused = [(e["step"], e["tool"], e["stage"], e["reason"]) for e in h.events("harness_action_refused")]
    assert refused == [(0, "click", "read_actuator", "read_without_target")]
    assert [e["args"] for e in h.events("harness_acquire_predispatch")] == [{"ref": "denial-details-button"}]
    assert [x for x in h.portal.log if x[0] in ("click", "click_text")] == [("click", "denial-details-button")]
