# -*- coding: utf-8 -*-
"""Agent-vs-harness checkpoint credit on HealthAdminBench.

Real run.run_task on the HealthAdminBench stand-ins (runner/tests/inline_hab_stub.py: a portal whose localStorage
state the test controls). The deterministic checkpoints are scored by the real scorer; a rubric checkpoint's judge is
stood in (a test double that passes), since only who authored the persisted field is under test. Pinned:

  * a flag the harness's ReadAcquire click (requester next_read) set is credited harness and does not count in the
    agent-credited outcome; a flag the agent's own click set is credited agent; a note the agent typed is credited agent
  * a flag recovery clicked after the agent's final (e.g. opening the remittance-image tab on a denial task) is
    credited harness with requester recovery; a note field the harness typed, even with the agent's text, is
    harness-involved (mixed)
  * the credit tracker adds no environment call (the state after each action comes from the executor's snapshot)

Without a credit record, or with the ReadAcquire click logged as origin "recovery" without a requester, these tests
fail.
"""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inline_hab_stub as S  # noqa: E402
from inline_hab_stub import click, fill, final, control, set_action  # noqa: E402

HAB = "HealthAdminBench"
PRIOR_AUTH_TASK = "HAB-emr-medium-1"      # prior-auth: the ReadAcquire source for clickedDiagnosesTab
DENIAL_TASK = "HAB-denial-easy-1"
NOTE = "Denial CO-50: medical necessity; route to clinical appeals with the records."


def _cp(cid, query):
    return {"id": cid, "type": "deterministic", "dimension": "Outcome", "subdimension": "workflow_compliance",
            "weight": 1.0, "check": {"method": "jmespath", "query": query, "expected": True}}


def _rubric(cid, template):
    return {"id": cid, "type": "llm_judge", "dimension": "Outcome", "subdimension": "evidence_auditability",
            "weight": 1.0, "check": {"verifier": "gateway_rubric_text", "rubric": "Does the note mention CO-50?",
                                     "student_answer": template, "context": "triage note"}}


def _harness(monkeypatch, tmp_path, task_id, checkpoints):
    h = S.Harness(monkeypatch, tmp_path)
    task = S.load_hab_task(task_id)
    task["checkpoints"] = checkpoints
    h.use_task(task)
    import scoring

    def run_checkpoint(cp, ctx):         # deterministic: the real scorer; rubric: a passing judge double
        base = {"id": cp["id"], "dimension": cp["dimension"], "subdimension": cp.get("subdimension"),
                "weight": float(cp.get("weight", 1.0))}
        if cp["type"] == "deterministic":
            return dict(scoring._ev_deterministic(cp, ctx, base), evaluator_type="deterministic")
        return dict(base, checkpoint_status="passed", pass_status="passed", score=1.0, score_eligible=True,
                    evaluator_kind="gateway_rubric_text", evaluator_type="llm_judge")
    monkeypatch.setattr(scoring, "run_checkpoint", run_checkpoint)
    return h


def _page_reads_inside_the_tracker(h):
    """Every page.evaluate and page.title call made while the credit tracker runs (constructor, observe, finalize);
    the tracker reads only the executor's snapshot (state_summary: cached portals and the local page.url), so this
    stays empty. Pass-throughs."""
    import credit
    inside, page_calls, total = [False], [], []
    for name in ("evaluate", "title"):
        orig = getattr(S.FakePage, name)

        def wrapped(page, *a, _orig=orig, _name=name, **k):
            total.append(_name)
            if inside[0]:
                page_calls.append(_name)
            return _orig(page, *a, **k)
        h.mp.setattr(S.FakePage, name, wrapped)
    for name in ("__init__", "observe", "finalize"):
        orig = getattr(credit.CheckpointCreditTracker, name)

        def traced(self, *a, _orig=orig, **k):
            inside[0] = True
            try:
                return _orig(self, *a, **k)
            finally:
                inside[0] = False
        h.mp.setattr(credit.CheckpointCreditTracker, name, traced)
    return page_calls, total


def _note_field(portal, key="triageNotes"):
    def on_fill(p, ref, text):
        p.emr().setdefault(key, []).append(text)
    return on_fill


def test_harness_read_click_flag_is_harness_credited_agent_flag_and_note_are_agent_credited(monkeypatch, tmp_path):
    h = _harness(monkeypatch, tmp_path, PRIOR_AUTH_TASK, [
        _cp("cp_agent_flag", "full_state.agentActions.viewedAuthLetter"),
        _cp("cp_harness_flag", "full_state.agentActions.clickedDiagnosesTab"),
        _rubric("cp_note", "{{full_state.triageNotes}}")])
    p = h.portal
    S.probe_tab(p)                                                  # ReadAcquire's tab: its click sets the flag
    p.controls.append(control("auth-letter-tab", label="Auth Letter"))
    p.controls.append(control("triage-note", role="textbox", label="Triage Note"))
    p.on_click["auth-letter-tab"] = set_action("viewedAuthLetter", text="Auth letter open")
    p.on_fill["triage-note"] = _note_field(p)
    h.stub_recovery([])                                              # no recovery actions in this run
    page_calls, all_page_calls = _page_reads_inside_the_tracker(h)
    h.run([S.reveal_then(p, click("auth-letter-tab")), fill("triage-note", NOTE), final("done"), final("done")],
          max_steps=12)
    assert h.net == [] and p.unscripted == []

    calls = [(e["tool"], e["args"].get("ref"), e["origin"], e.get("requester")) for e in h.events("tool_call")]
    assert calls[:3] == [("click", "auth-letter-tab", "agent", None),
                         ("click", S.PROBE_TAB, "harness", "next_read"),
                         ("fill", "triage-note", "agent", None)]
    cr = h.result["credit"]
    assert "error" not in cr, cr
    cps = cr["checkpoints"]
    assert (cps["cp_harness_flag"]["credit"], cps["cp_harness_flag"]["became_true_after"]["requester"]) == \
        ("harness", "next_read")
    assert (cps["cp_agent_flag"]["credit"], cps["cp_agent_flag"]["became_true_after"]["origin"]) == ("agent", "agent")
    assert cps["cp_note"]["credit"] == "agent" and cps["cp_note"]["leaves"][0]["typed_by"] == ["agent"]
    assert "full_state.agentActions.clickedDiagnosesTab" in [f["field"] for f in cr["harness_written"]["persisted_fields"]]
    assert "full_state.agentActions.viewedAuthLetter" not in [f["field"] for f in cr["harness_written"]["persisted_fields"]]

    # native: every checkpoint passes; agent-credited: the harness-credited flag does not count for the agent
    assert [c["checkpoint_status"] for c in h.result["checkpoints"]] == ["passed", "passed", "passed"]
    assert (cr["native_outcome"], cr["agent_credited_outcome"]) == (True, False)
    import credit
    assert credit.attribution(h.result, HAB)["harness"] == ["cp_harness_flag"]
    rep = credit.report_block([h.result], HAB)
    assert rep["native_success_not_agent_credited"] == [h.result["task_id"]]
    assert rep["native_passed_checkpoints_by_credit"] == {"agent": 2, "harness": 1}

    # cost: one observation per executed tool call plus the baseline and the episode end, and no page call is made
    # while the tracker runs (it reads the executor's post-action snapshot)
    assert cr["tracker"]["observations"] == len(h.events("tool_call")) + 2
    assert page_calls == [] and all_page_calls, (page_calls, len(all_page_calls))


def test_recovery_click_after_the_final_is_harness_credited_and_a_harness_typed_note_is_mixed(monkeypatch, tmp_path):
    h = _harness(monkeypatch, tmp_path, DENIAL_TASK, [
        _cp("cp_remittance", "full_state.agentActions.viewedRemittanceImage"),
        _rubric("cp_note", "{{full_state.triageNotes}}")])
    p = h.portal
    p.controls = [control("triage-note", role="textbox", label="Triage Note"),
                  control("tab-remittance", label="Remittance Image")]
    p.on_click = {"tab-remittance": set_action("viewedRemittanceImage", text="Remittance image")}
    p.on_fill = {"triage-note": _note_field(p)}
    import test_harness_action_service as AS
    AS._script(h, [[("click", {"ref": "tab-remittance"}), ("type", {"ref": "triage-note", "text": NOTE})]])
    h.run([fill("triage-note", NOTE), final("f0"), final("f1")], max_steps=12)
    assert h.net == [] and p.unscripted == []

    rec = [(e["tool"], e["origin"], e.get("requester")) for e in h.events("tool_call") if e.get("origin") == "harness"]
    assert ("click", "harness", "recovery") in rec and ("type", "harness", "recovery") in rec
    cps = h.result["credit"]["checkpoints"]
    assert cps["cp_remittance"]["credit"] == "harness"
    assert cps["cp_remittance"]["became_true_after"]["requester"] == "recovery"
    # the agent typed the note; recovery typed the agent's text into the field again: a harness-typed persisted value
    assert cps["cp_note"]["credit"] == "mixed", cps["cp_note"]
    assert (h.result["credit"]["native_outcome"], h.result["credit"]["agent_credited_outcome"]) == (True, False)
