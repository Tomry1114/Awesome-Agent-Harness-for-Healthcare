# -*- coding: utf-8 -*-
"""The harness action service (runner/harness/action_service.py).

Every recovery-origin environment action goes through one shell path: ceilings, mode, the irreversible-commit
check, the write-provenance check, kernel before_action with the authorization ceremony, the executor,
kernel after_action, an origin=harness (requester recovery) event, and the post-action content guard. Evidence-reveal
clicks are READs through the READ actuator.

The runs are real run.run_task episodes on the HealthAdminBench stand-ins (runner/tests/inline_hab_stub.py). Only
the recovery STRATEGY is scripted: the RecoveryKernel is replaced by a script that issues primitives through the real
GUI backend, so run_recovery_v3, the service, the kernel, the executor, GuiEnvReal and run.py's budget stay real.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inline_hab_stub as S  # noqa: E402
from inline_hab_stub import click, fill, final, control, compact, set_action  # noqa: E402

DENIAL_TASK = "HAB-denial-easy-1"
NOTE = "Reviewed the denial; route to clinical appeals."


def _harness(monkeypatch, tmp_path):
    h = S.Harness(monkeypatch, tmp_path)
    h.use_task(S.load_hab_task(DENIAL_TASK))
    return h


def _clean(h):
    assert h.net == [], h.net
    assert h.portal.unscripted == [], h.portal.unscripted
    assert h.runtime_errors() == [], h.runtime_errors()


def _script(h, per_invocation):
    """Invocation n issues per_invocation[n] = [(tool, args) | ("read", tool, args), ...] through the real GUI backend
    and records what each returned. Later invocations issue nothing."""
    entry = S.recovery_entry()
    invocations, queue = [], list(per_invocation)

    class ScriptedRecoveryKernel(object):
        def run_all_episodes(self, bench, wf_registry, substrate, driver, task, trajectory, goal, judge, events=None):
            rec = {"step": driver.step, "results": []}
            invocations.append(rec)
            for op in (queue.pop(0) if queue else []):
                if op[0] == "read":
                    rec["results"].append((op[1], substrate.driver.read(op[1], dict(op[2]))))
                else:
                    rec["results"].append((op[0], substrate.driver.call_tool(op[0], dict(op[1]))))
            return []
    h.mp.setattr(entry, "RecoveryKernel", ScriptedRecoveryKernel)
    return invocations


def _at(h, step, *types):
    return [compact(e) for e in h.traj if e.get("step") == step and e.get("event_type") in types]


def _refusals(h, step):
    return [(e["tool"], e["stage"], e["reason"]) for e in h.traj
            if e.get("step") == step and e.get("event_type") == "harness_action_refused"]


def _rec_calls(h, step):
    return [(e["tool"], e["args"]) for e in h.traj
            if e.get("step") == step and e.get("event_type") == "tool_call" and e.get("origin") == "harness"
            and e.get("requester") == "recovery"]


# =====================================================================================================================
# kernel before_action with the authorization ceremony
# =====================================================================================================================
def test_recovery_write_refused_by_the_kernel_is_not_executed(monkeypatch, tmp_path):
    """The page shows DEN-999; the task's subject is DEN-001. The entry snapshot runs (the kernel's after_action records
    the subject mismatch against the recovery action, without drawing the agent's intervention budget). The recovery
    `type` that follows is a subject-bearing write: the service mints a ledger authorization bound to it, the kernel's
    prospective scope guard BLOCKs it, the authorization is cancelled and the field is never filled."""
    h = _harness(monkeypatch, tmp_path)
    h.portal.text = "Denial DEN-001 CO-50"
    h.portal.controls = [control("triage-note", role="textbox", label="Triage Note")]
    invs = _script(h, [[("type", {"ref": "triage-note", "text": "CO-50"})]])

    def on_other_case(_rec):                  # after reset: the page on screen is another case
        h.portal.path = "/emr/denied/DEN-999"
        return final("f0")
    h.run([on_other_case], max_steps=12)
    _clean(h)

    assert invs[0]["results"] == [("type", {"error": "BLOCK:subject_scope_mismatch"})]
    assert _refusals(h, 0) == [("type", "kernel", "BLOCK:subject_scope_mismatch")]
    assert [t for t, _a in _rec_calls(h, 0)] == ["snapshot"]
    assert not [x for x in h.portal.log if x[0] == "fill"]
    decisions = [(e["stage"], e["decision"], e["effective"], e["rule_id"], e.get("action_origin"))
                 for e in h.events("harness_decision")]
    assert decisions[:2] == [("after_action", "REVISE", "REVISE", "subject_scope_mismatch", "harness"),
                             ("before_action", "BLOCK", "BLOCK", "subject_scope_mismatch", "harness")]
    assert {e.get("action_requester") for e in h.events("harness_decision")[:2]} == {"recovery"}
    kern = h.kernels[0]
    agent_charged = [i for i in kern.ledger.interventions if "action_origin" not in i and i["effective"] != "ALLOW"]
    assert kern._n_interventions == len(agent_charged)        # recovery's REVISE and BLOCK drew nothing
    assert [a.status for a in kern.ledger.mutation_authorizations] == ["CANCELLED"]
    assert kern.ledger.mutation_hold is False
    assert [e["step"] for e in h.events("harness_action_refused")] == [0]
    # the agent's governance metrics leave recovery's records and opportunities out
    agent_records = lambda xs: [x for x in xs if not x.get("action_origin")]
    metrics = h.result["harness"]["metrics"]
    assert metrics["n_proposed_actions"] == len(agent_records(kern.ledger.proposed_actions))
    assert metrics["n_interventions"] == len(agent_records(kern.ledger.interventions))
    assert kern.ledger.opportunities.get("harness:subject_bearing_action", 0) >= 1
    assert metrics["wrong_scope_opportunities"] == kern.ledger.opportunities.get("subject_bearing_action", 0)


def test_recovery_write_runs_with_a_dispatched_authorization(monkeypatch, tmp_path):
    """The same write on the task's own case runs: authorization minted, matched by verify_commit, reserved, dispatched
    by the executor, finalized after after_action, and recorded next to the tool_call."""
    h = _harness(monkeypatch, tmp_path)
    h.portal.text = "Denial DEN-001 CO-50"
    h.portal.controls = [control("triage-note", role="textbox", label="Triage Note")]
    invs = _script(h, [[("type", {"ref": "triage-note", "text": "CO-50"})]])
    h.run([final("f0")], max_steps=12)
    _clean(h)

    assert "error" not in invs[0]["results"][0][1]
    assert ("fill", "triage-note", "CO-50") in h.portal.log
    calls = [e for e in h.events("tool_call") if e.get("step") == 0 and e.get("tool") == "type"]
    assert len(calls) == 1 and calls[0]["origin"] == "harness" and calls[0]["audience"] == "harness"
    assert calls[0]["requester"] == "recovery"
    auth_evs = _at(h, 0, "harness_authorization")
    assert [(e["action_id"], e["authorization_id"], e["source"], e["tool"]) for e in auth_evs] == [
        (calls[0]["action_id"], "auth-1", "deterministic_gap", "type")]
    kern = h.kernels[0]
    assert kern.ledger.mutation_authorizations[0].status == auth_evs[0]["auth_status"]
    assert auth_evs[0]["auth_status"] in ("VERIFIED", "UNKNOWN", "FAILED")
    assert kern.ledger.mutation_hold is False and kern.ledger.pending_authorization is None
    tool_proposals = [p for p in kern.ledger.proposed_actions if p["action"] != "final"]   # the agent only finals
    assert tool_proposals and all((p.get("action_origin"), p.get("action_requester")) == ("harness", "recovery")
                                  for p in tool_proposals)


# =====================================================================================================================
# irreversible-commit check: a harness irreversible commit needs a non-empty payload, the agent's own attempt at the
# same commit, and that commit not landed (the fuller boundary cases are in runner/tests/test_hab_boundary.py)
# =====================================================================================================================
def _submit_page(portal, note_value=""):
    portal.controls = [control("triage-note", role="textbox", label="Triage Note", value=note_value),
                       control("submit-disposition-button", label="Submit Disposition")]
    portal.on_click = {"submit-disposition-button": set_action("selectedDisposition", "Route to Clinical Appeals")}


SUBMIT = ("submit", {"ref": "submit-disposition-button"})


def test_q13_empty_payload_refused(monkeypatch, tmp_path):
    h = _harness(monkeypatch, tmp_path)
    _submit_page(h.portal)
    invs = _script(h, [[SUBMIT]])
    h.run([final("f0")], max_steps=12)
    _clean(h)
    assert invs[0]["results"] == [("submit", {"error": "irreversible_commit_empty_payload"})]
    assert _refusals(h, 0) == [("submit", "irreversible_commit", "irreversible_commit_empty_payload")]
    assert ("click", "submit-disposition-button") not in h.portal.log
    assert h.kernels[0].ledger.mutation_authorizations == []        # refused before the kernel


def test_q13_payload_without_an_agent_attempt_refused(monkeypatch, tmp_path):
    """The form holds a value, but the agent never pressed Submit: the harness may not submit on its own."""
    h = _harness(monkeypatch, tmp_path)
    _submit_page(h.portal, note_value="prefilled by the portal")
    invs = _script(h, [[SUBMIT]])
    h.run([final("f0")], max_steps=12)
    _clean(h)
    assert invs[0]["results"] == [("submit", {"error": "irreversible_commit_no_agent_attempt"})]
    assert ("click", "submit-disposition-button") not in h.portal.log


def test_d2_the_agents_submit_that_did_not_land_is_completed_by_the_harness(monkeypatch, tmp_path):
    """The agent wrote the note and pressed Submit, which did not land: the recovery submit of the same control passes the
    irreversible-commit rule and the kernel ceremony, and runs; its commit is recorded with action_origin=harness after
    the agent's own (unverified) click commit."""
    h = _harness(monkeypatch, tmp_path)
    _submit_page(h.portal)
    agent_click = {"n": 0}
    lands = h.portal.on_click["submit-disposition-button"]

    def second_click_lands(portal, ref):
        agent_click["n"] += 1
        if agent_click["n"] > 1:
            lands(portal, ref)
    h.portal.on_click = {"submit-disposition-button": second_click_lands}
    invs = _script(h, [[SUBMIT]])
    h.run([fill("triage-note", NOTE), click("submit-disposition-button"), final("f0"), final("f1")], max_steps=12)
    _clean(h)
    submit_step = invs[0]["step"]
    assert "error" not in invs[0]["results"][0][1]
    assert h.portal.log.count(("click", "submit-disposition-button")) == 2
    assert [t for t, _a in _rec_calls(h, submit_step)] == ["snapshot", "submit"]
    assert [e["tool"] for e in _at(h, submit_step, "harness_authorization")] == ["submit"]
    commits = h.kernels[0].ledger.commit_history
    assert [(c["semantic_type"], c.get("action_origin"), c["verified"]) for c in commits] == [
        ("submit", None, False), ("submit", "harness", True)]


# =====================================================================================================================
# write provenance: a harness write binds a semantic value only from agent_commitment / authoritative_state / bound_evidence
# =====================================================================================================================
def test_q14_value_provenance(monkeypatch, tmp_path):
    h = _harness(monkeypatch, tmp_path)
    h.task["system_metadata"] = {"memberId": "ZZ-777"}
    h.use_task(h.task)
    h.portal.controls = [control("triage-note", role="textbox", label="Triage Note"),
                         control("member-input", role="textbox", label="Member ID")]
    ops = [("type", {"ref": "member-input", "text": "ZZ-777"}),         # system_metadata only
           ("type", {"ref": "member-input", "text": "QQ-123"}),         # nowhere
           ("type", {"ref": "member-input", "text": NOTE})]             # the agent's own value
    invs = _script(h, [[], ops])
    h.run([fill("triage-note", NOTE), final("f0"), final("f1")], max_steps=12)
    _clean(h)
    step = invs[1]["step"]
    assert [r for _t, r in invs[1]["results"][:2]] == [{"error": "value_from_system_metadata_only"},
                                                       {"error": "value_untraced"}]
    assert "error" not in invs[1]["results"][2][1]
    assert _refusals(h, step) == [("type", "write_provenance", "value_from_system_metadata_only"),
                                  ("type", "write_provenance", "value_untraced")]
    assert [x for x in h.portal.log if x[0] == "fill" and x[1] == "member-input"] == [("fill", "member-input", NOTE)]


# =====================================================================================================================
# post-action content guard: it restores through the executor and records it
# =====================================================================================================================
def test_q17_guard_restores_through_the_executor_and_records_it(monkeypatch, tmp_path):
    """A recovery click (not a READ) switches the surface; the portal clears the persisted triage note and the unsaved
    draft. The shell guard re-types the draft (a recorded recovery `type`) and writes the persisted note back
    (a recorded restore_agent_content), both charged, and records agent_content_guard."""
    persisted = "CO-50 medical necessity; appeal with the clinical records."
    h = _harness(monkeypatch, tmp_path)
    p = h.portal
    p.state["emr"] = {"agentActions": {"selectedDisposition": "Route to Clinical Appeals"}, "triageNotes": [persisted]}
    p.controls = [control("tab-history", label="History"),
                  control("triage-note", role="textbox", label="Triage Note", value="draft: CO-50")]

    def switch(portal, ref):
        portal.emr()["triageNotes"] = [""]
        portal.find("triage-note")["value"] = ""
    p.on_click = {"tab-history": switch}
    _script(h, [[("click", {"ref": "tab-history"})]])
    h.run([final("f0")], max_steps=12)
    _clean(h)

    assert [(e["tool"], e["args"], e["action_id"]) for e in h.events("tool_call") if e.get("step") == 0] == [
        ("snapshot", {}, "rec-0-1"), ("click", {"ref": "tab-history"}, "rec-0-2"),
        ("type", {"ref": "triage-note", "text": "draft: CO-50"}, "rec-0-3"),
        ("restore_agent_content", {"fields": ["full_state.triageNotes.0"]}, "rec-0-4")]
    assert _at(h, 0, "agent_content_guard") == [
        {"step": 0, "event_type": "agent_content_guard", "action_origin": "harness",
         "action_requester": "content_guard", "after_action_id": "rec-0-2",
         "dom_restored": ["triage-note"], "dom_lost": [], "persisted_fields": ["full_state.triageNotes.0"],
         "persisted_restored": 1, "status": "ok"}]
    assert p.storage_writes == [{"emr": [{"path": ["triageNotes", "0"], "value": persisted}]}]
    assert p.emr()["triageNotes"] == [persisted] and p.find("triage-note")["value"] == "draft: CO-50"
    assert h.result["budget"]["recovery_env_actions"] == len(h.events("tool_call"))


def test_q17_no_restore_when_nothing_was_cleared(monkeypatch, tmp_path):
    h = _harness(monkeypatch, tmp_path)
    h.portal.state["emr"] = {"triageNotes": ["kept"]}
    h.portal.controls = [control("tab-history", label="History"),
                         control("triage-note", role="textbox", label="Triage Note", value="draft")]
    _script(h, [[("click", {"ref": "tab-history"})]])
    h.run([final("f0")], max_steps=12)
    _clean(h)
    assert h.events("agent_content_guard") == [] and h.portal.storage_writes == []
    assert [t for t, _a in _rec_calls(h, 0)] == ["snapshot", "click"]


# =====================================================================================================================
# evidence-reveal clicks are READs through the READ actuator
# =====================================================================================================================
def test_q16_reveal_click_is_a_read_through_the_read_actuator(monkeypatch, tmp_path):
    h = _harness(monkeypatch, tmp_path)
    h.portal.controls = [control("tab-remittance", label="Remittance Image"),
                         control("attach-doc-1", label="Attach")]
    h.portal.on_click = {"tab-remittance": set_action("viewedRemittanceImage")}
    invs = _script(h, [[("read", "click", {"ref": "tab-remittance"}), ("read", "click", {"ref": "attach-doc-1"})]])
    h.run([final("f0")], max_steps=12)
    _clean(h)
    assert "error" not in invs[0]["results"][0][1]
    assert invs[0]["results"][1] == ("click", {"error": "read_not_read_only"})     # the manifest calls it a write
    assert ("click", "attach-doc-1") not in h.portal.log
    assert _at(h, 0, "harness_acquire_predispatch") == [
        {"step": 0, "event_type": "harness_acquire_predispatch", "stage": "recovery", "tool": "click",
         "args": {"ref": "tab-remittance"}, "acquire_id": "acq-recovery-0-0", "status": "ok"}]
    tel = [r for r in h.result["harness"]["audit"]["actuation"]["records"]["acquire"] if r["stage"] == "recovery"]
    assert [(r["tool"], r["capability"], r["executed"], r["succeeded"]) for r in tel] == [
        ("click", None, True, True)]
    assert _refusals(h, 0) == [("click", "read_actuator", "read_not_read_only")]


# =====================================================================================================================
# fixed ceilings, enforced per action
# =====================================================================================================================
def test_q12_total_cap_enforced_per_action(monkeypatch, tmp_path):
    """With budget to spare, one invocation asks for 25 clicks: its entry snapshot and 19 clicks run, the rest are
    refused at the 20-action cap (the cap is checked per action, not only when an invocation starts)."""
    h = _harness(monkeypatch, tmp_path)
    h.portal.controls = [control("noop", label="Refresh")]
    invs = _script(h, [[("click", {"ref": "noop"})] * 25])
    h.run([final("f0")], max_steps=40)
    _clean(h)
    assert [("ran" if "error" not in r else r["error"]) for _t, r in invs[0]["results"]] == (
        ["ran"] * 19 + ["recovery_total_cap"] * 6)
    assert _refusals(h, 0) == [("click", "ceiling", "recovery_total_cap")] * 6
    assert h.result["budget"]["recovery_env_actions"] == 20
    assert [x for x in h.portal.log if x == ("click", "noop")] == [("click", "noop")] * 19


def test_q12_reserve_is_a_shell_constant(monkeypatch, tmp_path):
    """MH_RECOVERY_EXTRA is ignored: the additive reserve is max_steps // 6 whatever the environment says."""
    entry = S.recovery_entry()
    h = _harness(monkeypatch, tmp_path)
    monkeypatch.setenv("MH_RECOVERY_EXTRA", "30")
    h.portal.controls = [control("noop", label="Refresh")]
    seen = []

    def spend(rec):
        backend = entry._build_gui_backend(rec["driver"])
        for _ in range(2):
            backend.call_tool("snapshot", {})
        seen.append(rec["driver"].can_execute_recovery_action())
        return {"episodes": []}
    h.stub_recovery([spend])
    h.run([click("noop")] * 12 + [final("f0")], max_steps=12)
    _clean(h)
    assert seen == [False]                                  # 12 agent + 2 recovery reaches 12 + 12 // 6
