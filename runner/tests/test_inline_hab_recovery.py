# -*- coding: utf-8 -*-
"""HealthAdminBench recovery behaviour that a replay of recorded trajectories cannot exercise.

Covered: recovery allowance boundaries, the content-preservation guard, agent re-entry, and the inner forced recovery
after tool_budget_exhausted.

Same stand-ins as test_inline_hab_gates.py (runner/tests/inline_hab_stub.py). Where a test needs recovery to do
something specific it replaces the smallest piece and says which:
  * real recovery                      -- content preservation, the re-entry reachability test, inner forced recovery;
  * run_recovery_v3 stubbed            -- re-entry rounds/order/suppression (the stub returns episode summaries);
  * RecoveryKernel scripted            -- allowance boundaries (run_recovery_v3, its GUI backend with the completion
                                          allowance, the executor, the recovery events and run.py's budget and cap
                                          stay real).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inline_hab_stub as S  # noqa: E402
from inline_hab_stub import click, final, control, compact, set_action  # noqa: E402

DENIAL_TASK = "HAB-denial-easy-1"
PRIOR_AUTH_TASK = "HAB-emr-medium-1"

_AUTHORING_MSG = ("Your submission cannot be finalized yet: the field '%s' has no grounded source in what you have "
                  "observed. Open the relevant records (the referral / clinical note / coverage) and re-evaluate '%s' "
                  "from them, then complete and submit the form. Do NOT reuse a prior value unless the records support "
                  "it.")
_EVIDENCE_MSG = ("Your submission cannot be finalized yet: a required decision is not grounded in the available "
                 "evidence. Open the relevant records (denial details / remittance / clinical note) and re-derive your "
                 "decision and note from them before submitting.")
_OBLIGATION_MSG = ("You are finalizing, but the task's required workflow is NOT yet complete and you still have %d "
                   "action(s) of budget left. Do NOT stop here. Re-read the current page and the task goal, then "
                   "continue toward the unmet objective: reach the required surface, complete the required "
                   "fields/actions, submit, and record the result. Take the next concrete action now.")
_BUDGET_MSG = ("Your exploration budget is used up. Do NOT call another tool; give your best final answer from the "
               "evidence already gathered.")


def _harness(monkeypatch, tmp_path, task_id):
    h = S.Harness(monkeypatch, tmp_path)
    h.use_task(S.load_hab_task(task_id))
    return h


def _clean(h):
    assert h.net == [], h.net
    assert h.portal.unscripted == [], h.portal.unscripted
    assert h.runtime_errors() == [], h.runtime_errors()


def _unmet_obligations(h):
    """The obligation nudge sees an unmet conjunction at every final. The prior-auth task composes nothing until the
    agent itself submits; these tests pin the re-entry / nudge / grounding order, so ctx.obligations() is held unmet
    instead."""
    import middleware_host as MH
    h.mp.setattr(MH, "obligations_of", lambda task, env, trajectory=None, workspace=None: (False, (), (False,)))


def _reentry_fb(msg):
    return {"decision": "REVISE", "stage": "reentry", "reason": msg}


def _rec_tool(step, seq, tool, args=None, status="ok", requester=None, episode=None):
    """A recovery tool call; its recovery_episode_id names the host hook call that opened the invocation."""
    return {"step": step, "event_type": "tool_call", "tool": tool, "args": args or {}, "status": status,
            "origin": "harness", "audience": "harness", "action_id": "rec-%d-%d" % (step, seq),
            "requester": requester or ("content_guard" if tool == "restore_agent_content" else "recovery"),
            "recovery_episode_id": episode or "before_final@%d" % step, "parent_agent_step": step}


# =====================================================================================================================
# Agent re-entry (run.py)
# =====================================================================================================================
def test_c4_reentry_reachable_with_real_recovery_and_budget_left(monkeypatch, tmp_path):
    """The real recovery on a form with a required empty field blocks with needs_new_content:<field>; with budget
    left, run.py hands the final back at steps 0 and 1. At step 2 recovery does not run at all (two consecutive
    BLOCKED episodes stop it in run.py), so there is nothing to re-enter on and the final goes on to the obligation
    nudge. (The 2-round cap itself is pinned with a stubbed recovery below.)"""
    h = _harness(monkeypatch, tmp_path, DENIAL_TASK)
    h.portal.controls = [control("disposition-input", role="textbox", label="Disposition", required=True),
                         control("submit-disposition-button", label="Submit Disposition")]
    h.run([final("a"), final("b"), final("c")], max_steps=12)
    _clean(h)
    for step, rnd in ((0, 1), (1, 2)):
        evs = h.at(step)
        rv3 = [e for e in evs if e["event_type"] == "recovery_v3"]
        assert len(rv3) == 1 and rv3[0]["episode_state"] == "BLOCKED_NEEDS_DECISION"
        assert rv3[0]["reason"] == "dynamic_block:needs_new_content:Disposition"
        assert evs[-1] == {"step": step, "event_type": "agent_reentry", "round": rnd, "kind": "authoring",
                           "field": "Disposition", "status": "ok"}
        assert h.feedback(step + 1) == _reentry_fb(_AUTHORING_MSG % ("Disposition", "Disposition"))
        assert h.agent.turns[step + 1]["last_result"] == {"feedback": _AUTHORING_MSG % ("Disposition", "Disposition")}
        assert h.agent.turns[step + 1]["last_observation"] == "agent_reentry"
    assert [e["step"] for e in h.events("agent_reentry")] == [0, 1]
    assert [e["step"] for e in h.events("recovery_v3")] == [0, 1]
    assert h.at(2) == [{"step": 2, "event_type": "obligation_nudge", "unmet": True, "decisions": [],
                        "attempt": 1, "status": "ok"}]


def test_c4_rounds_feedback_and_order_against_obligation_grounding_kernel(monkeypatch, tmp_path):
    """Stubbed recovery returns needs_new_content on every final while budget remains. Per final, in order: recovery
    runs; re-entry (2 rounds) turns the final back before the obligation nudge (3), the grounding audit (2) and the
    kernel's before_final, none of which runs or spends a round on a re-entry turn. Re-entry unlocks repair: the
    prior-auth read-acquire never dispatches the tab revealed at the first final."""
    h = _harness(monkeypatch, tmp_path, PRIOR_AUTH_TASK)
    h.portal.text = "Referral REF-2025-001 diagnosis M17.11"          # an observed code the empty note lacks
    S.probe_tab(h.portal)
    blocked = {"episodes": [{"goal_id": "gui-0", "episode_state": "BLOCKED_NEEDS_DECISION", "path": "execution",
                             "metrics_bucket": "correctly_blocked", "reason": "dynamic_block:needs_new_content:Member ID",
                             "blocked_argument": None, "re_entry_requested": True}]}
    calls = h.stub_recovery([blocked] * 8)
    _unmet_obligations(h)
    kernel_bf = h.spy_kernel_before_final()
    h.run([S.reveal_then(h.portal, final("f0"))] + [final("f%d" % i) for i in range(1, 8)], max_steps=12)
    _clean(h)

    assert [(c["step"], c["root_content"], c["lifecycle"]) for c in calls] == [
        (i, "f%d" % i, "before_final") for i in range(8)]
    per_step = [[compact(e) for e in h.traj if e.get("step") == i] for i in range(8)]
    reentry = lambda s, r: [{"step": s, "event_type": "agent_reentry", "round": r, "kind": "authoring",
                             "field": "Member ID", "status": "ok"}]
    nudge = lambda s, a: [{"step": s, "event_type": "obligation_nudge", "unmet": True, "decisions": [],
                           "attempt": a, "status": "ok"}]
    missing = ["M17", "M17.11"]                                       # a record id is not a note item
    gap = lambda s, r: [{"event_type": "grounding_gap", "missing": missing, "status": "ok", "step": s, "round": r}]
    assert per_step[0] == reentry(0, 1)
    assert per_step[1] == reentry(1, 2)
    assert per_step[2] == nudge(2, 1)
    assert per_step[3] == nudge(3, 2)
    assert per_step[4] == nudge(4, 3)
    assert per_step[5] == gap(5, 1)
    assert per_step[6] == gap(6, 2)
    assert [e["event_type"] for e in per_step[7]] == ["answer_attempt", "final_answer"]
    assert per_step[7][0]["answer"] == "f7" and per_step[7][0]["attempt_index"] == 1
    assert kernel_bf == [7]

    msg = _AUTHORING_MSG % ("Member ID", "Member ID")
    assert [h.feedback(t) for t in (1, 2)] == [_reentry_fb(msg)] * 2
    assert [h.agent.turns[t]["last_observation"] for t in (1, 2)] == ["agent_reentry"] * 2
    assert [h.feedback(t) for t in (3, 4, 5)] == [{"decision": "CONTINUE", "stage": "obligation",
                                                  "reason": _OBLIGATION_MSG % 12}] * 3
    assert [h.feedback(t)["stage"] for t in (6, 7)] == ["grounding", "grounding"]
    assert S.probe_dispatch_steps(h) == []


def test_c4_detection_first_matching_episode_and_kind(monkeypatch, tmp_path):
    """Detection walks the episodes in order and stops at the first match: needs_new_content (authoring, field after
    the marker) or re_entry_requested / decision_readiness / BLOCKED_NEEDS_DECISION (evidence, field =
    blocked_argument or ''). An authoring match with an empty field is recorded as authoring but gets the evidence
    text. A summary with no matching episode does not re-enter (the final goes on to the obligation nudge) and does
    not unlock repair: the read-acquire dispatches on the next turn."""
    h = _harness(monkeypatch, tmp_path, PRIOR_AUTH_TASK)
    S.probe_tab(h.portal)
    failed = {"goal_id": "gui-0", "episode_state": "FAILED", "reason": "verify_effect:False", "blocked_argument": None}
    summaries = [
        {"episodes": [failed]},                                                          # no match
        {"episodes": [failed,
                      {"episode_state": "PARTIAL", "reason": "dynamic_block:re_entry_requested:decision_readiness",
                       "blocked_argument": "viewedRemittanceImage"},
                      {"episode_state": "BLOCKED_NEEDS_DECISION", "reason": "dynamic_block:needs_new_content:Member ID",
                       "blocked_argument": None}]},
        {"episodes": [{"episode_state": "BLOCKED_NEEDS_DECISION", "reason": "dynamic_block:needs_new_content:",
                       "blocked_argument": "ignored"}]},
    ]
    calls = h.stub_recovery(summaries)
    _unmet_obligations(h)
    h.run([S.reveal_then(h.portal, final("f0")), final("f1"), final("f2"), final("f3")], max_steps=12)
    _clean(h)
    # step 1 is the harness read (no agent turn), so agent turn n is step n+1 from there on
    assert [c["step"] for c in calls] == [0, 2, 3, 4, 5, 6]
    assert h.at(0) == [{"step": 0, "event_type": "obligation_nudge", "unmet": True, "decisions": [],
                        "attempt": 1, "status": "ok"}]
    assert S.probe_dispatch_steps(h) == [1]
    assert h.at(2) == [{"step": 2, "event_type": "agent_reentry", "round": 1, "kind": "evidence",
                        "field": "viewedRemittanceImage", "status": "ok"}]
    assert h.at(3) == [{"step": 3, "event_type": "agent_reentry", "round": 2, "kind": "authoring", "field": "",
                        "status": "ok"}]
    assert h.feedback(2) == _reentry_fb(_EVIDENCE_MSG)
    assert h.feedback(3) == _reentry_fb(_EVIDENCE_MSG)
    assert [e["step"] for e in h.events("obligation_nudge")] == [0, 4, 5]
    assert [e["event_type"] for e in h.traj if e.get("step") == 6] == ["answer_attempt", "final_answer"]


def test_c4_suppressed_when_budget_spent_at_loop_start(monkeypatch, tmp_path):
    """The budget test uses the agent's own budget at the START of the loop iteration. At step 11 (11 agent actions)
    the stubbed recovery spends 12 actions through the real GUI backend and still re-enters; those 12 are harness
    actions and spend none of the agent's budget, so the agent's 12th action at step 12 runs. At step 13 the
    agent's budget is spent at loop start, so the same summary does not re-enter (and the obligation nudge is skipped
    too)."""
    entry = S.recovery_entry()
    h = _harness(monkeypatch, tmp_path, DENIAL_TASK)
    h.portal.controls = [control("noop", label="Refresh")]
    blocked = {"episodes": [{"episode_state": "BLOCKED_NEEDS_DECISION",
                             "reason": "dynamic_block:needs_new_content:Member ID", "blocked_argument": None}]}

    def spend(rec):
        backend = entry._build_gui_backend(rec["driver"])
        for _ in range(12):
            backend.call_tool("snapshot", {})
        return blocked
    calls = h.stub_recovery([spend, blocked])
    h.run([click("noop")] * 11 + [final("f0"), click("noop"), final("f1")], max_steps=12)
    _clean(h)
    # budget_open is recovery's own gate (agent + harness < max_steps + reserve 2): open at step 11, closed at step 13
    assert [(c["step"], c["budget_open"]) for c in calls] == [(11, True), (13, False)]
    assert [compact(e) for e in h.traj if e.get("step") == 11][-1] == {
        "step": 11, "event_type": "agent_reentry", "round": 1, "kind": "authoring", "field": "Member ID", "status": "ok"}
    assert len([e for e in h.traj if e.get("step") == 11 and e.get("origin") == "harness"]) == 12
    assert [(e["step"], e["origin"]) for e in h.traj if e.get("event_type") == "tool_call" and e.get("step") == 12] == [
        (12, "agent")]
    assert len(h.events("agent_reentry")) == 1
    assert h.events("obligation_nudge") == []
    assert [e["event_type"] for e in h.traj if e.get("step") == 13][-2:] == ["answer_attempt", "final_answer"]
    assert {k: h.result["budget"][k] for k in ("max_steps", "agent_env_actions", "harness_env_actions", "recovery_env_actions", "total_env_actions", "budget_exhausted")} == {
        "max_steps": 12, "agent_env_actions": 12, "harness_env_actions": 12, "recovery_env_actions": 12,
        "total_env_actions": 24, "budget_exhausted": True}


# =====================================================================================================================
# Inner forced recovery after tool_budget_exhausted (run.py)
# =====================================================================================================================
def test_c5_forced_recovery_runs_once_after_tool_budget_exhausted(monkeypatch, tmp_path):
    """An agent that never finals: 6 actions spend the budget, 4 runtime_budget refusals draw repair turns 1-4, the
    5th records tool_budget_exhausted and run.py runs the real recovery once as if the agent had finished (root
    content '', lifecycle before_final, recovery reserve re-granted to max_steps//6 = 1), then aborts."""
    h = _harness(monkeypatch, tmp_path, DENIAL_TASK)
    h.portal.controls = [control("noop", label="Refresh")]
    calls = h.spy_recovery()
    h.run([click("noop")] * 30, max_steps=6)
    _clean(h)
    assert [(e.get("step"), e["event_type"], e.get("origin")) for e in h.traj[:6]] == [
        (s, "tool_call", "agent") for s in range(6)]
    assert compact(h.traj[6]) == {"step": 10, "event_type": "tool_budget_exhausted", "status": "escalated"}
    tail = [compact(e) for e in h.traj[7:]]
    assert tail[:-1] == [_rec_tool(10, 1, "snapshot", episode="after_episode@6"),
                         _rec_tool(10, 2, "snapshot", episode="after_episode@6")]
    assert tail[-1]["step"] == 10 and tail[-1]["event_type"] == "recovery_v3"
    assert [(c["step"], c["root_content"], c["lifecycle"], c["budget_open"]) for c in calls] == [
        (10, "", "before_final", True)]
    assert [h.feedback(t) for t in (7, 8, 9, 10)] == [
        {"decision": "REVISE", "stage": "runtime_budget", "missing_obligations": [], "reason": _BUDGET_MSG}] * 4
    assert len(h.agent.turns) == 11                         # aborted: no turn after the forced recovery
    assert h.events("final_answer") == [] and h.events("agent_error") == []
    assert {k: h.result["budget"][k] for k in ("max_steps", "agent_env_actions", "harness_env_actions", "recovery_env_actions", "total_env_actions", "budget_exhausted")} == {
        "max_steps": 6, "agent_env_actions": 6, "harness_env_actions": 2, "recovery_env_actions": 2,
        "total_env_actions": 8, "budget_exhausted": True}


def test_c5_forced_recovery_skipped_once_twenty_recovery_actions_ran(monkeypatch, tmp_path):
    """The contrast: with 20 recovery actions already spent at an earlier final, the forced invocation returns at the
    cap check, so nothing follows tool_budget_exhausted. The agent's own budget is its 30 actions (the 20 recovery
    actions spend none of it), and the obligation nudge at the final draws no repair turn, so the 5th refusal after
    them stops the loop at step 35."""
    entry = S.recovery_entry()
    h = _harness(monkeypatch, tmp_path, DENIAL_TASK)
    h.portal.controls = [control("noop", label="Refresh")]

    def spend(rec):
        backend = entry._build_gui_backend(rec["driver"])
        for _ in range(20):
            backend.call_tool("snapshot", {})
        return {"episodes": []}
    calls = h.stub_recovery([spend])
    h.run([final("f0")] + [click("noop")] * 40, max_steps=30)
    _clean(h)
    assert [c["step"] for c in calls] == [0]
    assert h.at(0)[-1]["event_type"] == "obligation_nudge"
    assert [e.get("event_type") for e in h.traj[-2:]] == ["tool_call", "tool_budget_exhausted"]
    assert h.traj[-1]["step"] == 35 and h.traj[-2]["origin"] == "agent" and h.traj[-2]["step"] == 30
    assert h.result["budget"]["recovery_env_actions"] == 20 and h.result["budget"]["agent_env_actions"] == 30


# =====================================================================================================================
# Recovery allowance boundaries (run.py; workspaces/hab/h0/recovery/entry.py)
# =====================================================================================================================
def test_c2_completion_allowance_twelve_per_invocation_and_total_cap_per_action(monkeypatch, tmp_path):
    """max_steps=12 gives recovery a reserve of 12//6 = 2. The agent spends its 12 actions, then finals three times
    (the grounding audit turns the first two back). Once agent + recovery actions reach 14, only 12 more non-snapshot
    actions run per invocation (refusals are not executed or charged; snapshots are charged and not refused by the
    allowance). The 20-action cap is a shell ceiling checked before every action (runner/harness/action_service.py),
    not only when an invocation starts: the second invocation starts at 15 and stops at 20, its remaining clicks and
    its snapshot refused at the cap; the third is skipped."""
    h = _harness(monkeypatch, tmp_path, DENIAL_TASK)
    h.portal.text = "Denial DEN-001 CO-50"
    h.portal.controls = [control("noop", label="Refresh")]
    ops = [("click", {"ref": "noop"})] * 15 + [("snapshot", {})]
    invs = h.script_recovery_kernel([ops, ops, ops])
    h.run([click("noop")] * 12 + [final("f0"), final("f1"), final("f2")], max_steps=12)
    _clean(h)

    refused = {"error": "recovery_budget_exhausted"}
    capped = {"error": "recovery_total_cap"}
    assert [i["step"] for i in invs] == [12, 13]
    ran = lambda inv: ["refused" if r == refused else "capped" if r == capped else ("ran:%s" % t)
                       for t, r in inv["results"]]
    assert ran(invs[0]) == ["ran:click"] * 13 + ["refused"] * 2 + ["ran:snapshot"]
    assert ran(invs[1]) == ["ran:click"] * 4 + ["capped"] * 11 + ["capped"]
    rec_tools = lambda step: [e["tool"] for e in h.traj if e.get("step") == step and e.get("origin") == "harness"]
    assert rec_tools(12) == ["snapshot"] + ["click"] * 13 + ["snapshot"]
    assert rec_tools(13) == ["snapshot"] + ["click"] * 4
    assert rec_tools(14) == []
    assert [x for x in h.portal.log if x == ("click", "noop")] == [("click", "noop")] * (12 + 13 + 4)
    assert [e["step"] for e in h.events("grounding_gap")] == [12, 13]
    assert [e["event_type"] for e in h.traj if e.get("step") == 14] == ["answer_attempt", "final_answer"]
    assert {k: h.result["budget"][k] for k in ("max_steps", "agent_env_actions", "harness_env_actions", "recovery_env_actions", "total_env_actions", "budget_exhausted")} == {
        "max_steps": 12, "agent_env_actions": 12, "harness_env_actions": 20, "recovery_env_actions": 20,
        "total_env_actions": 32, "budget_exhausted": True}
    assert h.result["harness"]["recovery_env_actions"] == 20


def test_c2_total_cap_boundary_nineteen_runs_twenty_stops(monkeypatch, tmp_path):
    """With budget to spare, invocation 1 spends 19 recovery actions (its entry snapshot + 18 clicks), so invocation
    2 still starts (19 < 20) and spends 1 (its entry snapshot); invocation 3 is skipped (20 >= 20). The obligation
    nudge's remaining-budget text counts the recovery actions."""
    h = _harness(monkeypatch, tmp_path, DENIAL_TASK)
    h.portal.controls = [control("noop", label="Refresh")]
    invs = h.script_recovery_kernel([[("click", {"ref": "noop"})] * 18, [], [("click", {"ref": "noop"})] * 5])
    h.run([final("f%d" % i) for i in range(4)], max_steps=40)
    _clean(h)
    assert [i["step"] for i in invs] == [0, 1]
    assert [len([e for e in h.traj if e.get("step") == s and e.get("origin") == "harness"]) for s in range(4)] == [
        19, 1, 0, 0]
    assert [e["step"] for e in h.events("obligation_nudge")] == [0, 1, 2]
    assert [h.feedback(t)["reason"] for t in (1, 2, 3)] == [_OBLIGATION_MSG % 21, _OBLIGATION_MSG % 20,
                                                           _OBLIGATION_MSG % 20]
    assert [e["event_type"] for e in h.traj if e.get("step") == 3] == ["answer_attempt", "final_answer"]
    assert h.result["budget"]["recovery_env_actions"] == 20


# =====================================================================================================================
# Content-preservation guard (runner/harness/action_service.py; runner/environments.py)
# =====================================================================================================================
def test_c3_guard_restores_note_a_recovery_click_cleared_and_records_it(monkeypatch, tmp_path):
    """The agent chose a disposition, wrote a triage note (persisted) and has an unsaved draft in the note box, but
    never opened the remittance image. At its final the real recovery opens the remittance tab; the portal stub, like
    the native portal, clears the triage note on that surface switch. The guard re-types the draft through a recorded
    recovery `type` call and writes the persisted note back into localStorage. The guard is the shell's post-action
    guard (runner/harness/action_service.py), so that write goes through the executor, is charged, and is recorded
    (a restore_agent_content tool call and an agent_content_guard event)."""
    note = "CO-50 medical necessity; appeal with the clinical records."
    h = _harness(monkeypatch, tmp_path, DENIAL_TASK)
    p = h.portal
    p.state["emr"] = {"agentActions": {"selectedDisposition": "Route to Clinical Appeals",
                                       "viewedDenialDetails": True},
                      "triageNotes": [note]}
    p.controls = [control("tab-remittance", label="Remittance Image"),
                  control("triage-note", role="textbox", label="Triage Note", value="draft: CO-50")]

    def open_remittance(portal, ref):
        portal.emr()["agentActions"]["viewedRemittanceImage"] = True
        portal.emr()["triageNotes"] = [""]                  # the portal quirk: a surface switch wipes the note
        portal.find("triage-note")["value"] = ""
        portal.text = "Remittance image"
    p.on_click = {"tab-remittance": open_remittance}
    h.run([final("done")], max_steps=12)
    _clean(h)

    assert [compact(e) for e in h.traj if e.get("step") == 0 and e.get("event_type") == "tool_call"] == [
        _rec_tool(0, 1, "snapshot"), _rec_tool(0, 2, "snapshot"),
        _rec_tool(0, 3, "click", {"ref": "tab-remittance"}),
        _rec_tool(0, 4, "type", {"ref": "triage-note", "text": "draft: CO-50"}, requester="content_guard"),
        _rec_tool(0, 5, "restore_agent_content", {"fields": ["full_state.triageNotes.0"]}),
        _rec_tool(0, 6, "snapshot")]
    # the persisted restore: one direct localStorage write of the agent's own prior value
    assert p.storage_writes == [{"emr": [{"path": ["triageNotes", "0"], "value": note}]}]
    assert p.emr()["triageNotes"] == [note]
    assert p.find("triage-note")["value"] == "draft: CO-50"
    assert p.log[1:3] == [("click", "tab-remittance"), ("fill", "triage-note", "draft: CO-50")]
    # ... and it is recorded: the restore tool call above, and one event naming what was restored
    assert [compact(e) for e in h.events("agent_content_guard")] == [
        {"step": 0, "event_type": "agent_content_guard", "action_origin": "harness",
         "action_requester": "content_guard", "after_action_id": "rec-0-3",
         "dom_restored": ["triage-note"], "dom_lost": [], "persisted_fields": ["full_state.triageNotes.0"],
         "persisted_restored": 1, "status": "ok"}]
    rv3 = [e for e in h.traj if e.get("event_type") == "recovery_v3" and e.get("step") == 0]
    assert len(rv3) == 1 and sorted(rv3[0]) == ["created_id", "episode_state", "event_type", "goal_id",
                                               "metrics_bucket", "path", "reason", "status", "step", "surface",
                                               "verified_as", "verified_level"]
    assert h.result["budget"]["recovery_env_actions"] == len(h.events("tool_call"))
