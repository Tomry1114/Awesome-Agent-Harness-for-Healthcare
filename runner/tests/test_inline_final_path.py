# -*- coding: utf-8 -*-
"""Scripted tests for run.py's before_final branches that replaying recorded trajectories does not exercise.

Each test drives the real run.run_task through runner/tests/inline_loop_rig.py and asserts what the loop does: the
trajectory events and their order, the feedback the agent is handed next turn, the repair-turn accounting, the
delivered final and its flags, and the ledger records. The test-name prefixes group the branches:

    a5   a refused before_final ACQUIRE falls through to clean_commit
    a6   DEFER_FINAL: final_deferred, repair turns, record_defer, flagged delivery on exhaustion
    a7   candidate REVISE past the repair cap: kept_original_no_candidate
    a9   must_resolve on repair exhaustion: [WITHHELD] stub, or flagged delivery when the deliverable is env state
    a11  a before_final kernel hook that raises aborts under enforce (hook_error)

Substrate per test. PhysicianBench's final answer is not a commit point (hapi_fhir.yaml declares no final rule), so
no shell capability decides at its before_final and the scripted middleware's decision is the one run.py applies.
PB-pretransplant_covid_clearance names no output/ deliverable, so the PB deliverable middleware stays inert at
before_final.

insufficient_grounding counting has no test here: its only producer is verify_commit's claim-support path, which
repair mode full never reaches.
"""
import inline_loop_rig as L

PB_TASK = "PB-pretransplant_covid_clearance"
PB = "PhysicianBench"


def _pb(monkeypatch, tmp_path, **kw):
    return L.Rig(monkeypatch, tmp_path, PB, PB_TASK, protocol="function_calling", **kw)


def _rendered(decision, reason):
    """What the PB workspace agent program appends for a batch-level harness feedback (workspaces/pb/h0/agent.py)."""
    return "[HARNESS %s] %s\nRevise accordingly; the ENVIRONMENT observation below is still current." % (decision, reason)


# ---------------------------------------------------------------------------------------------------- a6: DEFER_FINAL
DEFER_REASON = "scripted: this final is not accepted yet"
DEFER_MESSAGE = "scripted defer message (the agent never sees message)"


def test_a6_defer_final_turns_then_flagged_delivery_on_exhaustion(monkeypatch, tmp_path):
    rig = _pb(monkeypatch, tmp_path)
    rig.hooks.before_final = lambda answer, ctx: L.decision(
        "DEFER_FINAL", rule_id="scripted_defer", reason=DEFER_REASON, feedback=DEFER_MESSAGE,
        extra={"control_only": True, "instruction_supplied": False})
    rig.replies = [L.fc_final("answer %d" % i) for i in range(1, 7)]
    rig.run(max_steps=10)

    kernel_feedback = {"decision": "DEFER_FINAL", "rule_id": "scripted_defer", "reason": DEFER_REASON,
                       "message": DEFER_MESSAGE}
    # five finals reach the kernel; the sixth scripted reply is never asked for
    assert len(rig.states) == 5 and len(rig.replies) == 1 and rig.exhausted == 0
    assert rig.types() == ["answer_attempt", "harness_decision", "final_deferred"] * 5 + ["final_answer"]
    deferred = rig.events("final_deferred")
    assert deferred == [
        {"step": i, "event_type": "final_deferred", "stage": "before_final", "capability": L.MIDDLEWARE_NAME,
         "final_deferred": True, "repair_turn_started": i < 4, "repair_turn": i + 1, "budget_exhausted": i == 4,
         "feedback": kernel_feedback, "status": "ok"} for i in range(5)]
    assert [(e["decision"], e["effective"], e["capability"]) for e in rig.events("harness_decision")] == \
        [("DEFER_FINAL", "DEFER_FINAL", L.MIDDLEWARE_NAME)] * 5
    assert rig.trajectory[-1] == {
        "step": 4, "event_type": "final_answer", "thought": "answer 5", "status": "ok",
        "verification_flag": "deferred_unresolved", "final_disposition": "flagged_delivery_defer_budget_exhausted",
        "origin": "agent", "final_answer_author": "agent",
        "answer_attempt_index": 5, "harness_feedback": kernel_feedback,
        "canonical_action": {"action_type": "final_answer", "content": "answer 5"}}

    # the agent is handed the deferral after each of the first four finals, with the before_final stage
    assert rig.feedback_at(0) is None
    for turn in range(1, 5):
        assert rig.feedback_at(turn) == dict(kernel_feedback, stage="before_final")
        assert rig.model_calls[turn]["messages"][-1] == {
            "role": "user", "content": _rendered("DEFER_FINAL", DEFER_REASON)}

    # record_defer: one record per deferral, the fifth one exhausted and without a repair turn
    assert rig.actuation_records() == [
        {"kind": "DEFER_FINAL", "stage": "before_final", "step": i, "capability": L.MIDDLEWARE_NAME,
         "final_deferred": True, "repair_turn_started": i < 4, "repair_turn": i + 1, "control_only": True,
         "instruction_supplied": False, "budget_exhausted": i == 4} for i in range(5)]
    act = rig.audit["actuation"]
    assert (act["defer_final_gated"], act["defer_final_repair_turns"], act["defer_final_exhausted"]) == (5, 4, 1)
    # the delivered answer is recorded on the ledger as an unverified final commit with the flag
    last = rig.audit["ledger"]["commit_history"][-1]
    assert (last["detail"], last["semantic_type"], last["verified"], last["verification_flag"], last["step"]) == \
        ("final_answer", "answer", None, "deferred_unresolved", 4)
    assert rig.result["harness"]["runtime_errors"] == []


# ---------------------------------------------------------------------------------------------------- a7: candidate REVISE
def test_a7_candidate_revise_past_the_repair_cap_keeps_the_original(monkeypatch, tmp_path):
    rig = _pb(monkeypatch, tmp_path)

    def before_final(answer, ctx):
        n = int(answer.split()[-1])
        if n < 5:   # four plain REVISEs spend the repair budget (distinct answers: no per-action revision cap)
            return L.decision("REVISE", rule_id="scripted_revise", reason="scripted revise %d" % n)
        return L.decision("REVISE", rule_id="scripted_candidate", reason="scripted candidate review",
                          feedback="scripted candidate message", deterministic=True,
                          extra={"candidate": True, "critique": "scripted critique"})
    rig.hooks.before_final = before_final
    rig.replies = [L.fc_final("answer %d" % i) for i in range(1, 7)]
    rig.run(max_steps=10)

    assert len(rig.states) == 5 and len(rig.replies) == 1
    assert rig.types() == ["answer_attempt", "harness_decision"] * 5 + ["final_answer"]
    for turn in range(1, 5):
        assert rig.feedback_at(turn) == {"decision": "REVISE", "rule_id": "scripted_revise",
                                         "reason": "scripted revise %d" % turn, "stage": "before_final"}
    candidate_feedback = {"decision": "REVISE", "rule_id": "scripted_candidate", "reason": "scripted candidate review",
                          "message": "scripted candidate message"}
    assert rig.events("harness_decision")[-1]["candidate"] is True
    assert rig.trajectory[-1] == {
        "step": 4, "event_type": "final_answer", "thought": "answer 5", "status": "ok",
        "verification_flag": "unverified_grounding", "final_disposition": "kept_original_no_candidate",
        "origin": "agent", "final_answer_author": "agent",
        "answer_attempt_index": 5, "harness_feedback": candidate_feedback,
        "canonical_action": {"action_type": "final_answer", "content": "answer 5"}}
    # the four plain REVISEs and the candidate REVISE, past the budget, each recorded once (one accounting)
    assert rig.actuation_records() == [
        {"kind": "REVISE", "stage": "before_final", "step": i, "capability": L.MIDDLEWARE_NAME,
         "revision_delivered": True, "repair_turn_started": i < 4, "repair_turn": i + 1,
         "suppressed_reason": (None if i < 4 else "repair_budget_exhausted"), "source": "kernel:REVISE"}
        for i in range(5)]
    last = rig.audit["ledger"]["commit_history"][-1]
    assert (last["verified"], last["verification_flag"], last["step"]) == (None, "unverified_grounding", 4)


# ---------------------------------------------------------------------------------------------------- a5: refused ACQUIRE
def test_a5_refused_before_final_acquire_commits_the_final_clean(monkeypatch, tmp_path):
    rig = _pb(monkeypatch, tmp_path)
    rig.hooks.before_final = lambda answer, ctx: L.decision(
        "ACQUIRE", rule_id="scripted_acquire", reason="scripted acquire",
        extra={"next_action": {"tool": "NotAPhysicianBenchTool", "args": {"region": "x"}, "read_only": True}})
    rig.replies = [L.fc_final("answer 1"), L.fc_final("answer 2")]
    rig.run(max_steps=10)

    assert len(rig.states) == 1 and rig.env_calls == []
    assert rig.types() == ["answer_attempt", "harness_decision", "harness_acquire_refused", "final_answer"]
    assert rig.events("harness_decision")[0]["effective"] == "ACQUIRE"
    assert rig.trajectory[2] == {"step": 0, "event_type": "harness_acquire_refused", "stage": "before_final",
                                 "tool": "NotAPhysicianBenchTool", "reason": "tool_not_in_substrate_vocabulary",
                                 "acquire_id": "acq-before_final-0-0", "status": "error"}
    assert rig.trajectory[3] == {"step": 0, "event_type": "final_answer", "thought": "answer 1", "status": "ok",
                                 "final_disposition": "clean_commit", "answer_attempt_index": 1,
                                 "origin": "agent", "final_answer_author": "agent",
                                 "canonical_action": {"action_type": "final_answer", "content": "answer 1"}}
    acq = rig.audit["actuation"]["records"]["acquire"]
    assert acq == [{"acquire_id": "acq-before_final-0-0", "stage": "before_final", "step": 0,
                    "tool": "NotAPhysicianBenchTool", "capability": L.MIDDLEWARE_NAME, "in_vocabulary": False,
                    "requested": True, "attempted": False, "executed": False, "succeeded": None,
                    "evidence_added": None, "failure_reason": "tool_not_in_substrate_vocabulary"}]
    assert rig.kernel.ledger.acquire_count == 0


# ---------------------------------------------------------------------------------------------------- a11: before_final hook error
def test_a11_before_final_hook_error_aborts_under_enforce(monkeypatch, tmp_path):
    rig = _pb(monkeypatch, tmp_path)
    rig.hooks.before_final = lambda answer, ctx: L.NotADecision()
    rig.replies = [L.fc_final("answer 1"), L.fc_final("answer 2")]
    rig.run(max_steps=10)

    assert len(rig.states) == 1
    assert [L.slim(e) for e in rig.trajectory] == [
        {"step": 0, "event_type": "answer_attempt", "attempt_index": 1, "answer_hash": rig.trajectory[0]["answer_hash"],
         "answer": "answer 1", "status": "ok", "origin": "agent"},
        {"step": 0, "event_type": "harness_escalation", "stage": "before_final", "reason": "hook_error",
         "status": "error"}]
    errors = rig.result["harness"]["runtime_errors"]
    assert len(errors) == 1 and errors[0].startswith("before_final: AttributeError(")
    assert rig.result["harness"]["status"] == "degraded"
    assert rig.audit["ledger"]["commit_history"] == []


# ---------------------------------------------------------------------------------------------------- a9: must_resolve on exhaustion
WITHHELD = ("[WITHHELD] The answer asserted a claim refuted by the validated evidence and the conflict was not "
            "resolved; no safe answer could be committed.")
UOC_FEEDBACK = {"decision": "REVISE", "rule_id": "process_output_inconsistency",
                "reason": "operational write 'fhir_service_request_create' did not complete; the answer must not "
                          "report success",
                "message": "Your action 'fhir_service_request_create' did NOT complete successfully (failed or "
                           "unknown). Either complete it successfully, or correct your answer to state it did not "
                           "land -- do not report success over a write that did not commit."}


def _failed_order_then_finals(monkeypatch, tmp_path, task_edit=None):
    """verify_commit's own must_resolve producer: an irreversible order whose tool call failed leaves an unresolved
    operational commit, so every later final gets a must_resolve REVISE (process_output_inconsistency)."""
    rig = _pb(monkeypatch, tmp_path, task_edit=task_edit)
    mrn = rig.task()["context"]["patient_ref"]
    rig.tool_result = lambda name, args: {"error": "HTTP 400 Bad Request"}
    rig.replies = [L.fc_tool("fhir_service_request_create", {"patient_reference": "Patient/" + mrn,
                                                             "code_display": "SARS-CoV-2 PCR"})]
    rig.replies += [L.fc_final("answer %d" % i) for i in range(1, 6)]
    rig.run(max_steps=10)
    assert len(rig.replies) == 1 and len(rig.states) == 5
    assert rig.types() == ["tool_call", "harness_decision", "harness_revision"] + \
        ["answer_attempt", "harness_decision"] * 4 + ["final_answer"]
    assert [(d["capability"], d["effective"], d.get("must_resolve")) for d in rig.events("harness_decision")[1:]] == \
        [("verify_commit", "REVISE", True)] * 4
    # within the repair budget the agent is handed the targeted REVISE and keeps working
    for turn in (2, 3, 4):
        assert rig.feedback_at(turn) == dict(UOC_FEEDBACK, stage="before_final")
    assert [(r["stage"], r["repair_turn"], r["repair_turn_started"]) for r in rig.actuation_records()] == \
        [("after_action", 1, True), ("before_final", 2, True), ("before_final", 3, True), ("before_final", 4, True),
         ("before_final", 5, False)]
    return rig


def test_a9_must_resolve_on_repair_exhaustion_withholds_the_answer(monkeypatch, tmp_path):
    rig = _failed_order_then_finals(monkeypatch, tmp_path)
    assert rig.trajectory[-1] == {
        "step": 4, "event_type": "final_answer", "thought": WITHHELD, "status": "ok",
        "origin": "harness", "requester": "withhold", "final_answer_author": "withheld_stub",
        "verification_flag": "abstained_unresolved_contradiction", "final_disposition": "abstained_unresolved_violation",
        "answer_attempt_index": 4, "harness_feedback": UOC_FEEDBACK,
        "canonical_action": {"action_type": "final_answer", "content": "answer 4"}}
    last = rig.audit["ledger"]["commit_history"][-1]
    assert (last["detail"], last["verified"], last["verification_flag"]) == \
        ("final_answer", None, "abstained_unresolved_contradiction")


def test_a9_env_state_deliverable_is_delivered_flagged_instead_of_withheld(monkeypatch, tmp_path):
    """The graded artifact is environment state (answer_kind portal_state_and_notes, HealthAdminBench's declaration).
    HAB's declared tools cannot produce a must_resolve, so the same PB producer runs under a task whose
    expected_outcome declares that answer kind: run.py reads only that field."""
    rig = _failed_order_then_finals(
        monkeypatch, tmp_path,
        task_edit=lambda t: t.__setitem__("expected_outcome", {"answer_kind": "portal_state_and_notes"}))
    assert rig.trajectory[-1] == {
        "step": 4, "event_type": "final_answer", "thought": "answer 4", "status": "ok",
        "origin": "agent", "final_answer_author": "agent",
        "verification_flag": "unresolved_contradiction",
        "final_disposition": "flagged_delivery_unresolved_contradiction",
        "answer_attempt_index": 4, "harness_feedback": UOC_FEEDBACK,
        "canonical_action": {"action_type": "final_answer", "content": "answer 4"}}
    last = rig.audit["ledger"]["commit_history"][-1]
    assert (last["verified"], last["verification_flag"]) == (None, "unresolved_contradiction")
