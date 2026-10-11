# -*- coding: utf-8 -*-
"""Scripted tests for run.py's tool-call branches that replaying recorded trajectories does not exercise.

Each test drives the real run.run_task through runner/tests/inline_loop_rig.py (model, env network, judge and scoring
stood in at their boundaries; kernel decisions from a scripted workspace middleware). The test-name prefixes group the
branches:

    d5   before_action REVISE / BLOCK / ESCALATE, repair_budget_exhausted, reference_repaired, hook_error
    d7   mutation authorization reserve / finalize (VERIFIED, UNKNOWN, FAILED) and authorization_reserve_failed
    d8   after_action ESCALATE and hook_error aborts
    d10  unknown-commit RECONCILE: enter, resolve committed / failed, budget abort, and the write block
    d11  the budget-exempt deliverable write slot past the tool budget
    d16  truncated text-protocol tool call: agent_error and the re-issue request

PhysicianBench runs the function-calling workspace agent; PB-pretransplant_covid_clearance names no deliverable, so
the PB deliverable middleware stays inert. d11 needs a deliverable (PB-buprenorphine_mood_support). d16 needs the text
protocol, which MedCTA runs.
"""
import json
import os

import pytest

import inline_loop_rig as L

PB = "PhysicianBench"
PB_TASK = "PB-pretransplant_covid_clearance"
BUDGET_TEXT = ("Your exploration budget is used up. Do NOT call another tool; give your best final answer from the "
               "evidence already gathered.")


def _pb(monkeypatch, tmp_path, task=PB_TASK, **kw):
    rig = L.Rig(monkeypatch, tmp_path, PB, task, protocol="function_calling", **kw)
    rig.mrn = rig.task()["context"]["patient_ref"]
    return rig


def _labs(rig, **extra):
    return L.fc_tool("fhir_observation_search_labs", dict({"patient": "Patient/" + rig.mrn}, **extra))


def _lab_result(rig, name, args):
    """A one-entry lab bundle for this patient; the id follows the call count so each read is new evidence."""
    n = len(rig.env_calls)
    return {"entries": [{"resourceType": "Observation", "id": "obs-%d" % n,
                         "subject": {"reference": "Patient/" + rig.mrn},
                         "code": {"text": "SARS-CoV-2 PCR result %d" % n}}], "total": 1}


def _tool_calls(rig):
    return [(e["step"], e["tool"]) for e in rig.events("tool_call")]


# ---------------------------------------------------------------------------------------------------- d5: before_action decisions
@pytest.mark.parametrize("kind", ["REVISE", "BLOCK"])
def test_d5_before_action_revise_or_block_skips_the_tool_and_aborts_after_the_repair_cap(monkeypatch, tmp_path, kind):
    rig = _pb(monkeypatch, tmp_path)
    rig.hooks.before_action = lambda action, ctx: L.decision(kind, rule_id="scripted_gate", reason="scripted %s" % kind,
                                                             feedback="scripted gate message")
    rig.replies = [_labs(rig, count=n) for n in range(1, 7)]    # distinct args: no per-action revision cap
    rig.run(max_steps=10)

    assert rig.env_calls == [] and rig.events("tool_call") == []
    assert len(rig.states) == 5 and len(rig.replies) == 1
    assert rig.types() == ["harness_decision"] * 5 + ["repair_budget_exhausted"]
    assert [(d["stage"], d["decision"], d["effective"]) for d in rig.events("harness_decision")] == \
        [("before_action", kind, kind)] * 5
    assert rig.trajectory[-1] == {"step": 4, "event_type": "repair_budget_exhausted", "stage": "before_action",
                                  "status": "escalated", "reason": "repair_budget_exhausted"}
    fb = {"decision": kind, "rule_id": "scripted_gate", "reason": "scripted %s" % kind, "message": "scripted gate message",
          "stage": "before_action"}
    for turn in range(1, 5):
        assert rig.feedback_at(turn) == fb
        # a before_action revision with no global repair fields is delivered as the refused call's own result
        assert rig.model_calls[turn]["messages"][-1] == {
            "role": "tool", "tool_call_id": "call_fhir_observation_search_labs", "name": "fhir_observation_search_labs",
            "content": "[HARNESS %s -- this call only] scripted %s Your other tool calls in this batch are still "
                       "running; re-issue ONLY this corrected call." % (kind, kind)}
    # the 5th draws past the budget: recorded as not started (one accounting for every repair-type result)
    assert rig.actuation_records() == [
        {"kind": "REVISE", "stage": "before_action", "step": i, "capability": L.MIDDLEWARE_NAME,
         "revision_delivered": True, "repair_turn_started": i < 4, "repair_turn": i + 1,
         "suppressed_reason": (None if i < 4 else "repair_budget_exhausted"), "source": "kernel:" + kind}
        for i in range(5)]
    assert rig.result["budget"]["agent_env_actions"] == 0


def test_d5_before_action_escalate_is_corrective_then_aborts_after_the_repair_cap(monkeypatch, tmp_path):
    rig = _pb(monkeypatch, tmp_path)
    rig.hooks.before_action = lambda action, ctx: L.decision("ESCALATE", rule_id="scripted_escalate",
                                                             reason="scripted escalate", feedback="scripted message")
    rig.replies = [_labs(rig, count=n) for n in range(1, 7)]
    rig.run(max_steps=10)

    kernel_fb = {"decision": "ESCALATE", "rule_id": "scripted_escalate", "reason": "scripted escalate",
                 "message": "scripted message"}
    assert rig.env_calls == [] and len(rig.states) == 5
    assert rig.types() == ["harness_decision", "harness_escalation"] * 5 + ["repair_budget_exhausted"]
    assert rig.events("harness_escalation") == [
        {"step": i, "event_type": "harness_escalation", "stage": "before_action", "feedback": kernel_fb,
         "status": "corrective"} for i in range(5)]
    assert rig.trajectory[-1] == {"step": 4, "event_type": "repair_budget_exhausted", "stage": "before_action",
                                  "status": "escalated", "reason": "escalation_unresolved"}
    for turn in range(1, 5):
        assert rig.feedback_at(turn) == dict(kernel_fb, stage="before_action")
    assert rig.actuation_records() == []          # an ESCALATE takes a repair turn but no record_revision


def test_d5_before_action_hook_error_aborts_without_running_the_tool(monkeypatch, tmp_path):
    rig = _pb(monkeypatch, tmp_path)
    rig.hooks.before_action = lambda action, ctx: L.NotADecision()
    rig.replies = [_labs(rig), L.fc_final("answer 1")]
    rig.run(max_steps=10)

    assert rig.env_calls == [] and len(rig.states) == 1
    assert rig.trajectory == [{"step": 0, "event_type": "harness_escalation", "stage": "before_action",
                               "reason": "hook_error", "status": "error"}]
    errors = rig.result["harness"]["runtime_errors"]
    assert len(errors) == 1 and errors[0].startswith("before_action: AttributeError(")


def test_d5_reference_integrity_strip_is_logged_as_reference_repaired_once(monkeypatch, tmp_path):
    rig = _pb(monkeypatch, tmp_path)
    created = {}

    def tool_result(name, args):
        rid = "sr-%d" % (len(created) + 1)
        created[rid] = args
        return {"resourceType": "ServiceRequest", "id": rid, "status": "active"}
    rig.tool_result = tool_result
    rig.get_result = lambda path: {"resourceType": "ServiceRequest", "id": path.rsplit("/", 1)[-1]}
    ref = "Encounter/enc-invented-777"
    rig.replies = [L.fc_tool("fhir_service_request_create", {"patient_reference": "Patient/" + rig.mrn,
                                                             "code_display": "SARS-CoV-2 PCR", "encounter_reference": ref}),
                   L.fc_tool("fhir_service_request_create", {"patient_reference": "Patient/" + rig.mrn,
                                                             "code_display": "Chest CT"}),
                   L.fc_final("answer 1")]
    rig.run(max_steps=10)

    assert rig.types() == ["reference_repaired", "reconciliation", "tool_call", "reconciliation", "tool_call",
                           "answer_attempt", "final_answer"]
    assert rig.trajectory[0] == {"step": 0, "event_type": "reference_repaired", "stage": "before_action",
                                 "path": "encounter_reference", "stripped_ref": ref, "status": "ok"}
    # the dispatched payload and the recorded tool call do not carry the stripped reference
    assert rig.env_calls[0] == ("fhir_service_request_create",
                                {"patient_reference": "Patient/" + rig.mrn, "code_display": "SARS-CoV-2 PCR"})
    assert rig.events("tool_call")[0]["args"] == rig.env_calls[0][1]
    assert rig.kernel.ledger._reference_repairs_pending == []


# ---------------------------------------------------------------------------------------------------- d7: mutation authorizations
UNCERTAIN = dict(rule_id="scripted_uncertain", reason="scripted uncertain finding", deterministic=False)


def _hold_then(rig, *replies):
    """A lab read whose after_action raises a non-deterministic REVISE: the kernel puts mutations under a hold, so
    the next covered order needs a scoped user_goal authorization (verify_commit RULE 3)."""
    rig.hooks.after_action = lambda action, ctx, result: (
        L.decision("REVISE", **UNCERTAIN) if action.get("tool") == "fhir_observation_search_labs" else None)
    rig.replies = [_labs(rig)] + list(replies)


def _order(rig, display="SARS-CoV-2 PCR"):
    return L.fc_tool("fhir_service_request_create", {"patient_reference": "Patient/" + rig.mrn, "code_display": display})


def _auths(rig):
    return [(a.authorization_id, a.source, a.allowed_tool, a.allowed_semantic_type, a.status)
            for a in rig.kernel.ledger.mutation_authorizations]


def _orders_return(rig, rid="sr-1"):
    def tool_result(name, args):
        if name == "fhir_service_request_create":
            return {"resourceType": "ServiceRequest", "id": rid, "status": "active"}
        return _lab_result(rig, name, args)
    return tool_result


def test_d7_authorization_verified_after_confirmed_read_back(monkeypatch, tmp_path):
    rig = _pb(monkeypatch, tmp_path)
    rig.tool_result = _orders_return(rig)
    rig.get_result = lambda path: {"resourceType": "ServiceRequest", "id": "sr-1"}
    _hold_then(rig, _order(rig), L.fc_final("answer 1"))
    rig.run(max_steps=10)

    assert rig.kernel.ledger.mutation_hold is True
    assert _tool_calls(rig) == [(0, "fhir_observation_search_labs"), (1, "fhir_service_request_create")]
    assert _auths(rig) == [("auth-1", "user_goal", "fhir_service_request_create", "create", "VERIFIED")]
    assert rig.events("authorization_reserve_failed") == []


def test_d7_authorization_unknown_when_read_back_does_not_confirm(monkeypatch, tmp_path):
    rig = _pb(monkeypatch, tmp_path)
    rig.tool_result = _orders_return(rig)
    rig.get_result = lambda path: {"error": "HTTP 404", "query": path}
    _hold_then(rig, _order(rig), L.fc_final("answer 1"))
    rig.run(max_steps=10)

    order = rig.events("tool_call")[1]
    assert (order["tool"], order["status"]) == ("fhir_service_request_create", "error")
    assert rig.events("reconciliation")[0]["confirmed"] is False
    assert _auths(rig) == [("auth-1", "user_goal", "fhir_service_request_create", "create", "UNKNOWN")]


def test_d7_authorization_failed_when_the_commit_does_not_verify(monkeypatch, tmp_path):
    """The same resource id comes back twice: the second order changes no observable state, so verify_commit says
    the commit did not verify (verification False) although the read-back confirmed it."""
    rig = _pb(monkeypatch, tmp_path)
    rig.tool_result = _orders_return(rig)
    rig.get_result = lambda path: {"resourceType": "ServiceRequest", "id": "sr-1"}
    _hold_then(rig, _order(rig), _order(rig, "Chest CT"), L.fc_final("answer 1"))
    rig.run(max_steps=10)

    assert [(e["capability"], e["rule_id"]) for e in rig.events("harness_decision") if e["stage"] == "after_action"] \
        == [(L.MIDDLEWARE_NAME, "scripted_uncertain"), ("verify_commit", "state_transition")]
    assert _auths(rig) == [("auth-1", "user_goal", "fhir_service_request_create", "create", "VERIFIED"),
                           ("auth-2", "user_goal", "fhir_service_request_create", "create", "FAILED")]


def test_d7_authorization_failed_on_an_explicit_failure_the_read_back_confirms(monkeypatch, tmp_path):
    """A covered write (write_file, chart_review's update commit point) that errors and whose read-back finds no
    file: result_status failed and confirmed False."""
    rig = _pb(monkeypatch, tmp_path)
    rig.tool_result = lambda name, args: _lab_result(rig, name, args)
    _hold_then(rig, L.fc_tool("write_file", {"file_path": "", "content": "scripted note"}), L.fc_final("answer 1"))
    rig.run(max_steps=10)

    write = rig.events("tool_call")[1]
    assert (write["tool"], write["status"], write["result"]) == ("write_file", "error", {"error": "path traversal rejected"})
    assert rig.events("reconciliation")[0]["confirmed"] is False
    assert _auths(rig) == [("auth-1", "user_goal", "write_file", "update", "FAILED")]


def test_d7_authorization_reserve_failed_refuses_the_action(monkeypatch, tmp_path):
    """verify_commit only matches an AVAILABLE authorization, so a failed reserve needs a capability after it to
    change the matched authorization's state first. The scripted middleware's hook reserves it through the test's own
    handle on the kernel ledger: workspace code has a read-only ledger view."""
    rig = _pb(monkeypatch, tmp_path)
    rig.tool_result = _orders_return(rig)

    def before_action(action, ctx):
        assert not hasattr(ctx.ledger, "pending_authorization")    # not a field workspace code reads
        led = rig.kernel.ledger
        auth = led.pending_authorization
        if action.get("tool") == "fhir_service_request_create" and auth is not None:
            assert led.reserve_authorization(auth)
        return None
    rig.hooks.before_action = before_action
    _hold_then(rig, _order(rig), L.fc_final("answer 1"))
    rig.run(max_steps=10)

    assert [c[0] for c in rig.env_calls] == ["fhir_observation_search_labs"]
    assert _tool_calls(rig) == [(0, "fhir_observation_search_labs")]
    assert rig.events("authorization_reserve_failed") == [
        {"step": 1, "event_type": "authorization_reserve_failed", "authorization_id": "auth-1",
         "auth_status": "RESERVED", "status": "ok"}]
    fb = {"decision": "BLOCK", "stage": "authorization", "reason": "authorization could not be reserved (not AVAILABLE)",
          "missing_obligations": []}
    assert rig.feedback_at(2) == fb
    assert rig.model_calls[2]["messages"][-1]["content"] == (
        "[HARNESS BLOCK -- this call only] authorization could not be reserved (not AVAILABLE) Your other tool calls "
        "in this batch are still running; re-issue ONLY this corrected call.")
    # the refusal takes no repair turn: only the hold-setting after_action REVISE is recorded
    assert [(r["stage"], r["repair_turn"]) for r in rig.actuation_records()] == [("after_action", 1)]
    assert _auths(rig) == [("auth-1", "user_goal", "fhir_service_request_create", "create", "RESERVED")]


# ---------------------------------------------------------------------------------------------------- d8: after_action aborts
def test_d8_after_action_escalate_terminates_the_run(monkeypatch, tmp_path):
    rig = _pb(monkeypatch, tmp_path)
    rig.tool_result = lambda name, args: _lab_result(rig, name, args)
    rig.hooks.after_action = lambda action, ctx, result: L.decision("ESCALATE", rule_id="scripted_post",
                                                                    reason="scripted post escalate",
                                                                    feedback="scripted post message")
    rig.replies = [_labs(rig), L.fc_final("answer 1")]
    rig.run(max_steps=10)

    assert len(rig.states) == 1
    assert rig.types() == ["tool_call", "harness_decision", "harness_escalation"]
    assert rig.trajectory[-1] == {"step": 0, "event_type": "harness_escalation", "stage": "after_action",
                                  "feedback": {"decision": "ESCALATE", "rule_id": "scripted_post",
                                               "reason": "scripted post escalate", "message": "scripted post message"},
                                  "status": "error"}
    assert rig.result["budget"]["agent_env_actions"] == 1


def test_d8_after_action_hook_error_aborts_before_the_tool_call_is_recorded(monkeypatch, tmp_path):
    rig = _pb(monkeypatch, tmp_path)
    rig.tool_result = lambda name, args: _lab_result(rig, name, args)
    rig.hooks.after_action = lambda action, ctx, result: L.NotADecision()
    rig.replies = [_labs(rig), L.fc_final("answer 1")]
    rig.run(max_steps=10)

    assert len(rig.states) == 1 and [c[0] for c in rig.env_calls] == ["fhir_observation_search_labs"]
    assert rig.trajectory == [{"step": 0, "event_type": "harness_escalation", "stage": "after_action",
                               "reason": "hook_error", "status": "error"}]
    assert rig.result["budget"]["agent_env_actions"] == 1
    errors = rig.result["harness"]["runtime_errors"]
    assert len(errors) == 1 and errors[0].startswith("after_action: AttributeError(")


# ---------------------------------------------------------------------------------------------------- d10: unknown-commit RECONCILE
RECONCILE_FB = {"decision": "RECONCILE", "rule_id": "scripted_reconcile", "reason": "scripted: commit state unknown",
                "message": "scripted reconcile message"}


def _message(rig, text, **extra):
    return L.fc_tool("fhir_communication_create_message",
                     dict({"patient_reference": "Patient/" + rig.mrn, "message_text": text}, **extra))


def _reconcile_first_message(rig):
    """RECONCILE at after_action of the first message only. A Communication create is reversible (no commit point,
    risk below R2), so no shell capability decides at its after_action and the scripted RECONCILE is applied."""
    rig.hooks.after_action = lambda action, ctx, result: (
        L.decision("RECONCILE", rule_id="scripted_reconcile", reason="scripted: commit state unknown",
                   feedback="scripted reconcile message")
        if (action.get("args") or {}).get("message_text") == "first" else None)


def _messages_and_labs(rig, with_id=True):
    def tool_result(name, args):
        if name == "fhir_communication_create_message":
            out = {"resourceType": "Communication", "status": "in-progress"}
            if with_id:
                out["id"] = "comm-%s" % args["message_text"]
            return out
        return _lab_result(rig, name, args)
    return tool_result


def test_d10_reconcile_enter_then_a_read_resolves_it_committed(monkeypatch, tmp_path):
    rig = _pb(monkeypatch, tmp_path)
    rig.tool_result = _messages_and_labs(rig)
    rig.get_result = lambda path: {"resourceType": "Communication", "id": path.rsplit("/", 1)[-1]}
    _reconcile_first_message(rig)
    rig.replies = [_message(rig, "first"), _labs(rig), L.fc_final("answer 1")]
    rig.run(max_steps=10)

    # the executor's own read-back of the message is recorded before its tool_call; the resolving read-back of the
    # pending commit is recorded before the read's tool_call
    assert rig.types() == ["reconciliation", "tool_call", "harness_decision", "reconcile_enter",
                           "reconcile_resolved", "tool_call", "answer_attempt", "final_answer"]
    assert rig.events("reconcile_enter") == [{"step": 0, "event_type": "reconcile_enter", "stage": "after_action",
                                              "feedback": RECONCILE_FB, "status": "ok"}]
    assert rig.feedback_at(1) == dict(RECONCILE_FB, stage="after_action")
    assert rig.events("reconcile_resolved") == [{"step": 1, "event_type": "reconcile_resolved", "outcome": "committed",
                                                 "detail": "Communication/comm-first confirmed on server",
                                                 "status": "ok"}]
    # the pending commit is read back against the FIRST message's result, once per resolving read
    assert rig.gets == ["/Communication/comm-first", "/Communication/comm-first"]
    assert [c[0] for c in rig.env_calls] == ["fhir_communication_create_message", "fhir_observation_search_labs"]


RECONCILE_RETRY_HINT = {"decision": "RECONCILE", "stage": "reconcile",
                        "reason": "Read-back confirms the prior commit did NOT land -- you may re-attempt it."}


def test_d10_reconcile_resolved_failed_hint_reaches_the_agent_with_the_read_result(monkeypatch, tmp_path):
    """reconcile_resolved(failed): the re-attempt hint goes out through pending_harness_feedback and the read's own
    result stays last_res, so the agent's next turn carries both. (Written to last_res, the hint would be overwritten by
    the read's result in the same iteration and never reach the agent.)"""
    rig = _pb(monkeypatch, tmp_path)
    rig.tool_result = _messages_and_labs(rig)
    gets = []

    def get_result(path):
        gets.append(path)
        if len(gets) == 1:
            return {"resourceType": "Communication", "id": "comm-first"}
        return {"error": "HTTP 404", "query": path}
    rig.get_result = get_result
    _reconcile_first_message(rig)
    rig.replies = [_message(rig, "first"), _labs(rig), L.fc_final("answer 1")]
    rig.run(max_steps=10)

    assert rig.events("reconcile_resolved") == [{"step": 1, "event_type": "reconcile_resolved", "outcome": "failed",
                                                 "detail": "Communication/comm-first NOT found on server read-back",
                                                 "status": "ok"}]
    lab = rig.events("tool_call")[1]
    assert rig.states[2]["last_result"] == lab["result"]
    assert rig.feedback_at(2) == RECONCILE_RETRY_HINT
    # what the model is sent on that turn: the read's result as the call's tool message, then the hint
    tool_msg, hint_msg = rig.model_calls[2]["messages"][-2:]
    assert tool_msg["role"] == "tool" and tool_msg["name"] == "fhir_observation_search_labs"
    assert json.loads(tool_msg["content"]) == lab["result"]
    assert hint_msg == {"role": "user", "content": "[HARNESS RECONCILE] Read-back confirms the prior commit did NOT "
                                                   "land -- you may re-attempt it.\nRevise accordingly; the ENVIRONMENT "
                                                   "observation below is still current."}
    assert rig.types()[-2:] == ["answer_attempt", "final_answer"]


RECONCILE_BLOCK = {"event_type": "reconcile_block", "reason": "unconfirmed commit -- read back before any new write",
                   "status": "ok"}
RECONCILE_BLOCK_TEXT = ("You have an UNCONFIRMED prior commit (it timed out). Use a READ/inspect tool to read back "
                        "whether it landed BEFORE any new write.")


def test_d10_reconcile_budget_abort_and_writes_are_blocked(monkeypatch, tmp_path):
    """The message result carries no id, so every read-back of the pending commit is inconclusive (confirmed None)
    and uses one unit of the reconcile budget (3). While the commit is unreconciled, a write -- an action the kernel's
    canonicalize classifies create / update / submit from the substrate manifest -- is refused before the kernel and
    never runs: here a Communication create (create) and a write_file (update). Blocked writes use no reconcile
    budget, and the second message is not taken for a read-back."""
    rig = _pb(monkeypatch, tmp_path)
    rig.tool_result = _messages_and_labs(rig, with_id=False)
    _reconcile_first_message(rig)
    rig.replies = [_message(rig, "first"), _message(rig, "second"),
                   L.fc_tool("write_file", {"file_path": "/workspace/output/note.txt", "content": "note"}),
                   _labs(rig), _labs(rig, count=2), _labs(rig, count=3), L.fc_final("answer 1")]
    rig.run(max_steps=10)

    # each refusal takes a repair turn
    assert rig.events("reconcile_block") == [
        dict(RECONCILE_BLOCK, step=1, tool="fhir_communication_create_message", repair_turn=1),
        dict(RECONCILE_BLOCK, step=2, tool="write_file", repair_turn=2)]
    # the refusal is what the agent sees in place of each write's result
    for turn in (2, 3):
        assert rig.states[turn]["last_result"] == {"feedback": RECONCILE_BLOCK_TEXT}
        assert rig.states[turn]["last_observation"] == "reconcile_pending"
    # neither write reached the kernel or the environment
    assert [p["tool"] for h, p in rig.hooks.calls if h == "before_action"] == [
        "fhir_communication_create_message", "fhir_observation_search_labs", "fhir_observation_search_labs",
        "fhir_observation_search_labs"]
    assert [c[0] for c in rig.env_calls] == ["fhir_communication_create_message", "fhir_observation_search_labs",
                                             "fhir_observation_search_labs", "fhir_observation_search_labs"]
    assert not os.path.exists(str(rig.tmp / "job" / "workspace" / "output" / "note.txt"))
    # the third inconclusive read-back exhausts the budget before that read's tool_call is recorded
    assert _tool_calls(rig) == [(0, "fhir_communication_create_message"), (3, "fhir_observation_search_labs"),
                                (4, "fhir_observation_search_labs")]
    assert rig.trajectory[-1] == {"step": 5, "event_type": "harness_escalation", "stage": "reconcile",
                                  "reason": "reconcile_budget_exhausted", "status": "error"}
    assert rig.events("reconcile_enter")[0]["step"] == 0 and rig.events("reconcile_resolved") == []
    assert len(rig.states) == 6 and rig.gets == []


# ---------------------------------------------------------------------------------------------------- d11: deliverable write slot
DELIVERABLE_TASK = "PB-buprenorphine_mood_support"
DELIVERABLE = "/workspace/output/management_plan.txt"
# The deliverable middleware recognises the required write by the argument write_file declares for the file:
# file_path, read from the upstream registry's schema (workspaces/pb/h0/middleware/deliverable.py _path_args).
# PbNativeEnv.call_tool also accepts an undeclared `path`; such a write gets no exempt slot.
EXEMPT_WRITE = L.fc_tool("write_file", {"file_path": DELIVERABLE, "content": "plan"})


def _deliverable_file(rig):
    return str(rig.tmp / "job" / "workspace" / "output" / os.path.basename(DELIVERABLE))


def test_d11_the_deliverable_write_runs_past_the_tool_budget(monkeypatch, tmp_path):
    rig = _pb(monkeypatch, tmp_path, task=DELIVERABLE_TASK)
    rig.tool_result = lambda name, args: _lab_result(rig, name, args)
    rig.replies = [_labs(rig), _labs(rig, count=2), EXEMPT_WRITE, _labs(rig, count=3), L.fc_final("answer 1")]
    rig.run(max_steps=2)

    assert _tool_calls(rig) == [(0, "fhir_observation_search_labs"), (1, "fhir_observation_search_labs"),
                                (2, "write_file")]
    assert rig.events("tool_call")[2]["status"] == "ok" and os.path.isfile(_deliverable_file(rig))
    assert rig.feedback_at(3) is None
    # the read after the budget is refused with the budget REVISE, not executed (write_file never reaches the
    # registry dispatch, so env_calls lists only the reads)
    assert [c[0] for c in rig.env_calls] == ["fhir_observation_search_labs", "fhir_observation_search_labs"]
    assert rig.feedback_at(4) == {"decision": "REVISE", "stage": "runtime_budget", "missing_obligations": [],
                                  "reason": BUDGET_TEXT}
    assert {k: rig.result["budget"][k] for k in ("max_steps", "agent_env_actions", "harness_env_actions",
                                                 "recovery_env_actions", "total_env_actions", "budget_exhausted")} == {
        "max_steps": 2, "agent_env_actions": 3, "harness_env_actions": 0, "recovery_env_actions": 0,
        "total_env_actions": 3, "budget_exhausted": True}
    assert rig.types()[-2:] == ["answer_attempt", "final_answer"]


def test_d11_the_exempt_slot_is_used_once_even_when_the_write_did_not_land(monkeypatch, tmp_path):
    rig = _pb(monkeypatch, tmp_path, task=DELIVERABLE_TASK)
    rig.tool_result = lambda name, args: _lab_result(rig, name, args)
    writes = []

    def before_action(action, ctx):
        if action.get("tool") == "write_file":
            writes.append(action)
            if len(writes) == 1:
                return L.decision("BLOCK", rule_id="scripted_block_write", reason="scripted block of the first write")
        return None
    rig.hooks.before_action = before_action
    rig.replies = [_labs(rig), _labs(rig, count=2), EXEMPT_WRITE, EXEMPT_WRITE, L.fc_final("answer 1")]
    rig.run(max_steps=2)

    # first write: inside the exempt slot, reaches the kernel and is blocked; second write: the slot is spent, so the
    # budget refuses it before the kernel sees it
    assert len(writes) == 1
    assert _tool_calls(rig) == [(0, "fhir_observation_search_labs"), (1, "fhir_observation_search_labs")]
    assert rig.feedback_at(3) == {"decision": "BLOCK", "rule_id": "scripted_block_write",
                                  "reason": "scripted block of the first write", "stage": "before_action"}
    assert rig.feedback_at(4) == {"decision": "REVISE", "stage": "runtime_budget", "missing_obligations": [],
                                  "reason": BUDGET_TEXT}
    assert rig.result["budget"]["agent_env_actions"] == 2
    assert rig.events("deliverable_missing") != []


def test_d11_a_deliverable_write_naming_an_undeclared_path_argument_gets_no_exempt_slot(monkeypatch, tmp_path):
    """write_file declares file_path, not `path`, so past the budget a write naming the deliverable through `path` is
    refused like any other tool call; only a write through the declared file_path gets the exempt slot."""
    rig = _pb(monkeypatch, tmp_path, task=DELIVERABLE_TASK)
    rig.tool_result = lambda name, args: _lab_result(rig, name, args)
    rig.replies = [_labs(rig), _labs(rig, count=2), L.fc_tool("write_file", {"path": DELIVERABLE, "content": "plan"}),
                   L.fc_final("answer 1")]
    rig.run(max_steps=2)

    assert _tool_calls(rig) == [(0, "fhir_observation_search_labs"), (1, "fhir_observation_search_labs")]
    assert [h for h, p in rig.hooks.calls if h == "before_action"] == ["before_action", "before_action"]
    assert rig.feedback_at(3) == {"decision": "REVISE", "stage": "runtime_budget", "missing_obligations": [],
                                  "reason": BUDGET_TEXT}


# ---------------------------------------------------------------------------------------------------- d16: truncated tool call
@pytest.mark.parametrize("raw,tool,feedback", [
    ('<tool_call>{"name": "RegionAttributeDescription", "arguments": {"region": "left lo',
     "RegionAttributeDescription",
     "Your `RegionAttributeDescription` tool call was CUT OFF before the JSON closed. Re-issue the COMPLETE "
     "`RegionAttributeDescription` call now -- do NOT drop the action you intended. If the payload was long (e.g. "
     "write_file), keep it focused so the JSON finishes."),
    ('<tool_call>{"arguments": {"region": "left lo',
     None,
     "Your previous tool call was CUT OFF before the JSON closed. Re-issue the COMPLETE tool call you intended -- do "
     "NOT drop it; keep any long content focused so the JSON finishes."),
], ids=["named_tool", "no_tool_name"])
def test_d16_truncated_tool_call_is_recorded_and_the_agent_is_asked_to_reissue(monkeypatch, tmp_path, raw, tool,
                                                                               feedback):
    rig = L.Rig(monkeypatch, tmp_path, "MedCTA", "MCTA-0")
    rig.replies = [raw, '<tool_call>{"name": "ImageDescription", "arguments": {}}</tool_call>',
                   "<answer>deep vein thrombosis</answer>"]
    rig.run(max_steps=5)

    assert rig.trajectory[0] == {"step": 0, "event_type": "agent_error", "error": "truncated_tool_call",
                                 "intended_tool": tool, "raw": raw[:200], "status": "error"}
    assert rig.types()[1:3] == ["tool_call", "answer_attempt"]
    assert rig.events("tool_call")[0]["step"] == 1 and len(rig.env_calls) == 1
    assert rig.states[1]["last_result"] == {"feedback": feedback}
    assert rig.states[1]["last_observation"] == "truncated_tool_call"
    assert rig.states[1]["harness_feedback"] is None
    assert rig.model_calls[1]["messages"][-1] == {"role": "user", "content": "SYSTEM: " + feedback}
    assert rig.actuation_records() == []
