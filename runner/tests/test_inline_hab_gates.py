# -*- coding: utf-8 -*-
"""HealthAdminBench gate behaviour that a replay of recorded trajectories cannot exercise.

Covered gates: obligation snapshot -> terminal lock, precommit audit and precommit_terminal, attachment block,
collection nudge, write_misroute.

Every test drives the real loop, run.run_task(bench="HealthAdminBench"), with the real GuiEnvReal, HABAdapter,
harness kernel and recovery entry; only the browser, model, judge, scoring and network are stood in for
(runner/tests/inline_hab_stub.py). What is asserted is what the loop does: trajectory events, the feedback the agent
receives on its next turn, whether the portal action ran, counters, ledger fields and result fields. Refactors of the
loop must keep these tests green; a changed feedback text, bound, ordering or loop semantics fails them.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inline_hab_stub as S  # noqa: E402
from inline_hab_stub import click, fill, final, control, compact, set_action  # noqa: E402

DENIAL_TASK = "HAB-denial-easy-1"         # denial_triage: obligation spec = disposition, Epic doc, denial + remittance
PRIOR_AUTH_TASK = "HAB-emr-medium-1"      # submit_auth_aetna: read-acquire sources + obligation spec
FAX_TASK = "HAB-denial-medium-8"          # fax_appeal
NO_SPEC_TASK = "HAB-emr-easy-1"           # no_auth_medicare: no obligation spec
EAPPEAL_TASK = "HAB-denial-medium-2"      # electronic_appeals, Payer A portal
EAPPEAL_B_TASK = "HAB-denial-medium-3"    # electronic_appeals, Payer B portal


def _harness(monkeypatch, tmp_path, task_id):
    h = S.Harness(monkeypatch, tmp_path)
    h.use_task(S.load_hab_task(task_id))
    return h


def _clean(h):
    assert h.net == [], h.net
    assert h.portal.unscripted == [], h.portal.unscripted
    assert h.runtime_errors() == [], h.runtime_errors()


def _page_ops(h, op, before_turn=None):
    """Page actions of one kind, optionally only those issued before agent turn `before_turn` started."""
    log = h.portal.log if before_turn is None else h.portal.log[:h.agent.turns[before_turn]["page_log_len"]]
    return [x for x in log if x[0] == op]


def _revisions(h, below_step=None):
    """The kernel ledger's REVISE actuation records (actuation.record_revision), optionally below a step."""
    from harness.actuation import actuations
    return [r for r in actuations(h.kernels[0].ledger)
            if r.get("kind") == "REVISE" and (below_step is None or r.get("step") < below_step)]


# =====================================================================================================================
# Precommit audit / precommit_terminal (adapters/__init__.py; run.py)
# =====================================================================================================================
_PRECOMMIT_MSG = ("Pre-commit check blocked this submission: these REQUIRED fields are still empty: %s. Fill each one "
                  "from the available records (patient demographics / referral / coverage / clinical note) before "
                  "submitting. Do NOT resubmit until they are complete.")
_PRECOMMIT_TERMINAL_MSG = ("Required fields (%s) are still empty after repeated attempts. Do NOT submit an incomplete "
                           "form; give your best final answer from the evidence gathered.")


def _precommit_page(portal):
    portal.controls = [
        control("member-id-input", role="textbox", label="Member ID", required=True),          # required, empty
        control("reason-code-input", role="textbox", label="Reason Code *"),                  # asterisk convention
        control("attest-checkbox", role="checkbox", label="Attest", required=True, checked=False),
        control("npi-input", role="textbox", label="Provider NPI", required=True, value="1234567890"),  # complete
        control("fax-input", role="textbox", label="Fax", required=True, disabled=True),       # disabled counts done
        control("hidden-input", role="textbox", label="Hidden", required=True, visible=False),  # invisible counts done
        control("submit-auth-button", label="Submit Authorization"),
    ]


def test_b5_precommit_three_audit_rounds_then_terminal_refusals(monkeypatch, tmp_path):
    h = _harness(monkeypatch, tmp_path, PRIOR_AUTH_TASK)
    _precommit_page(h.portal)
    S.probe_tab(h.portal)
    submit = click("submit-auth-button")
    h.run([S.reveal_then(h.portal, submit)] + [submit] * 5 + [final()], max_steps=12)
    _clean(h)

    empty = ["Member ID", "Reason Code *", "Attest"]
    fields = "Member ID, Reason Code *, Attest"
    audit = [{"step": s, "event_type": "precommit_audit", "round": s + 1, "blocked_tool": "click",
              "required_empty": empty, "status": "ok"} for s in range(3)]
    terminal = [{"step": s, "event_type": "precommit_terminal", "blocked_tool": "click", "required_empty": empty,
                 "status": "ok"} for s in range(3, 6)]
    assert [compact(e) for e in h.traj[:6]] == audit + terminal

    # the refused submits never reached the portal (nothing but the reset page load before the final's turn) and
    # were never recorded or charged as env actions
    assert [x[0] for x in h.portal.log[:h.agent.turns[6]["page_log_len"]]] == ["goto"]
    assert [e for e in h.events("tool_call") if e.get("origin") != "harness"] == []
    assert h.result["budget"]["agent_env_actions"] == 0
    # each refusal is a repair-type result: one repair turn and one record each; the 5th and 6th are past the budget
    assert [(r["stage"], r["step"], r["source"], r["repair_turn"], r["repair_turn_started"])
            for r in _revisions(h, below_step=6)] == [("before_tool", s, "host:Revision", s + 1, s < 4) for s in range(6)]

    # what the agent receives on the next turn
    for turn in (1, 2, 3):
        msg = _PRECOMMIT_MSG % fields
        assert h.feedback(turn) == {"decision": "REVISE", "stage": "precommit", "reason": msg}
        assert h.agent.turns[turn]["last_result"] == {"feedback": msg}
        assert h.agent.turns[turn]["last_observation"] == "precommit_block"
    for turn in (4, 5, 6):
        msg = _PRECOMMIT_TERMINAL_MSG % fields
        assert h.feedback(turn) == {"decision": "REVISE", "stage": "precommit_terminal", "reason": msg}
        assert h.agent.turns[turn]["last_result"] == {"feedback": msg}
        assert h.agent.turns[turn]["last_observation"] == "precommit_terminal"

    # precommit unlocks repair (_in_repair): the revealed, unmet read-evidence tab is never dispatched
    assert S.probe_dispatch_steps(h) == []
    assert h.traj[6]["step"] == 6                                         # the next thing that happens is the final


def test_b5_precommit_refusal_draws_a_repair_turn(monkeypatch, tmp_path):
    """One precommit refusal, then the budget is spent and the agent keeps calling tools. The runtime_budget refusal
    aborts once repair turns exceed 4. The precommit refusal draws repair turn 1 (every repair-type result is accounted
    the same way), so the abort comes on the refusal's 4th occurrence, at step 7."""
    h = _harness(monkeypatch, tmp_path, DENIAL_TASK)
    h.portal.controls = [control("disposition-input", role="textbox", label="Disposition", required=True),
                         control("submit-disposition-button", label="Submit Disposition"),
                         control("noop", label="Refresh")]
    h.run([click("submit-disposition-button")] + [click("noop")] * 8, max_steps=3)
    _clean(h)
    assert [(e.get("step"), e["event_type"]) for e in h.traj if e.get("origin") != "harness"
            and e.get("event_type") != "recovery_v3"] == [
        (0, "precommit_audit"), (1, "tool_call"), (2, "tool_call"), (3, "tool_call"), (7, "tool_budget_exhausted")]
    budget_msg = ("Your exploration budget is used up. Do NOT call another tool; give your best final answer from "
                  "the evidence already gathered.")
    for turn in (5, 6, 7):
        assert h.feedback(turn) == {"decision": "REVISE", "stage": "runtime_budget", "missing_obligations": [],
                                    "reason": budget_msg}
    assert len(h.agent.turns) == 8
    assert h.result["budget"]["agent_env_actions"] == 3


# =====================================================================================================================
# Attachment block (adapters/__init__.py; run.py)
# =====================================================================================================================
_ATTACH_MSG = ("You are about to SEND, but you have downloaded document(s) and attached NONE. Attach the document(s) "
               "this submission requires before you send -- and only the correct ones.")


def _fax_page(portal, attached):
    portal.state["fax"] = {"attachmentNames": list(attached), "faxRecipient": "555-0100"}
    portal.controls = [control("download-clinical-note", label="Download Records"),
                       control("send-fax-button", label="Send Fax")]
    portal.downloads = {"download-clinical-note": "clinical_note.pdf"}
    portal.path = "/fax-portal"

    def send(p, ref):   # a click on Send Fax is a classified commit: the portal persists the send, as the real one does
        p.state["fax"]["faxesSent"] = 1
        p.text = "Fax queued"
    portal.on_click = {"send-fax-button": send}


def test_b6_attachment_block_twice_then_send_proceeds(monkeypatch, tmp_path):
    h = _harness(monkeypatch, tmp_path, FAX_TASK)
    _fax_page(h.portal, attached=[])
    send = click("send-fax-button")
    h.run([{"type": "tool_call", "tool": "download", "args": {"ref": "download-clinical-note"}}, send, send, send,
           final()], max_steps=12)
    _clean(h)
    assert [compact(e) for e in h.traj[:4]] == [
        {"step": 0, "event_type": "tool_call", "tool": "download", "args": {"ref": "download-clinical-note"},
         "status": "ok", "origin": "agent", "audience": "agent"},
        {"event_type": "attachment_block", "status": "ok", "step": 1, "round": 1},
        {"event_type": "attachment_block", "status": "ok", "step": 2, "round": 2},
        {"step": 3, "event_type": "tool_call", "tool": "click", "args": {"ref": "send-fax-button"}, "status": "ok",
         "origin": "agent", "audience": "agent"},
    ]
    assert sorted(h.env._download_registry) == ["download-clinical-note"]
    # the two refused sends never reached the portal; the third ran once
    assert h.portal.log[1:3] == [("click", "download-clinical-note"), ("click", "send-fax-button")]
    for turn in (2, 3):
        assert h.feedback(turn) == {"decision": "REVISE", "stage": "attachment", "reason": _ATTACH_MSG}
        assert h.agent.turns[turn]["last_result"] == {"feedback": _ATTACH_MSG}
        assert h.agent.turns[turn]["last_observation"] == "attachment_block"
    assert h.feedback(4) is None
    assert h.result["budget"]["agent_env_actions"] == 2


def test_b6_no_block_when_something_is_attached(monkeypatch, tmp_path):
    h = _harness(monkeypatch, tmp_path, FAX_TASK)
    _fax_page(h.portal, attached=["clinical_note.pdf"])
    h.run([{"type": "tool_call", "tool": "download", "args": {"ref": "download-clinical-note"}},
           click("send-fax-button"), final()], max_steps=12)
    _clean(h)
    assert h.events("attachment_block") == []
    assert [compact(e)["tool"] for e in h.traj if e.get("step") in (0, 1)] == ["download", "click"]


# =====================================================================================================================
# Collection nudge (adapters/__init__.py; run.py)
# =====================================================================================================================
_COLLECTION_MSG = ("The diagnosis section of the authorization form has 1 code(s), but you saw 2 diagnosis code(s) in "
                   "the records. Go back and add the diagnosis codes you already observed -- do not invent any.")


def _auth_form(portal, dx_label="Diagnosis Code"):
    portal.text = "Referral diagnoses: M17.11 primary osteoarthritis, right knee; M25.561 pain in right knee"
    portal.state["emr"] = {"priorAuthForm": {"diagnosisCodes": ["M17.11"]}}
    portal.controls = [control("dx-input", role="textbox", label=dx_label, value="M17.11"),
                       control("submit-auth-button", label="Submit Authorization")]
    portal.path = "/payer-a/prior-auth"

    def submit(p, ref):  # a click on Submit is a classified commit: the portal persists the submission, as the real one does
        p.state["payerA"]["submissions"] = [{"requestType": "outpatient"}]
        p.text = "Authorization submitted"
    portal.on_click = {"submit-auth-button": submit}


def test_b7_collection_nudge_twice_then_submit_proceeds(monkeypatch, tmp_path):
    h = _harness(monkeypatch, tmp_path, PRIOR_AUTH_TASK)
    _auth_form(h.portal)
    submit = click("submit-auth-button")
    h.run([submit, submit, submit, final()], max_steps=12)
    _clean(h)
    assert [compact(e) for e in h.traj[:3]] == [
        {"event_type": "collection_nudge", "status": "ok", "step": 0, "round": 1},
        {"event_type": "collection_nudge", "status": "ok", "step": 1, "round": 2},
        {"step": 2, "event_type": "tool_call", "tool": "click", "args": {"ref": "submit-auth-button"}, "status": "ok",
         "origin": "agent", "audience": "agent"},
    ]
    assert h.portal.log[1:2] == [("click", "submit-auth-button")]                # only the third submit ran
    for turn in (1, 2):
        assert h.feedback(turn) == {"decision": "REVISE", "stage": "collection", "reason": _COLLECTION_MSG}
        assert h.agent.turns[turn]["last_result"] == {"feedback": _COLLECTION_MSG}
        assert h.agent.turns[turn]["last_observation"] == "collection_nudge"
    assert h.feedback(3) is None


def test_b7_cpt_list_checked_once_the_diagnosis_list_is_covered(monkeypatch, tmp_path):
    """Diagnosis first, then CPT: with the observed ICD code already on the form, the observed CPT/HCPCS codes (2)
    against the form's cptCodes (1) produce the cpt reminder."""
    h = _harness(monkeypatch, tmp_path, PRIOR_AUTH_TASK)
    _auth_form(h.portal)
    h.portal.text = "Referral diagnosis M17.11. Procedures: 67028 intravitreal injection; J2778 ranibizumab"
    h.portal.state["emr"]["priorAuthForm"]["cptCodes"] = ["67028"]
    h.portal.controls.insert(1, control("cpt-input", role="textbox", label="CPT Code", value="67028"))
    h.run([click("submit-auth-button"), final()], max_steps=12)
    _clean(h)
    msg = ("The cpt section of the authorization form has 1 code(s), but you saw 2 cpt code(s) in the records. Go back "
           "and add the cpt codes you already observed -- do not invent any.")
    assert compact(h.traj[0]) == {"event_type": "collection_nudge", "status": "ok", "step": 0, "round": 1}
    assert h.feedback(1) == {"decision": "REVISE", "stage": "collection", "reason": msg}


def test_b7_no_nudge_on_a_page_without_a_code_field(monkeypatch, tmp_path):
    h = _harness(monkeypatch, tmp_path, PRIOR_AUTH_TASK)
    _auth_form(h.portal, dx_label="Member ID")
    h.run([click("submit-auth-button"), final()], max_steps=12)
    _clean(h)
    assert h.events("collection_nudge") == []
    assert compact(h.traj[0])["tool"] == "click" and h.traj[0]["step"] == 0


def test_b6_b7_refusals_do_not_unlock_repair(monkeypatch, tmp_path):
    """Attachment and collection refusals are Nudges: recorded like every repair-type result but they leave
    _in_repair False, so on the next turn the prior-auth read-acquire dispatches the tab revealed during the refused
    turn. (Contrast: the precommit refusal and write_misroute unlock repair.)"""
    for gate in ("attachment", "collection"):
        mp = monkeypatch
        with mp.context() as m:
            h = _harness(m, tmp_path / gate, PRIOR_AUTH_TASK)
            if gate == "attachment":
                _fax_page(h.portal, attached=[])
                h.portal.controls[1] = control("submit-auth-button", label="Submit Authorization")
                first = [{"type": "tool_call", "tool": "download", "args": {"ref": "download-clinical-note"}}]
            else:
                _auth_form(h.portal)
                first = []
            S.probe_tab(h.portal)
            h.run(first + [S.reveal_then(h.portal, click("submit-auth-button")), final()], max_steps=12)
            _clean(h)
            gate_step = len(first)
            gate_event = {"attachment": "attachment_block", "collection": "collection_nudge"}[gate]
            assert [e["step"] for e in h.events(gate_event)] == [gate_step]
            assert S.probe_dispatch_steps(h) == [gate_step + 1]
            dispatched = [compact(e) for e in h.traj if e.get("step") == gate_step + 1]
            assert dispatched == [
                {"step": gate_step + 1, "event_type": "read_acquire_dispatch", "evidence_flag": "clickedDiagnosesTab",
                 "ref": S.PROBE_TAB, "op": "click", "attempt": 1, "status": "ok"},
                {"step": gate_step + 1, "event_type": "harness_acquire_dispatch", "tool": "click",
                 "args": {"ref": S.PROBE_TAB, "provenance": "harness-acquired-evidence",
                          "evidence_flag": "clickedDiagnosesTab"}, "status": "ok"},
                {"step": gate_step + 1, "event_type": "tool_call", "tool": "click",
                 "args": {"ref": S.PROBE_TAB, "provenance": "harness-acquired-evidence",
                          "evidence_flag": "clickedDiagnosesTab"}, "status": "ok", "origin": "harness",
                 "requester": "next_read",
                 "audience": "both"},
            ]
            assert [(r["source"], r["repair_turn"]) for r in _revisions(h) if r["step"] == gate_step] == [
                ("host:Nudge", None)]


# =====================================================================================================================
# write_misroute via field_intent (adapters/__init__.py; run.py)
# =====================================================================================================================
_MISROUTE_MSG = ("Your value landed in the 'Member ID' control, but your stated field intent was 'diagnosis code'. The "
                 "required 'Diagnosis Code' control (ref=dx-input) is still empty -- enter the value there instead. The "
                 "harness does not supply the value; you already have it.")


def _misroute_page(portal):
    portal.controls = [control("member-id-input", role="textbox", label="Member ID"),
                       control("dx-input", role="textbox", label="Diagnosis Code", required=True)]
    S.probe_tab(portal)


def test_b9_write_misroute_rounds_capped_at_four(monkeypatch, tmp_path):
    h = _harness(monkeypatch, tmp_path, PRIOR_AUTH_TASK)
    _misroute_page(h.portal)
    w = fill("member-id-input", "M17.11", field_intent="diagnosis code")
    h.run([S.reveal_then(h.portal, w)] + [w] * 4 + [final()], max_steps=12)
    _clean(h)
    tool_ev = {"event_type": "tool_call", "tool": "fill",
               "args": {"ref": "member-id-input", "text": "M17.11", "field_intent": "diagnosis code"},
               "status": "ok", "origin": "agent", "audience": "agent"}
    expected = []
    for s in range(5):
        expected.append(dict(tool_ev, step=s))
        if s < 4:
            expected.append({"step": s, "event_type": "write_misroute", "round": s + 1, "field_intent": "diagnosis code",
                             "landed_control": "Member ID", "landed_ref": "member-id-input", "correct_ref": "dx-input",
                             "correct_label": "Diagnosis Code", "status": "ok"})
    assert [compact(e) for e in h.traj[:9]] == expected
    # the write ran every time (post-write gate), and its receipt is what the gate read
    assert _page_ops(h, "fill", before_turn=5) == [("fill", "member-id-input", "M17.11")] * 5
    assert h.events("tool_call")[0]["result"]["write_receipt"] == {
        "ref": "member-id-input", "field": "diagnosis code", "control_label": "Member ID", "control_section": "",
        "attempted": "M17.11", "observed": "M17.11", "landed": True}
    for turn in (1, 2, 3, 4):
        assert h.feedback(turn) == {"decision": "REVISE", "stage": "misroute", "reason": _MISROUTE_MSG}
        assert h.agent.turns[turn]["last_result"] == {"feedback": _MISROUTE_MSG}
        assert h.agent.turns[turn]["last_observation"] == "write_misroute"
    assert h.feedback(5) is None
    # misroute unlocks repair: the tab revealed at turn 0 is never dispatched
    assert S.probe_dispatch_steps(h) == []


def test_b9_without_field_intent_no_misroute_and_read_acquire_continues(monkeypatch, tmp_path):
    h = _harness(monkeypatch, tmp_path, PRIOR_AUTH_TASK)
    _misroute_page(h.portal)
    h.run([S.reveal_then(h.portal, fill("member-id-input", "M17.11")), final()], max_steps=12)
    _clean(h)
    assert h.events("write_misroute") == []
    assert S.probe_dispatch_steps(h) == [1]


# =====================================================================================================================
# Obligation snapshot -> ledger -> terminal lock (run.py, obligation_gate.lock_inputs; kernel.after_action;
# verify_commit.py). The R2 lock needs an R2 action: a tool named `submit` (not among the task's declared tools -- the
# scripted agent calls it anyway) or a click on a commit button the shell's classifier confirms on the page. The
# observed-submission lock needs no R2 action (test_hab_boundary.py). The conjunction is the composition of the goal's
# requirements and the effects of the agent's own commits.
# =====================================================================================================================
_LOCK_FB = {"decision": "BLOCK", "rule_id": "post_verified_mutation",
            "reason": "the graded commit is already verified-complete; further mutation can only corrupt it",
            "message": "The task's commit is already verified complete -- do NOT mutate further; finalize.",
            "stage": "before_action"}
_REDUNDANT_FB = {"decision": "BLOCK", "rule_id": "redundant_commit",
                 "reason": "this irreversible commit already succeeded; re-executing risks corrupting it",
                 "message": "You already completed this commit successfully — do NOT re-submit/re-create it; "
                            "finalize instead.",
                 "stage": "before_action"}


def _denial_page(portal):
    portal.controls = [control("tab-denial-details", label="Denial Details"),
                       control("tab-remittance", label="Remittance Image"),
                       control("document-epic-button", label="Document in Epic"),
                       control("submit-disposition-button", label="Submit Disposition"),
                       control("triage-note", role="textbox", label="Triage Note")]
    portal.on_click = {"tab-denial-details": set_action("viewedDenialDetails"),
                       "tab-remittance": set_action("viewedRemittanceImage"),
                       "document-epic-button": set_action("documentedAppealInEpic"),
                       "submit-disposition-button": set_action("selectedDisposition", "Route to Clinical Appeals")}


SUBMIT = {"type": "tool_call", "tool": "submit", "args": {"ref": "submit-disposition-button"}}
NOTE = fill("triage-note", "Reviewed; route to appeals.")


def _block_event(rule_id, step):
    return {"event_type": "harness_decision", "step": step, "stage": "before_action", "mode": "enforce", "capability": "verify_commit",
            "decision": "BLOCK", "effective": "BLOCK", "rule_id": rule_id, "missing_obligations": [],
            "deterministic": True}


def test_b2_submission_last_locks_and_blocks_the_next_mutation(monkeypatch, tmp_path):
    h = _harness(monkeypatch, tmp_path, DENIAL_TASK)
    assert "submit" not in [t["name"] for t in h.task["available_tools"]]      # `submit` is not a declared tool
    _denial_page(h.portal)
    rows = h.spy_ledger()
    h.run([click("tab-denial-details"), click("tab-remittance"), click("document-epic-button"), SUBMIT, NOTE, final()],
          max_steps=12)
    _clean(h)
    # (step, tool, public_obligations_met, observed_submission_realized, terminal_locked, decision). Before the submit
    # the goal's two requirements (the denial opened, the triage note) hold at step 2; nothing locks without a submission.
    assert rows == [(0, "click", False, False, False, "ALLOW"), (1, "click", False, False, False, "ALLOW"),
                    (2, "click", True, False, False, "ALLOW"), (3, "submit", True, True, True, "ALLOW"),
                    (5, "snapshot", True, False, True, "ALLOW")]          # recovery's snapshot at the final
    kern = h.kernels[0]
    assert kern.ledger.terminal_locked is True and len(kern.ledger.completed_commits) == 1
    # the fill at step 4 is blocked before it runs, and draws repair turn 1 (run.py's kernel REVISE/BLOCK branch)
    assert [compact(e) for e in h.traj if e.get("event_type") == "harness_decision"] == [
        _block_event("post_verified_mutation", 4)]
    assert _revisions(h) == [{"kind": "REVISE", "stage": "before_action", "step": 4, "capability": "verify_commit",
                              "revision_delivered": True, "repair_turn_started": True, "repair_turn": 1,
                              "suppressed_reason": None, "source": "kernel:BLOCK"}]
    assert _page_ops(h, "fill", before_turn=5) == []
    assert [e["step"] for e in h.events("tool_call") if e["origin"] == "agent"] == [0, 1, 2, 3]
    assert h.feedback(5) == _LOCK_FB


def test_b2_submission_first_no_lock_later_mutation_runs_resubmit_blocked(monkeypatch, tmp_path):
    h = _harness(monkeypatch, tmp_path, DENIAL_TASK)
    _denial_page(h.portal)
    rows = h.spy_ledger()
    h.run([SUBMIT, click("tab-denial-details"), click("tab-remittance"), click("document-epic-button"), NOTE, SUBMIT,
           final()], max_steps=12)
    _clean(h)
    assert rows == [(0, "submit", False, True, False, "ALLOW"), (1, "click", False, False, False, "ALLOW"),
                    (2, "click", False, False, False, "ALLOW"), (3, "click", True, False, False, "ALLOW"),
                    (4, "fill", True, False, False, "ALLOW"),
                    (6, "snapshot", True, False, False, "ALLOW")]         # recovery's snapshot at the final
    kern = h.kernels[0]
    assert kern.ledger.terminal_locked is False and len(kern.ledger.completed_commits) == 1
    assert _page_ops(h, "fill", before_turn=6) == [("fill", "triage-note", "Reviewed; route to appeals.")]
    assert [compact(e) for e in h.traj if e.get("event_type") == "harness_decision"] == [_block_event("redundant_commit", 5)]
    assert h.feedback(6) == _REDUNDANT_FB
    assert _page_ops(h, "click", before_turn=6).count(("click", "submit-disposition-button")) == 1


def test_b2_the_declared_click_submit_is_a_commit_and_locks(monkeypatch, tmp_path):
    """The same order through the task's declared tools: the submission is a `click` on Submit Disposition, which the
    manifest's commit rule and the shell's classifier make an irreversible commit (R2), not an inspect (R0). It is
    verified on the state change, observed as the agent's submission, the composition holds, the task locks and the
    later note fill is blocked."""
    h = _harness(monkeypatch, tmp_path, DENIAL_TASK)
    _denial_page(h.portal)
    rows = h.spy_ledger()
    h.run([click("tab-denial-details"), click("tab-remittance"), click("document-epic-button"),
           click("submit-disposition-button"), NOTE, final()], max_steps=12)
    _clean(h)
    assert rows == [(0, "click", False, False, False, "ALLOW"), (1, "click", False, False, False, "ALLOW"),
                    (2, "click", True, False, False, "ALLOW"), (3, "click", True, True, True, "ALLOW"),
                    (5, "snapshot", True, False, True, "ALLOW")]          # recovery's snapshot at the final
    kern = h.kernels[0]
    assert kern.ledger.terminal_locked is True and len(kern.ledger.completed_commits) == 1
    assert [(c["action"], c["semantic_type"], c["verified"]) for c in kern.ledger.commit_history] == [
        ("click", "submit", True)]
    assert [compact(e) for e in h.traj if e.get("event_type") == "harness_decision"] == [
        _block_event("post_verified_mutation", 4)]
    assert _page_ops(h, "fill", before_turn=5) == []


def _without_completion_content(monkeypatch, tmp_path):
    import shutil
    import yaml
    import workspace as W
    ws = str(tmp_path / "hab_ws")
    shutil.copytree(W.H0["gui"], ws)
    mp = os.path.join(ws, "manifest.yaml")
    manifest = yaml.safe_load(open(mp, encoding="utf-8"))
    del manifest["completion_obligations"]
    open(mp, "w", encoding="utf-8").write(yaml.safe_dump(manifest, sort_keys=False))
    monkeypatch.setitem(W.H0, "gui", ws)


def test_b2_without_an_obligation_model_the_verified_submit_locks(monkeypatch, tmp_path):
    """No obligation model (a workspace without completion content): both lock inputs are None and a verified R2 submit
    locks, as on a benchmark without an obligation model."""
    _without_completion_content(monkeypatch, tmp_path)
    h = _harness(monkeypatch, tmp_path, NO_SPEC_TASK)
    _denial_page(h.portal)
    rows = h.spy_ledger()
    h.run([SUBMIT, NOTE, final()], max_steps=12)
    _clean(h)
    assert rows == [(0, "submit", None, None, True, "ALLOW"),
                    (2, "snapshot", None, None, True, "ALLOW")]           # recovery's snapshot at the final
    assert [compact(e) for e in h.traj if e.get("event_type") == "harness_decision"] == [
        _block_event("post_verified_mutation", 1)]
    assert _page_ops(h, "fill", before_turn=2) == []
    assert h.feedback(2) == _LOCK_FB


def test_b2_an_empty_composition_never_locks(monkeypatch, tmp_path):
    """With the obligation model, a verified R2 submit on a control no decision names, on a task whose goal composes
    nothing, leaves the composition empty: not met, nothing locks, the later fill runs."""
    h = _harness(monkeypatch, tmp_path, NO_SPEC_TASK)
    _denial_page(h.portal)
    h.portal.controls.append(control("submit-other-button", label="Submit"))
    h.portal.on_click["submit-other-button"] = set_action("otherSubmitted")
    rows = h.spy_ledger()
    h.run([{"type": "tool_call", "tool": "submit", "args": {"ref": "submit-other-button"}}, NOTE, final()],
          max_steps=12)
    _clean(h)
    assert rows[:2] == [(0, "submit", False, False, False, "ALLOW"), (1, "fill", False, False, False, "ALLOW")]
    assert h.kernels[0].ledger.terminal_locked is False
    assert _page_ops(h, "fill", before_turn=2) == [("fill", "triage-note", "Reviewed; route to appeals.")]


@pytest.mark.parametrize("task_id, filed_on, lands", [(EAPPEAL_TASK, "payerA", True), (EAPPEAL_TASK, "payerB", False),
                                                    (EAPPEAL_B_TASK, "payerB", True), (EAPPEAL_B_TASK, "payerA", False)])
def test_b2_electronic_appeals_submission_conjunct_is_the_tasks_payers_appeal(monkeypatch, tmp_path, task_id, filed_on,
                                                                              lands):
    """electronic_appeals: conjunct 0 is the appeal submitted on the payer the public goal names, read where
    GuiEnvReal.eval_state puts a payer slice's appeal actions, payer_<x>_state.full_state.appealActions (the native
    checkpoint's path; _build_payer_state has no top-level appealActions, so a conjunct reading
    payer_{a,b}_state.appealActions would never observe an appeal). An appeal filed on the task's own payer, with the
    rest of the conjunction met, is observed as the submission, meets the conjunction and locks after the R2 submit;
    the final is then accepted and no obligation nudge fires. One filed on the other payer's portal satisfies nothing:
    nothing is observed or locked, and recovery (whose conjunction is the same) engages."""
    h = _harness(monkeypatch, tmp_path, task_id)
    p = h.portal
    p.controls = [control("tab-denial-details", label="Denial Details"),
                  control("tab-remittance", label="Remittance Image"),
                  control("document-epic-button", label="Document in Epic"),
                  control("submit-appeal-button", label="Submit Appeal")]

    def appeal(portal, ref):
        portal.state[filed_on]["appealActions"] = {"submittedAppeal": True}
        portal.text = "Appeal submitted"
    p.on_click = {"tab-denial-details": set_action("viewedDenialDetails"),
                  "tab-remittance": set_action("viewedRemittanceImage"),
                  "document-epic-button": set_action("documentedAppealInEpic"),
                  "submit-appeal-button": appeal}
    rows = h.spy_ledger()
    h.run([click("tab-denial-details"), click("tab-remittance"), click("document-epic-button"),
           {"type": "tool_call", "tool": "submit", "args": {"ref": "submit-appeal-button"}}, final()], max_steps=12)
    _clean(h)
    prefix = [(0, "click", False, False, False, "ALLOW"), (1, "click", False, False, False, "ALLOW"),
              (2, "click", True, False, False, "ALLOW")]       # the goal's requirements hold, no submission composed yet
    st = h.env.eval_state()
    payer = {"payerA": "payer_a_state", "payerB": "payer_b_state"}[filed_on]
    assert st[payer]["full_state"]["appealActions"] == {"submittedAppeal": True}
    assert "appealActions" not in st["payer_a_state"] and "appealActions" not in st["payer_b_state"]
    if lands:
        assert rows == prefix + [(3, "submit", True, True, True, "ALLOW"),
                                 (4, "snapshot", True, False, True, "ALLOW")]    # recovery's snapshot at the final
        assert h.kernels[0].ledger.terminal_locked is True
        assert h.events("obligation_nudge") == []
    else:
        assert rows == prefix + [(3, "submit", False, False, False, "ALLOW")] + [
            (s, "snapshot", False, False, False, "ALLOW") for s in (4, 4, 5, 5)]  # recovery's snapshots at the finals
        assert h.kernels[0].ledger.terminal_locked is False
        assert [e["step"] for e in h.events("recovery_v3")] == [4, 5]
