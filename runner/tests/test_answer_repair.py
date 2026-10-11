# -*- coding: utf-8 -*-
"""verify_commit's repair policy is workspace kernel middleware, separate from the shell's veto.

The fixed shell (runner/harness/capabilities/verify_commit.py) audits a final answer once, under the verification
reserve, and keeps the veto: a localized hard defect is REVISE evidence_contradiction (must-resolve). What to do about a
REPAIRABLE gap is MedCTA's workspaces/medcta/h0/middleware/answer_repair.py, which reads the shell's audit from the
read-only ctx.final_audit and makes no judge call. A stand-in judge that fills every list of the audit template or none
never produces a gap-only audit (a filled audit always carries a localized hard defect), so these tests script the
audit and pin the split: the kernel hook with its budget and judge accounting, and the real run.run_task loop on MedCTA
through runner/tests/inline_loop_rig.py, with and without answer_repair in the workspace manifest.
"""
import json
import os
import shutil
import sys

import pytest
import yaml

_RUNNER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RUNNER not in sys.path:
    sys.path.insert(0, _RUNNER)

import harness as H                                                    # noqa: E402
import workspace as W                                                  # noqa: E402
from harness import decision as D                                      # noqa: E402
from harness.capability import FinalAudit                              # noqa: E402
from harness.capabilities.obligation_lifecycle import ObligationLifecycle   # noqa: E402
from harness.capabilities.scope_evidence import ScopeEvidenceBinding   # noqa: E402
from harness.capabilities.verify_commit import VerifyAndCommit         # noqa: E402
from harness.compiler import build_contract                            # noqa: E402
from harness.kernel import HarnessKernel                               # noqa: E402

AUDITOR = "You are a clinical answer auditor"
ENTRY = "middleware/answer_repair.py:AnswerRepair"


def _audit(addresses_task=True, hard=(), gaps=(), confidence=0.9):
    return json.dumps({"addresses_task": addresses_task, "hard_violations": list(hard),
                       "repairable_gaps": list(gaps), "confidence": confidence})


HARD = {"type": "contradiction", "claim": "LUL effusion", "evidence_ids": ["E1"], "reason": "LUL vs RUL"}
GAP_UNANSWERED = {"type": "unanswered", "claim": "x", "evidence_ids": ["E1"], "critique": "no stain named"}
GAP_OVER = {"type": "over_specific", "claim": "a 3.1 cm nodule", "evidence_ids": ["E1"], "critique": "size unsupported"}
AUDIT_CLEAN = _audit()
AUDIT_HARD = _audit(hard=[HARD])
AUDIT_GAP_NOT_ADDRESSED = _audit(addresses_task=False, gaps=[GAP_UNANSWERED], confidence=0.4)
AUDIT_GAP_CONFIDENT = _audit(gaps=[GAP_OVER], confidence=0.9)
AUDIT_GAP_LOW = _audit(gaps=[GAP_OVER], confidence=0.4)


def _answer_repair_class():
    return W.Workspace(W.H0["tool_sandbox"]).load("middleware/answer_repair.py", "AnswerRepair")


def _kernel(judge_reply, with_repair=True):
    """MedCTA's policy, the shell's capabilities that act at before_final, and (optionally) MedCTA's answer_repair
    loaded from its default workspace (so the kernel hands it the WorkspaceContextView). One validated image observation
    satisfies the grounding prerequisite, so the final reaches verify_commit's audit."""
    pol = H.load_policy(env_type="tool_sandbox", workspace=W.H0["tool_sandbox"])
    task = {"task_id": "m", "goal": "finding?", "context": {}, "environment": {"type": "tool_sandbox"}}
    contract = build_contract(task, env_type="tool_sandbox", policy=pol)
    prompts = []

    def judge(prompt):
        prompts.append(prompt)
        return judge_reply
    caps = [ScopeEvidenceBinding(), ObligationLifecycle(), VerifyAndCommit()]
    if with_repair:
        caps.append(_answer_repair_class()())
    k = HarnessKernel(contract, caps, policy=pol, env_type="tool_sandbox", judge_fn=judge,
                      budget={"max_semantic_checks": 5})
    k.after_action({"type": "tool", "tool": "ImageDescription", "args": {}}, "a 3cm spiculated RUL nodule",
                   None, None, step=0, result_ok=True)
    return k, prompts


def _pools(k):
    m = k.ctx.semantic_meter
    return m.pool_spent("reserve"), m.pool_spent("general"), m.refused_total()


# ------------------------------------------------------------------------------------------------ kernel hook
@pytest.mark.parametrize("with_repair", [True, False])
def test_the_shell_still_vetoes_a_localized_hard_defect(with_repair):
    k, prompts = _kernel(AUDIT_HARD, with_repair)
    eff = k.before_final("LUL effusion", step=1)
    assert eff.type == D.REVISE
    assert (eff.raw.capability, eff.raw.reason_code, eff.raw.rule_id) == (
        "verify_commit", "evidence_contradiction", "claim_supported_by_evidence")
    assert eff.raw.extra["must_resolve"] is True and eff.raw.extra["side_effecting"] is False
    assert k.ledger.pending_resolution["violation_type"] == "audit_hard" and k.ctx.verification is False
    # the veto is the only finding: answer_repair sees vetoed=True and stays silent
    assert [(f["capability"], f["reason_code"]) for f in k.ledger.findings] == [
        ("verify_commit", "evidence_contradiction")]
    assert k.ctx.final_audit.vetoed is True
    assert len(prompts) == 1 and prompts[0].startswith(AUDITOR) and _pools(k) == (1, 0, 0)


def test_answer_repair_emits_the_repair_revise_from_the_shell_audit():
    k, prompts = _kernel(AUDIT_GAP_NOT_ADDRESSED)
    eff = k.before_final("a nodule", step=1)
    assert eff.type == D.REVISE and eff.raw.capability == "answer_repair"
    # the full process_gap REVISE, field for field, raised under answer_repair's capability name
    assert eff.raw.to_dict() == {
        "decision": "REVISE", "stage": "before_final", "capability": "answer_repair", "rule_id": "process_gap",
        "reason_code": "process_gap", "reason": "process incomplete for this conclusion (unanswered): no stain named",
        "missing_obligations": [], "suggested_capabilities": [], "deterministic": False, "risk": None,
        "feedback": "Before you finalize, your PROCESS is incomplete: no stain named. Go back and DO the work with "
                    "your tools -- examine the distinguishing feature / rule out the alternative / gather the missing "
                    "evidence -- then answer again. You are NOT being told the answer is wrong; you are being told "
                    "the supporting work is not yet done.",
        "extra": {"audit": {"addresses_task": False, "hard_violations": [], "repairable_gaps": [GAP_UNANSWERED],
                            "confidence": 0.4, "reason": None},
                  "side_effecting": False, "process_gap": True, "gap_type": "unanswered",
                  "critique": "no stain named", "verification_flag": "unverified_grounding"}}
    assert list(eff.raw.extra) == ["audit", "side_effecting", "process_gap", "gap_type", "critique",
                                   "verification_flag"]
    # an acquisition-suitable gap is routed to the acquisition target
    assert k._workspace_state["pending_required_element"] == "no stain named"   # the workspace's own state
    assert k.ledger.pending_resolution is None and k.ctx.verification is None
    assert [(f["capability"], f["reason_code"]) for f in k.ledger.findings] == [("answer_repair", "process_gap")]
    # one audit, paid once from the verification reserve; the middleware made no judge call and spent nothing
    assert len(prompts) == 1 and prompts[0].startswith(AUDITOR) and _pools(k) == (1, 0, 0)


def test_the_confidence_threshold_moved_with_the_policy():
    hi, _p = _kernel(AUDIT_GAP_CONFIDENT)
    d = hi.before_final("a 3.1 cm nodule", step=1)
    assert (d.type, d.raw.capability, d.raw.reason_code) == (D.REVISE, "answer_repair", "process_gap")
    assert hi._workspace_state.get("pending_required_element") is None   # over_specific is no acquisition target
    lo, _p = _kernel(AUDIT_GAP_LOW)
    assert lo.before_final("a 3.1 cm nodule", step=1).type == D.ALLOW and lo.ctx.verification is True
    import harness.capabilities.verify_commit as VC
    assert not hasattr(VC, "REPAIR_MIN_CONF") and not hasattr(VerifyAndCommit, "_repair_min_conf")
    assert sys.modules[_answer_repair_class().__module__].REPAIR_MIN_CONF == 0.7


@pytest.mark.parametrize("audit", [AUDIT_GAP_NOT_ADDRESSED, AUDIT_GAP_CONFIDENT])
def test_without_answer_repair_a_gap_makes_no_repair_revise(audit):
    k, prompts = _kernel(audit, with_repair=False)
    eff = k.before_final("a nodule", step=1)
    assert eff.type == D.ALLOW and k.ledger.findings == []
    assert len(prompts) == 1 and _pools(k) == (1, 0, 0)
    # the shell's verification ignores gaps: True when the auditor says the answer addresses the task
    assert k.ctx.verification is (json.loads(audit)["addresses_task"] or None)
    assert k.ledger.commit_history[-1]["verified"] is k.ctx.verification


def test_the_audit_is_read_only_and_lives_for_its_hook():
    k, _p = _kernel(AUDIT_GAP_CONFIDENT)
    k.before_final("a 3.1 cm nodule", step=1)
    fa = k.ctx.final_audit
    assert isinstance(fa, FinalAudit) and fa.hook_serial == k.ctx.hook_serial
    assert (fa.postcondition, fa.side_effecting, fa.vetoed, fa.addresses_task, fa.confidence) == (
        "claim_supported_by_evidence", False, False, True, 0.9)
    with pytest.raises(AttributeError):
        fa.vetoed = True
    with pytest.raises(AttributeError):
        del fa.confidence
    got = fa.audit()
    got["repairable_gaps"].clear()
    fa.top_gap()["critique"] = "edited"
    assert fa.audit()["repairable_gaps"] == [GAP_OVER] and fa.top_gap() == GAP_OVER
    # workspace code sees the field through the view but cannot replace it
    view = k._ctx_for([c for c in k.capabilities if c.name == "answer_repair"][0])
    assert type(view).__name__ == "WorkspaceContextView" and view.final_audit is fa
    with pytest.raises(AttributeError):
        view.final_audit = None
    # the next hook clears it
    k.before_action({"type": "tool_call", "tool": "ImageDescription", "args": {}}, step=2)
    assert k.ctx.final_audit is None


def test_no_audit_no_repair_when_the_shell_escalates_for_lack_of_a_judge():
    pol = H.load_policy(env_type="tool_sandbox", workspace=W.H0["tool_sandbox"])
    task = {"task_id": "m", "goal": "finding?", "context": {}, "environment": {"type": "tool_sandbox"}}
    k = HarnessKernel(build_contract(task, env_type="tool_sandbox", policy=pol),
                      [ScopeEvidenceBinding(), ObligationLifecycle(), VerifyAndCommit(), _answer_repair_class()()],
                      policy=pol, env_type="tool_sandbox", judge_fn=None)
    k.after_action({"type": "tool", "tool": "ImageDescription", "args": {}}, "a nodule", None, None, step=0,
                   result_ok=True)
    eff = k.before_final("a nodule", step=1)
    assert (eff.type, eff.raw.capability, eff.raw.reason_code) == (D.ESCALATE, "verify_commit", "unverifiable_commit")
    assert k.ctx.final_audit is None and [f["capability"] for f in k.ledger.findings] == ["verify_commit"]


# ------------------------------------------------------------------------------------------------ manifests
def test_only_medcta_lists_answer_repair_first_after_the_shell():
    for env_type in ("fhir", "gui"):
        assert "answer_repair" not in [c.name for c in H.compose_capabilities(env_type, W.H0[env_type])]
        assert not os.path.exists(os.path.join(W.H0[env_type], "middleware", "answer_repair.py"))
    names = [c.name for c in H.compose_capabilities("tool_sandbox", W.H0["tool_sandbox"])]
    assert names.index("answer_repair") == names.index("verify_commit") + 1 == 4


def _without_answer_repair(src, dst):
    shutil.copytree(src, dst)
    mp = os.path.join(dst, "manifest.yaml")
    m = yaml.safe_load(open(mp, encoding="utf-8"))
    assert ENTRY in m["kernel_middleware"]
    m["kernel_middleware"].remove(ENTRY)
    with open(mp, "w", encoding="utf-8") as f:
        yaml.safe_dump(m, f, sort_keys=False)
    return dst


def test_a_workspace_may_drop_answer_repair(tmp_path):
    path = _without_answer_repair(W.H0["tool_sandbox"], str(tmp_path / "ws"))
    names = [c.name for c in H.compose_capabilities("tool_sandbox", workspace=path)]
    assert "answer_repair" not in names and "verify_commit" in names


# ------------------------------------------------------------------------------------------------ run.run_task (MedCTA)
def _rig(monkeypatch, tmp_path, audits, with_repair):
    """MCTA-0 through the real loop: one ImageDescription, then finals. The judge answers the audit prompt with the
    next scripted audit and every other harness prompt with nothing (their no-judge paths)."""
    import inline_loop_rig as L
    audits = list(audits)

    def judge(prompt):
        return audits.pop(0) if prompt.startswith(AUDITOR) else ""
    rig = L.Rig(monkeypatch, tmp_path, "MedCTA", "MCTA-0", judge=judge)
    if not with_repair:
        mp = os.path.join(rig.workspace, "manifest.yaml")
        m = yaml.safe_load(open(mp, encoding="utf-8"))
        m["kernel_middleware"].remove(ENTRY)
        with open(mp, "w", encoding="utf-8") as f:
            yaml.safe_dump(m, f, sort_keys=False)
    rig.replies = ['<tool_call>{"name": "ImageDescription", "arguments": {}}</tool_call>',
                   "<answer>answer one</answer>", "<answer>answer two</answer>", "<answer>answer three</answer>"]
    return rig, audits


def _decisions(rig):
    return [(e["capability"], e["decision"], e["effective"], e["rule_id"]) for e in rig.events("harness_decision")]


def _auditor_prompts(rig):
    return [p for p in rig.judge_prompts if p.startswith(AUDITOR)]


def test_loop_answer_repair_turns_a_gapped_final_back(monkeypatch, tmp_path):
    rig, left = _rig(monkeypatch, tmp_path, [AUDIT_GAP_CONFIDENT, AUDIT_CLEAN], with_repair=True)
    rig.run(max_steps=10)
    assert left == [] and len(_auditor_prompts(rig)) == 2
    assert rig.types() == ["tool_call", "answer_attempt", "harness_decision", "answer_attempt", "final_answer"]
    assert _decisions(rig) == [("answer_repair", "REVISE", "REVISE", "process_gap")]
    ev = rig.events("harness_decision")[0]
    assert (ev["process_gap"], ev["gap_type"], ev["critique"], ev["verification_flag"]) == (
        True, "over_specific", "size unsupported", "unverified_grounding")
    assert rig.feedback_at(2) == {
        "decision": "REVISE", "rule_id": "process_gap", "stage": "before_final",
        "reason": "process incomplete for this conclusion (over_specific): size unsupported",
        "message": "Before you finalize, your PROCESS is incomplete: size unsupported. Go back and DO the work with "
                   "your tools -- examine the distinguishing feature / rule out the alternative / gather the missing "
                   "evidence -- then answer again. You are NOT being told the answer is wrong; you are being told "
                   "the supporting work is not yet done."}
    assert [(r["kind"], r["capability"], r["repair_turn"]) for r in rig.actuation_records()] == [
        ("REVISE", "answer_repair", 1)]
    last = rig.trajectory[-1]
    assert (last["thought"], last["final_disposition"]) == ("answer two", "clean_commit")
    assert [(f["capability"], f["reason_code"]) for f in rig.audit["ledger"]["findings"]] == [
        ("answer_repair", "process_gap")]


def test_loop_without_answer_repair_commits_the_gapped_final(monkeypatch, tmp_path):
    rig, left = _rig(monkeypatch, tmp_path, [AUDIT_GAP_CONFIDENT, AUDIT_CLEAN], with_repair=False)
    rig.run(max_steps=10)
    assert left == [AUDIT_CLEAN] and len(_auditor_prompts(rig)) == 1
    assert rig.types() == ["tool_call", "answer_attempt", "final_answer"] and _decisions(rig) == []
    last = rig.trajectory[-1]
    assert (last["thought"], last["final_disposition"]) == ("answer one", "clean_commit")
    assert rig.audit["ledger"]["findings"] == []
    assert rig.audit["ledger"]["commit_history"][-1]["verified"] is True
    assert "answer_repair" not in [c.name for c in rig.kernel.capabilities]


@pytest.mark.parametrize("with_repair", [True, False])
def test_loop_the_shell_veto_fires_with_or_without_answer_repair(monkeypatch, tmp_path, with_repair):
    rig, left = _rig(monkeypatch, tmp_path, [AUDIT_HARD, AUDIT_CLEAN], with_repair=with_repair)
    rig.run(max_steps=10)
    assert left == [] and len(_auditor_prompts(rig)) == 2
    assert rig.types() == ["tool_call", "answer_attempt", "harness_decision", "answer_attempt", "final_answer"]
    assert _decisions(rig) == [("verify_commit", "REVISE", "REVISE", "claim_supported_by_evidence")]
    ev = rig.events("harness_decision")[0]
    assert ev["must_resolve"] is True and ev["resolution"]["violation_type"] == "audit_hard"
    assert rig.feedback_at(2)["message"] == ("Your answer has a hard defect (contradiction) on claim 'LUL effusion' "
                                             "that the evidence refutes -- remove, correct, or qualify it.")
    last = rig.trajectory[-1]
    assert (last["thought"], last["final_disposition"]) == ("answer two", "clean_commit")
