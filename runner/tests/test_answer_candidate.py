# -*- coding: utf-8 -*-
"""The answer-candidate strategy as middleware of MedCTA's default workspace (workspace.H0["tool_sandbox"]), and how
it reaches the judge: ctx.semantic_check(), one check from the kernel's semantic meter per decision. There is no
unbudgeted judge channel.

These tests pin, on a scripted context: where the strategy lives (only in the MedCTA workspace), what a candidate
evaluation costs, what it does when the pool refuses, which judge services host middleware has, and the hook-level
lifecycle of held answers, acquisitions and bounded revise rounds."""
import json
import os
import sys
from types import SimpleNamespace

import pytest

_RUNNER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RUNNER not in sys.path:
    sys.path.insert(0, _RUNNER)

import instrumentation as I                            # noqa: E402
import middleware_host as MH                           # noqa: E402
import workspace as W                                  # noqa: E402
from harness.capability import HarnessContext          # noqa: E402
from harness.decision import HarnessDecision           # noqa: E402
from harness.kernel import Effective                   # noqa: E402

TASK = {"task_id": "T", "goal": "What is the finding?", "context": {}}


def test_only_the_medcta_workspace_holds_the_strategy():
    names = {env: [type(m).__name__ for m in MH.load_middleware(W.H0[env], TASK).middlewares]
             for env in ("tool_sandbox", "fhir", "gui")}
    assert names["tool_sandbox"] == ["AnswerCandidate"]
    assert "AnswerCandidate" not in names["fhir"] and "AnswerCandidate" not in names["gui"]
    assert not os.path.exists(os.path.join(_RUNNER, "harness", "answer_surface.py"))
    import harness.engines.semantic as S
    assert not hasattr(S, "evaluate_candidate") and not hasattr(S, "build_merged_answer")


def _scripted_judge(prompts):
    """Answers the adoption gates so that a funded evaluation adopts B as a slot completion: identical decision
    signatures (no core change, so no signature comparison call), slot completion adopted, not a hedge; the claim
    merge returns a new text."""
    def judge(prompt):
        prompts.append(prompt)
        if prompt.startswith("Extract the CORE decision"):
            return json.dumps({"primary_conclusion": "mass", "polarity": "present", "target": "liver"})
        if prompt.startswith("A medical visual-QA task."):
            return json.dumps({"adopt": True, "slot": "stain", "why": "fills the stain"})
        if prompt.startswith("ORIGINAL answer A and REVISED answer B for the SAME clinical"):
            return json.dumps({"hedge": False, "why": "adds a finding"})
        if prompt.startswith("Produce the FINAL answer"):
            return "merged answer"
        raise AssertionError("unexpected judge prompt: %r" % prompt[:60])
    return judge


def _kernel(judge, verdicts, semantic_budget=10, allocation=None):
    """The kernel as on_answer's services use it: the real HarnessContext budget, and a before_final that bumps
    hook_serial as kernel.before_final does before answering with the next scripted verdict."""
    kctx = HarnessContext(ledger=None, contract=None, policy={}, judge_fn=judge,
                          semantic_budget=semantic_budget, semantic_allocation=allocation)
    verdicts = list(verdicts)

    def before_final(text, step=0):
        kctx.hook_serial += 1
        return verdicts.pop(0)
    # verify_final is the kernel's query (evaluate_final); here the same scripted verdicts
    return SimpleNamespace(ctx=kctx, ledger=None, before_final=before_final, evaluate_final=before_final)


def _metered_ctx(kernel, traj, errors, answers, evidence=(), inst=None):
    """A context whose on_answer services are the shell's own (answer_services) over `kernel`."""
    shell = {"answers": lambda: answers, "ledger": lambda: SimpleNamespace(evidence=list(evidence)),
             "contract_meta": lambda: {"goal": TASK["goal"]},
             "verdict_feedback": lambda v, stage: {"decision": v.type, "stage": stage, "reason": "fix it"}}
    shell.update(MH.answer_services(kernel, len(answers), traj, inst))
    return MH.MiddlewareContext(env=None, task=TASK, trajectory=traj, step=len(answers), max_steps=10, used_actions=0,
                                judge_factory=lambda consumer: kernel.ctx.budgeted_judge(consumer), errors=errors,
                                shell=shell)


def _hold_revise(ac, answers):
    ac.after_final_verdict(_ctx([], [], answers), answers[0].text,
                           _verdict("REVISE", {"candidate": True, "critique": "name the stain"}))


def test_a_candidate_evaluation_pays_one_check_and_the_claim_merge_another():
    prompts, traj, errors, inst = [], [], [], I.Instrumentation()
    kernel = _kernel(_scripted_judge(prompts), [_verdict("ALLOW"), _verdict("ALLOW")])
    kctx = kernel.ctx
    ac, answers = _strategy(), [MH.Answer(1, "h1", "a liver mass")]
    _hold_revise(ac, answers)
    answers.append(MH.Answer(2, "h2", "a liver mass, H&E stain"))
    out = ac.on_answer(_metered_ctx(kernel, traj, errors, answers, inst=inst), "a liver mass, H&E stain")
    assert isinstance(out, MH.Accept) and errors == []
    assert (out.text, out.disposition, out.reverified, out.flag) == (
        "merged answer", "adopted_slot_completion_stain+claimmerge", True, None)
    # four gate calls (signature A, signature B, slot completion, hedge) for one check; the merge is a second check
    assert len(prompts) == 5 and prompts[-1].startswith("Produce the FINAL answer")
    m = kctx.semantic_meter
    assert (m.pool_spent("general"), m.pool_spent("reserve"), m.refused_total()) == (2, 0, 0)
    # the price of a check: five judge calls for two units (runner/harness/semantic_budget.py, pricing)
    assert {k: v for k, v in m.record()["consumers"]["middleware"].items() if v} == {"checks": 2, "calls": 5,
                                                                                      "units": 2}
    assert inst.calls == [{"kind": "semantic_check", "hook": "on_answer", "step": 2, "granted": True}] * 2


@pytest.mark.parametrize("hold,evidence,disposition", [
    ("REVISE", [], "kept_original_core_flip_without_new_evidence"),
    ("ACQUIRE", [{"evidence_id": "ev-1", "status": "VALIDATED"}, {"evidence_id": "ev-2", "status": "VALIDATED"}],
     "kept_original_core_change_not_supported_by_delta")])
def test_a_spent_pool_leaves_the_gates_on_their_no_judge_path(hold, evidence, disposition):
    prompts, traj, errors, inst = [], [], [], I.Instrumentation()
    # an allocation whose general pool is empty; the verification reserve is not
    kernel = _kernel(_scripted_judge(prompts), [_verdict("ALLOW"), _verdict("REVISE")],
                     allocation={"pools": {"reserve": {"rest": True}, "general": {"cap": 0}}})
    kctx = kernel.ctx
    ac, answers = _strategy(), [MH.Answer(1, "h1", "a liver mass")]
    if hold == "REVISE":
        _hold_revise(ac, answers)
    else:
        ac.after_final_verdict(_ctx([], [], answers, evidence=evidence[:1]), "a liver mass", _verdict(
            "ACQUIRE", {"next_action": {"tool": "ImageDescription", "read_only": True, "args": {"region": "liver"}}}))
    answers.append(MH.Answer(2, "h2", "a liver mass, H&E stain"))
    out = ac.on_answer(_metered_ctx(kernel, traj, errors, answers, evidence=evidence, inst=inst),
                       "a liver mass, H&E stain")
    # the same scripted judge adopts B when the pool pays (test above); refused, the gates never call it and keep A
    assert prompts == [] and errors == []
    assert (out.text, out.disposition, out.intervention_type, out.reverified, out.flag) == (
        "a liver mass", disposition, hold, False, disposition)
    assert (kctx.semantic_meter.pool_spent("general"), kctx.semantic_meter.refused_total()) == (0, 1)
    assert inst.calls == [{"kind": "semantic_check", "hook": "on_answer", "step": 2, "granted": False}]


def test_a_check_taken_before_verify_final_is_refused_after_it():
    prompts = []
    kernel = _kernel(_scripted_judge(prompts), [_verdict("ALLOW")])
    ctx = _metered_ctx(kernel, [], [], [MH.Answer(1, "h1", "a")])
    judge = ctx.semantic_check()
    ctx.verify_final("a")                          # the kernel's evaluate_final: a new hook call
    assert judge("Extract the CORE decision") is None and prompts == []
    assert (kernel.ctx.semantic_meter.pool_spent("general"), kernel.ctx.semantic_meter.refused_total()) == (1, 1)


def test_host_middleware_reaches_the_judge_only_through_metered_services():
    prompts = []
    kernel = _kernel(lambda p: prompts.append(p) or "{}", [])
    kctx = kernel.ctx
    services = MH.answer_services(kernel, 3, [])
    assert set(services) == {"verify_final", "record_final", "plan_read", "semantic_check", "semantic_rounds"}
    assert [n for n in dir(MH.MiddlewareContext) if "judge" in n] == ["judge"]
    ctx = MH.MiddlewareContext(env=None, task=TASK, trajectory=[], step=3, max_steps=10, used_actions=0,
                               judge_factory=lambda consumer: kctx.budgeted_judge(consumer), errors=[], shell=services)
    m = kctx.semantic_meter
    assert ctx.judge("p1") == "{}" and m.spent_total() == 1                              # one unit per call
    check = ctx.semantic_check()
    assert check("p2") == "{}" and check("p3") == "{}" and m.spent_total() == 2         # one per check
    assert prompts == ["p1", "p2", "p3"]
    while m.draw("goal_spec", "call"):                                                   # spend the general pool
        pass
    assert ctx.judge("p4") is None and ctx.semantic_check() is None and prompts == ["p1", "p2", "p3"]
    kctx.judge_fn = None
    refused = m.refused_total()
    assert MH.answer_services(kernel, 3, [])["semantic_check"]("middleware") is None and m.refused_total() == refused
    assert MH.answer_services(None, 3, [])["semantic_check"]("middleware") is None
    with pytest.raises(RuntimeError):
        MH.MiddlewareContext(env=None, task={}, trajectory=[], step=0, max_steps=1, used_actions=0,
                             judge_factory=lambda: None, errors=[]).semantic_check()   # only in on_answer


def _strategy():
    return W.Workspace(W.H0["tool_sandbox"]).load("middleware/answer_candidate.py", "AnswerCandidate")(TASK)


def _ctx(traj, errors, answers, verdicts=(), evidence=()):
    verdicts = list(verdicts)
    shell = {"answers": lambda: answers, "ledger": lambda: SimpleNamespace(evidence=list(evidence)),
             "contract_meta": lambda: {"goal": TASK["goal"]},
             "verdict_feedback": lambda v, stage: {"decision": v.type, "stage": stage, "reason": "fix it"},
             "verify_final": lambda text: verdicts.pop(0), "plan_read": lambda read: True,
             "semantic_check": lambda consumer: None, "semantic_rounds": {"acquire": 2, "revise": 2}.__getitem__}
    return MH.MiddlewareContext(env=None, task=TASK, trajectory=traj, step=len(answers), max_steps=10, used_actions=0,
                                judge_factory=lambda consumer: None, errors=errors, shell=shell)


def _verdict(type_, extra=None, events=None):
    return Effective(type_, feedback={"reason": "r"}, raw=HarnessDecision(type_, capability="c", extra=extra or {}),
                     events=events or [])


def test_an_acquire_holds_the_earliest_answer_and_the_next_final_is_judged_against_it():
    ac, traj, errors = _strategy(), [], []
    answers = [MH.Answer(1, "h1", "first answer"), MH.Answer(2, "h2", "second answer")]
    acquire = _verdict("ACQUIRE", {"hypotheses": ["x", "y"],
                                   "next_action": {"tool": "ImageDescription", "read_only": True,
                                                   "args": {"region": "apex", "attribute": "margin"}}})
    ctx = _ctx(traj, errors, answers, evidence=[{"evidence_id": "ev-1"}])
    text = ac.after_final_verdict(ctx, "second answer", acquire)
    assert text.startswith("Your prior answer was not grounded in a direct observation of the image. A targeted "
                           "observation of apex (margin)") and '"second answer"' in text
    assert text.endswith("Distinguish specifically between: x; y.")
    assert ac.pending["root_answer"] == "first answer" and ac.pending["baseline_ev_ids"] == {"ev-1"}
    assert ac.pending["acquisition_goal"] == {"region": "apex", "attribute": "margin", "hypotheses": ["x", "y"]}
    assert ac.after_final_verdict(ctx, "second answer", _verdict("REVISE")) is None   # a plain REVISE holds nothing new

    # candidate B verifies; without a judge the gates keep A (a core change without new evidence), A does not verify
    b_events = [{"event_type": "b_verdict"}]
    answers.append(MH.Answer(3, "h3", "third answer"))
    ctx = _ctx(traj, errors, answers, verdicts=[_verdict("ALLOW", events=b_events), _verdict("REVISE")])
    out = ac.on_answer(ctx, "third answer")
    assert isinstance(out, MH.Accept)
    assert (out.text, out.disposition, out.intervention_type, out.reverified, out.flag) == (
        "first answer", "kept_original_core_flip_without_new_evidence", "ACQUIRE", False,
        "kept_original_core_flip_without_new_evidence")
    assert traj == b_events and errors == [] and ac.pending is None
    assert ac.on_answer(ctx, "fourth answer") is None


def test_a_candidate_revise_earns_bounded_re_answer_rounds():
    ac, traj, errors = _strategy(), [], []
    answers = [MH.Answer(1, "h1", "only answer")]
    ac.after_final_verdict(_ctx(traj, errors, answers), "only answer",
                           _verdict("REVISE", {"candidate": True, "critique": "name the stain"}))
    assert ac.pending == {"surface": "answer", "type": "REVISE", "root_answer": "only answer",
                          "critique": "name the stain", "baseline_ev_ids": set()}
    for n in (1, 2):
        out = ac.on_answer(_ctx(traj, errors, answers, verdicts=[_verdict("REVISE", {"critique": "c%d" % n})]), "b")
        assert isinstance(out, MH.Repair) and out.feedback == {"decision": "REVISE", "stage": "before_final",
                                                               "reason": "fix it"}
        assert ac.pending["rev_round"] == n and ac.pending["critique"] == "c%d" % n
    assert [e["event_type"] for e in traj] == ["adaptive_revise"] * 2 and traj[1]["round"] == 2
    out = ac.on_answer(_ctx(traj, errors, answers, verdicts=[_verdict("REVISE"), _verdict("ALLOW")]), "b")
    assert isinstance(out, MH.Accept) and (out.text, out.reverified, out.flag) == ("only answer", True, None)
