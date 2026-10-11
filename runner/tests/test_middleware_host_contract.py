# -*- coding: utf-8 -*-
"""The middleware host contract: what each middleware hook result does to the run loop.

A stub middleware returns the result inside the real run loop: run.run_task with a fake environment, a scripted agent,
and the real kernel with its three hooks stubbed so its verdicts are fixed. The tests read the trajectory, what the
agent was shown, and the context later hooks saw. Services and helpers without a loop effect of their own are tested
at their seam."""
import collections
import json
import os
import sys
from types import SimpleNamespace

import pytest

_RUNNER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RUNNER not in sys.path:
    sys.path.insert(0, _RUNNER)

import harness as H                                    # noqa: E402
import instrumentation as I                            # noqa: E402
import middleware_host as MH                           # noqa: E402
import run as R                                        # noqa: E402
from harness import actuation                          # noqa: E402
from harness import obligation_gate as OG              # noqa: E402
from harness.decision import HarnessDecision           # noqa: E402
from harness.kernel import Effective                   # noqa: E402
import workspace as W                                  # noqa: E402

TOOLS = [{"name": "ImageDescription", "signature": "ImageDescription()"},
         {"name": "OtherTool", "signature": "OtherTool()"}]
OUTPUT = {"output": "a 3cm spiculated RUL nodule on the CT", "ok": True}
TOOL = {"type": "tool_call", "tool": "ImageDescription", "args": {}}


def final(answer="RUL nodule"):
    return {"type": "final", "answer": answer}


class FakeEnv(object):
    type = "tool_sandbox"
    full_state = None

    def __init__(self):
        self.calls = []

    def reset(self):
        pass

    def available_tools(self):
        return TOOLS

    def call_tool(self, name, args):
        self.calls.append((name, args))
        return dict(OUTPUT)

    def capabilities(self):
        return {}

    def teardown(self):
        pass


class Agent(object):
    name = "scripted"

    def __init__(self, actions):
        self.actions, self.states = list(actions), []

    def act(self, state):
        self.states.append(state)
        return self.actions.pop(0) if self.actions else final("done")


def acquire(args=None, agent_text=None, capability="stub_capability", read_only=True, tool="ImageDescription"):
    """A kernel ACQUIRE verdict shaped as run.py receives it (Effective over the raising decision)."""
    extra = {"next_action": {"tool": tool, "read_only": read_only,
                             "args": args if args is not None else {"region": "r", "attribute": "a"}}}
    if agent_text is not None:
        extra["agent_text"] = agent_text
    raw = HarnessDecision("ACQUIRE", capability=capability, reason="needs a read", extra=extra)
    return Effective("ACQUIRE", feedback={"reason": "needs a read"}, raw=raw,
                     events=[{"event_type": "stub_verdict", "status": "ok"}])


def scripted(results):
    """A kernel hook returning `results` in order, then ALLOW."""
    results = list(results)
    return lambda *a, **k: results.pop(0) if results else Effective("ALLOW")


def types(traj):
    return [e.get("event_type") for e in traj]


def record_stub(verdict, step=None, flagged=False):
    """kernel.record_final over a scripted verdict: the verdict, recorded, with one recorded_verdict event (the real
    kernel's are harness_decision / shell_hook), so the trajectory shows where the shell recorded it."""
    if verdict is None:
        return None
    return Effective(verdict.type, feedback=verdict.feedback, raw=verdict.raw,
                     events=[{"event_type": "recorded_verdict", "status": "ok"}])


def loop_events(res):
    """The loop's events of a run, with the shell's trace (shell_* events) left out."""
    return [e for e in res.get("_trajectory", []) if not str(e.get("event_type", "")).startswith("shell_")]


def once(result):
    box = [result]
    return lambda *a: box.pop() if box else None


@pytest.fixture
def loop(monkeypatch, tmp_path):
    for k in ("MH_GACC", "MH_GACC_MODEL", "MH_HARNESS_JUDGE_MODEL", "MH_MM_JUDGE", "MH_MM_JUDGE_MODEL",
              "LLM_JUDGE_MODEL"):
        monkeypatch.delenv(k, raising=False)
    # run_task requires every judge's configuration (runner/judges.py): stand-in names, and no harness judge wired
    for k in ("MH_HARNESS_JUDGE_MODEL", "LLM_JUDGE_MODEL", "MH_JUDGE_MODEL", "MH_GACC_MODEL"):
        monkeypatch.setenv(k, "contract-stub-judge")
    monkeypatch.setattr(H, "_build_judge", lambda: (None, None))
    Run = collections.namedtuple("Run", "traj agent env kernel result")

    def run(actions, middlewares, max_steps=4, before_action=None, after_action=None, before_final=None,
            task_edit=None):
        task = {"task_id": "T", "goal": "What is the finding on the CT?", "context": {"text": "chest CT"},
                "environment": {"type": "tool_sandbox"}, "available_tools": TOOLS, "checkpoints": []}
        if task_edit is not None:
            task_edit(task)
        agent, env, box = Agent(actions), FakeEnv(), {}
        monkeypatch.setattr(R, "load_task", lambda bench, tid: task)
        monkeypatch.setattr(R.environments, "make_env", lambda *a, **k: env)
        monkeypatch.setattr(R.agents, "make_agent", lambda name, t, workspace=None: agent)
        monkeypatch.setattr(MH, "load_middleware",
                            lambda env_type, t, instrumentation=None: MH.MiddlewareStack(middlewares, instrumentation))
        build = H.build_kernel

        def build_stubbed(*a, **k):
            kernel = build(*a, **k)
            kernel.before_action = before_action or scripted([])
            kernel.after_action = after_action or scripted([])
            kernel.before_final = before_final or scripted([])
            kernel.evaluate_final = kernel.before_final      # verify_final: a query of the same scripted verdicts
            kernel.record_final = record_stub
            box["kernel"] = kernel
            return kernel
        monkeypatch.setattr(H, "build_kernel", build_stubbed)
        res = R.run_task("MedCTA", "T", agent_name="scripted", max_steps=max_steps, job_dir=str(tmp_path / "job"),
                         cleanup=False)
        return Run(loop_events(res), agent, env, box.get("kernel"), res)
    return run


# ------------------------------------------------------------------------------------------ Nudge / Revision
@pytest.mark.parametrize("result,decision,in_repair_after", [
    (MH.Nudge("obligation", "keep going", "obligation_unmet", "CONTINUE"), "CONTINUE", False),
    (MH.Revision("obligation", "keep going", "obligation_unmet"), "REVISE", True)])
def test_before_final_nudge_refuses_the_final_and_leaves_repair_locked(loop, result, decision, in_repair_after):
    """A Nudge leaves repair locked; a Revision unlocks it (both are recorded the same way)."""
    seen = []
    mw = SimpleNamespace(before_model=lambda ctx: seen.append(ctx.in_repair), before_final=once(result))
    r = loop([final("a"), final("b")], [mw])
    shown = r.agent.states[1]
    assert shown["harness_feedback"] == {"decision": decision, "stage": "obligation", "reason": "keep going"}
    assert shown["last_result"] == {"feedback": "keep going"} and shown["last_observation"] == "obligation_unmet"
    assert seen == [False, in_repair_after]
    assert [e["attempt_index"] for e in r.traj if e["event_type"] == "answer_attempt"] == [1]
    assert [e["thought"] for e in r.traj if e["event_type"] == "final_answer"] == ["b"]


def test_before_tool_nudge_refuses_the_action_and_leaves_repair_locked(loop):
    seen = []
    mw = SimpleNamespace(before_model=lambda ctx: seen.append(ctx.in_repair),
                         before_tool=once(MH.Nudge("attachment", "attach first", "attachment_block", "REVISE")))
    r = loop([TOOL, final()], [mw])
    assert r.env.calls == [] and "tool_call" not in types(r.traj)
    shown = r.agent.states[1]
    assert shown["harness_feedback"] == {"decision": "REVISE", "stage": "attachment", "reason": "attach first"}
    assert shown["last_result"] == {"feedback": "attach first"} and shown["last_observation"] == "attachment_block"
    assert seen == [False, False]


@pytest.mark.parametrize("hook,aborted", [("on_answer", True), ("before_final", False)])
def test_repair_counts_a_repair_turn_and_a_nudge_does_not(loop, hook, aborted):
    """Four refused finals, then a tool call past the spent budget. That refusal is a repair turn: it is the 5th
    (> MAX_REPAIR_TURNS, abort) only if the four refusals each counted one -- a Repair does, a Nudge does not. Both are
    recorded once each."""
    fb = {"decision": "REVISE", "stage": "before_final", "reason": "re-answer"}
    n = [0]

    def refuse(ctx, answer):
        n[0] += 1
        if n[0] > 4:
            return None
        return MH.Repair(fb) if hook == "on_answer" else MH.Nudge("x", "re-answer", "o", "REVISE")
    r = loop([TOOL] + [final("a")] * 4 + [TOOL, final("z")], [SimpleNamespace(**{hook: refuse})], max_steps=1)
    assert ("tool_budget_exhausted" in types(r.traj)) is aborted
    assert [e["thought"] for e in r.traj if e["event_type"] == "final_answer"] == ([] if aborted else ["z"])
    if hook == "on_answer":
        assert r.agent.states[2]["harness_feedback"] == fb
    records = [(a["stage"], a["source"], a["repair_turn"], a["repair_turn_started"])
               for a in r.kernel.ledger.actuation_telemetry]
    assert records == ([(hook, "host:Repair", n, True) for n in (1, 2, 3, 4)] if hook == "on_answer" else
                       [(hook, "host:Nudge", None, False)] * 4)


# ------------------------------------------------------------------------------------------ after_tool
def test_after_tool_post_revision_overwrites_what_the_action_left_and_unlocks_repair(loop):
    seen, got = [], []

    def after_tool(ctx, action, result):
        got.append((action["tool"], result, ctx.last_obs))
        return MH.PostRevision("write_verify", "did not land", "write_not_landed") if len(got) == 1 else None
    mw = SimpleNamespace(before_model=lambda ctx: seen.append(ctx.in_repair), after_tool=after_tool)
    r = loop([TOOL, final()], [mw])
    assert r.env.calls == [("ImageDescription", {})] and types(r.traj).count("tool_call") == 1
    assert got == [("ImageDescription", OUTPUT, json.dumps(OUTPUT, ensure_ascii=False))]
    shown = r.agent.states[1]
    assert shown["harness_feedback"] == {"decision": "REVISE", "stage": "write_verify", "reason": "did not land"}
    assert shown["last_result"] == {"feedback": "did not land"} and shown["last_observation"] == "write_not_landed"
    assert seen == [False, True]
    assert [e["step"] for e in r.traj if e["event_type"] == "final_answer"] == [1]   # no iteration skipped


# ------------------------------------------------------------------------------------------ next_read
def test_next_read_runs_as_a_harness_read_without_a_plan(loop):
    calls = []

    def next_read(ctx):
        calls.append((ctx.step, ctx.forced_pending))
        return MH.Read("ImageDescription", {"region": "r"}) if ctx.step == 0 else None
    r = loop([final()], [SimpleNamespace(next_read=next_read)])
    assert r.traj[0] == {"step": 0, "event_type": "harness_acquire_dispatch", "tool": "ImageDescription",
                         "args": {"region": "r"}, "status": "ok"}
    call = r.traj[1]
    assert (call["event_type"], call["step"], call["origin"], call["audience"]) == ("tool_call", 0, "harness", "both")
    assert call["requester"] == "next_read"
    assert "harness_acquire_predispatch" not in types(r.traj)
    assert r.env.calls == [("ImageDescription", {"region": "r"})]
    assert r.agent.states[0]["harness_acquired"] == {"tool": "ImageDescription", "args": {"region": "r"},
                                                    "output": OUTPUT["output"]}
    assert [t["acquire_id"] for t in r.kernel.ledger.acquire_telemetry] == [None]   # record_execution, as on HAB
    assert calls == [(0, False), (1, False)]


def test_a_next_read_runs_after_the_tool_budget_is_spent(loop):
    mw = SimpleNamespace(next_read=lambda ctx: MH.Read("ImageDescription", {"n": 1}) if ctx.step == 1 else None)
    r = loop([TOOL, final()], [mw], max_steps=1)
    assert r.env.calls == [("ImageDescription", {}), ("ImageDescription", {"n": 1})]
    assert r.agent.states[1]["harness_feedback"] is None   # no runtime_budget refusal


def test_queued_read_is_the_adapter_forced_action_dict():
    args = {"ref": "e12"}
    assert MH.queued_read(None) is None
    q = MH.queued_read(MH.Read("click", args))
    assert q == {"type": "tool_call", "tool": "click", "args": {"ref": "e12"}, "_harness_acquire": True,
                 "_requester": "next_read"}
    assert q["args"] is args
    with pytest.raises(TypeError):
        MH.queued_read({"tool": "click"})


# ------------------------------------------------------------------------------------------ answer surface
@pytest.mark.parametrize("flag", [None, "kept_original_unverified"])
def test_accept_ends_the_episode_with_the_shells_final_record(loop, flag):
    verified = []
    acc = MH.Accept("the chosen answer", "adopted_candidate", "ACQUIRE", False, flag)
    r = loop([final("first")], [SimpleNamespace(on_answer=lambda ctx, answer: acc)],
             before_final=lambda *a, **k: verified.append(a))
    fin = r.traj[-1]
    assert fin == {"step": 0, "event_type": "final_answer", "thought": "the chosen answer", "status": "ok",
                   "final_disposition": "adopted_candidate", "intervention_type": "ACQUIRE",
                   "answer_attempt_index": 1, "reverified": False,
                   # not the agent's answer ("first"): the candidate_merge strategy authored it
                   "origin": "harness", "requester": "candidate_merge", "final_answer_author": "candidate_merge",
                   "canonical_action": R._canon.canonical_action({"type": "final", "answer": "the chosen answer"},
                                                                 "tool_sandbox")}
    # the strategy verified nothing, so the shell verified the delivered text once; the verdict is clean
    assert types(r.traj).count("final_answer") == 1 and verified == [("the chosen answer",)]
    assert [c.get("verification_flag") for c in r.kernel.ledger.commit_history] == ([] if flag is None else [flag])


# ------------------------------------------------------------------------------------------ withhold on Accept text
WITHHELD = ("[WITHHELD] The answer asserted a claim refuted by the validated evidence and the conflict was not "
            "resolved; no safe answer could be committed.")
CONTRADICTION_FEEDBACK = {"decision": "REVISE", "rule_id": "claim_supported_by_evidence",
                          "reason": "answer has a hard defect (contradiction): the CT shows no nodule"}
ENV_STATE = lambda t: t.__setitem__("expected_outcome", {"answer_kind": "portal_state_and_notes"})   # noqa: E731


def final_verdict(kind):
    """A kernel before_final verdict as run.py receives it. contradiction: verify_commit's localized evidence
    contradiction (an effective REVISE whose decision carries must_resolve); revise: a REVISE without must_resolve;
    allow: ALLOW. Each carries one event, so the trajectory shows whose events were recorded."""
    if kind == "allow":
        return Effective("ALLOW", events=[{"event_type": "stub_verdict", "status": "ok"}])
    extra = {"side_effecting": False}
    if kind == "contradiction":
        extra["must_resolve"] = True
    raw = HarnessDecision("REVISE", capability="verify_commit", rule_id="claim_supported_by_evidence",
                          reason_code="evidence_contradiction" if kind == "contradiction" else "process_gap",
                          reason=CONTRADICTION_FEEDBACK["reason"], deterministic=False, extra=extra)
    return Effective("REVISE", feedback=dict(CONTRADICTION_FEEDBACK), raw=raw,
                     events=[{"event_type": "stub_verdict", "status": "ok"}])


def kernel_script(script, calls):
    """A kernel before_final answering each call with the verdict scripted for it, in order: [(text, kind)].
    kind "raises" raises. `calls` receives every text the kernel was asked about."""
    script = list(script)

    def before_final(text, step=0):
        calls.append(text)
        want, kind = script.pop(0)
        assert text == want, (text, want)
        if kind == "raises":
            raise RuntimeError("kernel down")
        return final_verdict(kind)
    return before_final


def withheld_final(text, env_state=False):
    """The final_answer the normal before_final path writes for a must_resolve violation on repair exhaustion
    (runner/tests/test_inline_final_path.py pins it there)."""
    canon = R._canon.canonical_action({"type": "final", "answer": text}, "tool_sandbox")
    if env_state:
        return {"step": 0, "event_type": "final_answer", "thought": text, "status": "ok",
                "origin": "agent", "final_answer_author": "agent",
                "verification_flag": "unresolved_contradiction",
                "final_disposition": "flagged_delivery_unresolved_contradiction",
                "answer_attempt_index": 1, "harness_feedback": CONTRADICTION_FEEDBACK, "canonical_action": canon}
    return {"step": 0, "event_type": "final_answer", "thought": WITHHELD, "status": "ok",
            "origin": "harness", "requester": "withhold", "final_answer_author": "withheld_stub",
            "verification_flag": "abstained_unresolved_contradiction",
            "final_disposition": "abstained_unresolved_violation",
            "answer_attempt_index": 1, "harness_feedback": CONTRADICTION_FEEDBACK, "canonical_action": canon}


def delivered_final(acc):
    return MH.accept_event(acc, 0, 1, R._canon, "tool_sandbox", ["candidate B"])   # the agent's one answer


@pytest.mark.parametrize("env_state", [False, True])
def test_q3_accept_over_a_must_resolve_contradiction_is_withheld(loop, env_state):
    """The KEEP_A shape: B verified clean, root A re-verified into a must_resolve evidence contradiction, A delivered
    flagged. A candidate finalize is repair exhaustion, so the shell withholds A exactly as the normal path does
    (a [WITHHELD] stub, recorded as the flagged final), or delivers it flagged when the deliverable is environment
    state. The Accept's own flag is not recorded: the withhold records its own, as on the normal path."""
    calls = []
    acc = MH.Accept("root A", "kept_original_core_flip_without_new_evidence", "REVISE", False,
                    "kept_original_core_flip_without_new_evidence")

    def on_answer(ctx, answer):
        assert ctx.verify_final(answer).type == "ALLOW"
        assert ctx.verify_final("root A").type == "REVISE"
        return acc
    r = loop([final("candidate B"), final("never asked")], [SimpleNamespace(on_answer=on_answer)],
             before_final=kernel_script([("candidate B", "allow"), ("root A", "contradiction")], calls),
             task_edit=ENV_STATE if env_state else None)
    assert calls == ["candidate B", "root A"]            # the shell read A's verdict; it did not ask again
    assert types(r.traj) == ["answer_attempt", "stub_verdict", "recorded_verdict", "final_answer"]
    assert len(r.agent.states) == 1
    assert r.traj[-1] == withheld_final("root A", env_state)
    flag = "unresolved_contradiction" if env_state else "abstained_unresolved_contradiction"
    assert [(c["detail"], c["verified"], c["verification_flag"]) for c in r.kernel.ledger.commit_history] == \
        [("final_answer", None, flag)]
    assert not getattr(r.kernel.ledger, "actuation_telemetry", None)   # no record_revision: not a repair turn


def test_q3_clean_accept_is_delivered(loop):
    """The delivered text's own verdict is clean, so the Accept is recorded as is, flag included. B's
    contradiction says nothing about A. The strategy did not verify A itself: the shell verifies the delivered text
    through the same service (not the first verify_final of this call, so its query events are not recorded); the
    delivered verdict is recorded."""
    calls = []
    acc = MH.Accept("root A", "kept_original_candidate_not_clearly_better", "REVISE", False, "kept_unverified")

    def on_answer(ctx, answer):
        assert ctx.verify_final(answer).type == "REVISE"
        return acc
    r = loop([final("candidate B"), final("never asked")], [SimpleNamespace(on_answer=on_answer)],
             before_final=kernel_script([("candidate B", "contradiction"), ("root A", "allow")], calls))
    assert calls == ["candidate B", "root A"]
    assert types(r.traj) == ["answer_attempt", "stub_verdict", "recorded_verdict", "final_answer"]
    assert len(r.agent.states) == 1
    assert r.traj[-1] == delivered_final(acc)
    assert [c["verification_flag"] for c in r.kernel.ledger.commit_history] == ["kept_unverified"]


@pytest.mark.parametrize("delivered,script,withheld", [
    # KEEP_A: B contradicted, A clean -> A delivered; A contradicted, B clean -> A withheld
    ("root A", [("candidate B", "contradiction"), ("root A", "allow")], False),
    ("root A", [("candidate B", "allow"), ("root A", "contradiction")], True),
    # ADOPT_B with a merge that did not verify: B is delivered on B's own verdict, never on the merge's
    ("candidate B", [("candidate B", "contradiction"), ("A merged with B", "allow")], True),
    ("candidate B", [("candidate B", "allow"), ("A merged with B", "contradiction")], False),
    # the merge delivered: its own verdict
    ("A merged with B", [("candidate B", "contradiction"), ("A merged with B", "allow")], False),
    # B, a merge and A all verified in one call, A delivered: A's verdict, whatever B and the merge got
    ("root A", [("candidate B", "contradiction"), ("A merged with B", "contradiction"), ("root A", "allow")], False),
    ("root A", [("candidate B", "allow"), ("A merged with B", "allow"), ("root A", "contradiction")], True),
    # the delivered text verified twice: the later verdict decides
    ("root A", [("candidate B", "allow"), ("root A", "contradiction"), ("root A", "allow")], False),
    ("root A", [("candidate B", "allow"), ("root A", "allow"), ("root A", "contradiction")], True),
    # a REVISE without must_resolve is not withheld (as on the normal path, where it is a flagged delivery)
    ("root A", [("candidate B", "allow"), ("root A", "revise")], False),
])
def test_q3_the_authoritative_verdict_is_the_last_verify_final_of_the_delivered_text(loop, delivered, script, withheld):
    calls = []
    acc = MH.Accept(delivered, "scripted_disposition", "ACQUIRE", True, None)

    def on_answer(ctx, answer):
        for text, _ in script:
            ctx.verify_final(text)
        return acc
    r = loop([final("candidate B"), final("never asked")], [SimpleNamespace(on_answer=on_answer)],
             before_final=kernel_script(script, calls))
    assert calls == [t for t, _ in script]               # every verdict came from the strategy's own calls
    assert r.traj[-1] == (withheld_final(delivered) if withheld else delivered_final(acc))


@pytest.mark.parametrize("shell_verdict", ["contradiction", "raises"])
def test_q3_a_strategy_cannot_opt_out_by_not_verifying_its_text(loop, shell_verdict):
    """No Accept field turns the withhold off, and not verifying the delivered text does not skip it: the shell
    verifies the text itself. A contradiction is withheld. A kernel that raises there is a before_final hook error,
    as on the normal path: the episode aborts with no final."""
    calls = []
    acc = MH.Accept("root A", "adopted_evidence_supported_core_change", "ACQUIRE", True, None)
    r = loop([final("candidate B"), final("never asked")], [SimpleNamespace(on_answer=lambda ctx, answer: acc)],
             before_final=kernel_script([("root A", shell_verdict)], calls))
    assert calls == ["root A"]
    if shell_verdict == "contradiction":
        assert types(r.traj) == ["answer_attempt", "recorded_verdict", "final_answer"]   # the shell's recorded verdict
        assert r.traj[-1] == withheld_final("root A")
        return
    assert types(r.traj) == ["answer_attempt", "harness_escalation"]
    assert r.traj[-1] == {"step": 0, "event_type": "harness_escalation", "stage": "before_final",
                          "reason": "hook_error", "status": "error"}
    assert r.kernel.ledger.commit_history == []


def test_read_from_on_answer_is_planned_at_before_final_readapt_and_queued(loop):
    fb = {"decision": "ACQUIRE", "stage": "before_final", "reason": "look again", "missing_obligations": []}
    verdict = acquire(args={"region": "apex", "attribute": "margin"}, capability="active_evidence")
    pending = []

    def on_answer(ctx, answer):
        if ctx.answers[-1].index > 1:
            return None
        v = ctx.verify_final(answer)
        na = actuation.next_action_of(v)
        return MH.Read(na["tool"], na.get("args"), feedback=fb, source=v)
    mw = SimpleNamespace(on_answer=on_answer, before_model=lambda ctx: pending.append(ctx.forced_pending))
    r = loop([final("a"), final("b")], [mw], before_final=scripted([verdict]))
    t = types(r.traj)
    assert t[:3] == ["answer_attempt", "stub_verdict", "harness_acquire_predispatch"]
    assert r.traj[2] == {"step": 0, "event_type": "harness_acquire_predispatch", "stage": "before_final_readapt",
                         "tool": "ImageDescription", "args": {"region": "apex", "attribute": "margin"},
                         "acquire_id": "acq-before_final_readapt-0-0", "status": "ok"}
    assert t[3:5] == ["harness_acquire_dispatch", "tool_call"]
    assert (r.traj[4]["requester"], r.traj[4]["requester_stage"]) == ("on_answer_read", "before_final_readapt")
    assert (r.traj[4]["step"], r.traj[4]["origin"], r.traj[4]["args"]) == (1, "harness", {"region": "apex",
                                                                                          "attribute": "margin"})
    assert r.agent.states[1]["harness_feedback"] == fb
    assert r.agent.states[1]["harness_acquired"]["tool"] == "ImageDescription"
    rec = r.kernel.ledger.acquire_telemetry[0]
    assert (rec["stage"], rec["capability"], rec["in_vocabulary"], rec["executed"]) == \
        ("before_final_readapt", "active_evidence", None, True)
    assert pending[:2] == [False, True]


def test_plan_read_in_the_hook_records_the_plan_before_what_the_hook_emits_next(loop):
    verdict = acquire()

    def on_answer(ctx, answer):
        if ctx.answers[-1].index > 1:
            return None
        v = ctx.verify_final(answer)
        read = MH.Read("ImageDescription", actuation.next_action_of(v).get("args"), source=v)
        assert ctx.plan_read(read) is True and read.plan.dispatchable
        ctx.emit({"event_type": "adaptive_acquisition", "round": 2})
        return read
    r = loop([final("a"), final("b")], [SimpleNamespace(on_answer=on_answer)], before_final=scripted([verdict]))
    assert types(r.traj)[:4] == ["answer_attempt", "stub_verdict", "harness_acquire_predispatch",
                                 "adaptive_acquisition"]
    assert types(r.traj).count("harness_acquire_predispatch") == 1 and r.traj[3]["step"] == 0
    assert r.env.calls == [("ImageDescription", {"region": "r", "attribute": "a"})]


def test_a_refused_plan_leaves_the_hook_free_to_decide_otherwise(loop):
    verdict = acquire(read_only=False)

    def on_answer(ctx, answer):
        v = ctx.verify_final(answer)
        if not ctx.plan_read(MH.Read("ImageDescription", {}, source=v)):
            return MH.Accept(answer, "kept_original_read_refused", "ACQUIRE", False)
    r = loop([final("a")], [SimpleNamespace(on_answer=on_answer)], before_final=scripted([verdict]))
    assert types(r.traj) == ["answer_attempt", "stub_verdict", "harness_acquire_refused", "recorded_verdict",
                             "final_answer"]
    assert r.traj[2]["reason"] == actuation.NOT_READ_ONLY and r.env.calls == []


def test_a_read_with_a_source_plans_exactly_as_its_verdict_would():
    v = acquire(args={"region": "apex"}, capability="active_evidence")
    led_a, led_b, traj = SimpleNamespace(validated_evidence_version=2), SimpleNamespace(validated_evidence_version=2), []
    pa = actuation.plan(v, "before_final_readapt", 5, ledger=led_a)
    na = actuation.next_action_of(v)
    pb = MH.plan_answer_read(MH.Read(na["tool"], na.get("args"), source=v), 5, led_b, traj)
    assert traj == pa.events and pb.forced_action == pa.forced_action and pb.forced_action["args"] is na["args"]
    assert led_b.acquire_telemetry == led_a.acquire_telemetry and led_b.acquire_count == led_a.acquire_count == 1
    sourceless = MH.plan_answer_read(MH.Read("OtherTool", {"x": 1}), 6, SimpleNamespace(), [])
    assert sourceless.dispatchable and sourceless.forced_action["tool"] == "OtherTool"
    read = MH.Read("OtherTool", {"x": 1})
    first = MH.plan_answer_read(read, 6, None, [])
    assert MH.plan_answer_read(read, 6, None, []) is first   # planned once


def test_on_answer_rejects_a_result_it_does_not_accept(loop):
    """The host refuses a result the hook does not accept, and the episode stops there. The refusal does not escape
    run_task as a bare TypeError that would leave the task without a result.json: through the crash path
    (runner/crash_result.py) it is the task's result, end_reason=harness_error, with the TypeError in its digest."""
    mw = SimpleNamespace(on_answer=lambda ctx, answer: MH.Nudge("x", "y", "z", "REVISE"))
    r = loop([final()], [mw])
    assert r.result["end_reason"] == "harness_error" and r.result["outcome_status"] == "harness_error"
    assert r.result["harness_error"]["type"] == "TypeError"
    assert "does not accept" in r.result["harness_error"]["message"]
    assert r.result["checkpoints"] == []


def _ctx(traj, shell=None, step=3):
    return MH.MiddlewareContext(env=None, task={}, trajectory=traj, step=step, max_steps=10, used_actions=0,
                                judge_factory=lambda: None, errors=[], shell=shell)


def test_verify_final_appends_only_the_first_verdicts_events():
    traj, verdicts = [], [Effective("REVISE", events=[{"event_type": "first"}]),
                          Effective("ALLOW", events=[{"event_type": "second"}])]
    ctx = _ctx(traj, {"verify_final": lambda text: verdicts.pop(0)})
    assert ctx.verify_final("A").type == "REVISE" and ctx.verify_final("B").type == "ALLOW"
    assert traj == [{"event_type": "first"}]

    def down(text):
        raise RuntimeError("judge down")
    traj2, ctx2 = [], None
    calls = [down, lambda text: Effective("ALLOW", events=[{"event_type": "later"}])]
    ctx2 = _ctx(traj2, {"verify_final": lambda text: calls.pop(0)(text)})
    with pytest.raises(RuntimeError):
        ctx2.verify_final("A")                 # a raised first call is still the first call
    assert ctx2.verify_final("B").type == "ALLOW" and traj2 == []
    with pytest.raises(RuntimeError):
        _ctx([]).verify_final("A")             # only in on_answer
    with pytest.raises(RuntimeError):
        _ctx([]).plan_read(MH.Read("t", {}))


def test_verify_final_is_the_kernels_query_and_record_final_its_recording_at_this_step():
    """verify_final is kernel.evaluate_final (a query); record_final records a verdict (kernel.record_final),
    evaluating the text first when no verdict was asked for."""
    calls = []
    verdict = Effective("ALLOW")
    kernel = SimpleNamespace(evaluate_final=lambda text, step=0: calls.append(("evaluate", text, step)) or verdict,
                             record_final=lambda v, step=None, flagged=False: calls.append(("record", v, step, flagged)),
                             ledger=None)
    services = MH.answer_services(kernel, 7, [])
    assert services["verify_final"]("B").type == "ALLOW" and calls == [("evaluate", "B", 7)]
    services["record_final"]("B", verdict, True)
    services["record_final"]("C", MH.UNASKED, False)
    assert calls[1:] == [("record", verdict, 7, True), ("evaluate", "C", 7), ("record", verdict, 7, False)]
    with pytest.raises(RuntimeError):
        MH.answer_services(None, 7, [])["verify_final"]("B")


@pytest.mark.parametrize("host_text,agent_text,expected", [
    (None, None, None), (None, "KERNEL TEXT", "KERNEL TEXT"), ("HOST TEXT", "KERNEL TEXT", "HOST TEXT")])
def test_acquire_text_at_before_final(loop, host_text, agent_text, expected):
    seen = []

    def after_final_verdict(ctx, answer, verdict):
        seen.append((answer, verdict.type))
        return host_text
    r = loop([final("a"), final("b")], [SimpleNamespace(after_final_verdict=after_final_verdict)],
             before_final=scripted([acquire(agent_text=agent_text)]))
    reason = r.agent.states[1]["harness_feedback"]["reason"]
    assert reason == (expected or "needs a read")   # with neither text: the decision's reason
    assert seen == [("a", "ACQUIRE")]   # once, for the non-ALLOW verdict only


def test_after_final_verdict_text_must_be_a_string():
    assert MH.agent_text(None) is None and MH.agent_text("") is None and MH.agent_text("t") == "t"
    with pytest.raises(TypeError):
        MH.agent_text({"reason": "t"})


# ------------------------------------------------------------------------------------------ decision text
@pytest.mark.parametrize("agent_text", [None, "READ THIS FIRST"])
def test_acquire_text_at_before_action(loop, agent_text):
    ba = scripted([acquire(args={"resourceType": "Observation"}, agent_text=agent_text)])
    r = loop([TOOL, TOOL, final()], [], before_action=ba)
    reason = r.agent.states[1]["harness_feedback"]["reason"]
    assert reason == (agent_text or "needs a read")   # without agent_text: the decision's reason


@pytest.mark.parametrize("agent_text", [None, "LOOK AT THIS"])
def test_acquire_text_at_after_action(loop, agent_text):
    r = loop([TOOL, final()], [], after_action=scripted([acquire(agent_text=agent_text)]))
    reason = r.agent.states[1]["harness_feedback"]["reason"]
    assert reason == (agent_text or "needs a read")   # without agent_text: the decision's reason


def test_decision_text_reads_the_raising_decision():
    assert MH.decision_text(acquire(agent_text="T")) == "T"
    assert MH.decision_text(acquire(agent_text="T").raw) == "T"
    assert MH.decision_text(acquire()) is None and MH.decision_text(acquire(agent_text="")) is None
    assert MH.decision_text(None) is None and MH.decision_text(Effective("ALLOW")) is None


# ------------------------------------------------------------------------------------------ context
def test_the_context_reads_live_shell_state(loop):
    rec, kept = collections.defaultdict(list), {}
    long_answer = "x" * 700

    def before_model(ctx):
        kept.setdefault("first", ctx)
        rec["model"].append((ctx.step, ctx.last_obs, ctx.observed_history, ctx.budget_spent_at_turn_start,
                             ctx.used_actions, ctx.live_used_actions, ctx.agent_env_actions,
                             ctx.recovery_env_actions, ctx.recovery_reserve, ctx.in_repair, ctx.forced_pending))

    def on_answer(ctx, answer):
        rec["answer"].append((ctx.answers, ctx.observed_history, ctx.obligations()))

    def after_episode(ctx):
        rec["end"].append((ctx.end_reason, kept["first"].used_actions, kept["first"].live_used_actions,
                           kept["first"].end_reason))
        rec["ledger"].append(ctx.ledger)
        rec["meta"].append(ctx.contract_meta)
    mw = SimpleNamespace(before_model=before_model, on_answer=on_answer, after_episode=after_episode)
    r = loop([TOOL, final(long_answer)], [mw], max_steps=1)
    obs = json.dumps(OUTPUT, ensure_ascii=False)
    assert rec["model"] == [(0, None, (), False, 0, 0, 0, 0, 0, False, False),
                            (1, obs, (), True, 1, 1, 1, 0, 0, False, False)]
    import hashlib
    assert rec["answer"] == [((MH.Answer(1, hashlib.sha1(long_answer.encode("utf-8")).hexdigest()[:12],
                                         long_answer),), (obs[:4000],), (True, None, None))]
    assert rec["end"] == [("final_answer", 0, 1, None)]   # a kept context: snapshot stays, live moves
    view = rec["ledger"][0]
    assert isinstance(view.evidence, list) and view.validated_evidence_version == r.kernel.ledger.validated_evidence_version
    view.evidence.append({"evidence_id": "forged"})
    assert {"evidence_id": "forged"} not in r.kernel.ledger.evidence
    with pytest.raises(AttributeError):
        view.evidence = []
    with pytest.raises(AttributeError):
        view.pending_authorization
    meta = rec["meta"][0]
    assert meta is not r.kernel.contract.meta and meta == r.kernel.contract.meta


def test_emit_stamps_the_step_and_keeps_a_copy():
    traj = []
    ctx = _ctx(traj, step=7)
    ev = {"event_type": "grounding_gap", "step": 99, "missing": ["X"]}
    ctx.emit(ev)
    ev["event_type"] = "changed"
    assert traj == [{"event_type": "grounding_gap", "step": 7, "missing": ["X"]}]
    with pytest.raises(ValueError):
        ctx.emit({"step": 1})


def test_emit_lands_in_the_run_trajectory(loop):
    mw = SimpleNamespace(before_model=lambda ctx: ctx.emit({"event_type": "note", "step": -1}) if ctx.step == 0 else None)
    r = loop([final()], [mw])
    assert r.traj[0] == {"event_type": "note", "step": 0}


def test_obligations_and_verdict_feedback_come_from_the_shell():
    assert _ctx([]).obligations() == (True, None, None)
    assert _ctx([], {"obligations": lambda: (False, "t", (True, False))}).obligations() == (False, "t", (True, False))
    v = Effective("REVISE", feedback={"reason": "fix the note", "suggested_capabilities": ["x"]})
    assert _ctx([], {"verdict_feedback": R._next_feedback}).verdict_feedback(v, "before_final") == \
        R._next_feedback(v, "before_final")
    with pytest.raises(RuntimeError):
        _ctx([]).verdict_feedback(v, "before_final")


class _StateEnv(object):
    def __init__(self, state):
        self.state, self.reads = state, 0

    def eval_state(self):
        self.reads += 1
        return self.state


def test_gui_obligations_is_the_met_check_and_the_snapshot_from_one_state_read():
    HAB = W.H0["gui"]
    content = OG.load_completion_obligations(HAB)
    goal = ("open denial den-1 in emr / " + " / ".join(p for _cid, when in content["requires"]
                                                         for p in ((when or {}).get("goal_mentions") or ())))
    actions = [{"event_type": "tool_call", "origin": "agent", "tool": "click",
                "args": {"ref": d["match"]["ref_pattern"].strip("^$")}} for d in content["decisions"].values()]
    for payer in ("", " payer b"):
        task = {"goal": goal + payer, "environment": {"type": "gui", "config": {}}}
        decisions, specs = OG.completion_conjunction(task, HAB, actions=actions)
        assert decisions == tuple(content["decisions"])
        assert len(specs) == len(content["decisions"]) + len(content["requires"])   # every conjunct is on
        for n in range(len(specs) + 1):
            state = {}
            for s in specs[:n]:
                node, parts = state, s["path"].split(".")
                for p in parts[:-1]:
                    node = node.setdefault(p, {}) if isinstance(node.get(p, {}), dict) else node
                node[parts[-1]] = ["x"] * 5
            env = _StateEnv(state)
            met, d, conj = OG.gui_obligations(task, env, HAB, actions=actions)
            assert env.reads == 1
            assert (met, d, conj) == (all(conj), decisions, OG.evaluate(specs, state))
            assert conj == (True,) * n + (False,) * (len(specs) - n)
            assert conj == OG.obligation_snapshot(task, _StateEnv(state), HAB, actions=actions)
    assert OG.gui_obligations({"environment": {"config": {}}}, _StateEnv({}), HAB) == (True, (), None)


def test_episode_end_reason():
    f = MH.episode_end_reason
    assert f([{"event_type": "tool_call"}], False, False) == "max_steps_exceeded"
    assert f([{"event_type": "final_answer"}], True, False) == "final_answer"
    assert f([{"event_type": "tool_budget_exhausted"}, {"event_type": "recovery_v3"},
              {"event_type": "tool_call", "origin": "recovery"}], False, True) == "tool_budget_exhausted"
    assert f([{"event_type": "harness_escalation", "status": "corrective"},
              {"event_type": "repair_budget_exhausted"}], False, True) == "repair_budget_exhausted"
    assert f([{"event_type": "harness_escalation", "status": "error"}], False, True) == "harness_escalation"
    assert f([{"event_type": "agent_error", "error": "truncated_tool_call"},
              {"event_type": "agent_error", "error": "invalid_action"}], True, False) == "invalid_action"
    assert f([{"event_type": "circuit_breaker"}], False, True) == "circuit_breaker"


# ------------------------------------------------------------------------------------------ instrumentation
def test_new_hooks_are_recorded_like_the_existing_ones(monkeypatch):
    inst = I.Instrumentation()
    monkeypatch.setattr(I.W, "component_path", lambda obj: "/elsewhere/workspaces/hab/h0/middleware/stub.py")

    class Stub(object):
        def next_read(self, ctx):
            return MH.Read("t", {})

        def on_answer(self, ctx, answer):
            return MH.Repair(None)

        def after_final_verdict(self, ctx, answer, verdict):
            return "text"

        def before_tool(self, ctx, action):
            return MH.Nudge("attachment", "fb", "o", "REVISE")

        def after_tool(self, ctx, action, result):
            return MH.PostRevision("write_verify", "fb", "o")
    stack, ctx = MH.MiddlewareStack([Stub()], inst), SimpleNamespace(step=4)
    stack.next_read(ctx)
    stack.on_answer(ctx, "a")
    stack.after_final_verdict(ctx, "a", None)
    stack.before_tool(ctx, {})
    stack.after_tool(ctx, {}, {})
    assert [(c["hook"], c["step"], c["returned"], c.get("stage"), c.get("label")) for c in inst.calls] == [
        ("next_read", 4, "Read", None, None), ("on_answer", 4, "Repair", None, None),
        ("after_final_verdict", 4, "str", None, None), ("before_tool", 4, "Nudge", "attachment", "REVISE"),
        ("after_tool", 4, "PostRevision", "write_verify", None)]
