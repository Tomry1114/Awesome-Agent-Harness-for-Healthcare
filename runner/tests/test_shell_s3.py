# -*- coding: utf-8 -*-
"""Turn inputs, reconcile in both directions, revision accounting and measurement details in the shell.

The loop tests drive the real run loop (run.run_task) with the scripted rig of test_middleware_host_contract.py (a
MedCTA-shaped fake environment, a scripted agent, the real kernel with its hooks stubbed). The rendering tests drive the
six H0 agent programs directly."""
import json
import os
import sys
from types import SimpleNamespace

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import test_middleware_host_contract as TC                               # noqa: E402
from test_middleware_host_contract import loop, TOOL, final, types      # noqa: E402,F401  (fixture)
import middleware_host as MH                                            # noqa: E402
from workspace import H0, Workspace, FUNCTION_CALLING, TEXT             # noqa: E402


# ------------------------------------------------------------------------------------------------ turn inputs
def test_every_input_of_a_turn_reaches_the_agent_and_is_consumed_only_after_its_act(loop):
    """The agent's tool call runs; the next two iterations each run a host next_read in place of the agent's turn; then
    the agent acts. All three results are handed to that act, in order. While the second read is being dispatched the
    agent has not acted since its tool call, so nothing is consumed yet; after the act all three are. A single
    last-result slot would show the agent only the second read."""
    reads, seen = [MH.Read("ImageDescription", {"region": "b"}), MH.Read("ImageDescription", {"region": "a"})], []

    def next_read(ctx):
        calls = [e for e in ctx.trajectory if e.get("event_type") == "tool_call"]
        if not calls or not reads:
            return None
        if len(calls) == 2:            # the tool call and the first read ran; the second read is next
            seen.append([c["delivery_record"]["consumed_by_agent"] for c in calls])
        return reads.pop()
    r = loop([TOOL, final("done")], [SimpleNamespace(next_read=next_read)])
    assert seen == [[False, False]]
    items = r.agent.states[1]["turn_inputs"]
    assert [(i["kind"], i["source"]) for i in items] == [
        ("tool_result", "agent"), ("harness_read_result", "next_read"), ("harness_read_result", "next_read")]
    assert [i["args"] for i in items] == [{}, {"region": "a"}, {"region": "b"}]
    calls = [e for e in r.traj if e["event_type"] == "tool_call"]
    assert [c["delivery_record"]["consumed_by_agent"] for c in calls] == [True, True, True]


def test_feedback_after_a_tool_result_keeps_the_result_in_the_turn(loop):
    """An after_tool PostRevision replaces the last-result view of the action that ran; the action's own result stays in
    the turn inputs before the feedback. The single views are unchanged."""
    mw = SimpleNamespace(after_tool=TC.once(MH.PostRevision("write_verify", "did not land", "write_not_landed")))
    r = loop([TOOL, final()], [mw])
    shown = r.agent.states[1]
    assert [(i["kind"], i.get("result")) for i in shown["turn_inputs"]] == [
        ("tool_result", TC.OUTPUT), ("feedback", {"feedback": "did not land"})]
    assert shown["last_result"] == {"feedback": "did not land"}
    assert shown["harness_feedback"] == {"decision": "REVISE", "stage": "write_verify", "reason": "did not land"}


# ------------------------------------------------------------------------------------------------ the H0 programs
PROGRAMS = [("pb", "fhir"), ("hab", "gui"), ("medcta", "tool_sandbox")]


class _Program(object):
    """An H0 agent program with a scripted model: every reply is one call of tool `x`."""

    def __init__(self, env_type, protocol):
        ws = Workspace(H0[env_type], protocol)
        ws.tool_description = lambda name: {"name": name, "description": name, "parameters": {"type": "object"}}
        ws.prompt_fragment = lambda name: ""
        self.protocol = protocol
        chat = ((lambda messages, tools: {"content": "", "tool_calls": [
                    {"id": "c%d" % len(messages), "type": "function", "function": {"name": "x", "arguments": "{}"}}]})
                if protocol == FUNCTION_CALLING else
                (lambda messages, max_new_tokens=400: '<tool_call>{"name": "x", "arguments": {}}</tool_call>'))
        self.obj = ws.program_class()({"goal": "q", "context": {}, "available_tools": [{"name": "x"}],
                                       "environment": {"type": env_type, "config": {}}}, ws, chat)

    def act(self, state):
        return self.obj.act(state)


def _result(text):
    return {"output": text, "observation": text}


def _tool_item(text):
    return {"kind": "tool_result", "source": "agent", "tool": "x", "args": {}, "result": _result(text),
            "observation": text}


def _read_item(text):
    return {"kind": "harness_read_result", "source": "next_read", "tool": "x", "args": {}, "result": _result(text),
            "observation": text, "requester": "next_read", "acquired": {"tool": "x", "args": {}, "output": text}}


def _state(items, **views):
    st = {"tools": [{"name": "x"}], "last_result": None, "last_observation": None, "harness_feedback": None,
          "harness_acquired": None}
    st.update(views)
    if items is not None:
        st["turn_inputs"] = items
    return st


SINGLE_TURNS = [
    lambda: ([_tool_item("RESULT-ONE")], {"last_result": _result("RESULT-ONE")}),
    lambda: ([{"kind": "feedback", "source": "nudge:before_final", "result": {"feedback": "keep going"},
               "observation": "obligation_unmet", "feedback": {"decision": "CONTINUE", "stage": "obligation",
                                                              "reason": "keep going"}}],
             {"last_result": {"feedback": "keep going"},
              "harness_feedback": {"decision": "CONTINUE", "stage": "obligation", "reason": "keep going"}}),
    lambda: ([_read_item("READ-ONE")], {"last_result": _result("READ-ONE"),
                                        "harness_acquired": {"tool": "x", "args": {}, "output": "READ-ONE"}}),
    lambda: ([{"kind": "feedback", "source": "kernel:before_action", "feedback": {
        "decision": "REVISE", "stage": "before_action", "reason": "fix the order", "evidence_needed": ["x"]}}],
             {"last_result": _result("RESULT-ONE"), "harness_feedback": {
                 "decision": "REVISE", "stage": "before_action", "reason": "fix the order", "evidence_needed": ["x"]}}),
]


@pytest.mark.parametrize("bench,env_type", PROGRAMS)
@pytest.mark.parametrize("protocol", [FUNCTION_CALLING, TEXT])
def test_a_turn_with_one_input_renders_as_before_and_every_further_input_is_shown_in_order(bench, env_type, protocol):
    """Each H0 program, over four single-input turns (a tool result, a Nudge, a harness read, a kernel REVISE): the
    conversation is byte-identical whether or not the state carries turn_inputs (without them the program reads the
    single-result views only). Then a turn with the agent's tool result followed by two harness reads: the tool result
    and both reads reach the conversation, in that order, not only the last read."""
    with_items, views_only = _Program(env_type, protocol), _Program(env_type, protocol)
    with_items.act(_state([], last_result=_result("first page")))
    views_only.act(_state(None, last_result=_result("first page")))
    for make in SINGLE_TURNS:
        items, views = make()
        with_items.act(_state(items, **views))
        views_only.act(_state(None, **views))
        assert json.dumps(with_items.obj.messages, sort_keys=True) == json.dumps(views_only.obj.messages, sort_keys=True)
    n = len(with_items.obj.messages)
    with_items.act(_state([_tool_item("AGENT-OWN-RESULT"), _read_item("HARNESS-READ-1"), _read_item("HARNESS-READ-2")],
                          last_result=_result("HARNESS-READ-2"),
                          harness_acquired={"tool": "x", "args": {}, "output": "HARNESS-READ-2"}))
    text = json.dumps(with_items.obj.messages[n:])
    positions = [text.find(s) for s in ("AGENT-OWN-RESULT", "HARNESS-READ-1", "HARNESS-READ-2")]
    assert -1 not in positions and positions == sorted(positions), (bench, protocol, positions)


# ------------------------------------------------------------------------------------------------ reconcile
def _classify_other_tool_as_create(monkeypatch):
    """The MedCTA-shaped rig's manifest maps no write; classify OtherTool as a reversible create everywhere the shell
    classifies an action (the reconcile write block, the harness action channel)."""
    import harness.semantics as SEM
    real = SEM.canonicalize_on_page

    def canon(action, manifest, controls):
        sem = real(action, manifest, controls)
        if (action or {}).get("tool") == "OtherTool":
            sem.semantic_type, sem.effect, sem.mapped = "create", "reversible", True
        return sem
    monkeypatch.setattr(SEM, "canonicalize_on_page", canon)


def _authorizing_kernel(monkeypatch, kernels, after):
    """A stub before_action that matches the harness write's bound authorization (as verify_commit does) and ALLOWs,
    and `after(action, origin)` as after_action."""
    import harness as H
    from harness.kernel import Effective
    build = H.build_kernel

    def spy(*a, **k):
        kern = build(*a, **k)
        kernels.append(kern)
        return kern
    monkeypatch.setattr(H, "build_kernel", spy)

    def before_action(action, *a, **k):
        auth_id = (action or {}).get("_mutation_authorization_id")
        if auth_id:
            kernels[0].ledger.pending_authorization = kernels[0].ledger.find_authorization_by_id(auth_id)
        return Effective("ALLOW")

    def after_action(action, *a, **k):
        return after(action, k.get("origin"))
    return before_action, after_action


def _reconcile_verdict():
    from harness.decision import HarnessDecision
    from harness.kernel import Effective
    return Effective("RECONCILE", feedback={"reason": "commit state unknown"},
                     raw=HarnessDecision("RECONCILE", capability="stub_capability", reason="unknown", rule_id="unknown"),
                     events=[{"event_type": "stub_verdict", "status": "ok"}])


WRITE_OTHER = {"type": "tool_call", "tool": "OtherTool", "args": {"target_path": "Note/1"}}


def test_an_unknown_harness_commit_blocks_the_agents_write(loop, monkeypatch):
    """A before_final middleware writes through the harness action channel; the kernel answers RECONCILE (the commit's
    effect is unknown) and refuses the final with a Nudge. The agent's next write is refused by the reconcile block,
    as after its own unknown commit, names the harness commit and takes a repair turn (the first: the Nudge draws
    none); it never reaches the environment. A harness commit with unknown effect leaves reconcile state like an
    agent commit does."""
    _classify_other_tool_as_create(monkeypatch)
    kernels, done = [], []
    from harness.kernel import Effective
    before, after = _authorizing_kernel(monkeypatch, kernels,
                                        lambda action, origin: _reconcile_verdict() if origin == "harness" else
                                        Effective("ALLOW"))

    def before_final(ctx, answer):
        if done:
            return None
        ctx.begin_harness_actions()
        done.append(ctx.request_action("OtherTool", {"target_path": "Note/9"}))
        return MH.Nudge("obligation", "not yet", "obligation_unmet", "CONTINUE")
    r = loop([final("a"), WRITE_OTHER, final("b")], [SimpleNamespace(before_final=before_final)],
             before_action=before, after_action=after, max_steps=6)
    assert r.env.calls == [("OtherTool", {"target_path": "Note/9"})]        # the harness write; not the agent's
    enter = [e for e in r.traj if e["event_type"] == "reconcile_enter"]
    assert [(e["commit_origin"], e["action_requester"]) for e in enter] == [("harness", "recovery")]
    blocks = [e for e in r.traj if e["event_type"] == "reconcile_block"]
    assert [(b["tool"], b["commit_origin"], b["repair_turn"]) for b in blocks] == [("OtherTool", "harness", 1)]
    assert r.agent.states[2]["last_result"]["feedback"].startswith("The harness has an UNCONFIRMED prior commit")


def test_the_agents_unknown_commit_restricts_harness_writes(loop, monkeypatch):
    """The agent's write gets RECONCILE at after_action. A before_final middleware then asks the harness action channel
    for a write: it is refused (stage reconcile, unconfirmed_commit_pending) and never runs. The agent's reconcile
    state binds harness writes too."""
    _classify_other_tool_as_create(monkeypatch)
    kernels, got = [], []
    from harness.kernel import Effective
    before, after = _authorizing_kernel(monkeypatch, kernels,
                                        lambda action, origin: _reconcile_verdict() if origin is None else
                                        Effective("ALLOW"))

    def before_final(ctx, answer):
        if not got:
            ctx.begin_harness_actions()
            got.append(ctx.request_action("OtherTool", {"target_path": "Note/9"}))
        return None
    r = loop([WRITE_OTHER, final("a")], [SimpleNamespace(before_final=before_final)],
             before_action=before, after_action=after, max_steps=6)
    assert r.env.calls == [("OtherTool", {"target_path": "Note/1"})]         # the agent's write only
    assert got == [{"error": "unconfirmed_commit_pending"}]
    assert [(e["stage"], e["reason"], e["action_requester"]) for e in r.traj
            if e["event_type"] == "harness_action_refused"] == [("reconcile", "unconfirmed_commit_pending", "recovery")]


# ------------------------------------------------------------------------------------------------ revision cap
@pytest.mark.parametrize("hook", ["before_final", "on_answer"])
def test_the_shell_applies_at_most_the_revision_cap(loop, hook):
    """A before_final middleware returns a Revision (or an on_answer middleware a Repair) for every final. The shell
    applies MAX_REVISIONS_PER_EPISODE of them; the next final goes on and the refusal is recorded, so the episode
    still ends with a final."""
    cap = getattr(MH, "MAX_REVISIONS_PER_EPISODE", 16)
    result = ((lambda ctx, answer: MH.Revision("precommit_terminal", "not yet", "precommit_terminal"))
              if hook == "before_final" else (lambda ctx, answer: MH.Repair({"decision": "REVISE", "reason": "again"})))
    r = loop([final("f%d" % i) for i in range(cap + 3)], [SimpleNamespace(**{hook: result})], max_steps=cap + 4)
    assert [e["thought"] for e in r.traj if e["event_type"] == "final_answer"] == ["f%d" % cap]
    refused = [e for e in r.traj if e["event_type"] == "harness_revision_refused"]
    assert [(e["hook"], e["reason"], e["cap"]) for e in refused] == [(hook, "revision_cap", cap)]
    assert r.result["budget"]["harness_ceilings"]["revisions_applied"] == cap


# ------------------------------------------------------------------------------------------------ accounting
def test_every_repair_type_result_has_one_accounting(loop):
    """A kernel REVISE at before_action, a host Revision at before_tool, a kernel REVISE at after_action with a host
    PostRevision after the same action, a Nudge at before_final and an on_answer Repair: each records one record_revision
    naming its source. All but the Nudge draw one repair turn from the one budget (the Repair is the 5th: past it,
    recorded as not started, and still applied) and unlock repair; the Nudge draws none and changes nothing about repair
    (it is workspace advice, bounded by its own cap). Host results and kernel verdicts share one accounting."""
    seen = []
    mw = SimpleNamespace(before_model=lambda ctx: seen.append(ctx.in_repair),
                         before_tool=TC.once(MH.Revision("retention", "keep what you had", "retention_block")),
                         after_tool=TC.once(MH.PostRevision("write_verify", "did not land", "write_not_landed")),
                         before_final=TC.once(MH.Nudge("obligation", "not yet", "obligation_unmet", "CONTINUE")),
                         on_answer=TC.once(MH.Repair({"decision": "REVISE", "reason": "re-answer"})))
    r = loop([TOOL, TOOL, TOOL, final("a"), final("b"), final("c")], [mw],
             before_action=TC.scripted([_verdict_revise("fix the order")]),
             after_action=TC.scripted([_verdict_revise("that read was not enough")]), max_steps=8)
    assert [(a["stage"], a["source"], a["repair_turn"], a["repair_turn_started"], a["suppressed_reason"])
            for a in r.kernel.ledger.actuation_telemetry if a.get("kind") == "REVISE"] == [
        ("before_action", "kernel:REVISE", 1, True, None), ("before_tool", "host:Revision", 2, True, None),
        ("after_action", "kernel:REVISE", 3, True, None), ("after_tool", "host:PostRevision", 4, True, None),
        ("before_final", "host:Nudge", None, False, None),
        ("on_answer", "host:Repair", 5, False, "repair_budget_exhausted")]
    assert seen[:2] == [False, True] and all(seen[1:])
    assert [e["thought"] for e in r.traj if e["event_type"] == "final_answer"] == ["c"]
    assert r.result["budget"]["harness_ceilings"]["repair_turns"] == 5


def test_a_nudge_draws_no_repair_turn_and_leaves_repair_locked(loop):
    """Ten Nudges (the Nudge cap) at finals, then a kernel REVISE at before_action. The Nudges are recorded but draw no
    repair turn and never unlock repair, so the REVISE is repair turn 1, within the budget, and the episode goes on to
    a final."""
    seen = []
    mw = SimpleNamespace(before_model=lambda ctx: seen.append(ctx.in_repair),
                         before_final=lambda ctx, answer: (MH.Nudge("obligation", "not yet", "obligation_unmet",
                                                                    "CONTINUE") if answer.startswith("n") else None))
    r = loop([final("n%d" % i) for i in range(10)] + [TOOL, final("done")], [mw],
             before_action=TC.scripted([_verdict_revise("fix the order")]), max_steps=16)
    recs = [(a["source"], a["repair_turn"], a["repair_turn_started"]) for a in r.kernel.ledger.actuation_telemetry]
    assert recs == [("host:Nudge", None, False)] * 10 + [("kernel:REVISE", 1, True)]
    assert seen[:11] == [False] * 11 and seen[11] is True
    assert "repair_budget_exhausted" not in types(r.traj)
    assert [e["thought"] for e in r.traj if e["event_type"] == "final_answer"] == ["done"]


def _verdict_revise(reason):
    from harness.decision import HarnessDecision
    from harness.kernel import Effective
    return Effective("REVISE", feedback={"reason": reason},
                     raw=HarnessDecision("REVISE", capability="stub_capability", reason=reason, rule_id="stub_rule"),
                     events=[{"event_type": "stub_verdict", "status": "ok"}])


# ------------------------------------------------------------------------------------------------ measurement details
@pytest.mark.parametrize("refuse", [False, True])
def test_the_forced_deliverable_writes_kernel_decision_events_are_recorded(loop, refuse):
    """(a) The kernel's harness_decision events of the forced deliverable write reach the trajectory, for an ALLOWed
    write and a refused one: the harness action channel extends the write's before_action / after_action events into
    the trajectory."""
    from harness.kernel import Effective
    WRITE = {"type": "tool_call", "tool": "write_file", "args": {"path": "/o/note.md", "content": "the agent's note"}}
    decision = {"event_type": "harness_decision", "id": "hdec-forced", "stage": "before_action", "status": "ok"}

    def before_action(action, *a, **k):
        if k.get("requester") == "forced_deliverable_write":
            return Effective("REVISE" if refuse else "ALLOW", feedback={"reason": "no"}, events=[dict(decision)])
        return Effective("ALLOW")

    def after_episode(ctx):
        ctx.forced_deliverable_write("write your deliverable")
    r = loop([final("a"), WRITE], [SimpleNamespace(after_episode=after_episode)], before_action=before_action)
    assert [e["id"] for e in r.traj if e.get("event_type") == "harness_decision"] == ["hdec-forced"]
    assert bool([e for e in r.traj if e.get("forced_deliverable")]) is not refuse


def test_the_kernels_last_write_reads_the_declared_path_parameter():
    """(b) PhysicianBench's write_file declares file_path (upstream); the kernel's last_write.path is read from
    file_path, so it is set for every native write, and an append extends the recorded content."""
    sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "harness", "tests"))
    import test_deliverable_single_source as DS
    import harness as H
    real = H._build_judge
    H._build_judge = lambda: (None, None)
    try:
        k = DS._kernel("fhir")
    finally:
        H._build_judge = real
    DS._after(k, 1, "write_file", {"file_path": "/workspace/output/plan.txt", "content": "part one. "})
    DS._after(k, 2, "write_file", {"file_path": "/workspace/output/plan.txt", "content": "part two.", "mode": "a"})
    assert k.ctx.last_write["path"] == "/workspace/output/plan.txt"
    assert k.ctx.last_write["content"] == "part one. part two."


def test_governance_reads_the_final_appended_content():
    """(c) With no deliverable file on disk the PhysicianBench extractor rebuilds the deliverable from the writes to the
    declared path (file_path), not from the last write alone. The final content is every write to the declared path,
    in order: an append extends, a "w" overwrites."""
    import governance as gov
    tm = {"source_benchmark": "PhysicianBench", "context": {"output_path": "/workspace/output/plan.txt"}}
    trace = [{"event_type": "tool_call", "tool": "write_file",
              "args": {"file_path": "/workspace/output/plan.txt", "content": "DRAFT"}},
             {"event_type": "tool_call", "tool": "write_file",
              "args": {"file_path": "/workspace/output/plan.txt", "content": "Assessment: stable. "}},
             {"event_type": "tool_call", "tool": "write_file",
              "args": {"file_path": "/workspace/output/plan.txt", "content": "Addendum: recheck TSH.", "mode": "a"}},
             {"event_type": "final_answer", "thought": "Saved."}]
    answer, ex = gov._agent_final_output_ex(trace, policy=gov.PB_POLICY, bundle_dir=None, task_manifest=tm)
    assert "Assessment: stable. Addendum: recheck TSH." in answer and "DRAFT" not in answer, answer
    assert "write_file.content[declared_output_path]" in ex["source_fields"]


def test_an_escalate_delivery_records_its_disposition_and_attempt(loop):
    """(d) The kernel's before_final ESCALATE on an epistemic final delivers it flagged; the final_answer event carries
    final_disposition and answer_attempt_index like every other delivery."""
    from harness.decision import HarnessDecision
    from harness.kernel import Effective
    esc = Effective("ESCALATE", feedback={"reason": "unresolved"},
                    raw=HarnessDecision("ESCALATE", capability="stub_capability", reason="unresolved", rule_id="x"),
                    events=[])
    r = loop([final("the answer")], [], before_final=TC.scripted([esc]))
    fa = [e for e in r.traj if e["event_type"] == "final_answer"]
    assert [(e["final_disposition"], e["answer_attempt_index"], e["verification_flag"]) for e in fa] == [
        ("flagged_delivery_escalate", 1, "unresolved_risk")]
