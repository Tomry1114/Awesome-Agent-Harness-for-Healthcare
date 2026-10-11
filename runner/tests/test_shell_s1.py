# -*- coding: utf-8 -*-
"""The harness action channel.

Every harness-origin environment action goes through one shell path (runner/harness/action_service.py): scope, the
ceilings, the manifest's read-only classification, write provenance, the kernel ceremony with origin, the executor, the
record and the content guard. These tests drive the real run loop (run.run_task) with the scripted rigs of
test_middleware_host_contract.py (a MedCTA-shaped fake environment, a scripted agent, the real kernel with its hooks
stubbed) and inline_hab_stub.py (the HealthAdminBench portal stand-in with the real GuiEnvReal, kernel and recovery entry)."""
import os
import sys
from types import SimpleNamespace

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import test_middleware_host_contract as TC                               # noqa: E402
from test_middleware_host_contract import loop, TOOL, final, types      # noqa: E402,F401  (fixture)
import inline_hab_stub as S                                             # noqa: E402
import harness as H                                                     # noqa: E402
import middleware_host as MH                                            # noqa: E402
from harness.decision import HarnessDecision                            # noqa: E402
from harness.kernel import Effective                                    # noqa: E402

WRITE = {"type": "tool_call", "tool": "write_file", "args": {"path": "/o/note.md", "content": "the agent's note"}}


def _verdict(kind, rule="stub_rule"):
    raw = HarnessDecision(kind, capability="stub_capability", reason=rule, rule_id=rule)
    return Effective(kind, feedback={"reason": rule}, raw=raw, events=[{"event_type": "stub_verdict", "status": "ok"}])


def _refusals(traj):
    return [(e.get("tool"), e.get("stage"), e.get("reason"), e.get("action_requester")) for e in traj
            if e.get("event_type") == "harness_action_refused"]


# ------------------------------------------------------------------------------------------------ read-only reads
@pytest.mark.parametrize("tool", ["type", "OtherTool"])
def test_a_next_read_of_a_tool_that_is_not_read_only_is_refused(loop, tool):
    """next_read asks, at every turn, for a `type` (no read rule of this manifest) or an unmapped tool. The channel
    refuses it (read_not_read_only) before the kernel: the environment is never called and the agent takes the turn.
    A loop that dispatched it as a read anyway would spend every step on it and never give the agent a turn."""
    mw = SimpleNamespace(next_read=lambda ctx: MH.Read(tool, {"ref": "e1", "text": "harness-authored text"}))
    r = loop([final("agent answer")], [mw], max_steps=4)
    assert r.env.calls == []
    assert _refusals(r.traj)[0] == (tool, "read_channel", "read_not_read_only", "next_read")
    assert "harness_acquire_dispatch" not in types(r.traj)
    assert [e["thought"] for e in r.traj if e["event_type"] == "final_answer"] == ["agent answer"]
    assert len(r.agent.states) == 1


# ------------------------------------------------------------------------------------------------ invocation scope
def test_an_action_from_a_saved_context_after_its_hook_returned_is_refused(loop):
    """A before_final middleware opens an invocation, keeps its context and returns a Nudge. From a later before_tool and
    from after_episode (after the final) the kept context asks for an action and opens again: the invocation ended when
    its hook returned, so the channel refuses the request (recorded) and the scope opens nothing."""
    box, got = {}, []

    def before_final(ctx, answer):
        if "ctx" not in box:
            box["step"] = ctx.begin_harness_actions()
            box["ctx"] = ctx
            return MH.Nudge("x", "keep going", "o", "CONTINUE")
        return None

    def before_tool(ctx, action):
        if "ctx" in box and not got:
            got.append(box["ctx"].request_action("ImageDescription", {"n": 7}))
        return None

    def after_episode(ctx):
        got.append(box["ctx"].begin_harness_actions())
        got.append(box["ctx"].request_action("ImageDescription", {"after": "end"}))
    r = loop([final("a"), TOOL, final("b")],
             [SimpleNamespace(before_final=before_final, before_tool=before_tool, after_episode=after_episode)],
             max_steps=4)
    assert box["step"] == 0
    assert ("ImageDescription", {"n": 7}) not in r.env.calls and ("ImageDescription", {"after": "end"}) not in r.env.calls
    assert got == [{"error": "harness_invocation_closed"}, None, {"error": "harness_invocation_closed"}]
    assert [x[2] for x in _refusals(r.traj)] == ["harness_invocation_closed"] * 2
    assert r.result["budget"]["harness_env_actions"] == 0


def test_begin_again_in_the_same_hook_resets_nothing(loop):
    """With the agent's budget spent, one invocation draws the per-invocation completion allowance (12). Calling
    begin_harness_actions again inside the same hook returns the open invocation and resets nothing, so the 13th action
    is still refused."""
    out = []

    def before_final(ctx, answer):
        if out:
            return None
        ctx.begin_harness_actions()
        out.extend(ctx.request_action("ImageDescription", {"i": i}).get("error") for i in range(13))
        out.append(ctx.begin_harness_actions())
        out.append(ctx.request_action("ImageDescription", {"i": "again"}).get("error"))
        return None
    r = loop([TOOL, final("a")], [SimpleNamespace(before_final=before_final)], max_steps=1)
    assert out[:12] == [None] * 12 and out[12] == "recovery_budget_exhausted"
    assert out[13] == 1 and out[14] == "recovery_budget_exhausted"
    assert ("ImageDescription", {"i": "again"}) not in r.env.calls


def test_the_post_stop_allowance_needs_a_reserve_the_adapter_grants(loop):
    """MedCTA's adapter grants harness actions no reserve: after tool_budget_exhausted the post-stop allowance is None,
    no invocation opens and nothing runs."""
    got = []

    def after_episode(ctx):
        got.append((ctx.end_reason, ctx.post_stop_allowance()))
        if got[0][1] is not None:
            got.append(ctx.request_action("ImageDescription", {"post": "stop"}))
    r = loop([TOOL] * 12, [SimpleNamespace(after_episode=after_episode)], max_steps=6)
    assert got == [("tool_budget_exhausted", None)]
    assert ("ImageDescription", {"post": "stop"}) not in r.env.calls


# ------------------------------------------------------------------------------------------------ budgets
def test_harness_reads_do_not_consume_the_agents_budget_or_repair_turns(loop):
    """A next_read asks for five reads; the kernel REVISEs the first two (origin harness). The agent's own two tool calls
    and its final all run within max_steps=2: harness reads are counted by the channel, not in the agent's budget, its
    loop iterations or its repair turns, and the refused reads hand the agent no REVISE. If harness REVISEs counted as
    the agent's repair turns and harness reads spent its budget, both of the agent's tool calls would be refused."""
    n, reads = [0], [0]

    def before_action(action, *a, **k):
        if k.get("origin") == "harness":
            n[0] += 1
            if n[0] <= 2:
                return _verdict("REVISE", "harness_read_revised")
        return Effective("ALLOW")

    def next_read(ctx):
        reads[0] += 1
        return MH.Read("ImageDescription", {"region": "r%d" % reads[0]}) if reads[0] <= 5 else None
    mw = SimpleNamespace(next_read=next_read)
    r = loop([TOOL, TOOL, final("done")], [mw], max_steps=2, before_action=before_action)
    agent_calls = [e for e in r.traj if e["event_type"] == "tool_call" and e["origin"] == "agent"]
    harness_calls = [e for e in r.traj if e["event_type"] == "tool_call" and e["origin"] == "harness"]
    assert len(agent_calls) == 2 and harness_calls
    assert [e["thought"] for e in r.traj if e["event_type"] == "final_answer"] == ["done"]
    assert "repair_budget_exhausted" not in types(r.traj) and "tool_budget_exhausted" not in types(r.traj)
    assert all((s.get("harness_feedback") or {}).get("decision") != "REVISE" for s in r.agent.states)
    b = r.result["budget"]
    assert b["agent_env_actions"] == 2 and b["harness_env_actions"] == len(harness_calls)
    assert b["harness_env_actions_by_requester"] == {"next_read": len(harness_calls)}
    assert b["total_env_actions"] == 2 + len(harness_calls) and b["budget_exhausted"] is True
    assert len(harness_calls) == 3 and len(r.agent.states) == 3
    assert [x[2] for x in _refusals(r.traj)] == ["REVISE:harness_read_revised"] * 2


# ------------------------------------------------------------------------------------------------ forced write
@pytest.mark.parametrize("case", ["revise", "raises", "safety_abort"])
def test_the_forced_deliverable_write_is_refused_after_a_revise_an_exception_or_a_safety_abort(loop, case):
    """after_episode forces the deliverable write. It is a harness write through the channel: a kernel REVISE refuses it,
    a raising before_action refuses it, and after a safety abort (an after_action ESCALATE ended the episode) the agent is
    not even asked. In none of the three does the write run."""
    def after_episode(ctx):
        ctx.forced_deliverable_write("write your deliverable")

    def before_action(action, *a, **k):
        if k.get("requester") == "forced_deliverable_write":
            if case == "raises":
                raise RuntimeError("kernel down")
            if case == "revise":
                return _verdict("REVISE", "do_not_write")
        return Effective("ALLOW")
    after = TC.scripted([_verdict("ESCALATE", "unsafe")]) if case == "safety_abort" else None
    script = [TOOL, WRITE] if case == "safety_abort" else [final("a"), WRITE]
    r = loop(script, [SimpleNamespace(after_episode=after_episode)], before_action=before_action, after_action=after)
    assert ("write_file", WRITE["args"]) not in r.env.calls
    blocked = [e for e in r.traj if e["event_type"] == "forced_deliverable_blocked"]
    assert len(blocked) == 1 and not [e for e in r.traj if e.get("forced_deliverable")]
    if case == "safety_abort":
        assert blocked[0]["reason"] == "after_safety:harness_escalation"
        assert len(r.agent.states) == 1                     # the forced turn was never asked for
    else:
        assert blocked[0]["reason"] == {"revise": "REVISE:do_not_write", "raises": "before_action_error"}[case]
        assert _refusals(r.traj) == [("write_file", "kernel", blocked[0]["reason"], "forced_deliverable_write")]


def test_the_forced_deliverable_write_runs_through_the_channel(loop):
    def after_episode(ctx):
        ctx.forced_deliverable_write("write your deliverable")
    r = loop([final("a"), WRITE], [SimpleNamespace(after_episode=after_episode)])
    ev = [e for e in r.traj if e.get("forced_deliverable")]
    assert r.env.calls == [("write_file", WRITE["args"])] and len(ev) == 1
    assert (ev[0]["origin"], ev[0]["requester"], ev[0]["content_author"]) == ("harness", "forced_deliverable_write",
                                                                             "agent")
    assert r.result["budget"]["harness_env_actions_by_requester"] == {"forced_deliverable_write": 1}
    assert r.result["budget"]["agent_env_actions"] == 0


# ------------------------------------------------------------------------------------------------ order
def test_a_refused_before_tool_leaves_no_reserved_authorization(loop, monkeypatch):
    """The kernel matches an authorization to the agent's tool call and ALLOWs it; host before_tool refuses the call with
    a Nudge. The authorization stays AVAILABLE: before_tool runs before the reserve, so no authorization is left
    RESERVED for an action that never ran."""
    kernels, auths = [], []
    build = H.build_kernel

    def spy(*a, **k):
        kern = build(*a, **k)
        kernels.append(kern)
        return kern
    monkeypatch.setattr(H, "build_kernel", spy)

    def before_action(action, *a, **k):
        L = kernels[0].ledger
        auth = L.mint_authorization(source="user_goal", allowed_semantic_type="inspect", allowed_tool=action.get("tool"),
                                    expected_postcondition={"covered_commit_point": True})
        auths.append(auth)
        L.pending_authorization = auth
        return Effective("ALLOW")
    mw = SimpleNamespace(before_tool=TC.once(MH.Nudge("attachment", "attach first", "attachment_block", "REVISE")))
    r = loop([TOOL, final()], [mw], before_action=before_action)
    assert r.env.calls == [] and "authorization_reserve_failed" not in types(r.traj)
    assert auths[0].status == "AVAILABLE"


# ------------------------------------------------------------------------------------------------ nudge cap
def test_the_shell_applies_at_most_the_nudge_cap(loop):
    """A before_final middleware nudges every final. The shell applies MAX_NUDGES_PER_EPISODE of them; the next final is
    accepted and the refusal recorded, so a middleware that always nudges cannot keep the episode from ending with a
    final."""
    cap = getattr(MH, "MAX_NUDGES_PER_EPISODE", 10)
    mw = SimpleNamespace(before_final=lambda ctx, answer: MH.Nudge("obligation", "not yet", "obligation_unmet",
                                                                  "CONTINUE"))
    r = loop([final("f%d" % i) for i in range(cap + 3)], [mw], max_steps=cap + 4)
    assert [e["thought"] for e in r.traj if e["event_type"] == "final_answer"] == ["f%d" % cap]
    refused = [e for e in r.traj if e["event_type"] == "harness_nudge_refused"]
    assert [(e["hook"], e["reason"], e["cap"]) for e in refused] == [("before_final", "nudge_cap", cap)]
    assert r.result["budget"]["harness_ceilings"]["nudges_applied"] == cap


# ------------------------------------------------------------------------------------------------ guard
def _hab(monkeypatch, tmp_path):
    h = S.Harness(monkeypatch, tmp_path)
    h.use_task(S.load_hab_task("HAB-denial-easy-1"))
    return h


def _script(h, ops):
    entry = S.recovery_entry()
    results = []

    class ScriptedRecoveryKernel(object):
        def run_all_episodes(self, bench, wf_registry, substrate, driver, task, trajectory, goal, judge, events=None):
            if results:
                return []
            for tool, args in ops:
                results.append((tool, substrate.driver.call_tool(tool, dict(args))))
            return []
    h.mp.setattr(entry, "RecoveryKernel", ScriptedRecoveryKernel)
    return results


def _clear(portal, ref):
    for c in portal.controls:
        if c.get("role") == "textbox":
            c["value"] = ""


def test_a_content_guard_restore_counts_against_the_ceiling(monkeypatch, tmp_path):
    """The agent types its note. Recovery's invocation spends 19 actions (the entry snapshot and 18 clicks), then a 20th
    click clears the note. Writing it back would be the 21st harness action: the restore is refused at the cap and named,
    and recovery stays at 20."""
    h = _hab(monkeypatch, tmp_path)
    p = h.portal
    p.controls = [S.control("noop", label="Refresh"), S.control("tab-history", label="History"),
                  S.control("triage-note", role="textbox", label="Triage Note")]
    p.on_click = {"tab-history": _clear}
    results = _script(h, [("click", {"ref": "noop"})] * 18 + [("click", {"ref": "tab-history"})])
    h.run([S.fill("triage-note", "CO-50: route to clinical appeals"), final("f0")], max_steps=40)
    assert all("error" not in r for _t, r in results)
    assert h.result["budget"]["recovery_env_actions"] == 20
    guard = h.events("agent_content_guard")
    assert len(guard) == 1 and guard[0]["dom_restored"] == [] and guard[0].get("refused") == ["triage-note"]
    assert [e["reason"] for e in h.events("harness_action_refused") if e["stage"] == "content_guard"] == [
        "recovery_total_cap"]
    assert not [e for e in h.events("tool_call") if e.get("requester") == "content_guard"]
    assert p.find("triage-note")["value"] == ""


def test_the_content_guard_restores_the_agents_text_and_skips_harness_authored_text(monkeypatch, tmp_path):
    """Two textboxes are cleared by one recovery click: the agent typed the note, recovery typed the other box (a value
    traced to the persisted state). Only the agent's note is written back (content_author agent); the harness-authored
    text is named and left empty."""
    h = _hab(monkeypatch, tmp_path)
    p = h.portal
    p.state["emr"] = {"denialReason": "CO-50 medical necessity"}
    p.controls = [S.control("tab-history", label="History"),
                  S.control("triage-note", role="textbox", label="Triage Note"),
                  S.control("reason-box", role="textbox", label="Denial Reason Summary")]
    p.on_click = {"tab-history": _clear}
    results = _script(h, [("type", {"ref": "reason-box", "text": "CO-50 medical necessity"}),
                          ("click", {"ref": "tab-history"})])
    h.run([S.fill("triage-note", "Route to clinical appeals"), final("f0")], max_steps=40)
    assert all("error" not in r for _t, r in results), results
    guard = h.events("agent_content_guard")
    assert len(guard) == 1
    assert guard[0]["dom_restored"] == ["triage-note"]
    assert guard[0]["harness_authored"] == {"dom": ["reason-box"], "persisted": []}
    restores = [e for e in h.events("tool_call") if e.get("requester") == "content_guard"]
    assert [(e["tool"], e["args"]) for e in restores] == [("type", {"ref": "triage-note",
                                                                   "text": "Route to clinical appeals"})]
    assert restores[0]["content_author"] == "agent"
    assert p.find("reason-box")["value"] == "" and p.find("triage-note")["value"] == "Route to clinical appeals"
