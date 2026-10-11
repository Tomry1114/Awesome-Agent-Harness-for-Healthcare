# -*- coding: utf-8 -*-
"""The middleware host (shell) and PhysicianBench's deliverable middleware in the H0 workspace.

A parity capture shows the default path byte-identical but does not exercise three things the host owns, so they are
pinned here: a regressive rewrite stopped through before_tool, the forced deliverable write at the end of an episode
(executed / refused by the harness / agent error), and the missing-deliverable report."""
import os
import sys

import pytest

_RUNNER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RUNNER not in sys.path:
    sys.path.insert(0, _RUNNER)

import middleware_host as MH                             # noqa: E402
from workspace import H0, Workspace                       # noqa: E402

GOAL = {"goal": "Write your assessment to output/note.md", "context": ""}


def _ctx(env, trajectory, errors=None, judge=None, forced=None, step=3, max_steps=100, used=10):
    return MH.MiddlewareContext(env=env, task=GOAL, trajectory=trajectory, step=step, max_steps=max_steps,
                                used_actions=used, judge_factory=lambda consumer: judge,
                                errors=errors if errors is not None else [], forced_deliverable_write=forced)


class _Env(object):
    def __init__(self, ws):
        self.workspace = str(ws)
        self.calls = []

    def available_tools(self):
        return ["write_file"]

    def call_tool(self, name, args):
        self.calls.append((name, args))
        open(os.path.join(self.workspace, os.path.basename(args["path"])), "w").write(args["content"])
        return {"written": args["path"]}


# ------------------------------------------------------------------------------------------------ host
class _M(object):
    def __init__(self, tag, out):
        self.tag, self.out = tag, out

    def before_final(self, ctx, answer):
        return self.out

    def deferred_errors(self):
        return [self.tag]


def test_first_non_none_wins_in_manifest_order():
    stack = MH.MiddlewareStack([_M("a", None), _M("b", MH.Revision("s", "fb", "o")), _M("c", MH.Revision("x", "y", "z"))])
    r = stack.before_final(None, "answer")
    assert (r.stage, r.feedback, r.obs) == ("s", "fb", "o")
    assert stack.deferred_errors() == ["a", "b", "c"]
    assert stack.before_tool(None, {}) is None and stack.exempt_from_budget(None, {}) is False


def test_each_substrate_gets_its_h0_middleware():
    assert [type(m).__name__ for m in MH.load_middleware(H0["fhir"], GOAL).middlewares] == ["DeliverableMiddleware"]
    assert [type(m).__name__ for m in MH.load_middleware(H0["gui"], GOAL).middlewares] == [
        "Recovery", "ObligationNudge", "GroundingAudit", "ReadAcquire",
        "PrecommitAudit", "AttachmentBlock", "CollectionNudge", "WriteVerify"]
    assert Workspace(H0["fhir"]).check([]) == []


# ------------------------------------------------------------------------------------------------ deliverable
def _deliverable():
    return MH.load_middleware(H0["fhir"], GOAL).middlewares[0]


def test_before_final_asks_for_the_missing_deliverable(tmp_path):
    d, traj = _deliverable(), []
    r = d.before_final(_ctx(_Env(tmp_path), traj), "done")
    assert r.stage == "deliverable" and "output/note.md" in r.feedback and r.obs == r.feedback
    assert traj[-1]["event_type"] == "deliverable_nudge"


def test_a_regressive_rewrite_is_stopped_through_before_tool(tmp_path):
    env, traj = _Env(tmp_path), [{"event_type": "tool_call", "tool": "fhir_observation_search_labs",
                                  "result": {"entries": [{"code": {"text": "TSH"}}, {"code": {"text": "Prolactin"}}]}}]
    A = "# Assessment\nTSH 8.1 elevated -> hypothyroidism.\nProlactin 30 normal.\nPlan: levothyroxine.\n"
    open(os.path.join(str(tmp_path), "note.md"), "w").write(A)
    d = _deliverable()
    d.arm_retention(_ctx(env, traj))
    B = "# Assessment\nTSH 8.1 elevated -> hypothyroidism.\nPlan: levothyroxine + TFT recheck.\n"
    r = d.before_tool(_ctx(env, traj, judge=lambda p: '{"dropped": ["Prolactin 30 result"]}'),
                      {"tool": "write_file", "args": {"file_path": "output/note.md", "content": B}})
    assert r.stage == "retention" and r.obs == "retention_block" and "Prolactin" in r.feedback
    assert d.before_tool(_ctx(env, traj), {"tool": "fhir_observation_search_labs", "args": {}}) is None


def test_the_one_over_budget_write_is_only_the_missing_deliverable(tmp_path):
    env, d = _Env(tmp_path), _deliverable()
    assert d.exempt_from_budget(_ctx(env, []), {"tool": "write_file", "args": {"file_path": "output/note.md"}})
    assert not d.exempt_from_budget(_ctx(env, []), {"tool": "write_file", "args": {"file_path": "output/other.md"}})
    open(os.path.join(str(tmp_path), "note.md"), "w").write("written")
    assert not d.exempt_from_budget(_ctx(env, []), {"tool": "write_file", "args": {"file_path": "output/note.md"}})


def test_the_deliverable_write_is_recognised_by_the_argument_the_tool_declares(tmp_path, monkeypatch):
    """The path argument comes from the substrate's published tool schemas (substrate_catalog.tool_parameters), not a
    name the middleware carries: the upstream registry declares file_path; a schema declaring another *_path name
    moves the recognition with it; an unreadable schema recognises nothing and reports why."""
    from harness import substrate_catalog as SC
    env, d = _Env(tmp_path), _deliverable()
    assert SC.tool_parameters("fhir")["write_file"] == ["content", "file_path", "mode"]
    assert d._path_args("write_file") == ["file_path"]
    assert not d.is_required_write({"tool": "write_file", "args": {"path": "output/note.md"}})

    monkeypatch.setattr(SC, "tool_parameters", lambda env_type: {"write_file": ["body", "target_path"]})
    d2 = _deliverable()
    assert d2.is_required_write({"tool": "write_file", "args": {"target_path": "output/note.md", "body": "x"}})
    assert not d2.is_required_write({"tool": "write_file", "args": {"file_path": "output/note.md"}})

    def unavailable(env_type):
        raise SC.SubstrateUnavailable("upstream tool registry not importable")
    monkeypatch.setattr(SC, "tool_parameters", unavailable)
    d3 = _deliverable()
    assert not d3.exempt_from_budget(_ctx(env, []), {"tool": "write_file", "args": {"file_path": "output/note.md"}})
    assert d3.deferred_errors() == [
        "deliverable_path_args: SubstrateUnavailable('upstream tool registry not importable')"]


# ------------------------------------------------------------------------------------------------ forced write
class _Agent(object):
    def __init__(self, action):
        self.action, self.states = action, []

    def act(self, state):
        self.states.append(state)
        if isinstance(self.action, Exception):
            raise self.action
        return self.action


class _Harness(object):
    """The kernel as the harness action channel calls it (a manifest that maps no write, so no authorization)."""
    manifest = {"actions": []}

    def __init__(self, decision=None):
        from harness.state import Ledger
        self.decision, self.after, self.ledger = decision, [], Ledger()

    def before_action(self, action, snap, step=0, origin=None, requester=None):
        self.origins = getattr(self, "origins", []) + [("before_action", origin, requester)]
        return self.decision

    def after_action(self, action, result, before, after, step=0, origin=None, requester=None, **kw):
        self.origins = getattr(self, "origins", []) + [("after_action", origin, requester)]
        self.after.append(action)


class _Decision(object):
    def __init__(self, t):
        self.type, self.events, self.raw = t, [], None


def _episode_end(tmp_path, agent, harness, end_reason="final_answer"):
    import canonical_schema
    from harness.action_service import HarnessActionService
    from harness.executor import ActionExecutor
    env, traj, d = _Env(tmp_path), [], _deliverable()
    svc = HarnessActionService(harness, ActionExecutor(canonical_schema, "fhir", lambda e: None, lambda e: None), env,
                               {}, traj, 100, agent_actions=lambda: 100, reserve=lambda: 0, errors=[], workspace=None)
    forced = lambda fb: MH.forced_deliverable_write(fb, agent, env, GOAL, traj, svc, 100, end_reason)
    d.after_episode(_ctx(env, traj, forced=forced, step=100))
    return env, traj


def test_the_forced_write_runs_through_the_harness(tmp_path):
    agent = _Agent({"type": "tool_call", "tool": "write_file", "args": {"path": "output/note.md", "content": "note"}})
    h = _Harness()
    env, traj = _episode_end(tmp_path, agent, h)
    assert "output/note.md" in agent.states[0]["last_observation"]
    assert env.calls == [("write_file", {"path": "output/note.md", "content": "note"})]
    ev = [e for e in traj if e.get("event_type") == "tool_call"][0]
    assert ev["forced_deliverable"] is True and ev["via_pipeline"] is True and ev["step"] == 100
    assert h.after and not any(e.get("event_type") == "deliverable_missing" for e in traj)


def test_a_forced_write_the_harness_blocks_does_not_run_and_is_reported_missing(tmp_path):
    agent = _Agent({"type": "tool_call", "tool": "write_file", "args": {"path": "output/note.md", "content": "note"}})
    env, traj = _episode_end(tmp_path, agent, _Harness(_Decision("BLOCK")))
    assert env.calls == []
    assert [e["event_type"] for e in traj] == ["harness_action_refused", "forced_deliverable_blocked",
                                               "deliverable_missing"]


def test_an_agent_error_in_the_forced_turn_is_recorded(tmp_path):
    env, traj = _episode_end(tmp_path, _Agent(RuntimeError("boom")), _Harness())
    assert [e["event_type"] for e in traj] == ["agent_error", "deliverable_missing"]
    assert traj[0]["error"] == "forced_deliverable_failed"


def test_a_non_write_answer_in_the_forced_turn_writes_nothing(tmp_path):
    env, traj = _episode_end(tmp_path, _Agent({"type": "final", "answer": "done"}), _Harness())
    assert env.calls == [] and [e["event_type"] for e in traj] == ["deliverable_missing"]
