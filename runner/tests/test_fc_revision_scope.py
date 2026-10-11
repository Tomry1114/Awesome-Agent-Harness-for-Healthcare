"""Draining a model turn's parallel tool calls one per step, and how a harness revision in the middle of that
batch is scoped. Every function-calling workspace agent program (the workspaces in workspace.H0: PhysicianBench,
HealthAdminBench, MedCTA) carries it; each is checked the same way."""
import os
import sys

import pytest

_RUNNER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RUNNER not in sys.path:
    sys.path.insert(0, _RUNNER)

from workspace import H0, Workspace                     # noqa: E402

TOOLS = ["metanephrines", "DST", "aldosterone", "CT", "orders", "x", "y", "z"]
ENV_TYPE = {"pb_h0_program": "fhir", "hab_h0_program": "gui", "medcta_h0_program": "tool_sandbox"}


class _Driver(object):
    """One interface over the programs: act(state), set the model's next reply, read the messages, read the queue,
    classify a feedback."""

    def __init__(self, kind):
        self.kind = kind
        ws = Workspace(H0[ENV_TYPE[kind]])
        ws.tool_description = lambda name: {"name": name, "description": name, "parameters": {"type": "object"}}
        cls = ws.program_class()
        self._reply = None
        self.obj = cls({"goal": "q", "context": {}, "available_tools": [{"name": t} for t in TOOLS]}, ws,
                       lambda messages, tools: self._reply)

    def reply(self, content, names=()):
        tcs = [{"id": "c%d" % i, "type": "function", "function": {"name": n, "arguments": "{}"}}
               for i, n in enumerate(names)]
        self._reply = {"content": content, "tool_calls": tcs}

    def act(self, state):
        return self.obj.act(state)

    @property
    def messages(self):
        return self.obj.messages

    @property
    def queue(self):
        return self.obj._queue

    def scope(self, fb):
        return self.obj.revision_scope(fb)


KINDS = sorted(ENV_TYPE)


@pytest.mark.parametrize("kind", KINDS)
def test_call_local_keeps_draining(kind):
    d = _Driver(kind)
    d.reply("", ["metanephrines", "DST", "aldosterone", "CT"])
    act0 = d.act({})
    assert act0["tool"] == "metanephrines" and len(d.queue) == 3
    fb = {"decision": "REVISE", "stage": "before_action", "reason": "resourceType must be Observation"}
    act1 = d.act({"harness_feedback": fb, "last_result": {"output": "STALE"}})
    assert act1["tool"] == "DST"                        # the batch kept draining, the model was not re-called
    assert len(d.queue) == 2
    c0 = [m for m in d.messages if m.get("tool_call_id") == "c0"][0]
    assert "this call only" in c0["content"] and "STALE" not in c0["content"]
    assert not any(m["role"] == "user" and "HARNESS" in m.get("content", "") for m in d.messages)
    d.act({"last_result": {"output": "obsD"}})
    d.act({"last_result": {"output": "obsA"}})
    d.reply("done")
    d.act({"last_result": {"output": "obsCT"}})
    ids = [m["tool_call_id"] for m in d.messages if m["role"] == "tool"]
    assert sorted(ids) == ["c0", "c1", "c2", "c3"]      # every id closed exactly once


@pytest.mark.parametrize("kind", KINDS)
def test_batch_global_flushes(kind):
    d = _Driver(kind)
    d.reply("", ["orders", "x", "y", "z"])
    d.act({})
    d.reply("re")
    d.act({"harness_feedback": {"decision": "REVISE", "stage": "grounding", "reason": "wrong subject"},
           "last_result": {"output": "o"}})
    assert d.queue == []
    assert any("[skipped: superseded" in m.get("content", "") for m in d.messages if m["role"] == "tool")
    assert any(m["role"] == "user" and "HARNESS" in m.get("content", "") for m in d.messages)


@pytest.mark.parametrize("kind", KINDS)
def test_classifier(kind):
    d = _Driver(kind)
    assert d.scope({"stage": "before_action", "reason": "bad param"}) == "CALL_LOCAL"
    assert d.scope({"stage": "before_action", "missing_obligations": ["x"]}) == "BATCH_GLOBAL"
    assert d.scope({"stage": "runtime_budget"}) == "TERMINAL"
    assert d.scope({"decision": "ACQUIRE", "stage": "before_action"}) == "BATCH_GLOBAL"
    assert d.scope({"stage": "write_verify"}) == "CALL_LOCAL"
    assert d.scope({"stage": "grounding"}) == "BATCH_GLOBAL"
