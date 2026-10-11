"""When MH_AGENT_WIRE_MODEL is set, that gateway-specific name for the agent model is sent as the request's model
field; otherwise the run's model name is sent. The run itself keeps the canonical model name either way."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import transport as T  # noqa: E402


def _sent(monkeypatch):
    import urllib.request
    seen = {}

    class _R(object):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"role": "assistant", "content": "ok"}}]}).encode()

    def fake(req, timeout=None):
        seen["body"] = json.loads(req.data.decode())
        return _R()
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    T.chat_fc([{"role": "user", "content": "hi"}], [], "gpt-5.6-sol", "https://gw.example", "k", None)
    return seen["body"]["model"]


def test_wire_name_when_set(monkeypatch):
    monkeypatch.setenv("MH_AGENT_WIRE_MODEL", "gpt-5.6-sol-plus")
    assert _sent(monkeypatch) == "gpt-5.6-sol-plus"


def test_model_name_otherwise(monkeypatch):
    monkeypatch.delenv("MH_AGENT_WIRE_MODEL", raising=False)
    assert _sent(monkeypatch) == "gpt-5.6-sol"
