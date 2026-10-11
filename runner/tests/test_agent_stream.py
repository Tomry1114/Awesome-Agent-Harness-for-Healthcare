"""MH_AGENT_STREAM=1 streams the agent's turn and assembles the message from the chunks; unset, the request is the
plain non-streamed one."""
import io, json

from runner import transport


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _sse(*chunks):
    return b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks) + b"data: [DONE]\n\n"


def _rig(monkeypatch, payload):
    sent = []

    def fake_urlopen(req, timeout=None):
        sent.append(json.loads(req.data.decode()))
        return _Resp(payload)
    monkeypatch.setattr(transport.urllib.request, "urlopen", fake_urlopen)
    return sent


def _call():
    return transport.chat_fc([{"role": "user", "content": "q"}], [], "gpt-5.6-sol", "http://relay", "k", "high")


def test_streamed_turn_is_assembled(monkeypatch):
    monkeypatch.setenv("MH_AGENT_STREAM", "1")
    sent = _rig(monkeypatch, _sse(
        {"choices": [{"delta": {"content": "thinking "}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "fhir_search", "arguments": "{\"a\""}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ": 1}"}}]}}]}))
    msg = _call()
    assert sent[0]["stream"] is True
    assert msg["content"] == "thinking "
    assert msg["tool_calls"][0]["function"] == {"name": "fhir_search", "arguments": "{\"a\": 1}"}


def test_unset_sends_the_plain_request(monkeypatch):
    monkeypatch.delenv("MH_AGENT_STREAM", raising=False)
    sent = _rig(monkeypatch, json.dumps({"choices": [{"message": {"role": "assistant", "content": "done"}}]}).encode())
    assert _call().get("content") == "done" and "stream" not in sent[0]
