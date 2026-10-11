"""A key the relay refuses for good (quota exhausted, expired, invalid) is dropped and the same turn goes out on
another key of the key file, so one dead key does not end the task; a transient failure (overload) is retried and is
not a refusal; a turn given up on one endpoint moves to the next; the audit does not read the refusal line as an
infra failure."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import transport as T  # noqa: E402
import workspace as W  # noqa: E402


class _Agent(W.WorkspaceAgent):
    def __init__(self, keys, key):                       # skip the workspace build: only _chat is under test
        self.model, self.base, self.reasoning = "MiniMax-M3", "https://relay.example", None
        self._key, self._spare_keys = key, [k for k in keys if k != key]
        self._spare_bases = []


def test_refused_key_switches_to_another(monkeypatch):
    seen = []

    def fake(messages, tools, model, base, key, reasoning):
        seen.append(key)
        if key == "dead":
            return {"content": "API_BRAIN_ERROR", "key_refused": True}
        return {"role": "assistant", "content": "ok"}
    monkeypatch.setattr(T, "chat_fc", fake)
    a = _Agent(["dead", "live"], "dead")
    assert a._chat([], []) == {"role": "assistant", "content": "ok"}
    assert seen == ["dead", "live"] and a._key == "live"
    a._chat([], [])
    assert seen[-1] == "live"                            # the dead key is not tried again in this task


def test_all_keys_refused_is_the_sentinel(monkeypatch):
    monkeypatch.setattr(T, "chat_fc", lambda *a: {"content": "API_BRAIN_ERROR", "key_refused": True})
    a = _Agent(["k1", "k2"], "k1")
    assert a._chat([], []) == {"content": "API_BRAIN_ERROR"}


def test_agent_keys(tmp_path, monkeypatch):
    kf = tmp_path / "keys"
    kf.write_text("a\n\nb\n")
    monkeypatch.setenv("MH_AGENT_KEY_FILE", str(kf))
    assert W.agent_keys() == ["a", "b"]
    monkeypatch.delenv("MH_AGENT_KEY_FILE")
    assert W.agent_keys() == []


class _Resp(Exception):
    pass


def _http(code, body):
    import io
    import urllib.error
    return urllib.error.HTTPError("u", code, "x", {}, io.BytesIO(body.encode()))


def _run_chat_fc(monkeypatch, err):
    import urllib.request
    calls = []

    def boom(req, timeout=None):
        calls.append(1)
        raise err()
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    monkeypatch.setattr(T.time, "sleep", lambda s: None)
    monkeypatch.setenv("MH_FC_RETRIES", "3")
    return T.chat_fc([{"role": "user", "content": "hi"}], [], "MiniMax-M3", "https://relay.example", "k", None), calls


def test_quota_exhausted_and_expired_are_refusals(monkeypatch):
    for code, body in ((429, '{"code":"API_KEY_QUOTA_EXHAUSTED","message":"API key \u989d\u5ea6\u5df2\u7528\u5b8c"}'),
                       (403, '{"code":"API_KEY_EXPIRED","message":"API key expired"}'),
                       (401, '{"code":"INVALID_API_KEY","message":"Invalid API key"}')):
        msg, calls = _run_chat_fc(monkeypatch, lambda: _http(code, body))
        assert msg.get("key_refused") is True and len(calls) == 1, (code, msg, calls)


def test_overload_is_not_a_refusal(monkeypatch):
    msg, calls = _run_chat_fc(monkeypatch, lambda: _http(503, "system cpu overloaded"))
    assert not msg.get("key_refused") and len(calls) == 3


def test_a_turn_given_up_on_one_endpoint_goes_to_the_next(monkeypatch):
    seen = []

    def fake(messages, tools, model, base, key, reasoning):
        seen.append(base)
        return {"content": "API_BRAIN_ERROR"} if base == "https://a" else {"role": "assistant", "content": "ok"}
    monkeypatch.setattr(T, "chat_fc", fake)
    a = _Agent(["k"], "k")
    a.base, a._spare_bases = "https://a", ["https://b"]
    assert a._chat([], []) == {"role": "assistant", "content": "ok"} and seen == ["https://a", "https://b"]
    a._chat([], [])
    assert seen[-1] == "https://b"                       # stays on the endpoint that answered


def test_a_refused_key_does_not_move_the_endpoint(monkeypatch):
    seen = []

    def fake(messages, tools, model, base, key, reasoning):
        seen.append(base)
        return {"content": "API_BRAIN_ERROR", "key_refused": True}
    monkeypatch.setattr(T, "chat_fc", fake)
    a = _Agent(["k"], "k")
    a.base, a._spare_bases = "https://a", ["https://b"]
    assert a._chat([], []) == {"content": "API_BRAIN_ERROR"} and seen == ["https://a"]


def test_agent_bases_are_split_and_normalised(monkeypatch):
    monkeypatch.setenv("MH_AGENT_BASE", "https://ai.x/v1, https://llm.x/ ,https://ai.x")
    assert W.agent_bases() == ["https://ai.x", "https://llm.x"]
