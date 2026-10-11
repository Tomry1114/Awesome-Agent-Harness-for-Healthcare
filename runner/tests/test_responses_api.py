"""Chat-completions calls carried over /v1/responses (runner/responses_api.py): the request and the answer translate
both ways (including tool calls, usage and an incomplete answer as finish_reason=length). A model is sent as chat
first and moves to /v1/responses once the gateway refuses the chat protocol (protocol_not_supported) or answers 524;
MH_RESPONSES_MODELS names models that always go there, and the gateway routes them accordingly."""
import io
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import responses_api as RA  # noqa: E402


def test_request_translation():
    body = RA.to_request({"model": "gpt-6-sol", "max_tokens": 900, "reasoning_effort": "high", "temperature": 0.2,
                          "tools": [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}],
                          "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "u"},
                                       {"role": "assistant", "content": None, "tool_calls": [
                                           {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
                                       {"role": "tool", "tool_call_id": "c1", "content": "file text"}]})
    assert body["instructions"] == "sys" and body["max_output_tokens"] == 900 and body["reasoning"] == {"effort": "high"}
    assert body["tools"][0]["name"] == "read_file" and body["stream"] is True
    assert [i.get("type") or i.get("role") for i in body["input"]] == ["user", "function_call", "function_call_output"]
    assert body["input"][2] == {"type": "function_call_output", "call_id": "c1", "output": "file text"}


def test_response_translation():
    d = RA.from_response({"id": "r", "model": "gpt-6-sol", "status": "completed",
                          "usage": {"input_tokens": 70, "output_tokens": 31, "output_tokens_details": {"reasoning_tokens": 10}},
                          "output": [{"type": "reasoning", "summary": []},
                                     {"type": "function_call", "call_id": "c9", "name": "read_file", "arguments": "{\"path\":\"a\"}"}]})
    ch = d["choices"][0]
    assert ch["finish_reason"] == "tool_calls" and ch["message"]["tool_calls"][0]["id"] == "c9"
    assert d["usage"]["prompt_tokens"] == 70 and d["usage"]["total_tokens"] == 101
    t = RA.from_response({"status": "incomplete", "output": [{"type": "message", "content": [{"type": "output_text", "text": "hi"}]}]})
    assert t["choices"][0]["message"]["content"] == "hi" and t["choices"][0]["finish_reason"] == "length"


class _SSE(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_post_reads_the_stream(monkeypatch):
    done = {"type": "response.completed", "response": {"status": "completed", "output": [
        {"type": "message", "content": [{"type": "output_text", "text": "PASS"}]}], "usage": {"input_tokens": 3, "output_tokens": 1}}}
    sse = ("event: response.created\ndata: {\"type\": \"response.created\"}\n\n"
           "event: response.completed\ndata: %s\n\n" % json.dumps(done)).encode()
    seen = {}

    def fake(req, timeout=None):
        seen["url"] = req.full_url
        return _SSE(sse)
    monkeypatch.setattr(RA.urllib.request, "urlopen", fake)
    d = RA.post("https://gw/v1", "k", "ua", {"model": "gpt-5.6-sol", "messages": [{"role": "user", "content": "q"}]}, 30)
    assert seen["url"] == "https://gw/v1/responses" and d["choices"][0]["message"]["content"] == "PASS"


def test_chat_first_and_a_refused_protocol_or_524_moves_the_model(monkeypatch):
    monkeypatch.delenv("MH_RESPONSES_MODELS", raising=False)
    monkeypatch.setattr(RA, "_FALLBACK", set())
    assert not RA.wants_responses("gpt-6-sol")
    assert not RA.note_failure("gpt-6-sol", 429, "busy") and not RA.wants_responses("gpt-6-sol")
    assert RA.note_failure("gpt-6-sol", 400, '{"code":"protocol_not_supported"}') and RA.wants_responses("gpt-6-sol")
    assert RA.note_failure("gpt-5.6-sol", 524, "") and RA.wants_responses("gpt-5.6-sol")
    monkeypatch.setenv("MH_RESPONSES_MODELS", "MiniMax-M3")
    assert RA.wants_responses("MiniMax-M3")


def test_gateway_routes_a_responses_model(monkeypatch):
    import gateway as G
    calls = []
    monkeypatch.setattr(RA, "post", lambda base, key, ua, body, t, deadline=None: calls.append((base, body["model"])) or
                        {"choices": [{"message": {"content": "PASS"}}]})
    monkeypatch.setenv("MH_JUDGE_KEY", "k")
    monkeypatch.setenv("MH_RESPONSES_MODELS", "gpt-5.6-sol")
    r = G.chat([{"role": "user", "content": "q"}], model="gpt-5.6-sol", judge=True, base="https://gw")
    assert r["ok"] and r["content"] == "PASS" and calls == [("https://gw", "gpt-5.6-sol")]
