# -*- coding: utf-8 -*-
"""A chat-completions call carried over the gateway's Responses API (/v1/responses).

Some OpenAI-compatible gateways serve certain models only on /v1/responses: a /v1/chat/completions call is refused
with HTTP 400 protocol_not_supported, or times out at the gateway with a 524 even for a trivial prompt, while the same
model answers on /v1/responses. The callers keep speaking chat completions -- messages, tools,
max_tokens, reasoning_effort in; {"choices": [{"message", "finish_reason"}], "usage", "model"} out -- and this module
translates both ways. The request is streamed: a long answer keeps the connection alive past a gateway's wait
(~120 s) for a reply that has not started, which otherwise ends in a 524.

    post(base, key, ua, chat_body, timeout) -> chat-completions-shaped dict; raises urllib.error.HTTPError as urlopen
    does, TimeoutError when the answer is still streaming at the deadline, RuntimeError on a failed response.
"""
import json
import os
import threading
import time
import urllib.error
import urllib.request

# Chat completions first (it is usually the faster path); a model whose chat call is refused as a protocol it does
# not serve, or times out at the gateway (524), goes to /v1/responses for the rest of the process (note_failure).
# MH_RESPONSES_MODELS (comma list) pins models there.
_FALLBACK = set()


def wants_responses(model):
    env = os.environ.get("MH_RESPONSES_MODELS")
    pinned = {m.strip() for m in env.split(",") if m.strip()} if env else set()
    return str(model or "") in pinned or str(model or "") in _FALLBACK


def note_failure(model, code, body):
    """A chat call of `model` failed with HTTP `code` and `body`: True when that means "use /v1/responses" (and from
    now on in this process), False when the failure is anything else."""
    low = (body.decode("utf-8", "replace") if isinstance(body, bytes) else str(body or "")).lower()
    if (code == 400 and ("protocol_not_supported" in low or "不支持 chat" in low)) or code == 524:
        _FALLBACK.add(str(model or ""))
        return True
    return False


def _content_parts(content, role):
    """chat content (a string or a list of text / image_url parts) -> Responses content."""
    if isinstance(content, str) or content is None:
        return content or ""
    out = []
    for p in content:
        if not isinstance(p, dict):
            continue
        if p.get("type") == "text":
            out.append({"type": "output_text" if role == "assistant" else "input_text", "text": p.get("text") or ""})
        elif p.get("type") == "image_url":
            url = (p.get("image_url") or {}).get("url") if isinstance(p.get("image_url"), dict) else p.get("image_url")
            out.append({"type": "input_image", "image_url": url})
    return out


def to_request(chat_body):
    """A chat-completions request body -> a Responses request body."""
    items, instructions = [], []
    for m in chat_body.get("messages") or []:
        role = m.get("role")
        if role == "system" and not items:
            instructions.append(m.get("content") if isinstance(m.get("content"), str) else "")
        elif role == "tool":
            items.append({"type": "function_call_output", "call_id": m.get("tool_call_id"),
                          "output": m.get("content") if isinstance(m.get("content"), str) else json.dumps(m.get("content"))})
        elif role == "assistant":
            if m.get("content"):
                items.append({"role": "assistant", "content": _content_parts(m.get("content"), "assistant")})
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                items.append({"type": "function_call", "call_id": tc.get("id"), "name": fn.get("name"),
                              "arguments": fn.get("arguments") or "{}"})
        else:                                           # user, or a system message after the conversation began
            items.append({"role": "developer" if role == "system" else "user",
                          "content": _content_parts(m.get("content"), role)})
    body = {"model": chat_body.get("model"), "input": items, "stream": True}
    if instructions:
        body["instructions"] = "\n\n".join(i for i in instructions if i)
    mt = chat_body.get("max_completion_tokens") or chat_body.get("max_tokens")
    if mt:
        body["max_output_tokens"] = int(mt)
    if chat_body.get("reasoning_effort"):
        body["reasoning"] = {"effort": chat_body["reasoning_effort"]}
    tools = []
    for t in chat_body.get("tools") or []:
        fn = t.get("function") if isinstance(t, dict) else None
        if fn:
            tools.append({"type": "function", "name": fn.get("name"), "description": fn.get("description") or "",
                          "parameters": fn.get("parameters") or {"type": "object", "properties": {}}})
    if tools:
        body["tools"] = tools
        body["tool_choice"] = chat_body.get("tool_choice") or "auto"
    return body


def from_response(resp):
    """A Responses response object -> a chat-completions response dict."""
    texts, calls, reasoning = [], [], []
    for it in resp.get("output") or []:
        t = it.get("type")
        if t == "message":
            for c in it.get("content") or []:
                if c.get("type") in ("output_text", "text"):
                    texts.append(c.get("text") or "")
        elif t == "function_call":
            calls.append({"id": it.get("call_id") or it.get("id"), "type": "function",
                          "function": {"name": it.get("name"), "arguments": it.get("arguments") or "{}"}})
        elif t == "reasoning":
            for s in it.get("summary") or []:
                if s.get("text"):
                    reasoning.append(s["text"])
    msg = {"role": "assistant", "content": "".join(texts) or None}
    if calls:
        msg["tool_calls"] = calls
    if reasoning:
        msg["reasoning_content"] = "\n".join(reasoning)
    status = resp.get("status")
    finish = "tool_calls" if calls else ("length" if status == "incomplete" else "stop")
    u = resp.get("usage") or {}
    pt, ct = int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0)
    usage = {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": int(u.get("total_tokens") or pt + ct)}
    rt = (u.get("output_tokens_details") or {}).get("reasoning_tokens")
    if rt is not None:
        usage["completion_tokens_details"] = {"reasoning_tokens": rt}
    return {"id": resp.get("id"), "model": resp.get("model"), "usage": usage,
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}]}


def _stream(req, timeout):
    final = None
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        # carry the body in the message and keep it readable: callers tell an out-of-funds 403 (a quota message) from a
        # transient upstream 403 (e.g. "BPS upstream response failed") only by the body
        body = e.read()
        e.read = lambda *a, _b=body: _b                  # still readable by the caller (the bound read is spent)
        e.msg = "%s -- %s" % (e.msg, body.decode("utf-8", "replace")[:300])
        raise
    with r:
        for raw in r:
            ln = raw.decode("utf-8", "ignore").strip()
            if not ln.startswith("data:"):
                continue
            pl = ln[5:].strip()
            if pl == "[DONE]":
                break
            try:
                ev = json.loads(pl)
            except ValueError:
                continue
            et = ev.get("type")
            if et in ("response.completed", "response.incomplete"):
                final = ev.get("response") or final
            elif et in ("response.failed", "error"):
                raise RuntimeError("responses stream failed: %s" % json.dumps(ev)[:300])
    if final is None:
        raise RuntimeError("responses stream ended without a completed response")
    return final


def post(base, key, ua, chat_body, timeout, deadline=1800):
    """One call: `timeout` bounds each socket read, `deadline` the whole streamed answer (a daemon thread is abandoned
    past it)."""
    base = (base or "").rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3].rstrip("/")
    req = urllib.request.Request(base + "/v1/responses", data=json.dumps(to_request(chat_body)).encode("utf-8"),
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                                          "User-Agent": ua, "Accept": "text/event-stream"})
    box = {}

    def _w():
        try:
            box["r"] = _stream(req, timeout)
        except Exception as e:
            box["e"] = e
    th = threading.Thread(target=_w, daemon=True)
    th.start()
    th.join(max(timeout, deadline) + 5)
    if th.is_alive():
        raise TimeoutError("responses call still streaming after %ds" % max(timeout, deadline))
    if "e" in box:
        raise box["e"]
    return from_response(box["r"])
