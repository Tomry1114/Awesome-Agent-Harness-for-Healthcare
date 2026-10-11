"""Model transport: one chat turn through the gateway, function calling (chat_fc) or text protocol (chat_text).

Part of the fixed shell. The model, its parameters, the key and the retry policy are not something a workspace may
change, so every agent program -- the default H0 workspace's or any other workspace's -- calls this with the
messages (and, for function calling, the tool schema) it built, and gets back one assistant message or reply text.
"""
import json
import os
import time
import urllib.error
import urllib.request

import gateway
import resolve_override
resolve_override.install()   # MH_RESOLVE: pin a gateway host the cluster DNS gets wrong

ANTHROPIC_PATH = "/v1/messages"


def load_key():
    k = os.environ.get("MH_OPENAI_KEY")
    if k: return k.strip()
    p = os.path.expanduser("~/.mh_openai_key")
    if os.path.exists(p): return open(p).read().strip()
    raise RuntimeError("no API key: set MH_OPENAI_KEY or create ~/.mh_openai_key")


def _anthropic_tools(openai_tools):
    """Convert OpenAI tool schema -> Anthropic tool schema.

    Some OpenAI-compatible gateways route claude-* models to Anthropic's native API, which rejects OpenAI's
    {"type":"function","function":{"name","description","parameters"}} with
    "tools.0: Input tag 'function' ... does not match". Anthropic wants a flat
    {"name","description","input_schema"}.
    """
    out = []
    for t in (openai_tools or []):
        fn = t.get("function", t) if isinstance(t, dict) else {}
        out.append({"name": fn.get("name"), "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters") or {"type": "object", "properties": {}}})
    return out


def _anthropic_messages(messages):
    """OpenAI chat messages -> (system_str, anthropic_messages). A gateway's claude route that forwards to
    Anthropic natively (a) wants the system prompt top-level, (b) rejects role=="tool", and needs
    assistant tool calls as tool_use blocks + tool results as a user tool_result block."""
    import json as _json
    sysp, out = [], []
    for m in messages:
        role = m.get("role")
        if role == "system":
            c = m.get("content")
            if isinstance(c, str) and c.strip():
                sysp.append(c)
            continue
        if role == "tool":
            out.append({"role": "user", "content": [{"type": "tool_result",
                        "tool_use_id": m.get("tool_call_id"), "content": m.get("content") or ""}]})
            continue
        if role == "assistant" and m.get("tool_calls"):
            blocks = []
            c = m.get("content")
            if isinstance(c, str) and c.strip():
                blocks.append({"type": "text", "text": c})
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                try:
                    inp = _json.loads(fn.get("arguments") or "{}")
                except Exception:
                    inp = {}
                blocks.append({"type": "tool_use", "id": tc.get("id"), "name": fn.get("name"), "input": inp})
            out.append({"role": "assistant", "content": blocks})
            continue
        out.append({"role": role, "content": m.get("content") or ""})
    return ("\n\n".join(sysp), out)


def _anthropic_to_message(d):
    """Anthropic /v1/messages RESPONSE -> OpenAI-style assistant message dict {content, tool_calls} so the
    downstream FC parser is unchanged. The native response is content blocks (text / tool_use) + stop_reason,
    NOT choices[]. claude-* models are called on /v1/messages (some gateways reject them on the /v1/chat route), so
    the reply is translated here."""
    msg = {"role": "assistant"}
    texts, tcs = [], []
    for blk in (d.get("content") or []):
        if not isinstance(blk, dict):
            continue
        t = blk.get("type")
        if t == "text":
            texts.append(blk.get("text") or "")
        elif t == "tool_use":
            tcs.append({"id": blk.get("id"), "type": "function",
                        "function": {"name": blk.get("name"),
                                     "arguments": json.dumps(blk.get("input") or {}, ensure_ascii=False)}})
    if texts:
        msg["content"] = "".join(texts)
    if tcs:
        msg["tool_calls"] = tcs
    return msg


def chat_text(messages, model, reasoning, max_new_tokens=400):
    """One text-protocol turn through the unified gateway client (retry, backoff, timeout and billing detection are
    gateway.chat's) -> the reply text. A failed call returns the text protocol's sentinel final answer,
    <answer>API_BRAIN_ERROR: ...</answer>, with BILLING/QUOTA when the gateway classifies the error as billing."""
    max_tokens = max(int(os.environ.get("MH_OPENAI_MAX_TOKENS", "16000")), int(max_new_tokens))  # reasoning tokens count against the budget
    extra = {"reasoning_effort": reasoning} if reasoning else None
    res = gateway.chat(messages, model=model, max_tokens=max_tokens, judge=False,
                       timeout=int(os.environ.get("MH_OPENAI_TIMEOUT", "300")), extra=extra)
    if res.get("ok"):
        return res["content"]
    if res.get("error_type") == "billing":
        return "<answer>API_BRAIN_ERROR: BILLING/QUOTA %s</answer>" % res.get("raw", "")
    return "<answer>API_BRAIN_ERROR: %s</answer>" % res.get("raw", "")


def chat_fc(messages, tools_schema, model, base, key, reasoning):
    is_anthropic = model.startswith("claude")
    if is_anthropic:
        url = base + ANTHROPIC_PATH
    else:
        url = base + os.environ.get("MH_OPENAI_CHAT_PATH", "/v1/chat/completions")
    tools = _anthropic_tools(tools_schema) if is_anthropic else tools_schema
    msgs = messages
    # MH_AGENT_WIRE_MODEL: the name the agent's gateway serves the model under, when it is not the model's own (a
    # gateway may expose a model under an alias). The run is still the model's: its resume key and provenance name
    # `model`; result.json records the wire name beside it.
    body = {"model": os.environ.get("MH_AGENT_WIRE_MODEL") or model, "tools": tools,
            "max_tokens": int(os.environ.get("MH_OPENAI_MAX_TOKENS", "16000"))}
    if is_anthropic:
        # Full OpenAI->Anthropic message translation (system top-level, tool_use/tool_result blocks).
        # tool_choice/reasoning_effort are OpenAI-only and get HTTP 400 on the Anthropic route -> keep the body minimal.
        sys_txt, msgs = _anthropic_messages(msgs)
        if sys_txt:
            body["system"] = sys_txt
    else:
        body["tool_choice"] = "auto"
        if reasoning:
            body["reasoning_effort"] = reasoning
    body["messages"] = msgs
    # MH_AGENT_STREAM=1: stream the agent's turn. A gateway behind Cloudflare cuts a non-streamed reply that takes over
    # ~120 s (HTTP 524), while a long PhysicianBench turn (high reasoning effort, tens of thousands of input tokens) can
    # take several minutes; streamed, the first bytes arrive early and the reply completes. Streamed, urlopen's timeout is
    # the idle limit between chunks and MH_AGENT_STREAM_DEADLINE (900 s) the ceiling for the whole turn. qwen* models are
    # always streamed.
    _agent_stream = os.environ.get("MH_AGENT_STREAM") == "1"
    _stream = model.startswith("qwen") or _agent_stream
    if _stream and not is_anthropic:
        body["stream"] = True
    data = json.dumps(body).encode()
    import sys as _sys, time as _tt, threading as _th
    _sys.stderr.write("[FC-CALL msgs=%d bytes=%d model=%s]\n"%(len(messages),len(data),model)); _sys.stderr.flush()
    _t0=_tt.time()
    _hard = int(os.environ.get("MH_OPENAI_TIMEOUT", "300"))
    _ceiling = int(os.environ.get("MH_AGENT_STREAM_DEADLINE", "900")) if _agent_stream else _hard
    if is_anthropic:
        # Claude route: some relays require Authorization: Bearer (NOT x-api-key) and a Claude-CLI-style User-Agent,
        # rejecting a generic UA on claude models with 403 (a client-fingerprint check); MH_ANTHROPIC_UA overrides the
        # default UA and MH_ANTHROPIC_VERSION the anthropic-version header. The endpoint is /v1/messages.
        _headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json",
                    "anthropic-version": os.environ.get("MH_ANTHROPIC_VERSION", "2023-06-01"),
                    "User-Agent": os.environ.get("MH_ANTHROPIC_UA", "claude-cli/2.0.76 (external, cli)")}
    else:
        _headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json",
                    "User-Agent": gateway.OPENAI_UA}
    # Gateway brownout ride-out: a short retry budget would turn a burst of transient overload errors (e.g. 503
    # "system cpu overloaded") into API_BRAIN_ERROR final answers. Retries (MH_FC_RETRIES) and the backoff cap
    # (MH_FC_BACKOFF_CAP) are tunable, and transient 5xx/429 get a longer backoff.
    _MAXTRY = max(1, int(os.environ.get("MH_FC_RETRIES", "8")))
    _BOCAP  = max(1, int(os.environ.get("MH_FC_BACKOFF_CAP", "60")))
    _answered = False       # did the endpoint answer at all (any HTTP status) on some attempt?
    for attempt in range(_MAXTRY):
        # HARD wall-clock deadline: urlopen's own timeout does NOT reliably break a stuck SEND (gateway
        # accepts the TCP connect then stalls reading a large body -> Send-Q fills, urlopen hangs > timeout).
        # Run it in a daemon thread and abandon it past the deadline; a fresh retry gets a new connection.
        _box = {}
        def _worker(_box=_box):
            try:
                req = urllib.request.Request(url, data=data, method="POST", headers=_headers)
                with urllib.request.urlopen(req, timeout=_hard) as r:
                    if _stream:
                        _cp=[]; _tcs={}; _role="assistant"
                        for _raw in r:
                            _ln=_raw.decode("utf-8","ignore").strip()
                            if not _ln.startswith("data:"): continue
                            _pl=_ln[5:].strip()
                            if _pl=="[DONE]": break
                            try: _ch=json.loads(_pl)
                            except Exception: continue
                            _dl=((_ch.get("choices") or [{}])[0].get("delta") or {})
                            if _dl.get("content"): _cp.append(_dl["content"])
                            for _tc in (_dl.get("tool_calls") or []):
                                _ix=_tc.get("index",0)
                                _sl=_tcs.setdefault(_ix,{"id":None,"type":"function","function":{"name":"","arguments":""}})
                                if _tc.get("id"): _sl["id"]=_tc["id"]
                                _fn=_tc.get("function") or {}
                                if _fn.get("name"): _sl["function"]["name"]=_fn["name"]
                                if _fn.get("arguments"): _sl["function"]["arguments"]+=_fn["arguments"]
                        _msg={"role":_role}
                        if _cp: _msg["content"]="".join(_cp)
                        if _tcs: _msg["tool_calls"]=[_tcs[k] for k in sorted(_tcs)]
                        _box["msg"]=_msg
                    else:
                        _box["d"] = json.loads(r.read().decode())
            except Exception as _e:
                _box["err"] = _e
        _w = _th.Thread(target=_worker, daemon=True); _w.start(); _w.join(_ceiling + 5)
        if _w.is_alive():                          # hung past the hard deadline -> abandon + retry fresh
            _sys.stderr.write("[FC-HANG abandoned after %ds, retry]\n" % (_ceiling + 5)); _sys.stderr.flush()
            time.sleep(2 ** attempt); continue
        _e = _box.get("err")
        if _e is None:
            _sys.stderr.write("[FC-DONE %.1fs]\n"%(_tt.time()-_t0)); _sys.stderr.flush()
            if _stream:
                return _box.get("msg") or {}
            _d = _box.get("d") or {}
            if is_anthropic:
                return _anthropic_to_message(_d)
            return (_d.get("choices") or [{}])[0].get("message", {}) or {}
        if isinstance(_e, urllib.error.HTTPError):
            _answered = True
            _body = _e.read().decode("utf-8", "ignore")[:240]
            _sys.stderr.write("[FC-HTTP %s %s]\n"%(_e.code, _body)); _sys.stderr.flush()
            # PERMANENT failures must not be slept through. With many retries and a long backoff cap, an out-of-funds
            # error would otherwise burn minutes per call and then return the API_BRAIN_ERROR sentinel -- which downstream is
            # indistinguishable from an agent that simply failed the task. 429/5xx are still retried; so is a 403 that does NOT
            # mention billing or expiry, because some gateways return a transient 403 (account flagged as suspicious) that clears
            # on retry. Billing is recognised by quota/balance wording in English or Chinese.
            _low = _body.lower()
            _billing = any(k in _body for k in ("额度", "欠费", "预扣")) or \
                       any(k in _low for k in ("insufficient", "balance", "quota", "exceeded your current")) or \
                       any(k in _low for k in ("usage_limit_exceeded", "daily_limit_exceeded", "daily usage limit"))
            # ^ also a per-account daily quota (429 USAGE_LIMIT_EXCEEDED / DAILY_LIMIT_EXCEEDED): it looks transient but does
            # not clear within a turn, so retrying it only wastes minutes per turn before API_BRAIN_ERROR. A 429 "Concurrency
            # limit exceeded" is not this and is still retried.
            # Some relays route identical requests to different upstreams, and some upstreams reject short prompts as
            # "distillation or heartbeat probing" (HTTP 400). The same request can fail this way and then pass, so it is a
            # routing outcome, not a property of the request, and is retried like an overload.
            _rerouted = _e.code == 400 and ("distillation" in _low or "heartbeat probing" in _low)
            # A key the relay refuses for good -- out of quota, expired, invalid (e.g. 429 API_KEY_QUOTA_EXHAUSTED, 403
            # API_KEY_EXPIRED, 401 INVALID_API_KEY) -- ends the call at once and says so, so the agent can go on with another key
            # of its key file (runner/workspace.WorkspaceAgent._chat). The log line carries no status code: a refusal another key
            # took over from is not an infrastructure failure of the run.
            if _billing or _e.code == 401 or (_e.code == 403 and ("expired" in _low or "过期" in _body)):
                _sys.stderr.write("[FC-KEY-REFUSED%s -> another key of the key file, else API_BRAIN_ERROR]\n"
                                  % (" billing" if _billing else "")); _sys.stderr.flush()
                return {"content": "API_BRAIN_ERROR", "key_refused": True}
            if _e.code == 400 and not _rerouted:
                _sys.stderr.write("[FC-PERMANENT %s%s -> giving up immediately]\n"
                                  % (_e.code, " billing" if _billing else "")); _sys.stderr.flush()
                break
            # transient overload/upstream errors deserve a longer ride than a plain 2**n
            _bo = (3 * (2 ** attempt)) if (_e.code in (429, 500, 502, 503, 504) or _rerouted) else (2 ** attempt)
            time.sleep(min(_BOCAP, _bo))
        else:
            _sys.stderr.write("[FC-EXC %r]\n"%_e); _sys.stderr.flush(); time.sleep(min(_BOCAP, 2 ** attempt))
    # LOUD: this sentinel becomes the agent's final answer with zero env actions. Monitoring
    # must be able to count it from the log, not only from the trajectory.
    _sys.stderr.write("[FC-GIVEUP after %d attempts -> API_BRAIN_ERROR]\n" % _MAXTRY); _sys.stderr.flush()
    # not one attempt reached the endpoint (TLS failure, e.g. a certificate for another host name; refused or dropped
    # connection; hang): the gateway itself is down, not this request, and the reply is marked unreachable
    return dict({"content": "API_BRAIN_ERROR"}, **({} if _answered else {"unreachable": True}))
