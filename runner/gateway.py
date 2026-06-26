"""Unified gateway HTTP client (Codex #2). ONE place for: key loading, base/UA, retry+backoff,
timeout, billing/quota detection, multimodal image_url, and structured error types -- so every agent,
judge, and VLM tool gets identical reliability/cost/failure semantics instead of 7 divergent copies
(retry 4 vs 5 vs 1, timeout 120/200/300). Migrate callers to gateway.chat() incrementally.

Env: MH_OPENAI_BASE (default micuapi) / MH_JUDGE_BASE (judges may differ) / MH_OPENAI_KEY|~/.xbai_key
     / MH_OPENAI_UA / MH_GATEWAY_TIMEOUT (default 300, also the HARD per-call deadline) / MH_GATEWAY_RETRIES (default 4)

ROBUSTNESS (anti-hang): a single chat() can NEVER block longer than MH_GATEWAY_TIMEOUT, no matter
how many retries -- there is a TOTAL monotonic deadline shared across all attempts + backoff, and
each attempt's socket timeout is clamped to the budget left. A half-open / black-hole socket (TCP
connect succeeds but the server never sends bytes) is the classic urlopen hang; we defend with
(1) a process-wide socket.setdefaulttimeout floor so even a stray raw urlopen elsewhere cannot wait
forever, and (2) the per-call + total deadline here. Timeouts / connection errors are retried with
small bounded backoff, then returned as a STRUCTURED {ok:False, error_type:"timeout"} so the caller
records an environment failure and MOVES ON instead of hanging the whole batch.
"""
import os, json, time, base64, socket, urllib.request, urllib.error

_BILLING = ("额度", "欠费", "预扣费")
_BILLING_EN = ("insufficient", "balance", "quota", "exceeded your current")
_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp"}

# Process-wide floor: any socket that never sets its own timeout (e.g. a stray raw urllib call in
# another module) inherits this instead of blocking forever. We only RAISE the floor toward our
# budget; we never shorten a stricter timeout a caller may have set. Applied at import so it is in
# force before any agent/judge makes a request.
try:
    _floor = float(os.environ.get("MH_SOCKET_DEFAULT_TIMEOUT", os.environ.get("MH_GATEWAY_TIMEOUT", "300")))
    _cur = socket.getdefaulttimeout()
    if _cur is None or _cur > _floor:
        socket.setdefaulttimeout(_floor)
except Exception:
    pass

# Errors that mean "the connection was slow/dead, not that the request was bad" -> safe to retry,
# and -> classified as a timeout-class environment failure once the budget is spent.
_TIMEOUT_EXC = (socket.timeout, TimeoutError)
_CONN_HINTS = ("timed out", "timeout", "connection refused", "connection reset", "reset by peer",
               "broken pipe", "not connect", "name or service not known", "temporary failure",
               "no route to host", "network is unreachable", "connection aborted", "errno 11001")


def _is_timeout_like(ex):
    if isinstance(ex, _TIMEOUT_EXC):
        return True
    # URLError wraps the real OSError in .reason; check both the wrapper and the cause text.
    reason = getattr(ex, "reason", None)
    if isinstance(reason, _TIMEOUT_EXC):
        return True
    s = (repr(ex) + " " + repr(reason)).lower()
    return any(h in s for h in _CONN_HINTS)


def load_key(judge=False, override=None):
    """Resolve the API key for a call. Precedence:
      explicit per-call override  >  MH_JUDGE_KEY (judge=True only)  >  MH_OPENAI_KEY  >  OPENAI_API_KEY
      >  ~/.xbai_key.
    This lets the agent brain, the VLM tool backend, and the judge use DIFFERENT keys within ONE run --
    e.g. a gemini/deepseek-only key for the agent (OPENAI_API_KEY) while the VLM (MH_VLM_API_KEY, passed as
    override) and the judge (MH_JUDGE_KEY) run gpt-5.x on a separate gpt-capable key."""
    if override:
        return override.strip()
    if judge:
        jk = os.environ.get("MH_JUDGE_KEY")
        if jk:
            return jk.strip()
    k = os.environ.get("MH_OPENAI_KEY") or os.environ.get("OPENAI_API_KEY")
    if k:
        return k.strip()
    p = os.path.expanduser("~/.xbai_key")
    return open(p).read().strip() if os.path.exists(p) else ""


def base_url(judge=False):
    b = (os.environ.get("MH_JUDGE_BASE") if judge else None) or os.environ.get("MH_OPENAI_BASE", "https://www.micuapi.ai")
    b = b.rstrip("/")
    return b[:-3].rstrip("/") if b.endswith("/v1") else b


def image_data_url(image_path):
    ext = os.path.splitext(image_path)[1].lower()
    with open(image_path, "rb") as f:
        return "data:%s;base64,%s" % (_MIME.get(ext, "image/jpeg"), base64.b64encode(f.read()).decode())


def chat(messages, model, max_tokens=1024, judge=False, timeout=None, retries=None, image_path=None, extra=None, key=None):
    """Return {"ok": bool, "content": str|None, "error_type": str|None, "raw": str}.
    error_type in {None, "billing", "auth", "http_4xx", "http_5xx", "empty", "timeout", "exception"} --
    structured, never a substring guess. On image_path, the LAST user message is made multimodal.

    HARD GUARANTEE: this call returns within ~`timeout` seconds (MH_GATEWAY_TIMEOUT) regardless of
    `retries` -- a shared monotonic deadline caps the total wall time, including backoff sleeps, so a
    stuck/half-open connection can never block a run indefinitely."""
    timeout = int(os.environ.get("MH_GATEWAY_TIMEOUT", "300")) if timeout is None else int(timeout)
    timeout = max(1, timeout)
    retries = int(os.environ.get("MH_GATEWAY_RETRIES", "4")) if retries is None else retries
    deadline = time.monotonic() + timeout  # TOTAL budget across every attempt + backoff
    msgs = [dict(m) for m in messages]
    if image_path and os.path.exists(image_path):
        for m in reversed(msgs):
            if m.get("role") == "user":
                c = m.get("content")
                txt = c if isinstance(c, str) else ""
                m["content"] = [{"type": "text", "text": txt},
                                {"type": "image_url", "image_url": {"url": image_data_url(image_path)}}]
                break
    body = {"model": model, "messages": msgs, "max_tokens": max_tokens}
    if extra:
        body.update(extra)
    data = json.dumps(body).encode()
    url = base_url(judge) + "/v1/chat/completions"
    ua = os.environ.get("MH_OPENAI_UA", "codex_cli_rs/0.20.0")
    last = ""
    auth_key = load_key(judge=judge, override=key)   # resolve once: per-call key > MH_JUDGE_KEY > globals
    timed_out = False  # sticky: did the latest failure look like a dead/slow connection?
    for attempt in range(max(1, retries)):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            last = last or "deadline_exhausted"
            break
        # Clamp this attempt's socket timeout to the budget left so urlopen itself cannot overrun
        # the total deadline (urlopen's timeout is the per-socket-op limit; the deadline is the cap).
        call_to = max(1, min(timeout, int(remaining) or 1))
        try:
            req = urllib.request.Request(url, data=data, method="POST", headers={
                "Authorization": "Bearer " + auth_key, "Content-Type": "application/json",
                "User-Agent": ua, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=call_to) as r:
                d = json.loads(r.read().decode())
            c = (d.get("choices") or [{}])[0].get("message", {}).get("content")
            if isinstance(c, list):
                c = "".join(x.get("text", "") for x in c if isinstance(x, dict))
            if c and c.strip():
                return {"ok": True, "content": c, "error_type": None, "raw": c[:300]}
            last = "empty_content"
            timed_out = False
        except urllib.error.HTTPError as e:
            try:
                eb = e.read().decode("utf-8", "ignore")[:300]
            except Exception:
                eb = ""
            last = "http_%s:%s" % (e.code, eb)
            low = eb.lower()
            if any(k in eb for k in _BILLING) or any(k in low for k in _BILLING_EN):
                return {"ok": False, "content": None, "error_type": "billing", "raw": last}
            if e.code in (400, 401):
                return {"ok": False, "content": None, "error_type": "auth", "raw": last}
            if 400 <= e.code < 500:
                return {"ok": False, "content": None, "error_type": "http_4xx", "raw": last}
            last = "http_5xx:" + last  # 5xx -> retry
            timed_out = False
        except Exception as ex:
            # socket.timeout / TimeoutError / URLError(connection refused|reset|DNS) -> retryable
            # connection-class failure that we report as a timeout once the budget is spent.
            timed_out = _is_timeout_like(ex)
            last = ("timeout:" if timed_out else "exception:") + repr(ex)[:160]
        # Bounded backoff, but never sleep past the total deadline (and skip the sleep after the
        # final attempt) so the deadline guarantee holds even with many retries.
        if attempt < max(1, retries) - 1:
            nap = min(min(12, 2 ** attempt), max(0.0, deadline - time.monotonic()))
            if nap <= 0:
                timed_out = True
                break
            time.sleep(nap)
    if last.startswith("empty"):
        et = "empty"
    elif timed_out:
        et = "timeout"
    elif "5xx" in last:
        et = "http_5xx"
    else:
        et = "exception"
    return {"ok": False, "content": None, "error_type": et, "raw": last or "no_response"}
