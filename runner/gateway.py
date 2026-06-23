"""Unified gateway HTTP client (Codex #2). ONE place for: key loading, base/UA, retry+backoff,
timeout, billing/quota detection, multimodal image_url, and structured error types -- so every agent,
judge, and VLM tool gets identical reliability/cost/failure semantics instead of 7 divergent copies
(retry 4 vs 5 vs 1, timeout 120/200/300). Migrate callers to gateway.chat() incrementally.

Env: MH_OPENAI_BASE (default micuapi) / MH_JUDGE_BASE (judges may differ) / MH_OPENAI_KEY|~/.xbai_key
     / MH_OPENAI_UA / MH_GATEWAY_TIMEOUT (default 300) / MH_GATEWAY_RETRIES (default 4)
"""
import os, json, time, base64, urllib.request, urllib.error

_BILLING = ("额度", "欠费", "预扣费")
_BILLING_EN = ("insufficient", "balance", "quota", "exceeded your current")
_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp"}


def load_key():
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


def chat(messages, model, max_tokens=1024, judge=False, timeout=None, retries=None, image_path=None, extra=None):
    """Return {"ok": bool, "content": str|None, "error_type": str|None, "raw": str}.
    error_type in {None, "billing", "auth", "http_4xx", "http_5xx", "empty", "exception"} -- structured,
    never a substring guess. On image_path, the LAST user message is made multimodal."""
    timeout = int(os.environ.get("MH_GATEWAY_TIMEOUT", "300")) if timeout is None else timeout
    retries = int(os.environ.get("MH_GATEWAY_RETRIES", "4")) if retries is None else retries
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
    for attempt in range(max(1, retries)):
        try:
            req = urllib.request.Request(url, data=data, method="POST", headers={
                "Authorization": "Bearer " + load_key(), "Content-Type": "application/json",
                "User-Agent": ua, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read().decode())
            c = (d.get("choices") or [{}])[0].get("message", {}).get("content")
            if isinstance(c, list):
                c = "".join(x.get("text", "") for x in c if isinstance(x, dict))
            if c and c.strip():
                return {"ok": True, "content": c, "error_type": None, "raw": c[:300]}
            last = "empty_content"
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
        except Exception as ex:
            last = "exception:" + repr(ex)[:160]
        time.sleep(min(12, 2 ** attempt))
    et = "empty" if last.startswith("empty") else ("http_5xx" if "5xx" in last else "exception")
    return {"ok": False, "content": None, "error_type": et, "raw": last}
