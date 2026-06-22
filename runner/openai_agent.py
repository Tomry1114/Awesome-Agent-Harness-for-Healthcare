"""OpenAI-compatible API agent (the BRAIN) — same <tool_call>/<answer> protocol as QwenToolAgent,
but the brain is a remote chat-completions model (e.g. gpt-5.5 via an OpenAI-compatible gateway)
instead of the local Qwen3-VL. Tool perception (MedCTA image tools) still runs the LOCAL VLM inside
the tool backend; this class only swaps the reasoning brain. Text-only brain across all 3 substrates.

Config (env):
  MH_OPENAI_BASE   default https://www.micuapi.ai   (calls {BASE}/v1/chat/completions; UA=MH_OPENAI_UA default codex_cli_rs/0.20.0)
  MH_OPENAI_MODEL  default gpt-5.5
  MH_OPENAI_KEY    API key; if unset, read ~/.xbai_key (chmod 600, never committed)
"""
import os, json, time, urllib.request, urllib.error
from qwen_agent import QwenToolAgent

def _load_key():
    k = os.environ.get("MH_OPENAI_KEY")
    if k: return k.strip()
    p = os.path.expanduser("~/.xbai_key")
    if os.path.exists(p): return open(p).read().strip()
    raise RuntimeError("no API key: set MH_OPENAI_KEY or create ~/.xbai_key")

class OpenAIToolAgent(QwenToolAgent):
    name = "gpt5"
    def __init__(self, task):
        super().__init__(task)
        self.base = os.environ.get("MH_OPENAI_BASE", "https://www.micuapi.ai").rstrip("/")
        self.model = os.environ.get("MH_OPENAI_MODEL", "gpt-5.5")
        self.reasoning = os.environ.get("MH_OPENAI_REASONING", "high")  # PhysicianBench official default = high
        self._key = _load_key()
    def _chat(self, messages, max_new_tokens=400):
        url = self.base + "/v1/chat/completions"
        body = {"model": self.model, "messages": messages,
                "max_tokens": max(2048, int(max_new_tokens))}
        if self.reasoning: body["reasoning_effort"] = self.reasoning
        data = json.dumps(body).encode()
        last = ""
        for attempt in range(5):
            try:
                req = urllib.request.Request(url, data=data, method="POST", headers={
                    "Authorization": "Bearer " + self._key, "Content-Type": "application/json",
                    "User-Agent": os.environ.get("MH_OPENAI_UA", "codex_cli_rs/0.20.0"), "Accept": "application/json"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    d = json.loads(r.read().decode())
                msg = (d.get("choices") or [{}])[0].get("message", {}) or {}
                content = msg.get("content")
                if isinstance(content, list):
                    content = "".join(x.get("text", "") for x in content if isinstance(x, dict))
                if content and content.strip():
                    return content
                last = "empty_content: " + json.dumps(d)[:200]
            except urllib.error.HTTPError as e:
                try: eb = e.read().decode()[:300]
                except Exception: eb = ""
                last = "http_%s: %s" % (e.code, eb)
                low = eb.lower()
                # billing/quota is non-retryable (backoff is useless) -> fail fast with a clear tag
                if any(k in eb for k in ("额度", "欠费", "预扣费")) or \
                   any(k in low for k in ("insufficient", "balance", "quota", "exceeded your current")):
                    return "<answer>API_BRAIN_ERROR: BILLING/QUOTA %s</answer>" % last
                if e.code in (400, 401):
                    break  # config/auth error -> no retry
                # non-billing 403 / 429 / 5xx -> fall through to backoff retry
            except Exception as e:
                last = "err: %s" % e
            time.sleep(min(16, 2 ** attempt))
        return "<answer>API_BRAIN_ERROR: %s</answer>" % last
