"""OpenAI-compatible API agent (the BRAIN) — same <tool_call>/<answer> protocol as QwenToolAgent,
but the brain is a remote chat-completions model (e.g. gpt-5.5 via an OpenAI-compatible gateway)
instead of the local Qwen3-VL. Tool perception (MedCTA image tools) still runs the LOCAL VLM inside
the tool backend; this class only swaps the reasoning brain. Text-only brain across all 3 substrates.

Config (env):
  MH_OPENAI_BASE   default https://www.micuapi.ai   (calls {BASE}/v1/chat/completions; UA=MH_OPENAI_UA default codex_cli_rs/0.20.0)
  MH_OPENAI_MODEL  default gpt-5.5
  MH_OPENAI_KEY    API key; if unset, read ~/.xbai_key (chmod 600, never committed)
"""
import os, json, time, re, urllib.request, urllib.error
import gateway
from qwen_agent import (QwenToolAgent, _MEDCTA_MM, _MEDCTA_MM_NOTOOLS,
                        resolve_medcta_image, image_data_url)

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
        # FAIRNESS #3: native function-calling protocol (fair standard; frontier models are
        # trained for it) instead of our text <tool_call> protocol. Opt-in MH_PROTOCOL=function_calling.
        self.fc = os.environ.get("MH_PROTOCOL", "text") == "function_calling"
        if self.fc:
            self._init_fc()
        # MedCTA single-system: inject the task image into the brain's first user message (default
        # MH_MEDCTA_IMAGE_VISIBLE=1) and override the system prompt to the image-visible / tools-optional
        # text. Runs AFTER _init_fc so the image survives the FC message rebuild. Tools-disabled ablation
        # (MH_MEDCTA_TOOLS_ENABLED=0) is handled in qwen __init__ (empty tool list) + FC schema below.
        self._apply_medcta_mm()
    def _init_fc(self):
        tools = self.task.get("available_tools", []) or []
        def _schema(sig):
            m = re.match(r"\(([^)]*)\)", sig or "")
            props = {}
            if m and m.group(1).strip():
                for part in m.group(1).split(","):
                    nm = part.strip().split(":")[0].split("=")[0].strip()
                    if nm and nm.lower() != "image":
                        props[nm] = {"type": "string"}
            return {"type": "object", "properties": props}
        self.tools_schema = [{"type": "function", "function": {
            "name": t.get("name"), "description": (t.get("signature") or "")[:200],
            "parameters": _schema(t.get("signature"))}} for t in tools if t.get("name")]
        # MedCTA tools-disabled ablation: present an empty tool list to the FC brain too.
        if self.et == "tool_sandbox" and not getattr(self, "medcta_tools_enabled", True):
            self.tools_schema = []
        base = {"fhir": "You are a clinical AI assistant with access to an EHR via FHIR API tools and a write_file tool.",
                "tool_sandbox": "You are a medical reasoning agent. The image is already loaded inside the perception tools.",
                "gui": "You are a web agent operating a healthcare portal."}.get(self.et, "You are a medical agent.")
        q = self.messages[-1]["content"] if self.messages else self.task.get("goal", "")
        self.messages = [{"role": "system", "content": base + " Use the available tools as needed; give your final answer as plain text when done."},
                         {"role": "user", "content": q}]
        self._fc_call_id = None

    def _apply_medcta_mm(self):
        """MedCTA single-system default: make gpt-5.5 see the task image directly and offer tools as
        optional. Only for env type == tool_sandbox with MH_MEDCTA_IMAGE_VISIBLE=1 (default)."""
        if self.et != "tool_sandbox" or not getattr(self, "medcta_image_visible", False):
            return
        if os.environ.get("MH_PROMPT_TRACK", "harness") == "native":
            return  # native track keeps its upstream-faithful prompt untouched
        img = resolve_medcta_image(self.task)
        # Override the system message with the image-visible / tools-optional prompt (matches the
        # text-protocol prompt qwen __init__ chose; needed because _init_fc rebuilt a generic one).
        tools = self.task.get("available_tools", []) or []
        if not getattr(self, "medcta_tools_enabled", True):
            tools = []
        tool_lines = "\n".join("- %s : %s" % (t.get("name"), t.get("signature", "")) for t in tools)
        prompt = _MEDCTA_MM if getattr(self, "medcta_tools_enabled", True) else _MEDCTA_MM_NOTOOLS
        sys_text = prompt.format(tools=tool_lines, patient="")
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0]["content"] = sys_text
        else:
            self.messages.insert(0, {"role": "system", "content": sys_text})
        if not img:
            return  # no resolvable image -> keep text-only (prompt still says "attached" harmlessly)
        try:
            url = image_data_url(img)
        except Exception:
            return
        # Find the first user message and convert its content to multimodal [text, image_url].
        for m in self.messages:
            if m.get("role") == "user":
                txt = m.get("content")
                if isinstance(txt, list):
                    txt = " ".join(x.get("text", "") for x in txt if isinstance(x, dict))
                m["content"] = [{"type": "text", "text": txt or ""},
                                {"type": "image_url", "image_url": {"url": url}}]
                break
        self._medcta_image_injected = True

    def _chat_fc(self, messages):
        url = self.base + "/v1/chat/completions"
        body = {"model": self.model, "messages": messages, "tools": self.tools_schema,
                "tool_choice": "auto", "max_tokens": int(os.environ.get("MH_OPENAI_MAX_TOKENS", "16000"))}
        if self.reasoning:
            body["reasoning_effort"] = self.reasoning
        data = json.dumps(body).encode()
        for attempt in range(5):
            try:
                req = urllib.request.Request(url, data=data, method="POST", headers={
                    "Authorization": "Bearer " + self._key, "Content-Type": "application/json",
                    "User-Agent": os.environ.get("MH_OPENAI_UA", "codex_cli_rs/0.20.0")})
                with urllib.request.urlopen(req, timeout=int(os.environ.get("MH_OPENAI_TIMEOUT", "300"))) as r:
                    d = json.loads(r.read().decode())
                return (d.get("choices") or [{}])[0].get("message", {}) or {}
            except Exception:
                time.sleep(2 ** attempt)
        return {"content": "API_BRAIN_ERROR"}

    def act_fc(self, state):
        if self._fc_call_id is not None:
            lr = state.get("last_result")
            obs = lr.get("output") if isinstance(lr, dict) and "output" in lr else lr
            if not isinstance(obs, str):
                obs = json.dumps(obs, ensure_ascii=False)
            obs = obs[:int(os.environ.get("MH_OBS_MAX_LEN", "10000"))]
            self.messages.append({"role": "tool", "tool_call_id": self._fc_call_id, "content": obs})
            self._fc_call_id = None
        msg = self._chat_fc(self.messages)
        tcs = msg.get("tool_calls") or []
        am = {"role": "assistant", "content": msg.get("content") or ""}
        if tcs:
            am["tool_calls"] = tcs
        self.messages.append(am)
        if tcs:
            tc = tcs[0]
            self._fc_call_id = tc.get("id")
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:
                args = {}
            return {"type": "tool_call", "tool": fn.get("name"), "args": args}
        return {"type": "final", "answer": msg.get("content") or ""}

    def act(self, state):
        if getattr(self, "fc", False):
            return self.act_fc(state)
        return super().act(state)

    def _chat(self, messages, max_new_tokens=400):
        # Migrated to the unified gateway HTTP client (Codex #2). Same brain model/prompts; gateway
        # owns retry/backoff/timeout/billing detection. Preserve the EXACT text contract: a plain
        # content string on success, an <answer>API_BRAIN_ERROR ...</answer> sentinel on failure
        # (BILLING/QUOTA variant when the gateway classifies the error as billing).
        max_tokens = max(int(os.environ.get("MH_OPENAI_MAX_TOKENS", "16000")), int(max_new_tokens))  # reasoning_effort eats budget; long write_file needs headroom
        extra = {"reasoning_effort": self.reasoning} if self.reasoning else None
        res = gateway.chat(messages, model=self.model, max_tokens=max_tokens, judge=False,
                           timeout=int(os.environ.get("MH_OPENAI_TIMEOUT", "300")), extra=extra)
        if res.get("ok"):
            return res["content"]
        if res.get("error_type") == "billing":
            return "<answer>API_BRAIN_ERROR: BILLING/QUOTA %s</answer>" % res.get("raw", "")
        return "<answer>API_BRAIN_ERROR: %s</answer>" % res.get("raw", "")
