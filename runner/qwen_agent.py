"""Real Qwen3-VL tool-calling agent (the BRAIN), environment-aware.

Same text brain + <tool_call>/<answer> protocol across all three substrates; only the system prompt
and the perceive-via-tools framing differ by environment.type:
  - tool_sandbox (MedCTA): cannot see image; must call image tools to perceive.
  - fhir (PhysicianBench): clinical agent; query FHIR for the given patient; write deliverables.
  - gui (HealthAdminBench): (uses observation snapshots; basic support).
Reasoning via vlm_backend.chat() (text-only). One action per act() (run.py protocol).
"""
import json, re, os

PROTOCOL = """Respond with EXACTLY ONE of these, nothing else:
  - a tool call:    <tool_call>{{"name": "<ToolName>", "arguments": {{...}}}}</tool_call>
  - a final answer: <answer>...</answer>
Output ONE action per turn. NEVER put more than one <tool_call> block in a single message.
Do NOT fabricate facts not returned by tools. Give the final answer only when you have enough evidence."""

SYS_BY_ENV = {
    "tool_sandbox": """You are a careful medical reasoning agent. You CANNOT see the image directly;
the ONLY way to perceive it is to call the perception tools below. Available tools:
{tools}
Rules:
- The image is ALREADY loaded inside the tools. Do NOT pass an "image" argument and do NOT paste any
  image text yourself. Call ImageDescription with empty arguments: {{"name":"ImageDescription","arguments":{{}}}}.
- Do NOT answer from the first global description alone. GROUND your answer: call
  RegionAttributeDescription (args: bbox [x1,y1,x2,y2], attribute) one or more times to inspect the
  specific structures/regions the question asks about before answering.
- Use OCR only when the question is about text printed in the image.
""" + PROTOCOL,
    "fhir": """You are a clinical agent working in an EHR. Use the FHIR tools to retrieve the patient's
data and complete the clinical task in the instructions. The patient resource id / MRN is: {patient}.
Each search tool is NAMED for what it returns (demographics/problems/labs/vitals/medications/notes/...) — call the specific tool you need; START with fhir_patient_search_demographics ONCE to confirm identity (this single demographics query is EXPECTED and scored). Pass patient={patient} to the clinical search tools. Use fhir_read(resourceType, id) ONLY for a specific resource you must inspect in full — the search tools ALREADY return the data you need, so do NOT read resources one-by-one (that wastes all your steps). get_lab_reference_range for lab interpretation, and the *_create tools to place orders / send messages / schedule.
IMPORTANT — do NOT spend all your steps retrieving. As soon as you have the data the task needs, WRITE the required deliverable with write_file(path, content) under the EXACT path the instructions specify, BEFORE finishing or running out of steps. Available tools:
{tools}
""" + PROTOCOL,
    "gui": """You are an agent operating a REAL web admin portal to complete the task. Each step you
receive an OBSERVATION: the visible page text plus a numbered list of interactive elements, e.g.
  [ref=3] button 'Submit Appeal'
  [ref=7] input[text] 'Reason'
Act with EXACTLY ONE tool call, addressing elements by their ref number:
  - navigate {{"url": "/path"}}
  - click {{"ref": N}}
  - type {{"ref": N, "text": "..."}}
  - select {{"ref": N, "value": "..."}}
  - submit {{"ref": N}}   (or submit {{}} for the page's main submit button)
  - snapshot {{}}         (re-read the current page)
ALWAYS read the OBSERVATION before acting. Available tools:
{tools}
""" + PROTOCOL,
}

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S)
# Perception tools whose image is provided by the backend; agent must not pass an image arg.
PERCEPTION_TOOLS = {"ImageDescription", "RegionAttributeDescription", "OCR"}


def _first_json_after(s, tag="<tool_call>"):
    """Extract the first balanced {...} JSON object after `tag` (robust to nested braces).
    Returns None if `tag` is ABSENT — callers must not treat tagless JSON (e.g. a structured
    answer inside <answer>...</answer>) as a tool call."""
    i = s.find(tag)
    if i < 0:
        return None
    j = s.find("{", i + len(tag))
    if j < 0:
        return None
    depth = 0; instr = False; esc = False
    for k in range(j, len(s)):
        c = s[k]
        if esc:
            esc = False; continue
        if c == "\\":
            esc = True; continue
        if c == '"':
            instr = not instr; continue
        if instr:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[j:k + 1]
    return None


class QwenToolAgent:
    name = "qwen"
    def __init__(self, task):
        self.task = task
        et = (task.get("environment") or {}).get("type", "tool_sandbox")
        self.et = et
        tools = task.get("available_tools", []) or []
        tool_lines = "\n".join("- %s : %s" % (t.get("name"), t.get("signature", "")) for t in tools)
        patient = (task.get("context") or {}).get("patient_ref") or ""
        sys = SYS_BY_ENV.get(et, SYS_BY_ENV["tool_sandbox"]).format(tools=tool_lines, patient=patient)
        q = str((task.get("context") or {}).get("text") or "") or str(task.get("goal") or "")
        self.messages = [{"role": "system", "content": sys}, {"role": "user", "content": q}]
        self._pending = False; self._last_tool = None
    def _chat(self, messages, max_new_tokens=400):
        from vlm_backend import get_backend
        return get_backend().chat(messages, max_new_tokens=max_new_tokens)

    def _parse(self, out):
        # Only treat as a tool call when the <tool_call> tag is actually present AND carries a name.
        # A final answer may itself contain JSON (e.g. <answer>{"diagnosis": "pneumonia"}</answer>) —
        # that must NOT be mis-parsed as a tool call. Otherwise fall through to the <answer> branch.
        if "<tool_call>" in out:
            raw = _first_json_after(out, "<tool_call>")
            if raw:
                try:
                    call = json.loads(raw)
                    name = call.get("name")
                    args = call.get("arguments", {}) or {}
                    if name:  # a tool_call with no name is not a valid action
                        # Protocol: image is supplied by the backend; strip any agent-passed image arg.
                        if name in PERCEPTION_TOOLS and isinstance(args, dict):
                            for k in ("image", "images", "image_path", "img"):
                                args.pop(k, None)
                        self._pending = True; self._last_tool = name
                        return {"type": "tool_call", "tool": name, "args": args}
                except Exception:
                    pass
            # <tool_call> tag opened but JSON missing/truncated/unparseable -> a CUT-OFF tool call
            # (e.g. an over-long write_file), NOT a chat answer. Flag so the runner retries, not loses it.
            if "<answer>" not in out:
                return {"type": "tool_call_truncated", "raw": out[:300]}
        a = ANSWER_RE.search(out)
        return {"type": "final", "answer": (a.group(1).strip() if a else out.strip())}

    def act(self, state):
        _lr0 = state.get("last_result")
        _fb = _lr0.get("feedback") if isinstance(_lr0, dict) else None
        if _fb:  # runner deliverable-enforcement feedback: incorporate and re-decide
            self.messages.append({"role": "user", "content": "SYSTEM: " + _fb})
            out = self._chat(self.messages, max_new_tokens=1500)
            self.messages.append({"role": "assistant", "content": out})
            return self._parse(out)
        if self.et == "gui":
            lr = state.get("last_result")
            if isinstance(lr, dict) and ("observation" in lr or "error" in lr):
                parts = []
                if lr.get("error"): parts.append("ERROR: %s" % lr["error"])
                if lr.get("observation"): parts.append(lr["observation"])
                self.messages.append({"role": "user", "content": ("OBSERVATION:\n" + "\n".join(parts))[:3800]})
            out = self._chat(self.messages, max_new_tokens=300)
            self.messages.append({"role": "assistant", "content": out})
            return self._parse(out)
        if self._pending:
            lr = state.get("last_result")
            obs = lr.get("output") if isinstance(lr, dict) and "output" in lr else lr
            if not isinstance(obs, str): obs = json.dumps(obs, ensure_ascii=False)[:int(os.environ.get("MH_OBS_MAX_LEN", "10000"))]  # was 1500; official caps tool output to LLM at 10k
            self.messages.append({"role": "user", "content": "TOOL RESULT (%s): %s" % (self._last_tool, obs)})
            self._pending = False
        out = self._chat(self.messages, max_new_tokens=1500)
        self.messages.append({"role": "assistant", "content": out})
        return self._parse(out)
