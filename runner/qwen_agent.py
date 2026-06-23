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
    "tool_sandbox": """You are a medical reasoning agent. Answer the question using the tools below.
The image is already loaded inside the perception tools (do not pass an "image" argument). Available tools:
{tools}
""" + PROTOCOL,  # FAIRNESS #2: neutral exposure of ALL tools, no strategy-teaching / no tool hiding (was: taught Image/Region/OCR usage and omitted GoogleSearch/Calculator -> measured our bias, not the model)
    "fhir": """You are a clinical AI assistant designed to support healthcare professionals.
You have access to an EHR system via FHIR API tools and can write files to disk.

Guidelines:
- Use the FHIR search tools to retrieve patient data before making clinical decisions.
- Use the FHIR create tools to place orders, send messages, or schedule appointments.
- Use the write_file tool to save deliverables (notes, assessments, reports) to disk.
- Be thorough: retrieve all relevant clinical data before writing your assessment.
- Be accurate: base your clinical reasoning on the actual patient data retrieved.
- Complete all tasks specified in the instruction before finishing.

Available tools:
{tools}
""" + PROTOCOL,  # FAIRNESS #2 (PB): upstream agent/prompts.py SYSTEM_PROMPT verbatim — no query-mechanism teaching, no "demographics is scored" hint, no obs-bug "do not read one-by-one" scaffolding (bug fixed)
    "gui": """You are an autonomous web agent that can interact with a healthcare admin portal by
performing actions. Complete the objective by analyzing the current page and selecting an action.
Each step you receive an OBSERVATION: the visible page text plus a numbered list of interactive
elements, e.g.
  [ref=3] button 'Submit Appeal'
  [ref=7] input[text] 'Reason'
Act with EXACTLY ONE tool call, addressing elements by their ref number:
  - navigate {{"url": "/path"}}
  - click {{"ref": N}}
  - type {{"ref": N, "text": "..."}}
  - select {{"ref": N, "value": "..."}}
  - submit {{"ref": N}}   (or submit {{}} for the page's main submit button)
  - back {{}}             (browser back, e.g. to return from an external portal)
  - scroll {{"direction": "down"}}
  - download {{"ref": N}}                  (download a document to the workspace)
  - upload {{"ref": N, "file_ref": "last"}}  (upload a previously downloaded file)
  - snapshot {{}}         (re-read the current page)
ALWAYS read the OBSERVATION before acting. Available tools:
{tools}
""" + PROTOCOL,  # FAIRNESS/FIDELITY (HAB): upstream framing ("autonomous web agent"); REGISTERED DEVIATION:
    # our ref=N + JSON protocol differs from upstream click([id]) bracket + ACTION:/KEY_INFO; download/upload
    # not yet supported (stage-2 file handling) -> tasks needing them are out of scope, see NATIVE_FIDELITY.md
}


# --- NATIVE prompt track (MH_PROMPT_TRACK=native): upstream-faithful system prompts. Clinical text is the
# official benchmark prompt verbatim; only the tool-calling MECHANISM (our text PROTOCOL) is appended as a
# REGISTERED deviation (we do not yet use native function-calling). See docs/PROMPT_PROVENANCE.md. ---
_PB_NATIVE = """You are a clinical AI assistant designed to support healthcare professionals.
You have access to an EHR system via FHIR API tools and can write files to disk.

Guidelines:
- Use the FHIR search tools to retrieve patient data before making clinical decisions.
- Use the FHIR create tools to place orders, send messages, or schedule appointments.
- Use the write_file tool to save deliverables (notes, assessments, reports) to disk.
- Be thorough: retrieve all relevant clinical data before writing your assessment.
- Be accurate: base your clinical reasoning on the actual patient data retrieved.
- Complete all tasks specified in the instruction before finishing.

Available tools:
{tools}
""" + PROTOCOL
_MEDCTA_NATIVE = '''You are an assistant who can utilize external tools to answer the user question. You have access to the following tools:
{tools}
''' + PROTOCOL
NATIVE_SYS_BY_ENV = {"fhir": _PB_NATIVE, "tool_sandbox": _MEDCTA_NATIVE}  # gui (HAB) = stage-2 (protocol+screenshot coupled)


# --- MedCTA single-system DEFAULT (image VISIBLE to the brain, tools OPTIONAL). This is the correct
# eval: gpt-5.5 is multimodal and sees the task image directly; the 5 tools are offered but NOT forced.
# tool_sandbox image-hidden prompt (SYS_BY_ENV["tool_sandbox"]) and the tools-disabled prompt below are
# now ABLATION configs of this one system, selected via MH_MEDCTA_IMAGE_VISIBLE / MH_MEDCTA_TOOLS_ENABLED.
_MEDCTA_MM = """You are a medical reasoning agent. The relevant medical image is ATTACHED to this conversation (you can see it directly). Answer the question about the image.
You also have access to these OPTIONAL tools:
{tools}
Use a tool ONLY if it genuinely helps (e.g. to zoom into a region, read embedded text, search a fact, or
compute). If you can answer from the image directly, just answer -- do NOT call tools you do not need.
""" + PROTOCOL
# Tools-disabled ablation (~ pure VQA): image visible, NO tools at all.
_MEDCTA_MM_NOTOOLS = """You are a medical reasoning agent. The relevant medical image is ATTACHED to this conversation (you can see it directly). Answer the question about the image directly.
No tools are available -- reason from the image alone.
""" + PROTOCOL


def medcta_config():
    """Resolve the MedCTA single-system config flags (only meaningful when env type == tool_sandbox).
    Defaults make the CORRECT eval the default: image visible to the brain + tools offered (optional)."""
    return (os.environ.get("MH_MEDCTA_IMAGE_VISIBLE", "1") == "1",
            os.environ.get("MH_MEDCTA_TOOLS_ENABLED", "1") == "1")


def resolve_medcta_image(task):
    """Resolve task.context.images[0].path to an absolute path under MH_MEDCTA_IMG_ROOT
    (same root convention as MedCTAToolSandbox._resolve_image). Returns None if absent."""
    imgs = (task.get("context") or {}).get("images") or []
    if not imgs:
        return None
    rel = imgs[0].get("path") or ""
    root = os.environ.get("MH_MEDCTA_IMG_ROOT", os.path.join(
        os.path.dirname(__file__), "..", "benchmark", "MedCTA", "opencompass", "data", "medcta_dataset"))
    fp = os.path.join(root, rel)
    return fp if os.path.exists(fp) else (rel if os.path.exists(rel) else None)


def image_data_url(path):
    """Read an image file and return a data: URL (base64) suitable for OpenAI multimodal image_url."""
    import base64, mimetypes
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return "data:%s;base64,%s" % (mime, b64)

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
        patient = (task.get("context") or {}).get("patient_ref") or ""
        _track = os.environ.get("MH_PROMPT_TRACK", "harness")
        # MedCTA single-system config: image-visible (default) -> multimodal prompt; tools optional
        # (default) vs disabled ablation -> hide the tool list. Flags only apply to tool_sandbox.
        self.medcta_image_visible, self.medcta_tools_enabled = (False, True)
        if et == "tool_sandbox":
            self.medcta_image_visible, self.medcta_tools_enabled = medcta_config()
            if not self.medcta_tools_enabled:
                tools = []  # tools-disabled ablation: present an empty tool list
        tool_lines = "\n".join("- %s : %s" % (t.get("name"), t.get("signature", "")) for t in tools)
        if et == "tool_sandbox" and self.medcta_image_visible and _track != "native":
            _prompt = _MEDCTA_MM if self.medcta_tools_enabled else _MEDCTA_MM_NOTOOLS
            sys = _prompt.format(tools=tool_lines, patient=patient)
        else:
            _tbl = NATIVE_SYS_BY_ENV if (_track == "native" and et in NATIVE_SYS_BY_ENV) else SYS_BY_ENV
            sys = _tbl.get(et, SYS_BY_ENV["tool_sandbox"]).format(tools=tool_lines, patient=patient)
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
