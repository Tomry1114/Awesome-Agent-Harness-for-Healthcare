# -*- coding: utf-8 -*-
"""HealthAdminBench H0 agent program, text protocol (<tool_call>/<answer>).

The agent HealthAdminBench runs for a text-protocol model: a GUI tool-protocol agent whose system prompt carries the
tool list plus the static portal operating guidance, with its own context window. It builds the conversation from this
workspace's components (system_prompt_text.md with the tool list spliced in at {tools}, then
prompt_fragments/portal_operating_guidance.md; tool_descriptions/), shows each page as an OBSERVATION capped at
upstream's accessibility-tree length, keeps a window of recent exchanges with only the newest page in full, asks the
shell's `chat` for one reply at a time, parses the reply into one action (repairing near-miss tool-call markup, and
flagging a cut-off tool call so the shell asks for a re-issue), and renders harness feedback and harness-acquired tool
results into the conversation. It never calls the environment or the network itself.

Contract with the shell (runner/workspace.py):
    AgentProgram(task, workspace, chat)
        task       public task fields: goal, context, available_tools, environment (type, public config)
        workspace  this directory's components for the text protocol (system_prompt, prompt_fragment(name),
                   tool_description(name))
        chat       chat(messages, max_new_tokens) -> the reply text; a failed call returns the shell's sentinel
                   "<answer>API_BRAIN_ERROR: ...</answer>"
    act(state) -> {"type": "tool_call", "tool", "args"} | {"type": "final", "answer"}
                  | {"type": "tool_call_truncated", "raw"}
        state      tools, last_result (the page: observation and/or error; or the shell's feedback), harness_feedback
                   (dict or None), harness_acquired (dict or None)
    protocol_variant_last
        the near-miss tag the last parsed tool call was repaired from, else None; the shell records it on the
        tool_call event and resets it
"""
import json
import re

OBS_MAX_LEN = 10000          # cap on a harness-acquired tool result shown to the model
TRAJ_WINDOW = 10             # upstream settings.agent.max_trajectory_length: exchanges kept verbatim
AXTREE_MAX = 50000           # upstream settings.agent.max_axtree_length
PORTAL_GUIDANCE = "portal_operating_guidance"   # the prompt fragment appended to the system prompt
STALE_PAGE = "(page from an earlier step -- omitted)"
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S)
_ACTION_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def _first_json_after(s, tag="<tool_call>"):
    """The first balanced {...} JSON object after `tag`, or None when the tag is absent (tagless JSON, e.g. a
    structured answer inside <answer>...</answer>, is not a tool call)."""
    i = s.find(tag)
    if i < 0:
        return None
    j = s.find("{", i + len(tag))
    if j < 0:
        return None
    depth = 0
    instr = False
    esc = False
    for k in range(j, len(s)):
        c = s[k]
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
            continue
        if c == '"':
            instr = not instr
            continue
        if instr:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[j:k + 1]
    return None


# Near-miss tool-call markup. Some models fall back to their native markup (DeepSeek's DSML with fullwidth bars) or
# misspell the tag (<tool_calls> with a JSON array, <toolcall>, <tool_action>, <tool_calling> with XML sub-tags).
# Unrecognised, the whole reply would become the final answer and end the episode. Tag detection is loose; acceptance
# is strict: the payload must name a tool this task offers.
_ALT_TAG_RE = re.compile(
    u"<\\s*(?:[|\uff5c]{1,2}\\s*DSML\\s*[|\uff5c]{1,2}[A-Za-z_]*"
    u"|tool[_\\s-]*call[A-Za-z_]*|toolcall[A-Za-z_]*"
    u"|tool[_\\s-]*action|tool[_\\s-]*use|function[_\\s-]*call[A-Za-z_]*)\\s*>", re.I)
_XML_NAME_RE = re.compile(r"<tool_name>\s*(.*?)\s*</tool_name>", re.S | re.I)
_XML_ARGS_RE = re.compile(r"<tool_(?:arguments|args)>\s*(\{.*?\})\s*</tool_(?:arguments|args)>", re.S | re.I)


def _first_json_value_at(s, start, max_gap=400):
    """The first balanced JSON object or array at/after `start`; None when it starts more than `max_gap` characters
    away (that is prose, not the tag's argument)."""
    cand = [p for p in (s.find("{", start), s.find("[", start)) if p >= 0]
    if not cand:
        return None
    j = min(cand)
    if j - start > max_gap:
        return None
    depth = 0
    instr = False
    esc = False
    for k in range(j, len(s)):
        c = s[k]
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
            continue
        if c == '"':
            instr = not instr
            continue
        if instr:
            continue
        if c in "{[":
            depth += 1
        elif c in "}]":
            depth -= 1
            if depth == 0:
                return s[j:k + 1]
    return None


def _alt_tool_call_seen(out):
    """True when the text opens a near-miss tool-call tag at all, whatever its payload."""
    return _ALT_TAG_RE.search(out) is not None


def _alt_tool_call(out, tool_names):
    """A tool call written in near-miss markup -> (name, args, tag), accepted only for a tool the task offers."""
    m = _ALT_TAG_RE.search(out)
    if not m:
        return None
    tag = m.group(0)
    name = args = None
    raw = _first_json_value_at(out, m.end())
    if raw:
        try:
            call = json.loads(raw)
        except Exception:
            call = None
        if call is None and '\\"' in raw:          # a payload with backslash-escaped quotes
            try:
                call = json.loads(raw.replace('\\"', '"'))
            except Exception:
                call = None
        if isinstance(call, list):                 # <tool_calls>[{...}]: one action per turn
            call = call[0] if (call and isinstance(call[0], dict)) else None
        if isinstance(call, dict):
            name = call.get("name") or call.get("tool") or call.get("function")
            if isinstance(name, dict):             # {"function": {"name": ..., "arguments": ...}}
                args = name.get("arguments")
                name = name.get("name")
            if args is None:
                for k in ("arguments", "args", "parameters", "input"):
                    if k in call:
                        args = call[k]
                        break
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = None
            if args is None:
                args = {}
    if not (isinstance(name, str) and isinstance(args, dict)):
        nm = _XML_NAME_RE.search(out)              # <tool_calling><tool_name>x</tool_name>...
        if not nm:
            return None
        name = nm.group(1).strip()
        am = _XML_ARGS_RE.search(out)
        try:
            args = json.loads(am.group(1)) if am else {}
        except Exception:
            args = {}
        if not isinstance(args, dict):
            return None
    if name in tool_names:
        return name, args, tag
    by_case = {t.lower(): t for t in tool_names}   # a re-cased tool name
    if name.lower() in by_case:
        return by_case[name.lower()], args, tag
    return None


def _tool_line(name, description):
    line = "- %s : %s" % (name, description.get("signature", ""))
    props = ((description.get("parameters") or {}).get("properties")) or {}
    if props:
        required = set(((description.get("parameters") or {}).get("required")) or [])
        line += "\n    params: " + "; ".join(
            "%s%s (%s)" % (k, "*" if k in required else "", str(v.get("description") or v.get("type") or "")[:70])
            for k, v in props.items())
    return line


def split_turn(state, acquired_shown):
    """The turn's inputs as this program renders them (runner/turn_inputs.py).

    The shell hands every input of the turn in arrival order as state["turn_inputs"]; last_result, harness_feedback and
    harness_acquired are the latest of each. A turn whose inputs those views all show (one input, or one of each) is
    rendered from the views alone. Otherwise the inputs up to the last one they would not show are rendered first, one
    at a time in order, each as this program renders that input alone; then the rest of the turn from the views.
    -> (earlier inputs, the state with the views over the rest, the rest carries a result, the turn carries a result)"""
    items = state.get("turn_inputs")
    if not isinstance(items, list) or not items:
        return [], state, False, False

    def last(pred):
        return max([i for i, it in enumerate(items) if pred(it)] or [-1])
    shown = {last(lambda it: it.get("feedback") is not None), last(lambda it: "result" in it)}
    if acquired_shown:
        shown.add(last(lambda it: it.get("kind") == "harness_read_result"))
    hidden = [i for i in range(len(items)) if i not in shown]
    if not hidden:
        return [], state, False, False
    earlier, rest = items[:hidden[-1] + 1], items[hidden[-1] + 1:]
    view = dict(state)
    view["harness_feedback"] = next((it["feedback"] for it in reversed(rest) if it.get("feedback") is not None), None)
    view["harness_acquired"] = next((it.get("acquired") for it in reversed(rest)
                                     if it.get("kind") == "harness_read_result"), None)
    results = [it["result"] for it in rest if "result" in it]
    view["last_result"] = results[-1] if results else None
    return earlier, view, bool(results), bool(results) or any("result" in it for it in earlier)


class AgentProgram(object):
    def __init__(self, task, workspace, chat):
        self._chat = chat
        tools = task.get("available_tools") or []
        lines = [_tool_line(t["name"], workspace.tool_description(t["name"])) for t in tools if t.get("name")]
        question = str((task.get("context") or {}).get("text") or "") or str(task.get("goal") or "")
        system = workspace.system_prompt.replace("{tools}", "\n".join(lines))
        self.messages = [{"role": "system", "content": system + "\n\n" + workspace.prompt_fragment(PORTAL_GUIDANCE)},
                         {"role": "user", "content": question}]
        self._tool_names = {str(t.get("name")) for t in tools if isinstance(t, dict) and t.get("name")}
        self.protocol_variant_last = None

    # ------------------------------------------------------------------ context window
    def _compact(self):
        """The text-protocol counterpart of the function-calling program's window (agent.py; arXiv:2604.09937 Sec 3.6,
        App A.5).

        Every page and every model turn is appended to the conversation, so without this an episode re-sends its whole
        history at each step; with pages capped at upstream's 50000 characters, single calls of a long episode grow to
        hundreds of thousands of tokens. Kept: the system prompt and the objective (the first two messages) and the last
        TRAJ_WINDOW exchanges. Older pages collapse to one line; older model turns keep only the action they chose,
        which is what upstream's RECENT ACTIONS block carries. Inside the window only the newest page stays in full,
        from the first step on: PAGE ELEMENTS describe the current page only (App A.5), and ten full pages are ~89k
        characters.
        """
        head, rest = self.messages[:2], self.messages[2:]
        keep = 2 * TRAJ_WINDOW               # one user page + one assistant turn per exchange
        old, kept = (rest[:-keep], rest[-keep:]) if len(rest) > keep else ([], rest)
        for m in old:
            c = str(m.get("content") or "")
            if len(c) <= 96:
                continue
            if m.get("role") == "assistant":
                mt = _ACTION_RE.search(c)
                m["content"] = ("ACTION: " + mt.group(1)[:160]) if mt else c[:96]
            elif c.startswith("OBSERVATION:"):
                m["content"] = STALE_PAGE
        last_page = None
        for i, m in enumerate(kept):
            if m.get("role") == "user" and str(m.get("content") or "").startswith("OBSERVATION:"):
                last_page = i
        for i, m in enumerate(kept):
            if i == last_page:
                continue
            c = str(m.get("content") or "")
            if m.get("role") == "user" and c.startswith("OBSERVATION:") and len(c) > 96:
                m["content"] = STALE_PAGE
        self.messages = head + old + kept

    # ------------------------------------------------------------------ parsing
    def _parse(self, out):
        self.protocol_variant_last = None
        # A tool call only when the <tool_call> tag is present and names a tool: an answer may itself contain JSON.
        if "<tool_call>" in out:
            raw = _first_json_after(out, "<tool_call>")
            if raw:
                try:
                    call = json.loads(raw)
                    name = call.get("name")
                    args = call.get("arguments", {}) or {}
                    if name:
                        return {"type": "tool_call", "tool": name, "args": args}
                except Exception:
                    pass
            # The tag opened but its JSON is missing, cut off or unparseable: a cut-off tool call, not an answer.
            if "<answer>" not in out:
                return {"type": "tool_call_truncated", "raw": out[:300]}
        # No usable canonical tag: before conceding a final answer, try near-miss markup. An explicit <answer> wins.
        if "<answer>" not in out:
            alt = _alt_tool_call(out, self._tool_names or set())
            if alt:
                name, args, tag = alt
                self.protocol_variant_last = tag
                return {"type": "tool_call", "tool": name, "args": args}
            # A near-miss tag with an unusable payload is a cut-off call too.
            if _alt_tool_call_seen(out):
                return {"type": "tool_call_truncated", "raw": out[:300]}
        a = ANSWER_RE.search(out)
        return {"type": "final", "answer": (a.group(1).strip() if a else out.strip())}

    # ------------------------------------------------------------------ one step
    def _render(self, fb, acquired, lr, show_result):
        """One set of inputs in this program's format: harness feedback `fb`, a harness-acquired result `acquired` and the
        result `lr` (not shown when `show_result` is False: an earlier input of the turn showed it). -> whether lr
        is the shell's feedback (the agent re-decides on it)."""
        if isinstance(fb, dict):                 # the harness verdict, in addition to the environment observation
            msg = "[HARNESS %s] %s" % (fb.get("decision", "REVISE"), fb.get("reason") or "")
            mo = fb.get("missing_obligations")
            if mo:
                msg += "; missing: " + json.dumps(mo, ensure_ascii=False)
            msg += "\nRevise accordingly; the ENVIRONMENT observation below is still current."
            self.messages.append({"role": "user", "content": msg})
        if isinstance(acquired, dict) and acquired.get("output") is not None:
            # A harness-dispatched tool result reaches the agent unconditionally, labelled as the harness's.
            out = acquired["output"]
            if not isinstance(out, str):
                out = json.dumps(out, ensure_ascii=False)
            args = acquired.get("args") or {}
            q = args.get("query") or args.get("region_query") or ""
            self.messages.append({"role": "user", "content": ("HARNESS-ACQUIRED TOOL RESULT (%s%s):\n%s" % (
                acquired.get("tool", "tool"), (" query=%s" % q) if q else "", out))[:OBS_MAX_LEN]})
        if not show_result:
            return False
        feedback = lr.get("feedback") if isinstance(lr, dict) else None
        if feedback:
            self.messages.append({"role": "user", "content": "SYSTEM: " + feedback})
            return True
        if isinstance(lr, dict) and ("observation" in lr or "error" in lr):
            parts = []
            if lr.get("error"):
                parts.append("ERROR: %s" % lr["error"])
            if lr.get("observation"):
                parts.append(lr["observation"])
            page = "OBSERVATION:\n" + "\n".join(parts)
            if len(page) > AXTREE_MAX:           # upstream's cap and its cut marker
                page = page[:AXTREE_MAX] + "\n... (truncated, scroll to see more elements)"
            self.messages.append({"role": "user", "content": page})
        return False

    def act(self, state):
        for t in state.get("tools") or []:      # the repair gate follows what the environment offers now
            n = t.get("name") if isinstance(t, dict) else t
            if isinstance(n, str) and n:
                self._tool_names.add(n)
        earlier, view, rest_result, turn_result = split_turn(state, acquired_shown=True)
        for item in earlier:
            self._render(item.get("feedback"), item.get("acquired"), item.get("result"), "result" in item)
        system = self._render(view.get("harness_feedback"), view.get("harness_acquired"), view.get("last_result"),
                              (not earlier) or rest_result or not turn_result)
        if earlier and not rest_result and turn_result:   # the turn's latest result was an earlier input
            lr = state.get("last_result")
            system = isinstance(lr, dict) and bool(lr.get("feedback"))
        if system:                               # the shell's feedback (e.g. re-issue a cut-off call): re-decide
            reply = self._chat(self.messages, max_new_tokens=1500)
            self.messages.append({"role": "assistant", "content": reply})
            return self._parse(reply)
        self._compact()
        reply = self._chat(self.messages, max_new_tokens=300)
        self.messages.append({"role": "assistant", "content": reply})
        return self._parse(reply)
