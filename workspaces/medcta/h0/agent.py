# -*- coding: utf-8 -*-
"""MedCTA default (h0) workspace agent program, function calling.

The agent for MedCTA runs whose model uses native function calling. It builds the conversation from
this workspace's components (system_prompt.md, tool_descriptions/: a tool's schema is derived from its signature), asks
the shell's `chat` for one assistant turn at a time, executes a turn's parallel tool calls one per step and renders
harness feedback into the conversation. It never calls the environment or the network itself.

Contract with the shell (runner/workspace.py):
    AgentProgram(task, workspace, chat)
        task       public task fields: goal, context, available_tools, environment (type, public config)
        workspace  this directory's components for function calling (system_prompt, tool_description(name))
        chat       chat(messages, tools_schema) -> one assistant message {content, tool_calls}
    act(state) -> {"type": "tool_call", "tool", "args"} | {"type": "final", "answer"}
        state      turn_inputs (every input of the turn, in order), last_result, harness_feedback (dict or None)
"""
import json
import re

OBS_MAX_LEN = 10000          # cap on a tool observation shown to the model

# How a mid-batch harness revision treats the sibling tool calls the model emitted in the same turn.
LOCAL_STAGES = frozenset({"write_verify", "misroute", "authorization"})
TERMINAL_STAGES = frozenset({"runtime_budget", "stagnation", "precommit_terminal"})


def _obs_truncate(s):
    """Cap a tool observation, and when it is actually cut, say so, so the agent knows the result is incomplete."""
    if isinstance(s, str) and len(s) > OBS_MAX_LEN:
        return s[:OBS_MAX_LEN] + ("\n[OUTPUT TRUNCATED -- %d of %d chars shown. Narrow the query with the "
                                  "date / code / count / page_limit filters to retrieve the specific records you need.]"
                                  % (OBS_MAX_LEN, len(s)))
    return s


def _schema(signature):
    """A parameter schema from a signature such as "(image,region)->text": one string property per argument, the
    image excluded (the backend supplies it)."""
    m = re.match(r"\(([^)]*)\)", signature or "")
    props = {}
    if m and m.group(1).strip():
        for part in m.group(1).split(","):
            name = part.strip().split(":")[0].split("=")[0].strip()
            if name and name.lower() != "image":
                props[name] = {"type": "string"}
    return {"type": "object", "properties": props, "required": []}


def _with_required(schema):
    """An explicit `required: []` where the schema has none: some providers reject a missing one."""
    if isinstance(schema, dict) and schema.get("type") == "object" and not isinstance(schema.get("required"), list):
        schema = dict(schema)
        schema["required"] = []
    return schema


def split_turn(state, acquired_shown):
    """The turn's inputs as this program renders them (runner/turn_inputs.py).

    The shell hands every input of the turn in arrival order as state["turn_inputs"]; last_result, harness_feedback and
    harness_acquired are the latest of each. A turn whose inputs those views all show (one input, or one of each) is
    rendered from those views alone. Otherwise the inputs up to the last one they would not show are rendered first,
    one at a time in order, each as this program renders that input alone; then the rest of the turn from the views.
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
        self.tools_schema = []
        for t in task.get("available_tools") or []:
            if not t.get("name"):
                continue
            d = workspace.tool_description(t["name"])
            self.tools_schema.append({"type": "function", "function": {
                "name": t["name"], "description": (d.get("description") or d.get("signature") or "")[:400],
                "parameters": _with_required(d.get("parameters") or _schema(d.get("signature")))}})
        question = str((task.get("context") or {}).get("text") or "") or str(task.get("goal") or "")
        self.messages = [{"role": "system", "content": workspace.system_prompt},
                         {"role": "user", "content": question}]
        self._call_id = None
        self._call_name = None
        self._queue = []                 # the rest of the current model turn's parallel tool calls
        self._answered = False           # an input of this turn already closed the call that ran
        self._held = []                  # later inputs of the turn, shown once the batch's tool results are in
        self._carried_result = None      # the last result as the previous turn was handed it

    # ------------------------------------------------------------------ harness feedback
    @staticmethod
    def revision_scope(fb):
        """CALL_LOCAL: revise only the current call, keep draining the batch. TERMINAL: stop issuing tools.
        BATCH_GLOBAL: flush the batch and re-plan. Reads only the public decision/stage and repair fields."""
        if not isinstance(fb, dict):
            return None
        override = str(fb.get("revision_scope") or "").upper()
        if override in ("CALL_LOCAL", "BATCH_GLOBAL", "TERMINAL"):
            return override
        decision = str(fb.get("decision") or "").upper()
        stage = fb.get("stage") or ""
        if decision == "ACQUIRE":
            return "BATCH_GLOBAL"            # a harness read re-decides the action
        if stage in TERMINAL_STAGES:
            return "TERMINAL"
        if stage in LOCAL_STAGES:
            return "CALL_LOCAL"
        if stage == "before_action":
            if fb.get("missing_obligations") or fb.get("evidence_needed") or fb.get("suggested_capabilities"):
                return "BATCH_GLOBAL"
            return "CALL_LOCAL"
        return "BATCH_GLOBAL"

    @staticmethod
    def local_revision_text(fb):
        """A revision delivered as the current call's own result: self-contained, because last_result may be
        stale for a call the harness blocked."""
        txt = "[HARNESS %s -- this call only] %s" % (fb.get("decision", "REVISE"), fb.get("reason") or "(revise this call)")
        mo = fb.get("missing_obligations")
        if mo:
            txt += "; missing: " + json.dumps(mo, ensure_ascii=False)
        txt += " Your other tool calls in this batch are still running; re-issue ONLY this corrected call."
        return txt

    def _next_call(self, tc):
        self._call_id = tc.get("id")
        fn = tc.get("function", {})
        self._call_name = fn.get("name")
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except Exception:
            args = {}
        return {"type": "tool_call", "tool": fn.get("name"), "args": args}

    # ------------------------------------------------------------------ one step
    def _render(self, fb, lr, own_result):
        """One set of inputs in this program's format: harness feedback `fb` and the result `lr` (`own_result`: lr is an
        input of this turn, not the result carried over from an earlier one). The call that just ran is closed by the
        first input of the turn; a later input of the same turn that would take its place instead follows as a user
        message once every tool call id of the batch has its result. -> fb as the caller acts on it (None once it
        was delivered on the call itself)."""
        scope = self.revision_scope(fb) if isinstance(fb, dict) else None
        # The tool result must immediately follow the assistant message that called it, so close the call that
        # just ran before any harness or user message.
        if self._call_id is not None:
            if scope == "CALL_LOCAL":
                self.messages.append({"role": "tool", "tool_call_id": self._call_id, "name": self._call_name,
                                      "content": self.local_revision_text(fb)})
            else:
                obs = lr.get("output") if isinstance(lr, dict) and "output" in lr else lr
                if not isinstance(obs, str):
                    obs = json.dumps(obs, ensure_ascii=False)
                self.messages.append({"role": "tool", "tool_call_id": self._call_id, "name": self._call_name,
                                      "content": _obs_truncate(obs)})
            self._call_id = None
            self._answered = True
        elif self._answered:
            if scope == "CALL_LOCAL":
                self._held.append({"role": "user", "content": self.local_revision_text(fb)})
            elif own_result:
                obs = lr.get("output") if isinstance(lr, dict) and "output" in lr else lr
                if not isinstance(obs, str):
                    obs = json.dumps(obs, ensure_ascii=False)
                self._held.append({"role": "user", "content": _obs_truncate(obs)})
        if scope == "CALL_LOCAL":
            fb = None                        # delivered on the call itself; keep draining
        if isinstance(fb, dict) and self._queue:
            # a batch-level revision: every remaining call id still needs a result
            for tc in self._queue:
                self.messages.append({"role": "tool", "tool_call_id": tc.get("id"),
                                      "name": tc.get("function", {}).get("name"),
                                      "content": "[skipped: superseded by the harness revision below]"})
            self._queue = []
        if isinstance(fb, dict):
            self._flush_held()
            msg = "[HARNESS %s] %s" % (fb.get("decision", "REVISE"), fb.get("reason") or "")
            mo = fb.get("missing_obligations")
            if mo:
                msg += "; missing: " + json.dumps(mo, ensure_ascii=False)
            msg += "\nRevise accordingly; the ENVIRONMENT observation below is still current."
            self.messages.append({"role": "user", "content": msg})
        return fb

    def _flush_held(self):
        self.messages.extend(self._held)
        self._held = []

    def act(self, state):
        earlier, view, rest_result, _turn_result = split_turn(state, acquired_shown=False)
        self._answered = False
        for item in earlier:
            self._render(item.get("feedback"), item["result"] if "result" in item else self._carried_result,
                         "result" in item)
        fb = self._render(view.get("harness_feedback"), view.get("last_result"), rest_result)
        self._carried_result = state.get("last_result")
        if not isinstance(fb, dict) and self._queue:
            return self._next_call(self._queue.pop(0))
        self._flush_held()
        reply = self._chat(self.messages, self.tools_schema)
        tcs = reply.get("tool_calls") or []
        am = {"role": "assistant", "content": reply.get("content") or ""}   # reasoning content is not echoed back
        if tcs:
            am["tool_calls"] = tcs
        self.messages.append(am)
        if tcs:
            self._queue = list(tcs[1:])
            return self._next_call(tcs[0])
        return {"type": "final", "answer": reply.get("content") or ""}
