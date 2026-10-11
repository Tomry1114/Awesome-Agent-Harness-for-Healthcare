# -*- coding: utf-8 -*-
"""HealthAdminBench H0 agent program, function calling.

The default function-calling GUI agent for HealthAdminBench. It follows the HealthAdminBench paper's context model (arXiv:2604.09937 Sec 3.6,
App A.5): per-task portal guidance in the system prompt, a KEY_INFO memory the agent writes with every action, a window
of recent turns with only the current page in full, a STEP counter, upstream's loop alert and the response-format
reminder in every page prompt. It builds the conversation from this workspace's components (system_prompt.md, a
template whose {portal_guidance} is assembled from prompt_fragments/; tool_descriptions/), asks the shell's `chat` for
one assistant turn at a time, executes a turn's parallel tool calls one per step and renders harness feedback into the
conversation. It never calls the environment or the network itself.

Contract with the shell (runner/workspace.py):
    AgentProgram(task, workspace, chat)
        task       public task fields: goal, context, available_tools, environment (type, public config:
                   payer_portal, challenge_type)
        workspace  this directory's components for function calling (system_prompt, prompt_fragment(name),
                   tool_description(name))
        chat       chat(messages, tools_schema) -> one assistant message {content, tool_calls}
    act(state) -> {"type": "tool_call", "tool", "args"} | {"type": "final", "answer"}
        state      turn_inputs (every input of the turn, in order), last_result (the first page on the first step),
                   harness_feedback (dict or None)
"""
import json
import re

TRAJ_WINDOW = 10             # upstream settings.agent.max_trajectory_length: turns kept verbatim
AXTREE_MAX = 50000           # upstream settings.agent.max_axtree_length
GUIDANCE_SLOT = "{portal_guidance}"
KEY_INFO_REMINDER = ("\n\nWith your next action also give key_info: a concise summary of NEW "
                     "task-relevant facts on this page (\"None\" if nothing new). Anything you do "
                     "not record is lost once it falls out of the window.")
STALE_PAGE = "(page from an earlier step -- see KEY INFORMATION in MEMORY)"
KEY_INFO_RE = re.compile(r"^\s*KEY[_ ]?INFO\s*:\s*(.+?)\s*$", re.I | re.M)
PAGE_MESSAGE_RE = re.compile(r"STEP: 1|STEP: \d+\n")   # a page shown as a user message (_wrap_obs)

# How a mid-batch harness revision treats the sibling tool calls the model emitted in the same turn.
LOCAL_STAGES = frozenset({"write_verify", "misroute", "authorization"})
TERMINAL_STAGES = frozenset({"runtime_budget", "stagnation", "precommit_terminal"})

# Upstream harness/healthcare_hints.py get_hints_for_task (DOM action space): action syntax, shared execution rules,
# the EMR block for the challenge type, and the block of the external portal the task involves.
EMR_DENIAL_TYPES = frozenset({"denial_triage", "electronic_appeals"})
PORTAL_FRAGMENTS = {"payer_a": "payer_a", "payer_b": "payer_b", "fax_portal": "fax_portal", "aetna": "payer_a",
                    "anthem": "payer_b", "payera": "payer_a", "payerb": "payer_b"}


def portal_guidance(workspace, portal, challenge_type):
    if challenge_type and challenge_type.lower() in EMR_DENIAL_TYPES:
        emr = "emr_denials"
    elif challenge_type and challenge_type.lower() == "workflow":
        emr = "emr_dme"
    else:
        emr = "emr_worklist"
    names = ["syntax_dom", "shared_execution", emr]
    if portal and PORTAL_FRAGMENTS.get(portal.lower()):
        names.append(PORTAL_FRAGMENTS[portal.lower()])
    blocks = [workspace.prompt_fragment(n) for n in names]
    return "\n\n".join(b.strip() for b in blocks if b.strip())


# Upstream loop detection (upstream HealthAdminBench harness/prompts.py:646-720 detector, :562-580 alert text),
# reimplemented because that module imports harness.config.settings, which collides with runner/harness.
def detect_loop(action_history):
    if not action_history:
        return {"any_loop": False, "severity": "none", "repeat_count": 0}
    recent = action_history[-8:]                                   # upstream's 8-action window
    repeat_count = 0
    if len(action_history) >= 2:
        last_action = action_history[-1]
        for i in range(len(action_history) - 1, -1, -1):
            if action_history[i] == last_action:
                repeat_count += 1
            else:
                break
    exact_repeat = repeat_count >= 2
    recent_clicks = [a for a in recent if a.startswith("click(")]
    same_click = len(recent_clicks) >= 2 and len(set(recent_clicks)) == 1
    recent_fills = [a for a in recent if a.startswith("fill(")]
    same_fill_field = False
    if len(recent_fills) >= 2:
        same_fill_field = len(set(a.split(",")[0] for a in recent_fills)) == 1
    oscillating = False
    if len(action_history) >= 6:
        l6 = action_history[-6:]
        if len(set(l6)) == 2 and l6[0] == l6[2] == l6[4] and l6[1] == l6[3] == l6[5]:
            oscillating = True
    severity = "none"
    if oscillating or (exact_repeat and repeat_count >= 6):
        severity = "critical"
    elif exact_repeat:
        severity = "critical" if repeat_count >= 4 else ("warning" if repeat_count >= 2 else "none")
    elif same_click or same_fill_field:
        severity = "warning"
    return {"exact_repeat": exact_repeat, "same_click": same_click, "same_fill_field": same_fill_field,
            "oscillating": oscillating,
            "any_loop": exact_repeat or same_click or same_fill_field or oscillating,
            "severity": severity, "repeat_count": repeat_count}


def loop_alert(loop_info):
    """Upstream prompts.py:562-580 wording, verbatim."""
    if not (loop_info and loop_info.get("any_loop")):
        return None
    if loop_info.get("severity") == "critical":
        return ("\n\U0001F6A8 CRITICAL: You have repeated the same action %d times!\n"
                "You MUST try a DIFFERENT action immediately:\n"
                "- scroll(down) to reveal new elements\n"
                "- click a DIFFERENT button/link\n"
                "- If truly stuck, call done() to end episode\n"
                "DO NOT repeat the same action again!" % loop_info.get("repeat_count", 0))
    return ("\n\u26A0\uFE0F WARNING: You appear to be repeating the same action.\n"
            "This usually means:\n"
            "- The element you clicked opened a new section/form\n"
            "- New elements are now visible in PAGE ELEMENTS\n"
            "- You need to look for NEW elements that weren't there before\n"
            "Carefully examine PAGE ELEMENTS below for new inputs, buttons, or tabs.")


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
        self.tools_schema = []
        for t in task.get("available_tools") or []:
            if not t.get("name"):
                continue
            d = workspace.tool_description(t["name"])
            self.tools_schema.append({"type": "function", "function": {
                "name": t["name"], "description": (d.get("description") or "")[:400],
                "parameters": _with_required(d.get("parameters"))}})
        config = (task.get("environment") or {}).get("config") or {}
        guidance = portal_guidance(workspace, config.get("payer_portal"), config.get("challenge_type"))
        question = str((task.get("context") or {}).get("text") or "") or str(task.get("goal") or "")
        self.messages = [{"role": "system", "content": workspace.system_prompt.replace(GUIDANCE_SLOT, guidance)},
                         {"role": "user", "content": question}]
        self._call_id = None
        self._call_name = None
        self._queue = []                 # the rest of the current model turn's parallel tool calls
        self._key_info = []              # the agent's KEY_INFO memory
        self._step = 0
        self._actions = []               # action strings for loop detection
        self._seeded = False
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

    # ------------------------------------------------------------------ memory and context
    def _harvest_key_info(self, content):
        """KEY_INFO lines into the memory: deduped, in order, 'None' entries dropped."""
        if not isinstance(content, str) or not content.strip():
            return
        for m in KEY_INFO_RE.finditer(content):
            v = (m.group(1) or "").strip()
            if not v or v.lower() in ("none", "none.", "n/a", "-"):
                continue
            if v not in self._key_info:
                self._key_info.append(v)

    @staticmethod
    def _turns(msgs):
        """A flat message list grouped into turns, each starting at an assistant message, so a turn is dropped whole
        (every tool_call id must be closed by a tool result inside the same turn)."""
        turns = []
        for m in msgs:
            if m.get("role") == "assistant" or not turns:
                turns.append([m])
            else:
                turns[-1].append(m)
        return turns

    def _compact(self):
        """Keep the system prompt and objective, the last TRAJ_WINDOW turns, and one [MEMORY] digest (KEY INFORMATION
        GATHERED SO FAR + STEP) in place of everything older. Inside the window only the most recent page stays in
        full; older results, long reasoning and the seeded first page shrink to stubs."""
        head, rest = self.messages[:2], self.messages[2:]
        turns = self._turns(rest)
        digest_prev = None
        # a previously built digest sits as a lone user message at the front of `rest`; absorb it
        if turns and turns[0] and turns[0][0].get("role") == "user" \
                and str(turns[0][0].get("content", "")).startswith("[MEMORY]"):
            digest_prev = turns.pop(0)
        dropped = len(turns) > TRAJ_WINDOW
        if dropped:
            turns = turns[-TRAJ_WINDOW:]
        for t in turns[:-1]:
            for m in t:
                if m.get("role") == "tool" and isinstance(m.get("content"), str) and len(m["content"]) > 64:
                    m["content"] = STALE_PAGE
                elif m.get("role") == "assistant" and isinstance(m.get("content"), str) and len(m["content"]) > 64:
                    # an assistant turn with neither content nor tool_calls is rejected by the gateway
                    m["content"] = "" if m.get("tool_calls") else "(reasoning from an earlier step)"
                elif m.get("role") == "user" and isinstance(m.get("content"), str) \
                        and PAGE_MESSAGE_RE.match(m["content"]) and len(m["content"]) > 64:
                    m["content"] = STALE_PAGE           # the seeded first page, or a page a later input of its turn showed
        if not dropped and digest_prev is None:
            self.messages = head + [m for t in turns for m in t]
            return
        parts = ["[MEMORY] KEY INFORMATION GATHERED SO FAR:"]
        parts += ["  - " + k for k in self._key_info] or ["  (nothing recorded yet)"]
        parts.append("STEP: %d" % self._step)
        digest = {"role": "user", "content": "\n".join(parts)}
        self.messages = head + [digest] + [m for t in turns for m in t]

    def _wrap_obs(self, obs):
        """One page prompt, upstream App A.5 shape: STEP, the loop alert, the page capped at AXTREE_MAX, and the
        response-format reminder. Used for the first page and every later tool result alike."""
        page = obs if len(obs) <= AXTREE_MAX else (obs[:AXTREE_MAX] + "\n... (truncated, scroll to see more elements)")
        pre = "STEP: %d" % (self._step + 1)
        alert = loop_alert(detect_loop(self._actions))
        if alert:
            pre += "\n" + alert
        return pre + "\n" + page + KEY_INFO_REMINDER

    def _next_call(self, tc):
        self._call_id = tc.get("id")
        fn = tc.get("function", {})
        self._call_name = fn.get("name")
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except Exception:
            args = {}
        return fn, args

    # ------------------------------------------------------------------ one step
    def _render(self, fb, lr, own_result):
        """One set of inputs in this program's format: harness feedback `fb` and the result `lr` (`own_result`: lr is an
        input of this turn, not the result carried over from an earlier one). The call that just ran is closed by the
        first input of the turn; a later input of the same turn that would take its place instead follows as a user
        message once every tool call id of the batch has its result. -> fb as the caller acts on it (None once it
        was delivered on the call itself)."""
        seeded_now = False
        if not self._seeded:
            # Upstream shows the current page in every prompt, step 1 included: the shell hands the first page in as
            # last_result, and there is no tool call id to hang it on, so it is seeded as a user message.
            self._seeded = True
            first = lr.get("output") if isinstance(lr, dict) and "output" in lr else lr
            if first is not None and not isinstance(first, str):
                first = json.dumps(first, ensure_ascii=False)
            if first and str(first).strip():
                self.messages.append({"role": "user", "content": self._wrap_obs(str(first))})
                self._answered = seeded_now = True
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
                                      "content": self._wrap_obs(obs)})
            self._call_id = None
            self._answered = True
        elif self._answered and not seeded_now:
            if scope == "CALL_LOCAL":
                self._held.append({"role": "user", "content": self.local_revision_text(fb)})
            elif own_result:
                obs = lr.get("output") if isinstance(lr, dict) and "output" in lr else lr
                if not isinstance(obs, str):
                    obs = json.dumps(obs, ensure_ascii=False)
                self._held.append({"role": "user", "content": self._wrap_obs(obs)})
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
            fn, args = self._next_call(self._queue.pop(0))
            return {"type": "tool_call", "tool": fn.get("name"), "args": args}
        self._flush_held()
        self._step += 1
        self._compact()
        reply = self._chat(self.messages, self.tools_schema)
        tcs = reply.get("tool_calls") or []
        self._harvest_key_info(reply.get("content"))
        am = {"role": "assistant", "content": reply.get("content") or ""}   # reasoning content is not echoed back
        if tcs:
            am["tool_calls"] = tcs
        self.messages.append(am)
        if tcs:
            self._queue = list(tcs[1:])
            fn, args = self._next_call(tcs[0])
            # key_info is a tool parameter, the channel the model cannot skip; the content harvest above is a second
            # source for models that also narrate it
            self._harvest_key_info("KEY_INFO: " + str(args.get("key_info") or ""))
            name = fn.get("name") or ""
            ref = str(args.get("ref") or args.get("id") or args.get("direction") or "")
            extra = args.get("text") or args.get("option") or args.get("key") or args.get("file")
            self._actions.append("%s(%s%s)" % (name, ref, (", " + str(extra)[:40]) if extra else ""))
            return {"type": "tool_call", "tool": fn.get("name"), "args": args}
        return {"type": "final", "answer": reply.get("content") or ""}
