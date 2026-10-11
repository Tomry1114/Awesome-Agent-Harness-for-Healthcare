# -*- coding: utf-8 -*-
"""Turn inputs: what the shell hands the agent program at its next turn, every input in the order it arrived.

Between two agent turns the loop can produce several inputs: the agent's own tool result, harness read results (a
host next_read, a kernel ACQUIRE, an on_answer Read) and feedback (a kernel verdict, a host Nudge / Revision /
PostRevision / on_answer feedback, a shell refusal). Each input is one typed item and none overwrites another, so a
later input of a kind cannot silently replace an earlier one within the same turn while both are recorded as
delivered. The agent program receives them all, in order, as state["turn_inputs"]. A tool event carried by an item is
marked consumed_by_agent only after the agent's act returns.

Items (dicts; every item has "kind" and "source"):
    observation          an environment observation that no action produced (the substrate's initial observation):
                         result, observation
    tool_result          the agent's own executed action: tool, args, result, observation
    harness_read_result  a harness read the harness action channel ran in place of an agent turn: tool, args, result,
                         observation, acquired ({tool, args, output}) and requester
    feedback             feedback (the structured dict handed to the agent, or None). A source that also sets what the
                         agent sees as the last result (a host Nudge / Revision / PostRevision, the truncated-call and
                         reconcile-block messages) carries result and observation as well

`result` present on an item means the item sets the last result. A host before_model Observation keeps its contract,
"set what the agent sees next": it replaces the result and observation of the latest result-bearing item of the turn
(or, when the turn has none yet, adds a feedback item with that result and no feedback dict).

The single-value state keys are views of the same inputs, so a turn with one input sees the same value in them as in
its one item:
    last_result / last_observation   the latest result and observation; they persist across turns
    harness_feedback                 the feedback dict of the turn's latest feedback item carrying one, else None
    harness_acquired                 the acquired record of the turn's latest harness read, else None
"""

OBSERVATION = "observation"
TOOL_RESULT = "tool_result"
HARNESS_READ_RESULT = "harness_read_result"
FEEDBACK = "feedback"
KINDS = (OBSERVATION, TOOL_RESULT, HARNESS_READ_RESULT, FEEDBACK)


class TurnInputs(object):
    """The loop's queue of inputs for the agent's next turn."""

    def __init__(self):
        self._items = []       # [item dict, the tool event it carries or None]

    # ---------------------------------------------------------------------------------------------- inputs
    def _add(self, item, event=None):
        self._items.append([item, event])
        return item

    def observation(self, result, observation, source):
        return self._add({"kind": OBSERVATION, "source": source, "result": result, "observation": observation})

    def tool_result(self, action, result, observation=None, source="agent"):
        """The agent's executed action. The observation is set once the loop has built it (set_event)."""
        return self._add({"kind": TOOL_RESULT, "source": source, "tool": (action or {}).get("tool"),
                          "args": (action or {}).get("args") or {}, "result": result, "observation": observation})

    def harness_read_result(self, action, result, observation, requester, event=None):
        """A harness read executed by the harness action channel. `acquired` is set by the loop once computed."""
        return self._add({"kind": HARNESS_READ_RESULT, "source": requester, "tool": (action or {}).get("tool"),
                          "args": (action or {}).get("args") or {}, "result": result, "observation": observation,
                          "acquired": None, "requester": requester}, event)

    def feedback(self, feedback, source, *view):
        """Feedback for the agent. `view` is (result, observation) when the source also sets the last result."""
        item = {"kind": FEEDBACK, "source": source, "feedback": feedback}
        if view:
            item["result"], item["observation"] = view
        return self._add(item)

    def set_event(self, item, event):
        """The tool event an item's delivery record belongs to."""
        for entry in self._items:
            if entry[0] is item:
                entry[1] = event
                return

    def replace_result(self, result, observation, source):
        """A host before_model Observation: what the agent sees next replaces the turn's latest result."""
        for item, _event in reversed(self._items):
            if "result" in item:
                item["result"], item["observation"] = result, observation
                item["replaced_by"] = source
                return item
        return self.feedback(None, source, result, observation)

    # ---------------------------------------------------------------------------------------------- the turn
    def pending(self):
        return bool(self._items)

    def items(self):
        return [item for item, _event in self._items]

    def state(self, last_result, last_observation):
        """The inputs handed to the agent's act: the ordered items plus the single-value views (last_result,
        last_observation, harness_feedback, harness_acquired)."""
        fb = next((i["feedback"] for i, _e in reversed(self._items)
                   if i["kind"] == FEEDBACK and i.get("feedback") is not None), None)
        acq = next((i.get("acquired") for i, _e in reversed(self._items) if i["kind"] == HARNESS_READ_RESULT), None)
        return {"last_observation": last_observation, "last_result": last_result, "harness_feedback": fb,
                "harness_acquired": acq, "turn_inputs": [dict(i) for i, _e in self._items]}

    def consumed(self):
        """The agent's act returned: every tool event handed in this turn was consumed; the queue starts empty."""
        for _item, event in self._items:
            rec = (event or {}).get("delivery_record") if isinstance(event, dict) else None
            if isinstance(rec, dict):
                rec["consumed_by_agent"] = True
        self._items = []
