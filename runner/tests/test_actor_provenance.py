"""A harness-dispatched read-acquire must NOT be scored as agent behaviour.

executor.build_event() defaults origin/audience to "agent"/"agent", so run.py must override them for a
harness-forced acquisition; otherwise the call lands in the trajectory tagged as an agent tool call.
substrate.map_trace derives `actor` from exactly those two fields, so without the tag the actor_scope="agent"
filters inside dim_tooling / dim_lifecycle / dim_observability have nothing to filter, and a harness capability
is charged, as agent waste, for the very evidence it forced. The test checks the tag at emission, the actor that
map_trace resolves from it, and that an untagged event would still read as the agent's.
"""
import os, sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "runner"))
sys.path.insert(0, os.path.join(_REPO, "runner", "scorers"))
os.chdir(_REPO)
import substrate as SUB
from harness.executor import ActionExecutor


class _Outcome(object):
    def __init__(self):
        self.res = {"total": 3, "entries": [1, 2, 3]}
        self.err = None
        self.state_before = "a"
        self.state_after = "a"


def _ev_for(action):
    ex = ActionExecutor.__new__(ActionExecutor)                      # no env needed: build_event only canonicalises
    ex.env_type = "fhir"
    ex._canon = type("C", (), {"canonical_action": staticmethod(lambda a, t: {"semantic_type": "read"}),
                               "canonical_result": staticmethod(lambda r: {"ok": True}),
                               "canonical_observation": staticmethod(lambda r, t: {"n": 3})})()
    hacq = bool(action.get("_harness_acquire"))
    return ex.build_event(action, _Outcome(), 1,
                          origin=("harness" if hacq else "agent"),
                          audience=("both" if hacq else "agent"))[0]


AGENT = {"type": "tool_call", "tool": "fhir_observation_search_labs", "args": {"patient": "P1"}}
FORCED = dict(AGENT, _harness_acquire=True, _requester="kernel_acquire", _requester_stage="after_action")

ev_a, ev_h = _ev_for(AGENT), _ev_for(FORCED)
assert ev_a["origin"] == "agent" and ev_a["audience"] == "agent", ev_a
assert ev_h["origin"] == "harness", "harness-dispatched call must be tagged at emission: %r" % ev_h.get("origin")
assert (ev_h["requester"], ev_h["requester_stage"]) == ("kernel_acquire", "after_action"), ev_h
assert "requester" not in ev_a, ev_a
assert ev_h["audience"] == "both", "the acquired result IS handed to the agent next turn"
assert ev_h["delivery_record"]["rendered_to_agent"] is True, "audience=both must keep the delivery record true"
print("1 emission: agent call origin=agent; forced acquire origin=harness (requester kernel_acquire), audience=both")

# --- the consequence: substrate must resolve the forced call to actor=harness, and the agent scope drop it
plugin = {"tool_semantics": {"fhir_observation_search_labs": {"event_role": "acquire"}}}
sem = SUB.map_trace([ev_a, ev_h], plugin)
actors = [s.get("actor") for s in sem]
assert actors == ["agent", "harness"], "map_trace must read the tags: %r" % (actors,)
agent_scope = [s for s in sem if s.get("actor", "agent") != "harness"]
assert len(agent_scope) == 1, "actor_scope='agent' must now exclude the harness-forced call"
print("2 substrate: actors=%r -> agent scope keeps %d of %d calls" % (actors, len(agent_scope), len(sem)))

# --- control: WITHOUT the tag both calls look like the agent's (the behaviour the tag prevents)
untagged = dict(ev_h, origin="agent", audience="agent")
sem_old = SUB.map_trace([ev_a, untagged], plugin)
assert [s.get("actor") for s in sem_old] == ["agent", "agent"]
assert len([s for s in sem_old if s.get("actor", "agent") != "harness"]) == 2, \
    "this is the defect: the harness's own read counted as agent behaviour"
print("3 untagged control: both read as agent -- the exact defect, still reproducible if the tag is dropped")

# --- the alternative harness origin spelling "recovery" (found in stored trajectories) also reads as harness
legacy = dict(ev_h, origin="recovery")
legacy.pop("requester")
assert [s.get("actor") for s in SUB.map_trace([ev_a, legacy], plugin)] == ["agent", "harness"]

print("ALL ACTOR-PROVENANCE ASSERTIONS PASSED")
