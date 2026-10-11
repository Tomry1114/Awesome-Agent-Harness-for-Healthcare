# -*- coding: utf-8 -*-
"""The origin model (fixed shell): who did an environment-affecting or scoring-relevant thing.

    origin     "agent" | "harness"
    requester  for origin "harness": the shell path that asked for it, one of REQUESTERS

Tagged where the thing is executed or recorded, never at the caller that asked for it:

    runner/harness/executor.py   build_event                  every executed tool call: an agent turn, a forced read
                                                              (run.py), a recovery action or a content-guard restore
                                                              (runner/harness/action_service.py)
    runner/middleware_host.py    forced_deliverable_write     the after_episode deliverable write
                                 accept_event                 an on_answer Accept (final_answer_author by text identity)
                                 withhold_unresolved_violation  the [WITHHELD] stub final
    runner/run.py                answer_attempt / final_answer events the loop writes (the agent's own answer)
    the kernel                   before_action / after_action of a harness-origin action record action_origin="harness"
                                 and action_requester on its ledger rows and decision events

A forced read carries its requester from the one place that builds it (middleware_host.queued_read for next_read,
harness.actuation.plan for kernel ACQUIREs and on_answer reads) as private keys of the queued action (_requester,
_requester_stage, _requester_capability), read at the execution point.

Trajectories in the legacy format tag every harness tool call origin "recovery" (forced reads included).
Every reader goes through is_harness_event, so both spellings read the same.
"""
AGENT = "agent"
HARNESS = "harness"
ORIGINS = (AGENT, HARNESS)

RECOVERY = "recovery"                     # HAB recovery strategy, through the harness action service
NEXT_READ = "next_read"                   # a workspace's turn-start read (HAB read_acquire.py, "ReadAcquire")
KERNEL_ACQUIRE = "kernel_acquire"         # a kernel ACQUIRE at before_action / after_action / before_final. PB's
#                                           forced broad read is this requester at stage after_action with capability
#                                           fhir_empty_targeted_search_broad_reacquire ("forced_read / broad_reacquire")
ON_ANSWER_READ = "on_answer_read"         # an on_answer Read (MedCTA answer_candidate.py adaptive acquisition)
CONTENT_GUARD = "content_guard"           # the action service writing agent-authored text back after a harness action
FORCED_DELIVERABLE_WRITE = "forced_deliverable_write"   # the after_episode write the harness forces past the budget
CANDIDATE_MERGE = "candidate_merge"       # an on_answer Accept delivering a text that is none of the agent's answers
WITHHOLD = "withhold"                     # the [WITHHELD] stub delivered in place of a refuted answer
REQUESTERS = (RECOVERY, NEXT_READ, KERNEL_ACQUIRE, ON_ANSWER_READ, CONTENT_GUARD, FORCED_DELIVERABLE_WRITE,
              CANDIDATE_MERGE, WITHHOLD)

# final_answer_author: who wrote the text of the delivered final
AUTHOR_AGENT = "agent"
AUTHOR_CANDIDATE_MERGE = "candidate_merge"
AUTHOR_WITHHELD_STUB = "withheld_stub"
AUTHOR_HARNESS = "harness"
AUTHORS = (AUTHOR_AGENT, AUTHOR_CANDIDATE_MERGE, AUTHOR_WITHHELD_STUB, AUTHOR_HARNESS)

LEGACY_HARNESS_ORIGINS = ("recovery",)    # the legacy spelling in older trajectories

# Event types a workspace may not emit through ctx.emit: every type the loop, the scorers, the 7-dim substrate or the
# aggregation reads to decide an outcome, a score, a credit or an episode end.
#   tool_call        scorers (Clinical Resource Competence, policy verifiers), trajectory.log, agent_tool_calls, substrate.map_trace,
#                    the action service's agent-commitment checks
#   final_answer     final_texts, governance / proxy verifiers, substrate, trajectory replays
#   answer_attempt   the final_texts fallback, trajectory replays
#   write_file       note_texts (write_file is a tool; an event of that type would read as one)
#   agent_error      substrate / lifecycle_exec agent-failure attribution, episode_end_reason
#   tool_budget_exhausted, repair_budget_exhausted, circuit_breaker   episode_end_reason (post_stop_allowance)
#   reconciliation, reconcile_resolved, reconcile_enter   commit read-back records (PB trajectory replay)
#   deliverable_budget_warning   substrate.map_trace maps it to a semantic event
RESERVED_EVENT_TYPES = frozenset((
    "tool_call", "final_answer", "answer_attempt", "write_file", "agent_error",
    "tool_budget_exhausted", "repair_budget_exhausted", "circuit_breaker",
    "reconciliation", "reconcile_resolved", "reconcile_enter", "deliverable_budget_warning"))
# harness_* : run.py hands every harness_* event to harness.ledger.governance.summarize (harness metrics);
#             the kernel writes harness_decision, which MedCTA trajectory replays read
# shell_* : the shell's trace: shell_hook (one per hook result the shell applies), shell_refusal (a budget
#           refusal), shell_final_query (a kernel evaluate_final verdict), shell_end_signal (an episode-end tool mapped to
#           the final). Named outside every scoring path above (no scorer, substrate map or aggregation reads them); a
#           workspace may not forge them.
RESERVED_EVENT_PREFIXES = ("harness_", "shell_")
# substrings: dim_lifecycle._trace_truncated reads any event type containing "budget" or "truncat" as a truncated
# episode; substrate.map_trace maps any type containing "escalat" to an escalation
RESERVED_EVENT_SUBSTRINGS = ("budget", "truncat", "escalat")
# provenance is the shell's to write
PROVENANCE_KEYS = ("origin", "requester", "requester_stage", "requester_capability", "action_origin",
                   "action_requester", "final_answer_author", "content_author")


def reserved_event_reason(event_type):
    """Why a workspace may not emit `event_type` (None when it may)."""
    et = str(event_type or "")
    low = et.lower()
    if et in RESERVED_EVENT_TYPES:
        return "reserved event type %r" % et
    for p in RESERVED_EVENT_PREFIXES:
        if low.startswith(p):
            return "reserved event type prefix %r" % p
    for s in RESERVED_EVENT_SUBSTRINGS:
        if s in low:
            return "event type %r contains the reserved substring %r" % (et, s)
    return None


def tag(ev, origin, requester=None, stage=None, capability=None):
    """Stamp origin (and requester for a harness origin) on an event in place; returns the event."""
    if origin not in ORIGINS:
        raise ValueError("origin must be one of %s, not %r" % (ORIGINS, origin))
    if origin == HARNESS:
        if requester not in REQUESTERS:
            raise ValueError("a harness-origin event needs a requester in %s, not %r" % (REQUESTERS, requester))
        ev["requester"] = requester
        if stage is not None:
            ev["requester_stage"] = stage
        if capability is not None:
            ev["requester_capability"] = capability
    elif requester is not None:
        raise ValueError("an agent-origin event has no requester (%r)" % (requester,))
    ev["origin"] = origin
    return ev


def queued_requester(action):
    """(requester, stage, capability) a queued forced action carries; (None, None, None) for an agent action."""
    if not isinstance(action, dict) or not action.get("_harness_acquire"):
        return None, None, None
    return action.get("_requester"), action.get("_requester_stage"), action.get("_requester_capability")


def kernel_kwargs(action):
    """The kernel before_action / after_action keywords for the action about to run: none for the agent's own action,
    origin + requester for a harness-dispatched one."""
    req, _stage, _cap = queued_requester(action)
    if req is None and not (isinstance(action, dict) and action.get("_harness_acquire")):
        return {}
    return {"origin": HARNESS, "requester": req}


def is_harness_event(ev):
    """A harness-origin event, in either spelling (origin harness, or the legacy origin "recovery" / audience harness)."""
    if not isinstance(ev, dict):
        return False
    o = ev.get("origin")
    return o == HARNESS or o in LEGACY_HARNESS_ORIGINS or ev.get("audience") == HARNESS


def requester_of(ev):
    """The requester of a harness-origin event; legacy tool calls without one read "legacy_recovery_origin"."""
    if not is_harness_event(ev):
        return None
    return ev.get("requester") or "legacy_recovery_origin"


def is_agent_tool_call(ev):
    return isinstance(ev, dict) and ev.get("event_type") == "tool_call" and not is_harness_event(ev)


def is_harness_tool_call(ev):
    return isinstance(ev, dict) and ev.get("event_type") == "tool_call" and is_harness_event(ev)
