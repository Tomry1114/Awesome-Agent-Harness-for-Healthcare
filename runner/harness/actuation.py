"""ONE actuation layer for the harness decisions that act: ACQUIRE, REVISE, DEFER_FINAL.

ACQUIRE gets an execution entry shared by every hook; REVISE and DEFER_FINAL get the delivery record that
makes their non-executing effect measurable. See the two section headers below.

--------------------------------------------------------------------------------------------------
ACQUIRE

ACQUIRE means exactly one thing, everywhere:

    "run a read-only tool for me and put the result in the normal evidence/observation path"

That definition is hook-independent. The hook only records WHEN the request was formed
(before_action / after_action / before_final); it must not decide WHETHER the request has an actuator.
Every call site (run.py before_final, run.py before_action, answer_surface.py's adaptive round,
run_driver.acquire, after_action) builds its forced action here, so an effective ACQUIRE at any hook produces
a tool call rather than a decision that, in the artifact, is indistinguishable from a capability that never fired.

What this module owns:
  * reading next_action off a decision (HarnessResult or raw HarnessDecision)
  * the admission gate: a tool name, read_only, and the existing acquisition budget
  * building the forced action that the NORMAL tool pipeline consumes
  * telemetry, at the granularity attribution needs

What it does NOT own: the per-hook feedback text and the per-hook loop control (deferring a commit,
holding an artifact baseline, seeding a comparison). Those are genuinely stage-specific and stay in run.py.

BUDGET. The acquisition budget is `ledger.acquire_count` -- the counter three built-in capabilities already
gate on (>=2, >=3) -- plus max_steps, since every acquisition costs a loop iteration. It is deliberately NOT
the env-action budget: run.py exempts _harness_acquire there so a harness read can never consume the agent's
own tool allowance.

TELEMETRY. effective != executed, so the record separates:
    requested      a hook produced an ACQUIRE decision
    attempted      it passed admission and was handed to the tool pipeline
    executed       the tool actually ran
    succeeded      the tool returned without error
    evidence_added it raised the validated-evidence high-water mark
plus failure_reason at whichever step it stopped. Reading only 'fired' collapses five different outcomes
into one number, and attribution cannot survive that.
"""


# Stopped-here reasons. Named, so that "it did not fire" does not cover all of them.
NO_NEXT_ACTION = "no_next_action"
NO_TOOL = "next_action_has_no_tool"
NOT_READ_ONLY = "next_action_not_read_only"
# The tool resolved, but this substrate does not expose it. Refusing here keeps the dispatch from dying
# inside the executor -- where it would read as a tool error, i.e. as the environment's fault, rather than
# as a resolution that could never have run.
NOT_IN_VOCABULARY = "tool_not_in_substrate_vocabulary"

TELEMETRY_FIELDS = ("requested", "attempted", "executed", "succeeded", "evidence_added")


class AcquirePlan:
    __slots__ = ("dispatchable", "forced_action", "events", "acquire_id",
                 "tool", "args", "failure_reason", "stage")

    def __init__(self, dispatchable, forced_action, events, acquire_id, tool, args, failure_reason, stage):
        self.dispatchable = dispatchable
        self.forced_action = forced_action
        self.events = events
        self.acquire_id = acquire_id
        self.tool = tool
        self.args = args
        self.failure_reason = failure_reason
        self.stage = stage


def next_action_of(decision):
    """The next_action carried by a HarnessResult or a raw HarnessDecision. {} when there is none.

    Both shapes appear at call sites: run.py holds HarnessResults (.raw.extra), the recovery path holds
    decisions (.extra). One reader, so a shape change cannot silently blank one hook and not another."""
    if decision is None:
        return {}
    extra = getattr(decision, "extra", None)
    if not extra:
        raw = getattr(decision, "raw", None)
        extra = getattr(raw, "extra", None) if raw is not None else None
    na = (extra or {}).get("next_action") if isinstance(extra, dict) else None
    return na if isinstance(na, dict) else {}


def capability_of(decision):
    for obj in (decision, getattr(decision, "raw", None)):
        if obj is None:
            continue
        cap = getattr(obj, "capability", None)
        if cap:
            return cap
    return None


def plan(decision, stage, step, ledger=None, vocabulary=None):
    """An ACQUIRE decision -> the plan to actuate it. Never raises; a refusal is a recorded result.

    The caller sets `_forced_action = plan.forced_action` when plan.dispatchable; the loop then runs it
    through the SAME tool pipeline as an agent action (real tool_call event, error normalization,
    read-back, after_action, provenance, last_obs update)."""
    tool = args = None
    reason = None
    na = next_action_of(decision)
    if not na:
        reason = NO_NEXT_ACTION
    else:
        tool = na.get("tool")
        args = na.get("args") or {}
        if not tool:
            reason = NO_TOOL
        elif not na.get("read_only"):
            reason = NOT_READ_ONLY

    spent = int(getattr(ledger, "acquire_count", 0) or 0) if ledger is not None else 0

    acquire_id = "acq-%s-%s-%s" % (stage, step, spent)
    in_vocab = None
    if tool and vocabulary:
        names = {(t.get("name") if isinstance(t, dict) else t) for t in vocabulary}
        in_vocab = tool in names
    # Only when the vocabulary is KNOWN and the tool is absent from it. `None` means the caller passed no
    # vocabulary, which is not evidence of anything and must not become a refusal.
    if reason is None and in_vocab is False:
        reason = NOT_IN_VOCABULARY

    rec = {"acquire_id": acquire_id, "stage": stage, "step": step, "tool": tool,
           "capability": capability_of(decision), "in_vocabulary": in_vocab,
           "requested": True, "attempted": reason is None, "executed": False,
           "succeeded": None, "evidence_added": None, "failure_reason": reason}
    _append(ledger, rec)

    if reason is not None:
        # A refused ACQUIRE is an EVENT, not a silent fall-through, so "no actuator" and "unroutable request"
        # read differently in a trajectory.
        return AcquirePlan(False, None, [{"step": step, "event_type": "harness_acquire_refused",
                                          "stage": stage, "tool": tool, "reason": reason,
                                          "acquire_id": acquire_id, "status": "error"}],
                           acquire_id, tool, args, reason, stage)

    if ledger is not None:
        try:
            ledger.acquire_count = spent + 1
        except Exception:
            pass

    forced = {"type": "tool_call", "tool": tool, "args": args, "_harness_acquire": True,
              "_acquire_id": acquire_id,
              # who asked (runner/harness/origin.py), read where the read executes (executor.build_event, the kernel)
              "_requester": "on_answer_read" if stage == "before_final_readapt" else "kernel_acquire",
              "_requester_stage": stage, "_requester_capability": capability_of(decision),
              "_valid_before": getattr(ledger, "validated_evidence_version", 0) if ledger is not None else 0}
    return AcquirePlan(True, forced,
                       [{"step": step, "event_type": "harness_acquire_predispatch", "stage": stage,
                         "tool": tool, "args": args, "acquire_id": acquire_id, "status": "ok"}],
                       acquire_id, tool, args, None, stage)


def record_execution(ledger, acquire_id, executed, succeeded=None, evidence_added=None, failure_reason=None):
    """Close the loop on a dispatched acquisition, from the point where the tool result is known."""
    rec = _find(ledger, acquire_id)
    if rec is None:
        rec = {"acquire_id": acquire_id, "stage": None, "step": None, "tool": None, "capability": None,
               "in_vocabulary": None, "requested": True, "attempted": True, "failure_reason": None}
        _append(ledger, rec)
    rec["executed"] = bool(executed)
    rec["succeeded"] = succeeded
    rec["evidence_added"] = evidence_added
    if failure_reason:
        rec["failure_reason"] = failure_reason
    elif executed and succeeded is False:
        rec["failure_reason"] = "tool_error"
    elif executed and succeeded and evidence_added is False:
        rec["failure_reason"] = "no_new_evidence"
    return rec


def records(ledger):
    return list(getattr(ledger, "acquire_telemetry", None) or [])


def summary(ledger):
    """The five counters + the reasons, so a report can say WHERE the chain stopped instead of "not fired"."""
    recs = records(ledger)
    out = {"acquire_%s" % f: sum(1 for r in recs if r.get(f) is True) for f in TELEMETRY_FIELDS}
    reasons = {}
    for r in recs:
        fr = r.get("failure_reason")
        if fr:
            reasons[fr] = reasons.get(fr, 0) + 1
    out["failure_reasons"] = reasons
    out["by_stage"] = {}
    for r in recs:
        s = r.get("stage") or "unknown"
        b = out["by_stage"].setdefault(s, {"requested": 0, "attempted": 0, "executed": 0})
        for f in ("requested", "attempted", "executed"):
            if r.get(f) is True:
                b[f] += 1
    return out


def _append(ledger, rec):
    if ledger is None:
        return
    try:
        if getattr(ledger, "acquire_telemetry", None) is None:
            ledger.acquire_telemetry = []
        ledger.acquire_telemetry.append(rec)
    except Exception:
        pass


def _find(ledger, acquire_id):
    if ledger is None or not acquire_id:
        return None
    for r in records(ledger):
        if r.get("acquire_id") == acquire_id:
            return r
    return None


# ==================================================================================================
# REVISE and DEFER_FINAL: the two primitives that hand control back to the agent.
#
# Neither executes anything itself, so "did it fire" says almost nothing about whether it did its job.
# What matters is whether the agent was actually given the instruction AND a turn in which to act on it.
# Splicing "[HARNESS] ..." into the observation string and setting feedback, with no repair turn, no bound,
# and no guarantee the episode has a turn left, is not recorded delivery. A primitive whose delivery cannot be
# distinguished from its non-delivery cannot be attributed.
# ==================================================================================================

REVISION_FIELDS = ("revision_delivered", "repair_turn_started")
DEFER_FIELDS = ("final_deferred", "repair_turn_started")


def record_revision(ledger, stage, step, capability=None, delivered=True,
                    repair_turn_started=False, repair_turn=None, suppressed_reason=None, source=None):
    """One repair-type result (a kernel REVISE / BLOCK, a host Nudge / Revision / PostRevision / Repair).
    Same shape at every hook, so they can be totalled against each other; `source` names the result
    ("kernel:REVISE", "host:Nudge", ...)."""
    rec = {"kind": "REVISE", "stage": stage, "step": step, "capability": capability,
           "revision_delivered": bool(delivered), "repair_turn_started": bool(repair_turn_started),
           "repair_turn": repair_turn, "suppressed_reason": suppressed_reason, "source": source}
    _append_act(ledger, rec)
    return rec


def record_defer(ledger, stage, step, capability=None, deferred=True,
                 repair_turn_started=False, repair_turn=None, control_only=None,
                 instruction_supplied=None, exhausted=False):
    """One DEFER_FINAL gate. control_only marks a deferral that demanded no specific repair."""
    rec = {"kind": "DEFER_FINAL", "stage": stage, "step": step, "capability": capability,
           "final_deferred": bool(deferred), "repair_turn_started": bool(repair_turn_started),
           "repair_turn": repair_turn, "control_only": control_only,
           "instruction_supplied": instruction_supplied, "budget_exhausted": bool(exhausted)}
    _append_act(ledger, rec)
    return rec


def actuations(ledger):
    return list(getattr(ledger, "actuation_telemetry", None) or [])


def actuation_summary(ledger):
    """Per-primitive counters over one episode. ACQUIRE's own five-rung chain comes from summary()."""
    recs = actuations(ledger)
    out = {"revise_delivered": sum(1 for r in recs if r.get("kind") == "REVISE" and r.get("revision_delivered")),
           "revise_repair_turns": sum(1 for r in recs if r.get("kind") == "REVISE" and r.get("repair_turn_started")),
           "revise_suppressed": sum(1 for r in recs if r.get("kind") == "REVISE" and r.get("suppressed_reason")),
           "defer_final_gated": sum(1 for r in recs if r.get("kind") == "DEFER_FINAL" and r.get("final_deferred")),
           "defer_final_repair_turns": sum(1 for r in recs if r.get("kind") == "DEFER_FINAL" and r.get("repair_turn_started")),
           "defer_final_exhausted": sum(1 for r in recs if r.get("kind") == "DEFER_FINAL" and r.get("budget_exhausted"))}
    by_stage = {}
    for r in recs:
        b = by_stage.setdefault(r.get("stage") or "unknown", {})
        b[r.get("kind")] = b.get(r.get("kind"), 0) + 1
    out["by_stage"] = by_stage
    return out


def _owner(rec):
    """The capability that owns a record. A record no capability raised (a workspace read, a harness action service
    read) has capability None; its key is the string "null", which is how JSON writes a None key anyway, so a breakdown
    that holds both kinds stays sortable and the written artifact is unchanged."""
    cap = rec.get("capability")
    return "null" if cap is None else cap


def executed_by_capability(ledger):
    """WHO executed what this episode -> {capability: n}. The breakdown, not just the total.

    Without it the total is unattributable after the fact: a count of executions says nothing about which
    capability's interventions they were."""
    out = {}
    for r in records(ledger):
        if r.get("executed") is True:
            k = _owner(r)
            out[k] = out.get(k, 0) + 1
    for r in actuations(ledger):
        k = _owner(r)
        if r.get("kind") == "REVISE" and r.get("revision_delivered") and r.get("repair_turn_started"):
            out[k] = out.get(k, 0) + 1
        elif r.get("kind") == "DEFER_FINAL" and r.get("final_deferred") and r.get("repair_turn_started"):
            out[k] = out.get(k, 0) + 1
    return out


def n_intervention_executed(ledger, capability=None):
    """How many times this episode an intervention ACTUALLY took effect. ONE definition, here, in the module
    that owns the telemetry -- so every reader of the artifact gets the same number.

    Per primitive, what "executed" means and why:

      ACQUIRE      the acquisition entered the executor. `requested` and `attempted` are earlier rungs and
                   neither of them touched the environment.
      REVISE       the instruction was delivered AND a repair turn actually started. Delivery alone is a
                   string in an observation; without the turn the agent never gets the chance to act on it
                   (REVISE is a repair turn, not a suffix).
      DEFER_FINAL  the final was refused AND the agent got another turn. On budget exhaustion the answer is
                   DELIVERED instead, so nothing was deferred and it does not count.

    This is deliberately NOT a quality measure. It answers "did an intervention occur", nothing else.

    `capability` RESTRICTS the count to one owner. Unfiltered, it is the TOTAL over every capability, which
    is the number for "what did this episode do" and not the number for "what did this capability do"."""
    by = executed_by_capability(ledger)
    if capability is not None:
        return int(by.get(capability, 0))
    return sum(by.values())


def episode_summary(ledger):
    """Everything ACQUIRE / REVISE / DEFER_FINAL did this episode, in one dict, for the run artifact.

    The two by_stage breakdowns are kept under distinct keys: a flat update() would drop the ACQUIRE
    one, which is the half that answers "did the acquisition reach a hook with an actuator"."""
    acq = dict(summary(ledger))
    act = dict(actuation_summary(ledger))
    out = {k: v for k, v in acq.items() if k != "by_stage"}
    out.update({k: v for k, v in act.items() if k != "by_stage"})
    out["by_stage_acquire"] = acq.get("by_stage") or {}
    out["by_stage_revise_defer"] = act.get("by_stage") or {}
    # Present on EVERY episode, including one where nothing happened -- a real 0 and a missing key are
    # different findings. The TOTAL answers "what did this episode do"; the per-capability breakdown beside
    # it answers "whose was it".
    out["n_intervention_executed"] = n_intervention_executed(ledger)
    out["n_intervention_executed_by_capability"] = executed_by_capability(ledger)
    # RAW records, so attribution is possible after the fact from the artifact alone. Counters could not
    # answer "whose was it", and `ledger.interventions` -- the only other list on disk -- is kept for a
    # different purpose and lacks the executed / repair_turn_started distinctions this definition uses.
    out["records"] = {"acquire": records(ledger), "actuation": actuations(ledger)}
    return out


def _append_act(ledger, rec):
    if ledger is None:
        return
    try:
        if getattr(ledger, "actuation_telemetry", None) is None:
            ledger.actuation_telemetry = []
        ledger.actuation_telemetry.append(rec)
    except Exception:
        pass
