# -*- coding: utf-8 -*-
"""Middleware host: the shell side of a workspace's middleware.

A workspace's middleware decides; the shell executes. Each hook receives a MiddlewareContext (read-only state
plus shell services) and returns a value whose effect on the loop the shell defines, identically for every
workspace. This file is the contract.

Hooks a middleware class may define (all optional; the class is constructed once per task with the task):
    before_model(ctx)               -> Observation | None   set what the agent sees next; the loop goes on
    next_read(ctx)                  -> Read | None          turn start, no harness read queued: the shell runs this
                                                            read in place of an agent turn
    before_final(ctx, answer)       -> Revision | Nudge | None   the final is not accepted; the agent gets a turn
    on_answer(ctx, answer)          -> Accept | Read | Repair | None
                                                            a final that reached the attempt point (answer_attempt
                                                            recorded), before the kernel's before_final
    after_final_verdict(ctx, answer, verdict) -> str | None the kernel's before_final did not ALLOW the final; its
                                                            events are recorded and the shell has not yet acted on
                                                            it. The text is what the agent is told for an ACQUIRE
    before_tool(ctx, action)        -> Revision | Nudge | None   the action does not run; the agent gets a turn
    after_tool(ctx, action, result) -> PostRevision | None  the action ran; last_res/last_obs are its result
Every input a result hands the agent (a feedback, a result) is one item of the agent's next turn, after the inputs already
queued for it, never in place of them (runner/turn_inputs.py).
    exempt_from_budget(ctx, action) -> bool                 this action may run after the tool budget is spent
                                                            (the shell still allows one such action per episode)
    after_episode(ctx)              -> None                 may ask the shell for a forced deliverable write, or take the
                                                            post-stop allowance and act through the harness action
                                                            service
    deferred_errors()               -> list[str]            judge outages to report as runtime errors

Order: the manifest's order; at each hook the first non-None result wins. What each result does is in its class.

Shell limits on what a result does:
    * a Revision (before_final, before_tool) or an on_answer Repair is applied at most MAX_REVISIONS_PER_EPISODE times
      per episode, together; past that the shell records harness_revision_refused and goes on as if the hook had returned
      None
    * every repair-type result (Nudge, Revision, PostRevision, Repair) is recorded the same way: one actuation
      record_revision naming its source. Revision, PostRevision and Repair also draw one repair turn from the
      task's single repair budget (the kernel's REVISE / BLOCK draw the same) and unlock repair (ctx.in_repair). A Nudge
      is workspace advice: it draws no repair turn and leaves ctx.in_repair unchanged; its only bound is its own cap.
      Only the kernel's verdicts and the shell's own refusals end or flag an episode when the repair budget is spent; a
      host result past it is still applied (bounded by its own cap) and recorded with repair_turn_started False
    * a Nudge is applied at most MAX_NUDGES_PER_EPISODE times per episode (before_final and before_tool together); past
      that the shell records harness_nudge_refused and goes on as if the hook had returned None
    * every environment action a result or a service asks for is a harness action: it goes through the harness action
      channel (runner/harness/action_service.py) and is counted there, never in the agent's budget
    * host before_tool / after_tool see the agent's own actions only; a harness read the loop dispatches does not reach them
"""
import collections
import copy
import json
import weakref


class Observation(object):
    __slots__ = ("last_res", "last_obs")

    def __init__(self, last_res, last_obs):
        self.last_res, self.last_obs = last_res, last_obs


class Revision(object):
    """The shell turns this into: last_res={"feedback": feedback}, last_obs=obs, a REVISE feedback for the agent
    tagged with `stage`, one repair-type accounting (a repair turn, the repair budget unlocked, a record_revision), and a
    new agent turn."""
    __slots__ = ("stage", "feedback", "obs")

    def __init__(self, stage, feedback, obs):
        self.stage, self.feedback, self.obs = stage, feedback, obs


class Nudge(object):
    """before_final / before_tool. The final or the action is refused: last_res={"feedback": feedback},
    last_obs=obs, feedback {"decision": label, "stage": stage, "reason": feedback}, and a new agent turn. Recorded as
    every repair-type result (one record_revision), but unlike Revision the repair budget is NOT unlocked
    (ctx.in_repair unchanged) and no repair turn is drawn: its only bound is MAX_NUDGES_PER_EPISODE. `label` is the
    decision word the agent's rendering shows (CONTINUE, REVISE)."""
    __slots__ = ("stage", "feedback", "obs", "label")

    def __init__(self, stage, feedback, obs, label):
        self.stage, self.feedback, self.obs, self.label = stage, feedback, obs, label


class PostRevision(object):
    """after_tool. The action already ran. last_res={"feedback": feedback}, last_obs=obs and the feedback
    {"decision": "REVISE", "stage": stage, "reason": feedback} are handed to the agent after the action's own result (both
    are inputs of its next turn; the last-result view is this one); one repair-type accounting (a repair turn, the repair
    budget unlocked, a record_revision). The iteration finishes normally: no skipped step."""
    __slots__ = ("stage", "feedback", "obs")

    def __init__(self, stage, feedback, obs):
        self.stage, self.feedback, self.obs = stage, feedback, obs


class Read(object):
    """A read the shell runs as a harness action (`_harness_acquire`) through the normal tool pipeline: kernel
    before_action and after_action, exemption from the budget refusal, provenance origin=harness with its requester
    (next_read, on_answer_read; runner/harness/origin.py) and audience=both, the result handed to the agent next turn as a
    harness_read_result input (the harness_acquired view), record_execution.

    next_read: queued as {"type": "tool_call", "tool": tool, "args": args, "_harness_acquire": True}, and the
        loop records harness_acquire_dispatch. There is no actuation.plan call (as on the HAB read-acquire path).
        `feedback` and `source` are not used.
    on_answer: planned by actuation.plan at stage before_final_readapt with no vocabulary, and queued if the plan
        is dispatchable. `feedback` (a structured feedback dict, or None) becomes the pending feedback; no repair
        turn is counted. `source` is the kernel verdict that asked for the read (a ctx.verify_final result): its
        next_action supplies read_only and its capability owns the telemetry record, exactly as when the verdict
        itself is planned. Without a source the read is admitted as read_only with no capability. The plan is
        made when the shell applies the result, or earlier when the middleware calls ctx.plan_read(read) to learn
        whether it is dispatchable; its events land in the trajectory when it is made.
    `args` is used by reference, as the kernel's next_action args are."""
    __slots__ = ("tool", "args", "feedback", "source", "plan")

    def __init__(self, tool, args=None, feedback=None, source=None):
        self.tool, self.args, self.feedback, self.source = tool, args, feedback, source
        self.plan = None   # the actuation.AcquirePlan, once the shell has planned it (on_answer only)


class Repair(object):
    """on_answer. The final is not accepted. Accounted as every repair-type result (a repair turn, the repair budget
    unlocked, a record_revision), and `feedback` (a structured feedback dict; None adds no feedback) goes to the
    agent."""
    __slots__ = ("feedback",)

    def __init__(self, feedback):
        self.feedback = feedback


class Accept(object):
    """on_answer. The episode ends with `text` delivered, unless the text's own verdict forbids it.

    The shell first takes and records the verdict of `text` (delivered_verdict: the last verdict ctx.verify_final
    returned for exactly this text during this on_answer call, else the kernel's evaluation of it; recorded as the real
    before_final verdict). A candidate finalize is repair exhaustion, so when that verdict is a must_resolve violation
    (must_resolve_violation) the shell withholds the text exactly as the normal before_final path does on exhaustion
    (withhold_unresolved_violation, including its flagged delivery when the deliverable is environment state), and
    disposition, intervention_type, reverified and flag are not used. Nothing on an Accept turns this off. If taking the
    verdict raises, that is a before_final hook error: harness_escalation hook_error, the episode aborts.
    Otherwise the shell records the final_answer event {thought: text, final_disposition: disposition,
    intervention_type, reverified, answer_attempt_index, canonical_action of {"type": "final", "answer": text}} and,
    when delivery_flag gives a flag (`flag`, or the verdict is not ALLOW), kernel.record_flagged_final(text, flag): one
    final-answer commit per delivery, clean or flagged."""
    __slots__ = ("text", "disposition", "intervention_type", "reverified", "flag")

    def __init__(self, text, disposition, intervention_type=None, reverified=None, flag=None):
        self.text, self.disposition, self.intervention_type = text, disposition, intervention_type
        self.reverified, self.flag = reverified, flag


Answer = collections.namedtuple("Answer", "index hash text")   # a final that reached the attempt point

# A Nudge refuses a final or an action without counting a repair turn, so nothing else bounds how many turns a
# workspace can take from the agent with it. The shell applies at most this many per episode. The default (H0)
# workspaces' own per-middleware Nudge bounds sum to 9 (obligation 3 + grounding 2 + collection 2 + attachment 2), so
# they never reach the cap.
MAX_NUDGES_PER_EPISODE = 10

# A host Revision (before_final, before_tool) or an on_answer Repair refuses a final or an action and gives the
# agent a turn; without this cap only the loop's iteration bound would limit how many (HAB precommit_terminal has no
# bound of its own). The shell applies at most this many per episode, together. The bounded Revision sources of the
# default (H0) PhysicianBench workspace sum to 15 (pre-final 3, freshness 2, specificity 2, commitment ledger 6,
# retention 2), so it never reaches the cap.
MAX_REVISIONS_PER_EPISODE = 16


# ---- read-only views
# A view object holds nothing that leads back to the shell: what it reads is kept in _BACKING, keyed by the view, which
# only this module reads. Every value handed out is a copy.
_BACKING = weakref.WeakKeyDictionary()


def readonly_copy(obj):
    """A copy of JSON-shaped shell data for workspace code: plain dicts, lists and tuples are rebuilt, str / int / float /
    bool / None are shared (immutable), anything else is copy.deepcopy'd. Faster than deepcopy on trajectories."""
    t = type(obj)
    if t is dict:
        return {k: readonly_copy(v) for k, v in obj.items()}
    if t is list:
        return [readonly_copy(v) for v in obj]
    if t is tuple:
        return tuple(readonly_copy(v) for v in obj)
    if obj is None or t in (str, int, float, bool):
        return obj
    return copy.deepcopy(obj)


class _View(object):
    __slots__ = ("__weakref__",)

    def __setattr__(self, name, value):
        raise AttributeError("%s is read-only to workspace code (tried to set %s)" % (type(self).__name__, name))

    def __delattr__(self, name):
        raise AttributeError("%s is read-only to workspace code (tried to delete %s)" % (type(self).__name__, name))


def _backing(view):
    return _BACKING[view]


class LedgerView(_View):
    """The kernel ledger as host middleware may read it: copies of a fixed set of fields, no writes."""
    FIELDS = ("evidence", "validated_evidence_version", "observations", "acquire_count", "successful_acquisitions",
              "acquisition_attempts")
    __slots__ = ()

    def __init__(self, ledger):
        _BACKING[self] = ledger

    def __getattr__(self, name):
        if name not in LedgerView.FIELDS:
            raise AttributeError("the ledger view has no field %r" % (name,))
        return copy.deepcopy(getattr(_backing(self), name, None))


class TrajectoryView(_View):
    """The run's trajectory as workspace code reads it: a live sequence (it sees events appended after it was taken)
    whose items are copies; no append, no item assignment. Workspace code records through ctx.emit."""
    __slots__ = ()

    def __init__(self, trajectory):
        _BACKING[self] = trajectory

    def __len__(self):
        return len(_backing(self))

    def __bool__(self):
        return bool(_backing(self))

    def __getitem__(self, index):
        items = _backing(self)
        if isinstance(index, slice):
            return [readonly_copy(e) for e in items[index]]
        return readonly_copy(items[index])

    def __iter__(self):
        items = _backing(self)
        for i in range(len(items)):   # live: an event appended during the iteration is reached, as with the list
            yield readonly_copy(items[i])

    def __reversed__(self):
        items = _backing(self)
        for i in range(len(items) - 1, -1, -1):
            yield readonly_copy(items[i])

    def __contains__(self, item):
        return any(e == item for e in _backing(self))

    def index(self, item):
        for i, e in enumerate(_backing(self)):
            if e == item:
                return i
        raise ValueError("event not in the trajectory")

    def count(self, item):
        return sum(1 for e in _backing(self) if e == item)


class _ContextState(object):
    """What a MiddlewareContext is built from. Shell-only: workspace code holds the context, never this."""
    __slots__ = ("env", "task", "trajectory", "step", "max_steps", "used_actions", "judge_factory", "errors", "forced",
                 "shell", "verify_calls", "verdicts", "action_step", "opened", "consumer")


class MiddlewareContext(_View):
    """What a hook can read and the services it can use. Nothing on it leads back to the shell's objects:
    the environment, the full task, the trajectory list, the ledger and the judge stay in the shell; workspace code
    gets copies, read-only views and the narrow services below.

    Fixed at construction: step, max_steps and `used_actions` (agent + harness env actions when the context was
    built; the default deliverable middleware reads it with that meaning).

        task                        the PUBLIC task (runner/workspace.py public_task), a copy: goal, context,
                                    available_tools, environment type and public config. No checkpoints, reference,
                                    expected outcome, scoring, policy, native_config or task_type
        trajectory                  TrajectoryView: the run's trajectory, live, items copied; no writes (emit records)

    There is no `env`. Read-only views of the environment, provided by the shell, each a copy and none able to act:
        controls()                  the live control model of the current page ([{ref, role, label, name, required,
                                    value, ...}]; [] where the environment has none; a failed read yields what the
                                    environment's reader yields -- [] from the harness action service's, which run.py
                                    provides)
        persisted_state()           the environment's persisted state as its obligation / evaluation view
                                    (env.eval_state(): the portals' localStorage state), or None where the environment
                                    has none. Exceptions of the read propagate
        downloads()                 the file names (basenames) the environment downloaded this episode, in order
        output_files()              the agent's output directory: None where the environment has none, else
                                    {"exists": the directory exists, "files": {name: {"size": bytes, "text": content}}}
                                    for its regular files, in directory-listing (glob) order

    Read-only, read from the shell when accessed (None where the shell did not provide them, e.g. a context built
    in a unit test):
        observed_history            every non-empty last_obs at the top of a turn, str(last_obs)[:4000], in order
        last_obs                    what the agent will see next
        budget_spent_at_turn_start  the agent's budget was spent at the top of this turn: agent_env_actions >= max_steps
                                    (harness actions are never charged to it)
        live_used_actions           agent_env_actions + harness_env_actions now (every env action of the episode)
        agent_env_actions           the agent's own executed env actions
        harness_env_actions         harness env actions the harness action channel executed, every requester
        recovery_env_actions        of which requested through harness-action invocations (recovery), restores included
        recovery_reserve            the extra actions reserved for recovery
        in_repair                   the repair budget is unlocked
        forced_pending              a harness read is queued for the next iteration
        answers                     tuple of Answer(index, hash, text): finals that reached the attempt point
        end_reason                  after_episode only: why the loop stopped (episode_end_reason)
        ledger                      LedgerView of the kernel ledger, or None without a kernel
        contract_meta               a copy of the kernel contract's meta (goal, goal_spec, ...), or None

    Services:
        judge                       the shell's budgeted judge (or None), fetched on access: one unit of the
                                    general semantic pool per call
        error(message)              record a runtime error the run reports
        emit(event)                 append an event to the trajectory, stamped with this context's step (with the
                                    step of the harness-action invocation this context opened, when it opened one)
        record_budget_warning()     the one reserved record a workspace may ask for: the deliverable budget warning
                                    (event deliverable_budget_warning, which the 7-dim substrate reads), stamped by the
                                    shell with this context's step and remaining = max_steps - used_actions; at most
                                    once per episode -> whether it was recorded
        obligations()               (met, decisions, conjuncts) from the shell's obligation model: the task's
                                    completion conjunction, composed from the public goal and the agent's own
                                    decisions in the trajectory, over one state read (decisions: the ids of the
                                    decisions the agent took). (True, None, None) where the benchmark declares no
                                    obligation model
        verdict_feedback(verdict, stage)  the structured feedback dict the shell hands the agent for a kernel
                                    verdict (decision, stage, reason = the raiser's reason, the rest of its feedback)
        verify_final(text)          on_answer only: the kernel's verdict on `text` as a QUERY (kernel.evaluate_final):
                                    nothing is recorded and no revision or intervention counter moves, however
                                    often it is asked. Exceptions propagate. Only the first call through a context
                                    appends the verdict's events (shell_final_query) to the trajectory. The shell
                                    keeps every verdict returned, with its text: an Accept is applied under the
                                    verdict of the delivered text, which the shell then records as the real one
                                    (delivered_verdict)
        plan_read(read)             on_answer only: plan a Read now (see Read); returns whether it is dispatchable
        semantic_check()            on_answer only: one semantic check (HarnessContext.semantic_check) -> a judge
                                    for the calls one decision makes, or None without a judge or when the meter
                                    refuses the unit (counted there, once). The judge answers until the kernel's next
                                    hook call, so a check taken before a verify_final is refused after it. Recorded in
                                    instrumentation.jsonl (kind semantic_check)
        semantic_rounds(name)       on_answer only: the round bound `name` of the workspace's allocation (manifest
                                    semantic_budget.rounds)
        forced_deliverable_write(feedback)  after_episode only

    Harness actions (the shell's harness action channel, runner/harness/action_service.py). A middleware acts on
    the environment only through these. The channel decides whether each action may run (scope, ceilings, the
    manifest's read-only classification, the irreversible-commit checks, write provenance, the kernel with
    origin=harness, the READ actuator), runs it through the executor, records it (origin=harness, requester=recovery)
    and guards agent-authored content after it. These need an invocation this context opened, and only while the hook
    that opened it runs: the shell ends the invocation when that middleware's hook returns, and a context kept past
    its hook opens nothing (begin_harness_actions / post_stop_allowance -> None) and acts through nothing
    (request_action / read -> {"error": "harness_invocation_closed"}, recorded; recovery_state -> {}, read_back -> None,
    harness_actions_open -> False):
        begin_harness_actions()     before_final only: open one invocation at this step -> the step, or None when the
                                    shell admits none (no kernel; a per-task cap already spent). Called again while the
                                    invocation is open it returns the same step and resets nothing
        post_stop_allowance()       after_episode only: once per episode, only when the loop stopped at
                                    tool_budget_exhausted and only where the adapter grants the additive reserve,
                                    re-grant the recovery reserve (the recovery ceiling) and open one invocation at the
                                    step the loop stopped at -> that step, or None
        request_action(tool, args)  one harness action -> its result, or {"error": <refusal reason>}
        read(tool, args)            an evidence-reveal READ through the READ actuator -> likewise
        recovery_state()            read-only view: the persisted portal state (a page read, not an action)
        read_back(tool, args, result)  read-only view: the environment's active read-back of a write's result
        harness_actions_open()      read-only view: whether the budget ceiling admits a harness action now
    """
    __slots__ = ()

    def __init__(self, env, task, trajectory, step, max_steps, used_actions, judge_factory, errors,
                 forced_deliverable_write=None, shell=None):
        st = _ContextState()
        st.env, st.task, st.trajectory = env, task, trajectory
        st.step, st.max_steps, st.used_actions = step, max_steps, used_actions
        st.judge_factory, st.errors = judge_factory, errors
        st.forced = forced_deliverable_write
        st.shell = dict(environment_views(env), **(shell or {}))
        st.verify_calls = 0
        st.verdicts = []      # (text, verdict) of each verify_final that returned, in call order; the shell's (delivered_verdict reads it)
        st.action_step = None   # the step of the harness-action invocation this context has open
        st.opened = False       # this context opened an invocation (a request after it ended is refused, not an error)
        st.consumer = None      # the host middleware whose hook runs (MiddlewareStack): its judge units' consumer
        _BACKING[self] = st

    def __getattr__(self, name):
        if name == "env":
            raise AttributeError("workspace code has no environment: read it through ctx.controls(), "
                                 "ctx.persisted_state(), ctx.downloads(), ctx.output_files(), and act through the "
                                 "harness action service")
        raise AttributeError("the middleware context has no %r" % (name,))

    def _live(self, name, default=None):
        fn = _backing(self).shell.get(name)
        return default if fn is None else fn()

    @property
    def task(self):
        from workspace import public_task
        return public_task(_backing(self).task)

    @property
    def trajectory(self):
        return TrajectoryView(_backing(self).trajectory)

    @property
    def step(self):
        return _backing(self).step

    @property
    def max_steps(self):
        return _backing(self).max_steps

    @property
    def used_actions(self):
        return _backing(self).used_actions

    @property
    def judge(self):
        """A judge that pays one unit per call, drawn as the consumer middleware:<Class> of the middleware whose hook
        is running (harness/semantic_budget.py), or None."""
        st = _backing(self)
        return st.judge_factory(st.consumer or "middleware")

    def error(self, message):
        _backing(self).errors.append(message)

    def forced_deliverable_write(self, feedback):
        forced = _backing(self).forced
        if forced is None:
            raise RuntimeError("forced_deliverable_write is only available in after_episode")
        return forced(feedback)

    # ---- read-only views of the environment
    def controls(self):
        return readonly_copy(self._live("controls", []) or [])

    def persisted_state(self):
        return readonly_copy(self._live("persisted_state"))

    def downloads(self):
        return list(self._live("downloads", ()) or ())

    def output_files(self):
        return readonly_copy(self._live("output_files"))

    # ---- read-only shell state
    @property
    def observed_history(self):
        return tuple(self._live("observed_history", ()))

    @property
    def last_obs(self):
        return readonly_copy(self._live("last_obs"))

    @property
    def budget_spent_at_turn_start(self):
        return self._live("budget_spent_at_turn_start")

    @property
    def agent_env_actions(self):
        return self._live("agent_env_actions")

    @property
    def recovery_env_actions(self):
        return self._live("recovery_env_actions")

    @property
    def harness_env_actions(self):
        return self._live("harness_env_actions")

    @property
    def live_used_actions(self):
        a, h = self.agent_env_actions, self.harness_env_actions
        return None if a is None or h is None else a + h

    @property
    def recovery_reserve(self):
        return self._live("recovery_reserve")

    @property
    def in_repair(self):
        return self._live("in_repair")

    @property
    def forced_pending(self):
        return self._live("forced_pending")

    @property
    def answers(self):
        return tuple(Answer(*a) for a in self._live("answers", ()))

    @property
    def end_reason(self):
        return self._live("end_reason")

    @property
    def ledger(self):
        led = self._live("ledger")
        return None if led is None else LedgerView(led)

    @property
    def contract_meta(self):
        return copy.deepcopy(self._live("contract_meta"))

    # ---- services
    def emit(self, event):
        """Append `event` to the trajectory with this context's step. The trajectory keeps a copy.

        Refused with ValueError: an event type the loop, a scorer or the aggregation reads (runner/harness/origin.py
        RESERVED_EVENT_TYPES: tool_call, final_answer, answer_attempt, write_file, agent_error, the episode-end and
        read-back records; the harness_ prefix; a type containing budget, truncat or escalat), and an event carrying
        a provenance key (origin, requester, action_origin, final_answer_author, ...): the shell records who did what."""
        if not isinstance(event, dict) or not event.get("event_type"):
            raise ValueError("emit needs a dict with an event_type")
        from harness import origin as _origin
        why = _origin.reserved_event_reason(event.get("event_type"))
        if why is not None:
            raise ValueError("emit refuses %s: the shell records it (runner/harness/origin.py)" % why)
        keys = sorted(k for k in event if k in _origin.PROVENANCE_KEYS)
        if keys:
            raise ValueError("emit refuses the provenance keys %s: the shell records origin" % ", ".join(keys))
        st = _backing(self)
        rec = readonly_copy(dict(event))
        rec["step"] = st.step if st.action_step is None else st.action_step
        st.trajectory.append(rec)

    def record_budget_warning(self):
        """The deliverable budget warning record (see the class docstring) -> True when recorded, False when the
        episode already has one."""
        st = _backing(self)
        once = st.shell.get("budget_warning_once")
        if once is not None and not once():
            return False
        st.trajectory.append({"step": st.step, "event_type": "deliverable_budget_warning",
                              "remaining": st.max_steps - st.used_actions, "status": "ok"})
        return True

    def obligations(self):
        """(met, decisions, conjuncts) from the shell's obligation model (see the class docstring)."""
        out = self._live("obligations")
        return (True, None, None) if out is None else readonly_copy(out)

    def verdict_feedback(self, verdict, stage):
        fn = _backing(self).shell.get("verdict_feedback")
        if fn is None:
            raise RuntimeError("verdict_feedback is not available in this context")
        return fn(verdict, stage)

    def verify_final(self, text):
        st = _backing(self)
        fn = st.shell.get("verify_final")
        if fn is None:
            raise RuntimeError("verify_final is only available in on_answer")
        st.verify_calls += 1
        first = st.verify_calls == 1
        verdict = fn(text)
        st.verdicts.append((text, verdict))
        if first and verdict is not None and getattr(verdict, "events", None):
            st.trajectory.extend(verdict.events)
        return verdict

    def plan_read(self, read):
        fn = _backing(self).shell.get("plan_read")
        if fn is None:
            raise RuntimeError("plan_read is only available in on_answer")
        return fn(read)

    def semantic_check(self):
        fn = _backing(self).shell.get("semantic_check")
        if fn is None:
            raise RuntimeError("semantic_check is only available in on_answer")
        return fn(_backing(self).consumer or "middleware")

    def semantic_rounds(self, name):
        fn = _backing(self).shell.get("semantic_rounds")
        if fn is None:
            raise RuntimeError("semantic_rounds is only available in on_answer")
        return fn(name)

    # ---- harness actions
    def begin_harness_actions(self):
        return self._open_actions("begin_harness_actions", "before_final")

    def post_stop_allowance(self):
        return self._open_actions("post_stop_allowance", "after_episode")

    def _open_actions(self, name, hook):
        st = _backing(self)
        fn = st.shell.get(name)
        if fn is None:
            raise RuntimeError("%s is only available in %s" % (name, hook))
        step = fn()
        if step is not None:
            st.action_step, st.opened = step, True
        return step

    def _harness_action(self, name):
        st = _backing(self)
        fn = st.shell.get(name)
        if fn is None or (st.action_step is None and not st.opened):
            raise RuntimeError("%s needs a harness-action invocation this context opened (begin_harness_actions in "
                               "before_final, post_stop_allowance in after_episode)" % name)
        return fn

    def request_action(self, tool, args):
        return self._harness_action("request_action")(tool, readonly_copy(args))

    def read(self, tool, args):
        return self._harness_action("read")(tool, readonly_copy(args))

    def recovery_state(self):
        return readonly_copy(self._harness_action("recovery_state")())

    def read_back(self, tool, args, result):
        return readonly_copy(self._harness_action("read_back")(tool, readonly_copy(args), readonly_copy(result)))

    def harness_actions_open(self):
        return self._harness_action("harness_actions_open")()


def context_state(ctx):
    """The shell's record behind a MiddlewareContext (shell code only: delivered_verdict, tests)."""
    return _backing(ctx)


def environment_views(env):
    """The read-only environment views of a middleware context, over `env` (see MiddlewareContext). run.py passes the
    harness action service's control model as `controls`, the same reader the service itself uses."""
    def controls():
        return env.list_controls() if hasattr(env, "list_controls") else []

    def persisted_state():
        return env.eval_state() if hasattr(env, "eval_state") else None

    def downloads():
        import os
        return [os.path.basename(str((v or {}).get("path") or ""))
                for v in (getattr(env, "_download_registry", {}) or {}).values()]

    def output_files():
        import glob
        import os
        ws = getattr(env, "workspace", "") or ""
        if not ws:
            return None
        files = {}
        if os.path.isdir(ws):
            for p in glob.glob(os.path.join(ws, "*")):
                if os.path.isfile(p):
                    try:
                        with open(p, encoding="utf-8", errors="ignore") as f:
                            text = f.read()
                    except Exception:
                        text = ""
                    files[os.path.basename(p)] = {"size": os.path.getsize(p), "text": text}
        return {"exists": os.path.isdir(ws), "files": files}

    return {"controls": controls, "persisted_state": persisted_state, "downloads": downloads,
            "output_files": output_files}


def hook_call_id(hook, step):
    """The id of one host hook call: each host hook runs at most once per loop iteration (after_episode once per task), so
    hook@step names it. The shell_hook event of the call carries it, and the harness action service stamps it on the
    actions a recovery invocation opened in that call as recovery_episode_id (run.py)."""
    return "%s@%s" % (hook, step)


def hook_event(ctx, hook, middleware, summary):
    """The one shell_hook trajectory event of a host hook call whose result the shell applies (a non-None result;
    True from exempt_from_budget), recorded when the hook returns: layer host, hook, step, origin harness, requester
    the middleware, call_id and the result summary (instrumentation.summarize). A context not built by the shell
    (a unit test's stand-in) records nothing."""
    st = _BACKING.get(ctx) if isinstance(ctx, MiddlewareContext) else None
    if st is None:
        return
    st.trajectory.append({"event_type": "shell_hook", "layer": "host", "hook": hook, "step": st.step,
                          "origin": "harness", "requester": "middleware:%s" % type(middleware).__name__,
                          "call_id": hook_call_id(hook, st.step), "result": summary})


def _as_consumer(ctx, middleware):
    """The judge units a context draws while `middleware`'s hook runs are that middleware's."""
    st = _BACKING.get(ctx) if isinstance(ctx, MiddlewareContext) else None
    if st is not None:
        st.consumer = "middleware:%s" % type(middleware).__name__


class MiddlewareStack(object):
    def __init__(self, middlewares, instrumentation=None):
        self.middlewares = list(middlewares)
        self.instrumentation = instrumentation

    def _note(self, m, hook, ctx, out):
        from instrumentation import summarize
        if out is not None and out is not False:
            hook_event(ctx, hook, m, summarize(out))
        if self.instrumentation is None:
            return
        extra = {"stage": out.stage} if isinstance(out, (Revision, Nudge, PostRevision)) else {}
        if isinstance(out, Nudge):
            extra["label"] = out.label
        if out is not None:
            extra["summary"] = summarize(out)
        returned = out if (out is None or isinstance(out, bool)) else type(out).__name__
        self.instrumentation.call(m, hook, getattr(ctx, "step", None), returned, **extra)

    def _first(self, hook, ctx, *args):
        # the loop's own action and result dicts never reach workspace code (a kernel verdict is passed as is)
        args = tuple(readonly_copy(a) if isinstance(a, (dict, list)) else a for a in args)
        for m in self.middlewares:
            fn = getattr(m, hook, None)
            if fn is None:
                continue
            _as_consumer(ctx, m)
            try:
                out = fn(ctx, *args)
            finally:
                end_invocation(ctx)      # the invocation this middleware's hook opened ends when the hook returns
            self._note(m, hook, ctx, out)
            if out is not None:
                return out
        return None

    def before_model(self, ctx):
        return self._first("before_model", ctx)

    def next_read(self, ctx):
        return self._first("next_read", ctx)

    def before_final(self, ctx, answer):
        return self._first("before_final", ctx, answer)

    def on_answer(self, ctx, answer):
        return self._first("on_answer", ctx, answer)

    def after_final_verdict(self, ctx, answer, verdict):
        return self._first("after_final_verdict", ctx, answer, verdict)

    def before_tool(self, ctx, action):
        return self._first("before_tool", ctx, action)

    def after_tool(self, ctx, action, result):
        return self._first("after_tool", ctx, action, result)

    def exempt_from_budget(self, ctx, action):
        action = readonly_copy(action)
        for m in self.middlewares:
            fn = getattr(m, "exempt_from_budget", None)
            if fn is None:
                continue
            _as_consumer(ctx, m)
            out = bool(fn(ctx, action))
            self._note(m, "exempt_from_budget", ctx, out)
            if out:
                return True
        return False

    def after_episode(self, ctx):
        for m in self.middlewares:
            fn = getattr(m, "after_episode", None)
            if fn is not None:
                _as_consumer(ctx, m)
                try:
                    fn(ctx)
                finally:
                    end_invocation(ctx)
                self._note(m, "after_episode", ctx, None)

    def deferred_errors(self):
        out = []
        for m in self.middlewares:
            fn = getattr(m, "deferred_errors", None)
            if fn is not None:
                out.extend(fn() or [])
        return out


# ---- shell helpers run.py applies results with (kept here so the loop's hunks stay at the hook sites)
def queued_read(read):
    """The forced action a next_read Read becomes (None -> None)."""
    if read is None:
        return None
    if not isinstance(read, Read):
        raise TypeError("next_read must return Read or None, not %r" % (type(read).__name__,))
    return {"type": "tool_call", "tool": read.tool, "args": read.args, "_harness_acquire": True,
            "_requester": "next_read"}     # who asked (runner/harness/origin.py), read where it executes


class _ReadRequest(object):
    """What actuation.plan reads off a decision (extra.next_action, capability), built from an on_answer Read."""
    __slots__ = ("extra", "capability")

    def __init__(self, read):
        from harness import actuation
        src = read.source
        na = dict(actuation.next_action_of(src)) if src is not None else {"read_only": True}
        na["tool"], na["args"] = read.tool, read.args
        self.extra = {"next_action": na}
        self.capability = actuation.capability_of(src) if src is not None else None


def plan_answer_read(read, step, ledger, trajectory):
    """Plan an on_answer Read once (stage before_final_readapt, no vocabulary) and record the plan's events.
    Returns the plan."""
    if read.plan is None:
        from harness import actuation
        read.plan = actuation.plan(_ReadRequest(read), "before_final_readapt", step, ledger=ledger)
        trajectory.extend(read.plan.events)
    return read.plan


def answer_services(harness, step, trajectory, instrumentation=None):
    """The on_answer-only services of a MiddlewareContext."""
    def verify_final(text):
        """A query: the kernel's evaluate_final, which records nothing and moves no counter."""
        if harness is None:
            raise RuntimeError("verify_final needs the harness kernel")
        return harness.evaluate_final(text, step=step)

    def record_final(text, verdict, flagged):
        """The real before_final verdict on a delivered `text`: the query `verdict` recorded (kernel.record_final), or
        when this on_answer call asked for none (UNASKED), the kernel's evaluation of the text recorded the same way."""
        if harness is None:
            raise RuntimeError("record_final needs the harness kernel")
        if verdict is UNASKED:
            verdict = harness.evaluate_final(text, step=step)
        return harness.record_final(verdict, step=step, flagged=flagged)

    def plan_read(read):
        return plan_answer_read(read, step, getattr(harness, "ledger", None), trajectory).dispatchable

    def semantic_check(consumer):
        """One semantic check for `consumer` (the middleware whose on_answer runs), the unit the kernel middleware pays:
        a judge for every call of one decision, all or nothing. A refusal -> None, counted once by the kernel's meter
        (harness/semantic_budget.py). The kernel's evaluate_final bumps hook_serial, so the judge must be taken after the
        verify_final that precedes the decision; kept past a later verify_final it is refused call by call
        (HarnessContext.semantic_check)."""
        kernel_ctx = getattr(harness, "ctx", None)
        if kernel_ctx is None or not kernel_ctx.has_judge():
            return None
        judge = kernel_ctx.semantic_check(consumer)
        if instrumentation is not None:
            instrumentation.semantic_check("on_answer", step, judge is not None)
        return judge

    def semantic_rounds(name):
        """The round bound `name` of the workspace's allocation (manifest semantic_budget.rounds)."""
        if harness is None:
            raise RuntimeError("semantic_rounds needs the harness kernel")
        return harness.ctx.semantic_meter.rounds(name)
    return {"verify_final": verify_final, "record_final": record_final, "plan_read": plan_read,
            "semantic_check": semantic_check, "semantic_rounds": semantic_rounds}


def harness_action_services(service, at_step=None, trajectory=None, end_reason=None):
    """The harness-action services of a middleware context, backed by the task's harness action channel. A
    before_final context (at_step) may open an invocation at its step; an after_episode context (trajectory, end_reason)
    may take the post-stop allowance, which exists only when the loop stopped at tool_budget_exhausted, at the step of
    that record.

    The services hold one scope (HarnessActionService.scope) for the hook call the context is built for. The stack
    ends the invocation the scope opened when the middleware hook that opened it returns (end_invocation), and run.py
    closes the scope when the hook is done (close_scope): the channel refuses a request through an ended invocation, and a
    closed scope opens nothing."""
    scope = service.scope()

    def open_(fn):
        return lambda *a: fn(*a) if service.invocation_open(scope) else None
    out = {"request_action": lambda tool, args: service.request_action(tool, args, scope=scope),
           "read": lambda tool, args: service.read(tool, args, scope=scope),
           "recovery_state": lambda: service.read_recovery_state() if service.invocation_open(scope) else {},
           "read_back": open_(service.read_back),
           "harness_actions_open": lambda: bool(service.invocation_open(scope) and service.budget_open()),
           "end_invocation": lambda: service.end_invocation(scope),
           "close_scope": lambda: service.close_scope(scope)}
    if trajectory is None:
        out["begin_harness_actions"] = lambda: service.begin(at_step, scope=scope)
    else:
        def post_stop_allowance():
            if end_reason != "tool_budget_exhausted":
                return None
            stops = [e.get("step") for e in trajectory if e.get("event_type") == "tool_budget_exhausted"]
            return service.post_stop_allowance(stops[-1], scope=scope) if stops else None
        out["post_stop_allowance"] = post_stop_allowance
    return out


def end_invocation(ctx):
    """The middleware hook running with `ctx` returned: the invocation it opened ends (MiddlewareStack)."""
    st = _BACKING.get(ctx) if isinstance(ctx, MiddlewareContext) else None
    if st is None:
        return
    fn = st.shell.get("end_invocation")
    if fn is not None:
        fn()
    st.action_step = None


def close_scope(ctx):
    """The loop is done with the hook `ctx` was built for: its harness-action scope opens nothing more (run.py)."""
    st = _BACKING.get(ctx) if isinstance(ctx, MiddlewareContext) else None
    if st is None:
        return
    fn = st.shell.get("close_scope")
    if fn is not None:
        fn()
    st.action_step = None


def nudge_admitted(counter, nudge, hook, step, trajectory):
    """Whether the shell applies this Nudge. `counter` is the episode's one-slot count of applied Nudges. Past
    MAX_NUDGES_PER_EPISODE the Nudge is refused: harness_nudge_refused is recorded and the loop goes on as if the hook had
    returned None."""
    if counter[0] >= MAX_NUDGES_PER_EPISODE:
        trajectory.append({"step": step, "event_type": "harness_nudge_refused", "hook": hook,
                           "stage": getattr(nudge, "stage", None), "label": getattr(nudge, "label", None),
                           "reason": "nudge_cap", "cap": MAX_NUDGES_PER_EPISODE, "status": "refused"})
        return False
    counter[0] += 1
    return True


def revision_admitted(counter, result, hook, step, trajectory):
    """Whether the shell applies this Revision or Repair. `counter` is the episode's one-slot count of applied ones.
    Past MAX_REVISIONS_PER_EPISODE the result is refused: harness_revision_refused is recorded and the loop goes on as if
    the hook had returned None."""
    if counter[0] >= MAX_REVISIONS_PER_EPISODE:
        trajectory.append({"step": step, "event_type": "harness_revision_refused", "hook": hook,
                           "result": type(result).__name__, "stage": getattr(result, "stage", None),
                           "reason": "revision_cap", "cap": MAX_REVISIONS_PER_EPISODE, "status": "refused"})
        return False
    counter[0] += 1
    return True


def accept_event(accept, step, attempt_index, canon, env_type, agent_answers=()):
    """The final_answer event of an Accept. Its author (runner/harness/origin.py): the agent when the delivered text is
    one of the agent's own answers that reached the attempt point (`agent_answers`, string equality); otherwise the
    answer-candidate mechanism that produced it: final_answer_author candidate_merge, origin harness, requester
    candidate_merge."""
    from harness import origin as _origin
    ev = {"step": step, "event_type": "final_answer", "thought": accept.text, "status": "ok",
          "final_disposition": accept.disposition, "intervention_type": accept.intervention_type,
          "answer_attempt_index": attempt_index, "reverified": accept.reverified,
          "canonical_action": canon.canonical_action({"type": "final", "answer": accept.text}, env_type)}
    if accept.text in tuple(agent_answers or ()):
        ev["final_answer_author"] = _origin.AUTHOR_AGENT
        return _origin.tag(ev, _origin.AGENT)
    ev["final_answer_author"] = _origin.AUTHOR_CANDIDATE_MERGE
    return _origin.tag(ev, _origin.HARNESS, _origin.CANDIDATE_MERGE)


# ---- the evidence-contradiction withhold: one definition for the normal before_final path and an Accept
WITHHELD_THOUGHT = ("[WITHHELD] The answer asserted a claim refuted by the validated evidence and the conflict was not "
                    "resolved; no safe answer could be committed.")


def must_resolve_violation(verdict):
    """A kernel before_final verdict no final may be delivered over once repair is exhausted: an effective REVISE or
    BLOCK whose raising decision carries must_resolve and is not a candidate review. verify_commit raises it for a
    localized evidence contradiction (the adequacy audit's hard defect) and for process-output inconsistency."""
    if verdict is None or getattr(verdict, "type", None) not in ("REVISE", "BLOCK"):
        return False
    extra = (getattr(verdict, "raw", None) and getattr(verdict.raw, "extra", None)) or {}
    return bool(extra.get("must_resolve")) and not extra.get("candidate")


UNASKED = object()   # delivered_verdict: no verify_final of the delivered text in this on_answer call


def delivered_verdict(ctx, text, flagged=False):
    """The REAL before_final verdict an Accept of `text` is applied under, recorded once.

    Which verdict: the LAST one ctx.verify_final returned during this on_answer call for exactly `text` (string
    equality). A verdict on another text says nothing about the delivered one, in either direction: candidate B's
    when root A is delivered, the merge's when B is delivered after the merge failed to verify, B's and A's when the
    merge is delivered. Of two verdicts on the same text the later was reached on the later kernel state. A verdict
    from an earlier on_answer call does not count: reads may have changed the evidence since. When this call has no
    verdict for `text` (never asked, or every call raised) the kernel evaluates it, so leaving the delivered text
    unverified is no way around the withhold.

    ctx.verify_final is a query (kernel.evaluate_final). The delivery is the verdict the shell applies, so it is
    recorded here with kernel.record_final: proposal, findings, intervention and counters, and the final-answer commit
    when the verdict is ALLOW and the text is not delivered `flagged` (delivery_flag; record_flagged_final records that
    commit). Its events go to the trajectory. -> the recorded Effective. Exceptions propagate."""
    st = _backing(ctx)
    query = UNASKED
    for said, verdict in reversed(st.verdicts):
        if said == text:
            query = verdict
            break
    fn = st.shell.get("record_final")
    if fn is None:
        raise RuntimeError("delivered_verdict is only available in on_answer")
    real = fn(text, query, bool(flagged))
    if real is not None and getattr(real, "events", None):
        st.trajectory.extend(real.events)
    return real


def delivery_flag(accept, verdict):
    """The verification flag an Accept of `accept.text` is delivered with, applied under its recorded verdict: the
    Accept's own flag; else, when that verdict is not ALLOW (a candidate adopted without a clean verdict), the flag the
    normal before_final path gives a final delivered over that verdict type (ESCALATE: its verification_flag or
    unresolved_risk; DEFER_FINAL: deferred_unresolved; otherwise unverified_grounding); else None (a clean commit)."""
    if accept.flag is not None:
        return accept.flag
    vtype = getattr(verdict, "type", None)
    if verdict is None or vtype == "ALLOW":
        return None
    if vtype == "ESCALATE":
        extra = (getattr(verdict, "raw", None) and getattr(verdict.raw, "extra", None)) or {}
        return extra.get("verification_flag", "unresolved_risk")
    return "deferred_unresolved" if vtype == "DEFER_FINAL" else "unverified_grounding"


def withhold_unresolved_violation(harness, trajectory, errors, answer, feedback, step, attempt_index, canonical_action,
                                  deliverable_is_env_state):
    """End the episode over a final whose verdict is a must_resolve violation and that has no repair left: the normal
    before_final path on repair exhaustion, and an Accept (a candidate finalize is repair exhaustion). A confirmed
    conflict is not eligible for flagged delivery: the answer is WITHHELD, the final_answer carries a stub and the
    stub is recorded as the flagged final (safe abstention, never the violating answer). Exception: when the graded
    artifact is state the agent already wrote to the environment, a stub cannot un-write it, so abstention prevents
    nothing and forfeits the run; the answer is delivered and the contradiction flagged. `feedback` is the verdict's,
    `canonical_action` the answer's. The caller ends the loop."""
    if deliverable_is_env_state:
        trajectory.append({"step": step, "event_type": "final_answer", "thought": answer, "status": "ok",
                           "origin": "agent", "final_answer_author": "agent",   # runner/harness/origin.py
                           "verification_flag": "unresolved_contradiction",
                           "final_disposition": "flagged_delivery_unresolved_contradiction",
                           "answer_attempt_index": attempt_index, "harness_feedback": feedback,
                           "canonical_action": canonical_action})
        recorded, flag = answer, "unresolved_contradiction"
    else:
        trajectory.append({"step": step, "event_type": "final_answer", "thought": WITHHELD_THOUGHT, "status": "ok",
                           "origin": "harness", "requester": "withhold", "final_answer_author": "withheld_stub",
                           "verification_flag": "abstained_unresolved_contradiction",
                           "final_disposition": "abstained_unresolved_violation",
                           "answer_attempt_index": attempt_index, "harness_feedback": feedback,
                           "canonical_action": canonical_action})
        recorded, flag = "[WITHHELD: unresolved evidence contradiction]", "abstained_unresolved_contradiction"
    try:
        harness.record_flagged_final(recorded, flag=flag, step=step)
    except Exception as e:
        errors.append("record_flagged_final: %r" % e)


def unexpected(hook, out):
    return TypeError("%s returned %r, which this hook does not accept" % (hook, type(out).__name__))


def agent_text(text):
    """after_final_verdict's result as the shell uses it: a non-empty str, or None."""
    if text is None:
        return None
    if not isinstance(text, str):
        raise unexpected("after_final_verdict", text)
    return text or None


def decision_text(decision):
    """The agent text the raising kernel middleware put on its decision (extra["agent_text"]), read the way
    actuation.next_action_of reads next_action; None when it supplied none."""
    extra = getattr(decision, "extra", None)
    if not extra:
        raw = getattr(decision, "raw", None)
        extra = getattr(raw, "extra", None) if raw is not None else None
    text = extra.get("agent_text") if isinstance(extra, dict) else None
    return text if isinstance(text, str) and text else None


_LOOP_END_EVENTS = ("final_answer", "harness_infrastructure_error", "repair_budget_exhausted",
                    "tool_budget_exhausted", "circuit_breaker")


def episode_end_reason(trajectory, finished, aborted):
    """Why the loop stopped, for after_episode. "max_steps_exceeded" when it ran out of iterations; otherwise the
    record that ended it, found backwards past anything appended after it (forced recovery's own events):
    final_answer, harness_infrastructure_error, harness_escalation (status error), repair_budget_exhausted,
    tool_budget_exhausted, circuit_breaker, or the agent_error kinds invalid_action / bad_action_type."""
    if not finished and not aborted:
        return "max_steps_exceeded"
    for ev in reversed(trajectory):
        et = ev.get("event_type")
        if et in _LOOP_END_EVENTS or (et == "harness_escalation" and ev.get("status") == "error"):
            return et
        if et == "agent_error" and ev.get("error") in ("invalid_action", "bad_action_type"):
            return ev.get("error")
    return None


def obligations_of(task, env, trajectory, workspace):
    """The shell's obligation model, harness.obligation_gate.gui_obligations (the task's completion conjunction over
    the agent's decisions in `trajectory`, the one composition the terminal lock and recovery read too), over the
    completion content of the run's `workspace`."""
    from harness.obligation_gate import gui_obligations
    return gui_obligations(task, env, workspace, actions=trajectory)


def load_middleware(workspace, task, instrumentation=None):
    """The middleware of the run's workspace (runner/run.py resolve_workspace), instantiated for this task with the
    public task (runner/workspace.py public_task; each middleware its own copy). Empty when the run has no workspace
    (None)."""
    from workspace import Workspace, public_task
    if not workspace:
        return MiddlewareStack([], instrumentation)
    return MiddlewareStack([cls(public_task(task)) for cls in Workspace(workspace).middleware_classes()],
                           instrumentation)


# The ends after which the harness forces nothing: a safety abort -- an escalation that stopped the episode
# (harness_escalation with status error: an ESCALATE at before_final / after_action, the reconcile budget; a
# before_action escalation left unresolved past the repair budget) -- or a harness error (the kernel failed to build, a
# kernel hook raised).
def safety_abort(trajectory, end_reason):
    """-> "safety" | "harness_error" when the episode ended on a safety abort or a harness error, else None. Read from
    the record that ended the loop (episode_end_reason's)."""
    if end_reason == "harness_infrastructure_error":
        return "harness_error"
    if end_reason not in ("harness_escalation", "repair_budget_exhausted"):
        return None
    for ev in reversed(trajectory):
        et = ev.get("event_type")
        if et == "harness_escalation" and ev.get("status") == "error":
            return "harness_error" if ev.get("reason") == "hook_error" else "safety"
        if et == "repair_budget_exhausted":
            return "safety" if ev.get("reason") == "escalation_unresolved" else None
    return None


def forced_deliverable_write(feedback, agent, env, task, trajectory, service, max_steps, end_reason=None, turn=None):
    """Give the agent one more turn to write its deliverable, and run that write as a harness action through the harness
    action channel: the ceilings, the kernel's before_action with origin harness and the authorization
    ceremony, the executor, after_action and the record. Only a write_file is executed. The channel refuses it on any
    kernel verdict other than ALLOW (a REVISE included) and when a kernel hook raises; the refusal is recorded as
    forced_deliverable_blocked (and harness_action_refused). Nothing is forced after a safety abort or a harness error
    (safety_abort): the agent is not asked, forced_deliverable_blocked records why.

    The write is the harness's (origin harness, requester forced_deliverable_write; the text is the agent's,
    content_author agent): it runs past the agent's budget only because the harness forces it, and it is counted in the
    channel, never in the agent's budget (runner/harness/origin.py)."""
    why = safety_abort(trajectory, end_reason)
    if why is not None:
        trajectory.append({"step": max_steps, "event_type": "forced_deliverable_blocked", "decision": None,
                           "reason": "after_%s:%s" % (why, end_reason), "tool": "write_file", "status": "ok",
                           "origin": "harness", "requester": "forced_deliverable_write"})
        return
    try:
        state = {"goal": task.get("goal"), "context": task.get("context"), "tools": env.available_tools()}
        if turn is None:
            state.update({"last_observation": feedback, "last_result": {"feedback": feedback}})
            a = agent.act(state)
        else:   # this turn's input after whatever the loop left undelivered, in order (runner/turn_inputs.py)
            turn.feedback(None, "forced_deliverable_write", {"feedback": feedback}, feedback)
            state.update(turn.state({"feedback": feedback}, feedback))
            a = agent.act(state)
            turn.consumed()
        if isinstance(a, dict) and a.get("type") == "tool_call" and a.get("tool") == "write_file":
            out = service.forced_write(a, max_steps)
            if out.refusal is not None:
                verdict = getattr(out.verdict, "type", None)
                trajectory.append({"step": max_steps, "event_type": "forced_deliverable_blocked",
                                   "decision": verdict if verdict is not None else "refused", "reason": out.refusal,
                                   "tool": "write_file", "status": "ok",
                                   "origin": "harness", "requester": "forced_deliverable_write"})
    except Exception as fe:
        trajectory.append({"step": max_steps, "event_type": "agent_error",
                           "error": "forced_deliverable_failed", "raw": repr(fe)[:120], "status": "error"})
