"""Capability base class — Modules A / B / C all implement this 3-hook interface.

The kernel calls each registered capability at three stages and combines their decisions by precedence:
  before_action(action, ctx)  -> HarnessDecision | None     (prospective gate)
  after_action(action, result, before_state, after_state, ctx) -> HarnessDecision | None  (retrospective)
  before_final(answer, ctx)   -> HarnessDecision | None     (final answer = a commit point)

`ctx` is a HarnessContext (ledger + contract + risk + policy). A capability mutates the LEDGER
(records evidence/obligations) but returns only a decision; it never executes the action itself.
Returning None == no opinion (== ALLOW for that capability).
"""
import copy as _copy
import weakref as _weakref

from .decision import HarnessDecision, ALLOW
from .semantic_budget import CALL, CHECK, MAX_JUDGE_CALLS_PER_UNIT, SemanticMeter


class Capability:
    name = "capability"

    def on_contract(self, ctx):
        """Hook once after the contract is compiled (e.g. declare obligations into the ledger)."""
        return None

    def before_action(self, action, ctx):
        return None

    def after_action(self, action, result, before_state, after_state, ctx):
        return None

    def before_final(self, answer, ctx):
        return None

    # helpers for subclasses
    def _decide(self, type, **kw):
        return HarnessDecision(type, capability=self.name, **kw)


class HarnessContext:
    """What every capability + the kernel share for one task."""
    __slots__ = ("ledger", "contract", "policy", "step", "env_type", "risk_of", "observation", "last_raw_decision",
                 "judge_fn", "judge_model", "semantic_meter", "manifest", "sem", "risk",
                 "result_ok", "result_status", "verification", "last_observation", "observed_subject", "displayed_subject",
                 "action_key", "evidence_version_before", "current_state", "final_is_commit",
                 "hook_serial",
                 "final_audit",  # FinalAudit of the final under review (verify_commit), None at every other hook
                 "last_write",   # the deliverable the agent has written; nothing else retains it
                 "_trace_sink", "_trace_pending")   # shell events (kernel.bind_trajectory); shell only

    def __init__(self, ledger, contract, policy, env_type=None, risk_of=None,
                 judge_fn=None, judge_model=None, semantic_budget=0, semantic_allocation=None, manifest=None):
        self.ledger = ledger
        self.contract = contract
        self.policy = policy or {}
        self.step = 0
        self.env_type = env_type
        self.risk_of = risk_of      # optional; the kernel sets ctx.sem + ctx.risk per action
        self.observation = None     # canonical_observation of the most recent action (after_action only)
        self.last_raw_decision = None  # the RAW combined before_action winner (before budgets); recovery reads raw+effective
        self.manifest = manifest or {}   # substrate manifest (adapter layer); tool->semantic mapping
        self.sem = None             # SemanticAction of the current action (set by the kernel)
        self.risk = None            # risk tier of the current action (set by the kernel)
        self.result_ok = None       # did the current action's tool result succeed? (set by the kernel)
        self.result_status = None   # tri-state ok|failed|unknown (timeout/ambiguous) -> unknown commit = ESCALATE
        self.current_state = None   # STRUCTURED env snapshot at before_action (pre-commit form/resource state)
        self.final_is_commit = None  # is the terminal answer the manifest-declared graded commit?
        self.verification = None     # tri-state commit verification: True/False/None (set by verify_commit)
        self.final_audit = None      # verify_commit's adequacy audit of the final under review (FinalAudit, read-only)
        self.last_observation = None  # the most recent canonical_observation
        self.action_key = None       # canonical id of the current action (dedup before/after interventions)
        self.observed_subject = None   # subject displayed by THIS action's observation (manifest-projected)
        self.displayed_subject = None  # last-known displayed subject, carried into before_action (sticky)
        self.judge_fn = judge_fn    # injected judge: callable(prompt:str) -> str|None (INDEPENDENT model)
        self.judge_model = judge_model
        # the task's one semantic meter (harness/semantic_budget.py). `semantic_budget` is the shell's total;
        # `semantic_allocation` the workspace manifest's semantic_budget section (None: the default allocation). Every judge
        # unit is drawn, refused and counted there, whoever asks: goal-spec compilation, the final verification,
        # kernel middleware, host middleware.
        self.semantic_meter = SemanticMeter(semantic_budget, semantic_allocation)
        self.last_write = None      # {path, content, tool, content_from_arg, path_from_arg, n_paths}
        self.hook_serial = 0        # bumped by the kernel at every hook call; a semantic_check judge lives for one
        self._trace_sink = None     # where shell events go (HarnessKernel.bind_trajectory); None: held in _trace_pending
        self._trace_pending = []

    def bind_trace(self, sink):
        """Send shell events to `sink` (a trajectory's append), after the ones held so far."""
        self._trace_sink = sink
        pending, self._trace_pending = self._trace_pending, []
        for ev in pending:
            sink(ev)

    def shell_event(self, event):
        """One shell trace event (shell_hook, shell_refusal): to the bound trajectory, else held until one is bound."""
        if self._trace_sink is not None:
            self._trace_sink(event)
        else:
            self._trace_pending.append(event)

    def _refused(self, consumer, unit, reason):
        """A semantic unit the meter refused (reason meter) or a judge called past the hook that paid for it (expired):
        one shell_refusal event (the meter counts it in audit n_semantic_refused)."""
        self.shell_event({"event_type": "shell_refusal", "kind": "semantic_unit", "step": self.step,
                          "hook_serial": self.hook_serial, "consumer": consumer, "unit": unit, "reason": reason,
                          "status": "refused"})

    def has_judge(self):
        """Whether the harness has a judge at all, budget aside."""
        return self.judge_fn is not None

    def semantic_check(self, consumer):
        """One semantic check for `consumer` (harness/semantic_budget.py: one unit for the calls of one decision) -> a
        judge, or None: no judge, or the meter refused the unit. The judge answers only during the hook call that paid
        for it; kept for a later hook it returns None and the meter counts the call as expired, so one payment cannot
        be replayed for free. It answers at most MAX_JUDGE_CALLS_PER_UNIT calls: a call past that returns None and
        the meter counts it once (over_ceiling). The judge names its model (judge_model), which the semantic cache keys
        on."""
        if self.judge_fn is None:
            return None
        meter = self.semantic_meter
        if not meter.draw(consumer, CHECK):
            self._refused(consumer, CHECK, "meter")
            return None
        fn, serial, ctx = self.judge_fn, self.hook_serial, self
        answered = [0]

        def judge(*a, **kw):
            if ctx.hook_serial != serial:
                meter.expired(consumer)
                ctx._refused(consumer, CHECK, "expired")
                return None
            if answered[0] >= MAX_JUDGE_CALLS_PER_UNIT:
                meter.over_ceiling(consumer)
                return None
            answered[0] += 1
            meter.answered(consumer, unit_calls=answered[0])
            return fn(*a, **kw)
        judge.judge_model = self.judge_model
        return judge

    def budgeted_judge(self, consumer):
        """A judge for `consumer` that pays one unit per call (harness/semantic_budget.py) -> callable, or None without
        a judge. A refused call returns None: every caller already has that no-judge branch, the kernel's fail-closed
        reading (no budget -> no intervention -> ALLOW)."""
        if self.judge_fn is None:
            return None
        ctx, meter = self, self.semantic_meter

        def _budgeted(*a, **kw):
            if ctx.judge_fn is None:
                return None
            if not meter.draw(consumer, CALL):
                ctx._refused(consumer, CALL, "meter")
                return None
            meter.answered(consumer)
            return ctx.judge_fn(*a, **kw)
        _budgeted.judge_model = self.judge_model
        return _budgeted


class FinalAudit(object):
    """The shell's adequacy audit of the final answer under review, as workspace kernel middleware reads it.

    verify_commit.before_final audits a final once (one judge call, paid from the verification reserve) and vetoes a
    localized hard defect itself. It records the audit as ctx.final_audit for the kernel middleware that runs after it
    in the same before_final hook call; the kernel clears the field at every hook, with ctx.verification, so it is
    None wherever no audit ran for the current final. Whether and how to repair a repairable gap is the workspace's
    (e.g. MedCTA: workspaces/medcta/h0/middleware/answer_repair.py), decided from this record without a second judge call.

    Read-only: the audit is kept as JSON text and every read returns a new copy, so a middleware can change neither
    what the shell decided nor what a later middleware reads.

        hook_serial      ctx.hook_serial of the before_final hook call that ran the audit
        postcondition    the claim-support postcondition type the audit checked (the rule_id of the shell's veto)
        side_effecting   verify_commit's operational-write flag for this final
        vetoed           the shell raised the audit's localized hard defect (REVISE evidence_contradiction)
        addresses_task   the auditor's True / False / None
        confidence       the auditor's confidence, unrounded
        audit()          SemanticAudit.to_dict(): the dict the shell's own decision carries under extra["audit"]
        top_gap()        SemanticAudit.top_gap(): the first localized repairable gap, or None
    """
    __slots__ = ("hook_serial", "postcondition", "side_effecting", "vetoed", "addresses_task", "confidence",
                 "_audit_json", "_gap_json")

    def __init__(self, hook_serial, postcondition, side_effecting, audit, vetoed):
        import json
        for name, value in (("hook_serial", hook_serial), ("postcondition", postcondition),
                            ("side_effecting", side_effecting), ("vetoed", bool(vetoed)),
                            ("addresses_task", audit.addresses_task), ("confidence", audit.confidence),
                            ("_audit_json", json.dumps(audit.to_dict())), ("_gap_json", json.dumps(audit.top_gap()))):
            object.__setattr__(self, name, value)

    def audit(self):
        import json
        return json.loads(self._audit_json)

    def top_gap(self):
        import json
        return json.loads(self._gap_json)

    def __setattr__(self, name, value):
        raise AttributeError("the final audit is read-only (tried to set %s)" % name)

    def __delattr__(self, name):
        raise AttributeError("the final audit is read-only (tried to delete %s)" % name)


class KernelLedgerView(object):
    """The kernel ledger as workspace kernel middleware has it: copies of the fields in
    FIELDS, the subject, and the one lifecycle a workspace capability legitimately records -- its findings (the Scoped
    Repair lifecycle: repair_decision, open_finding, mark_delivered, mark_attempted, resolve_finding) and its advisory
    findings (record_advisory). Everything else (terminal_locked, pending_resolution, authorizations, commits,
    interventions, evidence) is the shell's: no attribute can be set and no other method exists. What goes in is
    copied, what comes out is a copy."""
    FIELDS = ("active_subject", "evidence", "observations", "acquire_count", "successful_acquisitions",
              "acquisition_attempts", "interventions", "commit_history", "repair_findings", "advisories",
              "validated_evidence_version", "evidence_version")
    __slots__ = ("__weakref__",)

    def __init__(self, ledger):
        _VIEW_BACKING[self] = ledger

    def __getattr__(self, name):
        if name not in KernelLedgerView.FIELDS:
            raise AttributeError("the ledger view has no field %r (a workspace records findings through its lifecycle "
                                 "methods and keeps its own state in ctx.workspace_state)" % (name,))
        return _copy.deepcopy(getattr(_VIEW_BACKING[self], name, None))

    def __setattr__(self, name, value):
        raise AttributeError("the ledger is read-only to workspace code (tried to set %s)" % name)

    def __delattr__(self, name):
        raise AttributeError("the ledger is read-only to workspace code (tried to delete %s)" % name)

    def subject_id(self):
        return _VIEW_BACKING[self].subject_id()

    def subject_ref(self):
        return _copy.deepcopy(_VIEW_BACKING[self].subject_ref())

    # ---- the finding lifecycle (runner/harness/state.py)
    def record_advisory(self, finding_dict):
        _VIEW_BACKING[self].record_advisory(_copy.deepcopy(finding_dict))

    def repair_decision(self, finding, current_projection):
        mode, rec = _VIEW_BACKING[self].repair_decision(finding, current_projection)
        return mode, _copy.deepcopy(rec)

    def open_finding(self, finding, baseline_projection, step):
        return _copy.deepcopy(_VIEW_BACKING[self].open_finding(_copy.deepcopy(finding),
                                                               _copy.deepcopy(baseline_projection), step))

    def mark_delivered(self, finding_id, projection, step):
        return _copy.deepcopy(_VIEW_BACKING[self].mark_delivered(finding_id, _copy.deepcopy(projection), step))

    def mark_attempted(self, finding_id, projection, step):
        return _copy.deepcopy(_VIEW_BACKING[self].mark_attempted(finding_id, _copy.deepcopy(projection), step))

    def resolve_finding(self, finding_id):
        return _copy.deepcopy(_VIEW_BACKING[self].resolve_finding(finding_id))


class WorkspaceContextView(object):
    """The harness context as kernel middleware from a workspace sees it: copies of what
    the shell's own capabilities read (READ), the ledger as KernelLedgerView, the metered judge services, and the
    workspace's own per-task state. No raw judge, no semantic meter, no budget counters, nothing settable, and no
    attribute that leads back to the HarnessContext.

        READ fields             copies (final_audit is itself read-only and is handed out as is)
        ledger                  KernelLedgerView
        has_judge()             whether the harness has a judge at all
        semantic_check()        one semantic check -> a judge for this hook call, or None
        budgeted_judge()        a judge that pays one unit per call, or None
                                both draw as the consumer kernel_middleware:<Class> (the middleware's class), from the
                                pools the workspace's semantic_budget gives it (harness/semantic_budget.py)
        workspace_state         a dict the run's workspace kernel middleware share for the task (the shell never reads
                                it)
    The shell's audit of the final under review is `final_audit` (FinalAudit, read-only), so repair policy needs no
    judge call of its own."""
    READ = ("step", "env_type", "sem", "risk", "result_ok", "result_status", "current_state", "final_is_commit",
            "verification", "final_audit", "observation", "last_observation", "action_key", "observed_subject",
            "displayed_subject", "last_write", "hook_serial", "contract", "manifest", "policy", "last_raw_decision",
            "judge_model")
    HIDDEN = ("judge_fn", "semantic_meter")
    __slots__ = ("__weakref__",)

    def __init__(self, ctx, workspace_state=None, consumer="kernel_middleware"):
        _VIEW_BACKING[self] = (ctx, KernelLedgerView(ctx.ledger), {} if workspace_state is None else workspace_state,
                               consumer)

    def __getattr__(self, name):
        ctx = _VIEW_BACKING[self][0]
        if name in WorkspaceContextView.HIDDEN:
            raise AttributeError("%s belongs to the shell; workspace code judges through semantic_check() or "
                                 "budgeted_judge()" % name)
        if name not in WorkspaceContextView.READ:
            raise AttributeError("the harness context offers workspace code no %r" % (name,))
        value = getattr(ctx, name, None)
        return value if isinstance(value, FinalAudit) else _copy.deepcopy(value)

    def __setattr__(self, name, value):
        raise AttributeError("the harness context is read-only to workspace code (tried to set %s)" % name)

    def __delattr__(self, name):
        raise AttributeError("the harness context is read-only to workspace code (tried to delete %s)" % name)

    @property
    def ledger(self):
        return _VIEW_BACKING[self][1]

    @property
    def workspace_state(self):
        return _VIEW_BACKING[self][2]

    def has_judge(self):
        return _VIEW_BACKING[self][0].has_judge()

    def semantic_check(self):
        return _VIEW_BACKING[self][0].semantic_check(_VIEW_BACKING[self][3])

    def budgeted_judge(self):
        return _VIEW_BACKING[self][0].budgeted_judge(_VIEW_BACKING[self][3])


# View -> what it reads. Only this module reads it; a view object has no attribute leading to the shell's objects.
_VIEW_BACKING = _weakref.WeakKeyDictionary()
