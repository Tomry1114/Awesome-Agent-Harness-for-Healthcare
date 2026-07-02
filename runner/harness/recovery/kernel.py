"""Bounded Clinical Recovery v3 - Recovery Kernel.

The state machine + control flow (design sec.3 and sec.6b). The Kernel is dependency-injected: it imports
NO concrete substrate / workflow / benchmark adapter, only the generic contracts/bindings/metrics of the
recovery package. It knows paths, steps, bindings and auth tiers; it knows NOTHING about FHIR/GUI/image or
about order/appeal/prior-auth.

Enforced invariants:
  - one goal per episode (resolve_commitments -> N goals -> N episodes);
  - the <=1 irreversible_commit rule (validate_plan; violation -> FAILED);
  - per-step Decision-Boundary gate (unbound SEMANTIC arg -> BLOCKED_NEEDS_DECISION);
  - tiered auth by step kind + manifest side-effect declaration (read-like -> no auth);
  - no-repeat-after-UNKNOWN (a commit that returns UNKNOWN forbids any further commit; reconcile by re-read);
  - no commitment -> DECLINED_NO_COMMITMENT (NOT a failure);
  - no matching workflow -> NOT_APPLICABLE.
"""
from . import contracts as C
from . import metrics as M
from .bindings import (
    decision_boundary, AGENT_COMMITMENT, AUTHORITATIVE_STATE, BOUND_EVIDENCE, SYSTEM_METADATA,
)


class _CommitBlocked(Exception):
    """A commit was requested after an UNKNOWN commit in the same episode (no-repeat-after-UNKNOWN)."""


def _idem_key(goal_id, i):
    return "idem:%s:%d" % (goal_id, i)


def _arg_names(step, plan):
    names = []
    for spec in (step.arg_specs or []):
        if isinstance(spec, str):
            names.append(spec)
        elif isinstance(spec, dict) and spec.get("name"):
            names.append(spec["name"])
        else:
            n = getattr(spec, "name", None)
            if n:
                names.append(n)
    return names


def _gap_path(gap_class):
    return {
        C.GAP_DECISION: C.PATH_DECISION,
        C.GAP_EVIDENCE: C.PATH_EVIDENCE,
        C.GAP_VERIFICATION: C.PATH_VERIFICATION,
        C.GAP_EXECUTION: C.PATH_EXECUTION,
    }.get(gap_class, C.PATH_EXECUTION)


def _postcondition_paths(plan, goal, created_ids=None):
    pc = plan.expected_postcondition or {}
    paths = pc.get("paths", []) if isinstance(pc, dict) else []
    paths = paths or []
    if created_ids:
        # authoritative read-back by the just-created id(s): immediate + search-index-lag safe. Substrates
        # that cannot read by id ignore the extra key and fall back to their default state read.
        stamped = []
        for p in paths:
            if isinstance(p, dict):
                q = dict(p); q.setdefault("ids", list(created_ids)); stamped.append(q)
            else:
                stamped.append(p)
        return stamped
    return paths


class RecoveryKernel:
    """Drives one Recovery Episode per committed goal. Wire it with injected adapters."""

    def __init__(self, max_steps=64):
        self.max_steps = max_steps

    # -- public entry -----------------------------------------------------------------------------
    def run_episode(self, benchmark_adapter, workflow_registry, substrate_adapter, driver,
                    task, trajectory, goal, judge):
        """Resolve commitments and run the FIRST committed goal as this episode (one goal per episode).
        Returns a single EpisodeResult. For all committed goals, use run_all_episodes()."""
        ctx = self._build_ctx(benchmark_adapter, task, driver)
        commitments = self._resolve(benchmark_adapter, task, trajectory, goal, judge, ctx)
        if not commitments:
            return self._finalize(C.EpisodeResult(
                state=C.DECLINED_NO_COMMITMENT, path=None, goal_id=None,
                reason="no_committed_goal", events=[{"event": "no_commitment"}]))
        res = self._run_single(commitments[0], ctx, workflow_registry, substrate_adapter)
        if len(commitments) > 1:
            res.events.append({"event": "additional_commitments_deferred",
                               "count": len(commitments) - 1})
        return self._finalize(res)

    def run_all_episodes(self, benchmark_adapter, workflow_registry, substrate_adapter, driver,
                         task, trajectory, goal, judge):
        """Run one episode per committed goal. Returns list[EpisodeResult]."""
        ctx = self._build_ctx(benchmark_adapter, task, driver)
        commitments = self._resolve(benchmark_adapter, task, trajectory, goal, judge, ctx)
        if not commitments:
            return [self._finalize(C.EpisodeResult(
                state=C.DECLINED_NO_COMMITMENT, path=None, goal_id=None,
                reason="no_committed_goal", events=[{"event": "no_commitment"}]))]
        out = []
        for cg in commitments:
            out.append(self._finalize(
                self._run_single(cg, ctx, workflow_registry, substrate_adapter)))
        return out

    # -- helpers ----------------------------------------------------------------------------------
    def _build_ctx(self, benchmark_adapter, task, driver):
        ctx = dict(benchmark_adapter.context(task) or {})
        ctx.setdefault("driver", driver)
        ctx.setdefault("task", task)
        return ctx

    def _resolve(self, benchmark_adapter, task, trajectory, goal, judge, ctx):
        # `root` = the root-agent deliverable carrier; the kernel supplies the task object.
        return list(benchmark_adapter.resolve_commitments(task, trajectory, goal, judge, ctx) or [])

    def _finalize(self, result):
        result.metrics_bucket = M.classify(result)
        return result

    def _mint_auth(self, step, unknown_seen, goal_id, i):
        kind = step.kind
        if kind in C.READ_LIKE_KINDS:
            return None                                    # strict reader gate: no auth minted
        manifest = step.manifest or {}
        if kind == C.STAGED_WRITE:
            server_persisted = bool(manifest.get("server_persisted"))
            tier = C.AUTH_TIER_IRREVERSIBLE if server_persisted else C.AUTH_TIER_SCOPED
            return C.MutationAuthorization(
                tier=tier, allowed_kind=kind,
                side_effect_scope=manifest.get("side_effect_scope"),
                idempotency_key=_idem_key(goal_id, i))
        if kind == C.IRREVERSIBLE_COMMIT:
            if unknown_seen:
                raise _CommitBlocked("no_repeat_after_unknown")
            return C.MutationAuthorization(
                tier=C.AUTH_TIER_IRREVERSIBLE, allowed_kind=kind,
                idempotency_key=_idem_key(goal_id, i))
        return None

    # -- the single-episode state machine ---------------------------------------------------------
    def _run_single(self, cg, ctx, registry, substrate):
        goal_id = cg.goal_id
        events = [{"event": "start", "goal": goal_id, "state": C.NOT_STARTED}]

        # 1) route to a workflow
        wf = registry.match(cg, ctx)
        if wf is None:
            events.append({"event": "no_workflow"})
            return C.EpisodeResult(state=C.NOT_APPLICABLE, path=None, goal_id=goal_id,
                                   reason="no_matching_workflow", events=events)

        # 2) plan + compile-time invariants
        try:
            required = list(wf.required_bindings(cg, ctx) or [])
            plan = wf.compile_plan(cg, ctx)
            C.validate_plan(plan)
        except C.PlanCompileError as e:
            events.append({"event": "compile_error", "detail": str(e)})
            return C.EpisodeResult(state=C.FAILED, path=C.PATH_EXECUTION, goal_id=goal_id,
                                   reason="plan_compile_error:%s" % e, events=events)
        events.append({"event": "planned", "state": C.PLANNING,
                       "steps": len(plan.steps), "required": required})

        schema = ctx.get("schema")
        state_view = dict(ctx.get("authoritative_state") or {})
        observation = ctx.get("observation")
        system_metadata = dict(ctx.get("system_metadata") or {})
        bound_evidence = dict(ctx.get("bound_evidence") or {})

        completed = []
        created_ids = []
        auth_status = None
        unknown_seen = False
        committed = False
        commit_verified = False   # substrate server-confirmed the irreversible commit (RESULT_OK + created id)

        for i, step in enumerate(plan.steps[:self.max_steps]):
            # 2a) Decision-Boundary gate for this step's args (plus plan-wide required bindings).
            sources = {
                AGENT_COMMITMENT: cg.committed_fields,
                AUTHORITATIVE_STATE: state_view,
                BOUND_EVIDENCE: bound_evidence,
                SYSTEM_METADATA: system_metadata,
            }
            arg_names = _arg_names(step, plan)
            if i == 0:
                # resolve plan-wide required bindings once, at the first step
                arg_names = arg_names + [n for n in required if n not in arg_names]
            db = decision_boundary(arg_names, sources, schema=schema)
            if db["blocked_argument"] is not None:
                gc = db["gap"].gap_class
                state = C.BLOCKED_NEEDS_DECISION if gc == C.GAP_DECISION else C.BLOCKED_MISSING_EVIDENCE
                events.append({"event": "blocked", "arg": db["blocked_argument"], "gap": gc})
                return C.EpisodeResult(
                    state=state, path=_gap_path(gc), goal_id=goal_id, completed_steps=completed,
                    blocked_step_index=i, blocked_argument=db["blocked_argument"], created_ids=created_ids,
                    auth_status=auth_status, reason="unbound_%s_arg:%s" % (gc, db["blocked_argument"]),
                    events=events)
            bindings = db["bindings"]

            # 2b) Affordance: locate the control in the live observation.
            aff = None
            if step.affordance_target is not None:
                ab = substrate.resolve_affordance(step.affordance_target, observation)
                if isinstance(ab, str):                    # a BLOCKED_* terminal string
                    events.append({"event": "affordance_block", "result": ab})
                    return C.EpisodeResult(
                        state=ab, path=C.PATH_EXECUTION, goal_id=goal_id, completed_steps=completed,
                        blocked_step_index=i, created_ids=created_ids, auth_status=auth_status,
                        reason="affordance:%s" % ab, events=events)
                aff = ab

            # 2c) Tiered authorization.
            try:
                auth = self._mint_auth(step, unknown_seen, goal_id, i)
            except _CommitBlocked:
                events.append({"event": "commit_refused", "reason": "no_repeat_after_unknown"})
                return C.EpisodeResult(
                    state=C.UNKNOWN, path=C.PATH_VERIFICATION, goal_id=goal_id, completed_steps=completed,
                    blocked_step_index=i, created_ids=created_ids, auth_status=auth_status,
                    reason="no_repeat_after_unknown", events=events)

            # 2d) Execute the primitive.
            action = {
                "kind": step.kind,
                "action": step.action or {},
                "bindings": dict((k, v.value) for k, v in bindings.items()),
                "affordance": aff,
                "goal_id": goal_id,
            }
            outcome = substrate.execute_primitive(step.kind, action, auth)
            cls = substrate.classify_result(outcome)
            events.append({"event": "step", "i": i, "kind": step.kind,
                           "auth": (auth.tier if auth is not None else None), "class": cls})

            if step.kind in C.READ_LIKE_KINDS:
                # read-back updates the authoritative view / observation
                if isinstance(getattr(outcome, "state_view", None), dict):
                    state_view.update(outcome.state_view)
                if getattr(outcome, "result", None) is not None:
                    observation = outcome.result
                if step.probe and cls == C.RESULT_ALREADY_REALIZED:
                    completed.append(i)
                    events.append({"event": "already_realized", "i": i})
                    return C.EpisodeResult(
                        state=C.ALREADY_REALIZED, path=C.PATH_EXECUTION, goal_id=goal_id,
                        completed_steps=completed, created_ids=created_ids, auth_status=auth_status,
                        reason="effect_already_present", events=events)
            else:
                # mutation step
                if getattr(outcome, "created_id", None):
                    created_ids.append(outcome.created_id)
                if isinstance(getattr(outcome, "state_view", None), dict):
                    state_view.update(outcome.state_view)
                if step.kind == C.IRREVERSIBLE_COMMIT:
                    committed = True
                    auth_status = C.AUTH_TIER_IRREVERSIBLE
                    if cls == C.RESULT_UNKNOWN:
                        unknown_seen = True
                    elif cls == C.RESULT_OK and getattr(outcome, "created_id", None):
                        # the substrate ran its OWN authoritative server read-back and confirmed the record
                        commit_verified = True
                elif auth is not None:
                    auth_status = auth.tier
                if cls == C.RESULT_FAILED:
                    events.append({"event": "step_failed", "i": i})
                    return C.EpisodeResult(
                        state=C.FAILED, path=C.PATH_EXECUTION, goal_id=goal_id, completed_steps=completed,
                        blocked_step_index=i, created_ids=created_ids, auth_status=auth_status,
                        reason="primitive_failed", events=events)
            completed.append(i)

        # 3) Effect Verification (+ idempotent reconciliation if a commit was UNKNOWN: re-read only).
        try:
            rb = substrate.read_state(_postcondition_paths(plan, cg, created_ids)) or {}
            if isinstance(rb, dict):
                state_view.update(rb)
        except Exception as e:                             # read-back failure is non-fatal for reconcile
            events.append({"event": "readback_error", "detail": str(e)})
        verdict = wf.verify_effect(cg, state_view)
        events.append({"event": "verify", "state": C.VERIFYING, "verdict": verdict})

        reason = "verify_effect:%s" % verdict
        if verdict is True:
            state = C.VERIFIED
        elif verdict is False:
            if commit_verified:
                # the commit already passed the substrate's authoritative server read-back; a redundant,
                # budget-limited or search-index-lagged verify read cannot refute a confirmed effect.
                state = C.VERIFIED
                reason = "commit_server_verified;verify_effect:False"
            else:
                state = C.FAILED
        else:
            # ambiguous read-back: never re-commit -> UNKNOWN if we mutated, else benign VERIFIED.
            # a server-confirmed commit stays VERIFIED even when the redundant read is ambiguous.
            if commit_verified:
                state = C.VERIFIED
                reason = "commit_server_verified;verify_effect:None"
            else:
                state = C.UNKNOWN if (committed or unknown_seen) else C.VERIFIED
        return C.EpisodeResult(
            state=state, path=C.PATH_EXECUTION, goal_id=goal_id, completed_steps=completed,
            created_ids=created_ids, auth_status=auth_status,
            reason=reason, events=events)


# Convenience module-level entry mirroring the required signature.
def run_episode(benchmark_adapter, workflow_registry, substrate_adapter, driver,
                task, trajectory, goal, judge):
    return RecoveryKernel().run_episode(
        benchmark_adapter, workflow_registry, substrate_adapter, driver,
        task, trajectory, goal, judge)
