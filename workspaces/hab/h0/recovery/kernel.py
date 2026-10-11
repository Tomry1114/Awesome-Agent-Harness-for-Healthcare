"""Bounded Clinical Recovery - Recovery Kernel (workspace strategy, HealthAdminBench default workspace).

The state machine + control flow. The Kernel is dependency-injected: it imports
NO concrete substrate / workflow / benchmark adapter, only the generic contracts/bindings/metrics of the
recovery package. It knows paths, steps, bindings and auth tiers; it knows NOTHING about FHIR/GUI/image or
about order/appeal/prior-auth.

Enforced invariants:
  - one goal per episode (resolve_commitments -> N goals -> N episodes);
  - tiered auth by step kind + manifest side-effect declaration (read-like -> no auth); the strategy's own tiers --
    what gates execution is the shell's ledger authorization in the harness action service;
  - no-repeat-after-UNKNOWN (a commit that returns UNKNOWN forbids any further commit; reconcile by re-read); the
    shell refuses any harness irreversible commit after an unconfirmed one for the rest of the episode
    (irreversible_commit_after_unconfirmed);
  - no commitment -> DECLINED_NO_COMMITMENT (NOT a failure);
  - no matching workflow -> NOT_APPLICABLE.
"""
import workspace_api as _W

C = _W.load_relative(__file__, "contracts.py")
M = _W.load_relative(__file__, "metrics.py")


class _CommitBlocked(Exception):
    """A commit was requested after an UNKNOWN commit in the same episode (no-repeat-after-UNKNOWN)."""


def _idem_key(goal_id, i):
    return "idem:%s:%d" % (goal_id, i)


class RecoveryKernel:
    """Drives one Recovery Episode per committed goal. Wire it with injected adapters."""

    def __init__(self, max_steps=64):
        self.max_steps = max_steps

    # -- public entry -----------------------------------------------------------------------------
    def run_all_episodes(self, benchmark_adapter, workflow_registry, substrate_adapter, driver,
                         task, trajectory, goal, judge, events=None):
        """Run one episode per committed goal. Returns list[EpisodeResult]."""
        ctx = self._build_ctx(benchmark_adapter, task, driver)
        commitments = self._resolve(benchmark_adapter, task, trajectory, goal, judge, ctx)
        if not commitments:
            return [self._finalize(C.EpisodeResult(
                state=C.DECLINED_NO_COMMITMENT, path=None, goal_id=None,
                reason="no_committed_goal", events=[{"event": "no_commitment"}]))]
        out = []
        _consec_blocked = 0
        for cg in commitments:
            r = self._finalize(self._run_single(cg, ctx, workflow_registry, substrate_adapter))
            out.append(r)
            # Bound UNPRODUCTIVE churn WITHOUT cutting productive recovery. A BLOCKED_UNRESOLVED_AFFORDANCE
            # episode wandered and changed nothing; 2 in a row means the remaining commitments would wander the
            # same way -> stop (e.g. a fax task whose episodes keep getting lost in orders/history pages, which can
            # turn a correct agent answer incorrect). ANY other outcome (VERIFIED / PARTIAL / acquire / DECLINED) RESETS the
            # counter, so a productive episode -- e.g. the denial acquire that flips viewedRemittanceImage -- is
            # never cut. Recovery runs at before_final (after the agent), so stopping the wander preserves the
            # agent's own F->T and reduces harness-induced T->F.
            if r.state == C.BLOCKED_UNRESOLVED_AFFORDANCE:
                _consec_blocked += 1
                if _consec_blocked >= 2:
                    if events is not None:
                        events.append({"event": "recovery_churn_stop", "after_episodes": len(out),
                                       "reason": "2_consecutive_blocked_unresolved_affordance"})
                    break
            else:
                _consec_blocked = 0
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

    # -- dynamic closed-loop episode (observe -> decide next action -> execute -> re-observe) -------
    def _run_dynamic(self, cg, ctx, wf, substrate, events=None):
        goal_id = cg.goal_id
        events = events if events is not None else [{"event": "start", "goal": goal_id}]
        driver_shim = getattr(substrate, "driver", None)
        history = []
        _fail_tags = {}
        _total_fail = 0
        created_ids = []
        committed = False
        commit_verified = False
        unknown_seen = False
        auth_status = None

        for i in range(self.max_steps):
            # OBSERVE: fresh observation string + live control model + persisted state.
            observation = None
            try:
                snap = substrate.execute_primitive(
                    C.READ, {"kind": C.READ, "action": {"op": "snapshot"}, "affordance": None}, None)
                res = getattr(snap, "result", None)
                observation = res.get("observation") if isinstance(res, dict) else res
            except Exception:
                pass
            controls = []
            if driver_shim is not None and hasattr(driver_shim, "list_controls"):
                try:
                    controls = driver_shim.list_controls() or []
                except Exception:
                    controls = []
            try:
                state_view = substrate.read_state([]) or {}
            except Exception:
                state_view = {}

            # DECIDE the ONE next action.
            try:
                decision = wf.next_action(cg, ctx, observation, controls, state_view, history) or {}
            except Exception as e:
                events.append({"event": "next_action_error", "detail": repr(e)})
                decision = {"kind": "block", "reason": "next_action_error"}
            dk = decision.get("kind")
            events.append({"event": "decide", "i": i, "kind": dk,
                           "reason": decision.get("reason"), "tag": decision.get("tag")})

            if dk == "done":
                break
            if dk == "block":
                return C.EpisodeResult(
                    state=C.BLOCKED_NEEDS_DECISION, path=C.PATH_EXECUTION, goal_id=goal_id,
                    completed_steps=history, created_ids=created_ids, auth_status=auth_status,
                    reason="dynamic_block:%s" % decision.get("reason"), events=events)
            step = decision.get("step")
            if step is None:
                break

            # AFFORDANCE: the control model (list_controls) already carries the fresh data-mh-ref that THIS
            # iteration's snapshot set -> click by that ref directly, so the acted-on element is exactly the
            # one the decision was made from. (Re-resolving the role/label against the separate observation
            # string mismatches, causing spurious BLOCKED_UNRESOLVED_AFFORDANCE.) Fall back to observation
            # resolution only when no ref is bound.
            aff = None
            if step.affordance_target is not None:
                _tgt = step.affordance_target
                _ref = _tgt.get("ref") if isinstance(_tgt, dict) else None
                if _ref is not None:
                    aff = C.AffordanceBinding(target_spec=_tgt, ref=_ref, observation_hash="")
                else:
                    ab = substrate.resolve_affordance(_tgt, observation)
                    if isinstance(ab, str):
                        events.append({"event": "affordance_block", "result": ab, "target": _tgt})
                        return C.EpisodeResult(
                            state=ab, path=C.PATH_EXECUTION, goal_id=goal_id, completed_steps=history,
                            created_ids=created_ids, auth_status=auth_status,
                            reason="affordance:%s" % ab, events=events)
                    aff = ab

            # AUTH (scoped for staged, single-use irreversible for a commit).
            try:
                auth = self._mint_auth(step, unknown_seen, goal_id, i)
            except _CommitBlocked:
                return C.EpisodeResult(
                    state=C.UNKNOWN, path=C.PATH_VERIFICATION, goal_id=goal_id,
                    completed_steps=history, created_ids=created_ids, auth_status=auth_status,
                    reason="no_repeat_after_unknown", events=events)

            # EXECUTE.
            action = {"kind": step.kind, "action": step.action or {}, "bindings": {},
                      "affordance": aff, "goal_id": goal_id}
            outcome = substrate.execute_primitive(step.kind, action, auth)
            cls = substrate.classify_result(outcome)
            events.append({"event": "exec", "i": i, "kind": step.kind, "class": cls,
                           "tag": decision.get("tag")})
            if getattr(outcome, "created_id", None):
                created_ids.append(outcome.created_id)
            if step.kind == C.IRREVERSIBLE_COMMIT:
                committed = True
                auth_status = C.AUTH_TIER_IRREVERSIBLE
                if cls == C.RESULT_UNKNOWN:
                    unknown_seen = True
                elif cls == C.RESULT_OK:
                    # A RESULT_OK commit is confirmed (executed + reconciled). created_id is a
                    # FHIR-specific server-echo, NOT a requirement -- a GUI submit reconciles via a snapshot and
                    # carries no created_id, so requiring it would score real GUI submits as unconfirmed.
                    commit_verified = True
            _tag = decision.get("tag") or ("step%d" % i)
            if cls == C.RESULT_OK:
                history.append(_tag)                          # only a SUCCEEDED action counts as done
            else:
                _fail_tags[_tag] = _fail_tags.get(_tag, 0) + 1
                _total_fail += 1
                if _fail_tags[_tag] >= 2:
                    history.append(_tag)                      # give up retrying this specific action after 2 tries
                if _total_fail >= 4:                          # runaway guard: too many failed actions -> stop, verify
                    events.append({"event": "dynamic_abort", "reason": "too_many_failed_actions"})
                    break

        # VERIFY against the workflow's outcome-related postcondition.
        try:
            state_view = substrate.read_state([]) or {}
        except Exception:
            state_view = {}
        verdict = wf.verify_effect(cg, state_view)
        events.append({"event": "verify", "verdict": verdict, "committed": committed})
        reason = "verify_effect:%s" % verdict
        # The COMPOSITE postcondition (verify_effect over verify_all) is AUTHORITATIVE. A single
        # irreversible commit being server-confirmed does NOT verify the whole workflow -- payer submit landed
        # but EHR documentation missing is PARTIAL (effect recovered, task incomplete), never VERIFIED.
        if verdict is True:
            state = C.VERIFIED
        elif verdict is False:
            if commit_verified:
                state = C.PARTIAL
                reason = "partial_effect;commit_verified;verify_effect:False"
            else:
                state = C.FAILED
        else:
            if commit_verified:
                state = C.PARTIAL
                reason = "partial_effect;commit_verified;verify_effect:None"
            else:
                # Verdict None means the composite could NOT be verified. With nothing landed
                # that is UNKNOWN, never a free VERIFIED (an already-realized goal is caught earlier as done).
                state = C.UNKNOWN
        verified_level = None
        if state == C.VERIFIED:
            try:
                _pc = wf.payload_covered(cg, state_view) if hasattr(wf, "payload_covered") else None
            except Exception:
                _pc = None
            verified_level = "payload" if _pc is True else "effect"
            reason = reason if "level=" in reason else (reason + ";level=" + verified_level)
        return C.EpisodeResult(
            state=state, path=C.PATH_EXECUTION, goal_id=goal_id, completed_steps=history,
            created_ids=created_ids, auth_status=auth_status, reason=reason, events=events,
            verified_level=verified_level)

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

        # 1b) dynamic closed loop: the workflow's next_action drives observe->decide->execute->re-observe.
        return self._run_dynamic(cg, ctx, wf, substrate, events)
