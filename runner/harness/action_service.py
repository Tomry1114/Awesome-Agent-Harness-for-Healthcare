# -*- coding: utf-8 -*-
"""Harness action channel: the one path every harness-origin environment action takes (fixed shell).

A workspace strategy, a workspace next_read, a kernel ACQUIRE or the deliverable enforcement decides WHAT
to do; this channel decides WHETHER it may run, runs it, records it and guards the agent's content after it. The
requesters (runner/harness/origin.py):

    recovery                   ctx.request_action / ctx.read, inside the invocation a hook opened (begin_harness_actions
                               in before_final, post_stop_allowance in after_episode) and only while that hook runs
    next_read, kernel_acquire, on_answer_read
                               the reads the loop dispatches for a host next_read Read, a kernel ACQUIRE (PB's forced
                               broad read included) and an on_answer Read (dispatch_read)
    forced_deliverable_write   the after_episode deliverable write (forced_write)
    content_guard              the restore of agent-authored text after any of the above

For every request, in order:

  0. scope                 an invocation request needs its hook's invocation still open. The shell ends the invocation
                           when the middleware hook that opened it returns and closes the hook's scope when the loop is
                           done with the hook, so a context kept past its hook cannot act, and a second begin in the same
                           hook call resets nothing
  1. ceilings              shell constants, checked before every action, restores included: HARNESS_TOTAL_CAP harness env
                           actions per task over every requester, RECOVERY_TOTAL_CAP of them requested through invocations
                           (their restores included); past the agent's budget plus the additive reserve (max_steps //
                           RECOVERY_RESERVE_DIVISOR where the adapter grants a reserve), RECOVERY_COMPLETION_ALLOWANCE
                           non-snapshot actions per invocation. Harness actions are counted here, never in the agent's
                           budget
  2. classification        the manifest classifies the action on the page it is issued on (semantics.canonicalize_on_page).
                           A READ must be read-only: a mapped rule of semantic type read / inspect with effect none
                           (read_not_read_only; a loop read that is an irreversible commit is
                           irreversible_commit_on_read_channel). An invocation READ must also name its target and
                           passes actuation.plan at stage "recovery"
  3. irreversible commit   the harness never makes a new irreversible commit on its own initiative. It runs only when
                           no harness commit of the episode is unconfirmed (the memory is the episode's; begin() never
                           resets it), the page holds a non-empty payload, the agent itself attempted the same commit
                           (same control on the same portal, agent origin) and that commit has not landed
                           (obligation_gate.decision_landed)
  4. write provenance      a semantic value traces to agent_commitment / authoritative_state / bound_evidence,
                           never to system_metadata alone
  5. kernel before_action  origin harness with the requester; must be ALLOW. A write carries a ledger authorization bound
                           by id (_mutation_authorization_id) that verify_commit adjudicates under a mutation hold, then the
                           channel reserves it. A raising hook refuses the action (before_action_error)
  6. the executor          dispatches the reserved authorization immediately before the env call
  7. kernel after_action   with the obligation snapshot run.py takes for agent actions; the authorization is finalized
  8. the tool_call event   origin=harness, requester, audience (both for a loop read, whose result the agent is handed)
  9. content guard         content the action cleared is written back through the executor and recorded (a content_guard
                           `type` for a control still on the page, restore_agent_content for persisted state). Never
                           the harness's own text: a value is harness-authored when the last executed write
                           naming that control was a harness write with that text, or, for persisted content and a
                           control no write named, when only the harness ever wrote that text; it is skipped and named
                           (harness_authored). A restore of the agent's own text carries content_author agent. A restore
                           is a harness action: it passes the per-task caps, and one that no longer fits is refused and
                           named, never run past a cap

A refusal is recorded as harness_action_refused and handed back as {"error": <reason>} (dispatch_read and forced_write
return a DispatchOutcome). Kernel decisions on a harness-origin action are recorded with action_origin=harness and the
requester and draw neither the agent's intervention nor its revision budget (HarnessKernel._decide_effective). A
post-action ESCALATE, or an exception in after_action, stops the channel for the rest of the task.
"""
import collections
import re

from . import actuation as _act
from .authorization import action_target_path, MUTATION_TYPES
from .origin import is_harness_event as _is_harness

HARNESS_TOTAL_CAP = 40                  # harness env actions per task, every requester, restores included
RECOVERY_TOTAL_CAP = 20                 # of which requested through invocations (recovery), their restores included
RECOVERY_COMPLETION_ALLOWANCE = 12      # non-snapshot actions per invocation once agent + harness pass the budget
RECOVERY_RESERVE_DIVISOR = 6            # additive reserve = max_steps // 6 where the adapter grants one; the post-stop
                                        # re-grant is the same
READ_ONLY_TYPES = ("read", "inspect")   # a READ's manifest semantic types (effect none)

STAGE_SETTLE_MS = 400                   # after a page-changing action, before the next read
READ_SETTLE_MS = 1200                   # before reading persisted state (the portal writes it in a post-render effect)
_SETTLE_TOOLS = ("click", "submit", "select", "upload", "navigate", "type")
_VALUE_TOOLS = ("type", "fill", "select")
_TARGET_KEYS = ("ref", "selector", "target", "id", "field", "url")   # what a READ may name as the thing it reads
RESTORE_TOOL = "restore_agent_content"
_ORIGIN = "harness"                      # every action this channel runs (runner/harness/origin.py) ...
_REQUESTER = "recovery"                 # ... asked for by a workspace strategy inside an invocation
_GUARD_REQUESTER = "content_guard"      # ... or by the post-action content guard
_FORCED_WRITE = "forced_deliverable_write"
INVOCATION, LOOP, FORCED = "invocation", "loop", "forced"   # the request kinds


def read_only(sem):
    """A READ the manifest admits: a mapped rule of semantic type read / inspect whose effect is none."""
    return (bool(getattr(sem, "mapped", False)) and getattr(sem, "semantic_type", None) in READ_ONLY_TYPES
            and getattr(sem, "effect", None) in (None, "none"))


class HookScope(object):
    """The harness-action scope of one hook call. The context services of that call hold it: `live` until the loop
    is done with the hook (HarnessActionService.close_scope), `invocation` the invocation this scope opened while it is
    open (ended when the middleware hook that opened it returns)."""
    __slots__ = ("live", "invocation", "opened")

    def __init__(self):
        self.live, self.invocation, self.opened = True, None, False


class DispatchOutcome(object):
    """What dispatch_read / forced_write report. `refusal` is the reason a refused request did not run (None when it
    ran); `verdict` the kernel's non-ALLOW before_action verdict that refused it, if that was the reason; `outcome`,
    `event`, `obs` and `after` (the kernel's after_action verdict) belong to an executed action."""
    __slots__ = ("refusal", "verdict", "outcome", "event", "obs", "after")

    def __init__(self, refusal=None, verdict=None, outcome=None, event=None, obs=None, after=None):
        self.refusal, self.verdict, self.outcome, self.event, self.obs, self.after = (refusal, verdict, outcome, event,
                                                                                      obs, after)

    @property
    def executed(self):
        return self.refusal is None


class _ReadRequest(object):
    """The READ actuator's input shape for a harness read the service was asked for. No kernel capability raised it, so
    its telemetry record has capability None (like the HAB read-acquire records) and is attributed by stage "recovery".
    (A string next to those None owners would also make actuation's per-capability dict unsortable.)"""

    def __init__(self, tool, args):
        self.extra = {"next_action": {"tool": tool, "args": dict(args or {}), "read_only": True}}
        self.capability = None


class HarnessActionService(object):
    def __init__(self, harness, executor, env, task, trajectory, max_steps, agent_actions, reserve, errors,
                 obligations=None, observation=None, *, workspace, reserve_granted=None, reconcile=None):
        self.h = harness
        self.workspace = workspace                   # the run's workspace: its completion content names the decisions
        self.ex = executor
        self.env = env
        self.task = task if isinstance(task, dict) else {}
        self.trajectory = trajectory
        self.max_steps = max_steps
        self._agent_actions = agent_actions          # () -> the agent's own env actions so far
        self._reserve = reserve                      # () -> the substrate's additive recovery reserve
        self._reserve_granted = reserve_granted      # whether the adapter grants the reserve (None: reserve() > 0)
        self.errors = errors                         # run.py's harness runtime error list
        self._obligations = obligations              # () -> conjunct tuple | None; None when no obligation model
        self._observation = observation              # () -> what the kernel sees at before_action
        self.step = 0
        self.episode_id = None
        self._seq = 0
        self._allowance_used = 0
        self._open = None                            # the open invocation (a token), or None
        self._commit_attempts = []                   # the EPISODE's harness irreversible commits: (control, landed)
        self._regranted = None                       # the post-stop re-grant of the reserve, once given
        self._post_stop_granted = False
        self.n_total = 0                             # harness env actions executed, every requester (restores included)
        self.n_invocation = 0                        # of which for invocation requests (with their restores)
        self.by_requester = collections.Counter()    # executed harness env actions by requester
        self.stopped = None
        self.reconcile = reconcile                   # the episode's reconcile state, shared with the agent

    # ---------------------------------------------------------------------------------------------- lifecycle
    def scope(self):
        """A new hook-call scope for a middleware context's harness-action services (middleware_host)."""
        return HookScope()

    def begin(self, step, scope=None):
        """Open one invocation at an agent final -> `step`, or None when no harness action could run: a closed scope, no
        kernel, or a ceiling already spent. A scope whose invocation is still open gets that invocation back: nothing is
        reset. The completion allowance and the action_id sequence are per invocation; the counters, the memory of the
        harness's irreversible commits and the post-stop re-grant are the episode's and are never reset."""
        if scope is not None and not scope.live:
            return None
        if self.h is None:
            return None
        if scope is not None and scope.invocation is not None and scope.invocation is self._open:
            return self.step
        if self.n_total >= HARNESS_TOTAL_CAP or self.n_invocation >= RECOVERY_TOTAL_CAP:
            return None
        token = object()
        self._open = token
        if scope is not None:
            scope.invocation, scope.opened = token, True
        self.step = step
        self._seq = 0
        self._allowance_used = 0
        return step

    def post_stop_allowance(self, step, scope=None):
        """The loop stopped at tool_budget_exhausted: once per task, and only where the adapter grants the additive
        reserve, re-grant it at its ceiling (max_steps // RECOVERY_RESERVE_DIVISOR) and open one invocation at the
        stop step (begin). -> the step, or None."""
        if self.h is None or self._post_stop_granted or not self.reserve_granted():
            return None
        if scope is not None and not scope.live:
            return None
        self._post_stop_granted = True
        self._regranted = self.max_steps // RECOVERY_RESERVE_DIVISOR
        return self.begin(step, scope)

    def end_invocation(self, scope=None):
        """The middleware hook that opened the invocation returned: it is closed (a request through it is refused)."""
        if scope is None:
            self._open = None
            return
        if scope.invocation is not None and scope.invocation is self._open:
            self._open = None
        scope.invocation = None

    def close_scope(self, scope):
        """The loop is done with the hook: its scope opens nothing more."""
        self.end_invocation(scope)
        scope.live = False

    def invocation_open(self, scope=None):
        if self._open is None:
            return False
        return scope is None or scope.invocation is self._open

    def reserve_granted(self):
        if self._reserve_granted is not None:
            return bool(self._reserve_granted)
        try:
            return bool(self._reserve())
        except Exception:
            return False

    def reserve(self):
        """The additive recovery reserve now: the substrate's, or the post-stop re-grant once it was given."""
        return self._reserve() if self._regranted is None else self._regranted

    def budget_open(self):
        try:
            return (self._agent_actions() + self.n_total) < (self.max_steps + self.reserve())
        except Exception:
            return False                              # admission unknown -> closed (a harness action never fails open)

    def harness_actions(self):
        return self.n_total

    def recovery_actions(self):
        return self.n_invocation

    def counts(self):
        """The record run.py writes: harness env actions in total, through invocations, and by requester."""
        return {"harness_env_actions": self.n_total, "recovery_env_actions": self.n_invocation,
                "harness_env_actions_by_requester": dict(sorted(self.by_requester.items()))}

    # ---------------------------------------------------------------------------------------------- read-only views
    def list_controls(self):
        """The live control model of the current page ([{ref, role, label, name, required, value, ...}])."""
        page = getattr(self.env, "page", None)
        if page is None:
            return []
        try:
            from adapters.hab.gui_control_model import CONTROL_JS
            ctrls = page.evaluate(CONTROL_JS)
            return ctrls if isinstance(ctrls, list) else []
        except Exception:
            return []

    def read_recovery_state(self):
        """Persisted portal state (a page read, not a tool call) plus the page's current control values as a separate
        draft_ui_state namespace: a draft value is never persisted state."""
        page = getattr(self.env, "page", None)
        try:
            page.wait_for_timeout(READ_SETTLE_MS)
        except Exception:
            pass
        import json as _json
        fs, pa, pb, fax = {}, {}, {}, {}
        try:
            raw = page.evaluate(
                "() => { try { return localStorage.getItem('portals_state'); } catch(e){ return null; } }")
            st = _json.loads(raw) if raw else None
            if isinstance(st, dict):
                fs = st.get("emr") or {}
                pa = st.get("payerA") or {}
                pb = st.get("payerB") or {}
                fax = st.get("fax") or {}
        except Exception:
            fs = getattr(self.env, "full_state", None) or {}
        draft = {}
        try:
            for c in (self.list_controls() or []):
                nm = c.get("name") or c.get("label")
                val = c.get("value")
                if nm and val not in (None, "", [], "on"):
                    draft[str(nm)] = val
        except Exception:
            pass
        return {"full_state": fs, "payer_a_state": pa, "payer_b_state": pb, "fax": fax, "draft_ui_state": draft}

    def read_back(self, tool, args, result):
        """The environment's active read-back of a write's result (env.reconcile_write) -> {confirmed, detail} or None.
        A state read, not an action: nothing is dispatched or charged."""
        fn = getattr(self.env, "reconcile_write", None)
        if not callable(fn):
            return None
        try:
            return fn(tool, args or {}, result)
        except Exception:
            return None

    # ---------------------------------------------------------------------------------------------- requests
    def request_action(self, tool, args, scope=None):
        return self._invocation_request(tool, args, False, scope)

    def read(self, tool, args, scope=None):
        """An evidence-reveal READ: the READ actuator's admission and telemetry around the same path."""
        return self._invocation_request(tool, args, True, scope)

    def _invocation_request(self, tool, args, read, scope):
        args = dict(args or {})
        if not self.invocation_open(scope):
            return self._refuse(tool, args, "scope", "harness_invocation_closed", _REQUESTER, self.step)
        out = self._act(tool, args, _REQUESTER, read, self.step, INVOCATION)
        if out.refusal is not None:
            return {"error": out.refusal}
        res = out.outcome.res
        return res if isinstance(res, dict) else {"result": res}

    def dispatch_read(self, action, step):
        """A read the loop dispatches for a harness requester (the queued action of a next_read Read, a kernel ACQUIRE or
        an on_answer Read, carrying _harness_acquire and its _requester keys) -> DispatchOutcome. The loop records
        harness_acquire_dispatch once the read is admitted (ceilings and classification), before the kernel; it hands
        the executed read's result to the agent and closes its READ actuator record. A refusal closes that record."""
        from .origin import queued_requester
        requester = queued_requester(action)[0] or "next_read"
        return self._act(action.get("tool"), dict(action.get("args") or {}), requester, True, step, LOOP, queued=action)

    def forced_write(self, action, step):
        """The after_episode deliverable write (middleware_host.forced_deliverable_write) -> DispatchOutcome. A write like
        any harness write: ceilings, the kernel ceremony with the authorization, the executor, after_action and the
        record (content_author agent: the text is the agent's)."""
        return self._act(action.get("tool"), dict(action.get("args") or {}), _FORCED_WRITE, False, step, FORCED,
                         event_extra={"forced_deliverable": True, "content_author": "agent", "via_pipeline": True})

    def _act(self, tool, args, requester, read, step, kind, queued=None, event_extra=None):
        L = getattr(self.h, "ledger", None)
        manifest = getattr(self.h, "manifest", None)
        loop = kind == LOOP
        acquire_id = queued.get("_acquire_id") if loop else None

        def refused(stage, reason, verdict=None):
            self._refuse(tool, args, stage, reason, requester, step)
            if acquire_id is not None and L is not None:
                _act.record_execution(L, acquire_id, executed=False, failure_reason="refused:%s" % reason)
            return DispatchOutcome(refusal=reason, verdict=verdict)

        if self.stopped:
            return refused("service", "harness_actions_stopped:%s" % self.stopped)
        if L is None or not manifest:
            return refused("service", "no_substrate_manifest")
        reason, use_allowance = self._ceiling(tool, kind)
        if reason:
            return refused("ceiling", reason)
        from .semantics import canonicalize_on_page, page_evidence_needed
        action = {"type": "tool_call", "tool": tool, "args": args}
        page = self.list_controls() if page_evidence_needed(action, manifest) else None
        sem = canonicalize_on_page(action, manifest, page)   # a GUI commit is classified on the page it is issued on
        write = getattr(sem, "semantic_type", None) in MUTATION_TYPES
        irreversible = write and getattr(sem, "effect", None) == "irreversible"
        valid_before = None
        if read:
            if loop and irreversible:
                return refused("read_channel", "irreversible_commit_on_read_channel")
            if not read_only(sem):
                return refused("read_channel" if loop else "read_actuator", "read_not_read_only")
            if not loop:
                if not any(args.get(k) not in (None, "") for k in _TARGET_KEYS):
                    return refused("read_actuator", "read_without_target")
                plan = _act.plan(_ReadRequest(tool, args), _REQUESTER, step, ledger=L, vocabulary=self._vocabulary())
                self._extend(plan.events)
                acquire_id = plan.acquire_id if plan.dispatchable else None
                if not plan.dispatchable:
                    return refused("read_actuator", plan.failure_reason)
                valid_before = L.validated_evidence_version
        if write and self.reconcile is not None and self.reconcile.active:
            # an unconfirmed commit of either origin restricts every write until the agent reads it back
            return refused("reconcile", "unconfirmed_commit_pending")
        controls = page
        commit = None
        if irreversible:
            controls = self.list_controls() if controls is None else controls
            why, commit = self._irreversible_commit_refusal(action, controls)
            if why:
                return refused("irreversible_commit", why)
        if tool in _VALUE_TOOLS and _written_value(tool, args) not in (None, ""):
            controls = self.list_controls() if controls is None else controls
            why = self._provenance_refusal(tool, args, controls)
            if why:
                return refused("write_provenance", why)
        if loop:   # admitted: the loop's dispatch record, before the kernel (as run.py recorded it)
            self.trajectory.append({"step": step, "event_type": "harness_acquire_dispatch", "tool": tool,
                                    "args": queued.get("args"), "status": "ok"})

        # kernel before_action, with the authorization ceremony for a write
        auth = None
        bound = queued if loop else dict(action)
        if write:
            auth = L.mint_authorization(source="deterministic_gap", allowed_semantic_type=sem.semantic_type,
                                        allowed_tool=tool, allowed_effect=getattr(sem, "effect", None),
                                        target_path=action_target_path(sem, action),
                                        expected_postcondition={"harness_action": tool, "check": "state_transition"})
            bound["_mutation_authorization_id"] = auth.authorization_id
            bound["_recovery"] = "recovery_v3"
        prior_hold = (L.mutation_hold, L.mutation_hold_origin)
        if write and not L.mutation_hold:   # verify_commit adjudicates the bound authorization only under a hold
            L.set_mutation_hold(capability="harness_action_service")
        ob_before = self._obligation_snapshot()
        hb, hb_error = None, None
        try:
            hb = self.h.before_action(bound, self._observation() if self._observation else None, step=step,
                                      origin=_ORIGIN, requester=requester)
        except Exception as e:
            hb_error = e
        finally:
            matched = getattr(L, "pending_authorization", None)
            L.pending_authorization = None
            if write:
                L.mutation_hold, L.mutation_hold_origin = prior_hold
        if hb_error is not None:
            self.errors.append("harness_action.before_action: %r" % (hb_error,))
            if auth is not None:
                L.cancel_authorization(auth)
            return refused("kernel", "before_action_error")
        if hb is not None:
            self._extend(hb.events)
        repairs = getattr(L, "_reference_repairs_pending", None)
        if repairs:
            for r in repairs:
                self.trajectory.append({"step": step, "event_type": "reference_repaired", "stage": "before_action",
                                        "action_origin": _ORIGIN, "action_requester": requester, "path": r.get("path"),
                                        "stripped_ref": r.get("value"), "status": "ok"})
            L._reference_repairs_pending = []
        eff = getattr(hb, "type", "ALLOW") if hb is not None else "ALLOW"
        if eff != "ALLOW":
            if auth is not None:
                L.cancel_authorization(auth)
            raw = getattr(hb, "raw", None)
            return refused("kernel", "%s:%s" % (eff, getattr(raw, "rule_id", None) or getattr(raw, "reason_code", None)),
                           verdict=hb)
        if auth is not None:
            if matched is not auth:
                L.cancel_authorization(auth)
                return refused("authorization", "authorization_not_matched")
            if not L.reserve_authorization(auth):
                L.cancel_authorization(auth)
                return refused("authorization", "authorization_not_reserved")

        # the executor
        exec_action = queued if loop else {"type": "tool_call", "tool": bound["tool"], "args": bound.get("args", {})}
        before_content = self._content_snapshot() if (tool != "snapshot" and self._guards_content()) else None
        outcome = self.ex.execute_and_normalize(exec_action, self.env, ledger=(L if auth is not None else None),
                                                auth=auth)
        if outcome.err == "authorization_not_dispatchable":   # the env was not called
            L.cancel_authorization(auth)
            return refused("authorization", "authorization_not_dispatchable")
        if use_allowance:
            self._allowance_used += 1
        self._charge(requester, kind)
        self._settle(tool)
        if irreversible:   # the episode remembers whether this harness commit landed (its decision's effect read back)
            self._commit_attempts.append((commit, self._landed(action)))
        if loop and outcome.recon and outcome.recon.get("confirmed") is not None:   # the read-back record run.py wrote
            self.trajectory.append({"step": step, "event_type": "reconciliation", "tool": tool,
                                    "confirmed": outcome.recon.get("confirmed"), "detail": outcome.recon.get("detail"),
                                    "status": "ok"})

        # kernel after_action
        self._set_obligations(ob_before)
        hpost, after_ok = None, True
        try:
            hpost = self.ex.run_after_action(self.h, bound, outcome, step, origin=_ORIGIN, requester=requester)
        except Exception as e:
            after_ok = False
            self.stopped = "after_action_error"
            self.errors.append("harness_action.after_action: %r" % (e,))
        ev, obs = self.ex.build_event(exec_action, outcome, step, origin=_ORIGIN, audience=("both" if loop else "harness"),
                                      requester=(None if loop else requester))
        if event_extra:
            ev.update(event_extra)
        if kind == INVOCATION:
            self._emit(ev, step)
        else:
            self.trajectory.append(ev)
        if hpost is not None:
            self._extend(hpost.events)
            if hpost.type == "ESCALATE":
                self.stopped = "after_action_escalate"
            elif hpost.type == "RECONCILE" and write and self.reconcile is not None:
                # a harness commit whose effect is unknown restricts the agent's writes as the agent's own
                # would, until the agent reads it back (run.py)
                self.reconcile.enter({"tool": tool, "args": args}, outcome.res, _ORIGIN, requester=requester, step=step)
                self.trajectory.append({"step": step, "event_type": "reconcile_enter", "stage": "after_action",
                                        "commit_origin": _ORIGIN, "action_requester": requester,
                                        "feedback": hpost.feedback, "status": "ok"})
        if auth is not None:
            self._finalize(auth, outcome, hpost, after_ok)
            self.trajectory.append({"step": step, "event_type": "harness_authorization", "action_origin": _ORIGIN,
                                    "action_requester": requester,
                                    "action_id": ev.get("action_id"), "authorization_id": auth.authorization_id,
                                    "source": auth.source, "tool": tool, "auth_status": auth.status, "status": "ok"})
        if acquire_id is not None and not loop:
            _act.record_execution(L, acquire_id, executed=True, succeeded=(not outcome.err),
                                  evidence_added=(L.validated_evidence_version > valid_before))
        if before_content is not None:
            self._guard(before_content, ev.get("action_id"), step, kind)
        return DispatchOutcome(outcome=outcome, event=ev, obs=obs, after=hpost)

    # ---------------------------------------------------------------------------------------------- shell checks
    def _ceiling(self, tool, kind, need=1):
        """Ceilings. -> (refusal reason or None, whether the action draws the post-exhaustion allowance). `need` is the
        number of actions asked for at once. An invocation request draws the per-invocation allowance once the agent's and
        the harness's actions pass the budget plus the reserve; a loop read, a forced write or a restore is its own
        one-action request and is bounded by the per-task caps."""
        invocation = kind == INVOCATION
        cap = self._cap_refusal(kind, need)
        if cap:
            return cap, False
        if not invocation or self.budget_open() or tool == "snapshot":   # a snapshot is a re-read: never refused by the allowance
            return None, False
        if self._allowance_used >= RECOVERY_COMPLETION_ALLOWANCE:
            return "recovery_budget_exhausted", False
        return None, True

    def _cap_refusal(self, kind, need=1):
        """The per-task caps alone (a restore): -> the reason, or None."""
        if kind == INVOCATION and self.n_invocation + need > RECOVERY_TOTAL_CAP:
            return "recovery_total_cap"
        if self.n_total + need > HARNESS_TOTAL_CAP:
            return "harness_total_cap"
        return None

    def _charge(self, requester, kind):
        self.n_total += 1
        if kind == INVOCATION:
            self.n_invocation += 1
        self.by_requester[requester] += 1

    def _irreversible_commit_refusal(self, action, controls):
        """Irreversible-commit rule: the harness never makes a new irreversible commit on its own initiative; it may help
        complete a commit only when the agent itself attempted that same commit and it did not land. For every harness
        origin that reaches this service. -> (refusal reason or None, the commit's control identity). In order:
          irreversible_commit_after_unconfirmed   a harness irreversible commit earlier in the EPISODE (any invocation)
                                                  was not read back as landed: one completion attempt, never a repeat
                                                  after an unknown or failed one
          irreversible_commit_empty_payload       no filled payload control on the page
          irreversible_commit_unknown_surface     no page is known to issue it on
          irreversible_commit_no_agent_attempt    no agent-origin tool call on the same control on the same portal
                                                  (commit_control) earlier in the episode, whatever its status
          irreversible_commit_already_landed      the effect of the decision the control names already holds
          irreversible_commit_landing_unknown     no decision of the workspace's completion content names the control, so
                                                  whether it landed cannot be read (obligation_gate.decision_landed)"""
        if any(landed is not True for _c, landed in self._commit_attempts):
            return "irreversible_commit_after_unconfirmed", None
        if not payload_present(controls):
            return "irreversible_commit_empty_payload", None
        surface = self._current_surface()
        if surface is None:
            return "irreversible_commit_unknown_surface", None
        want = commit_control(action, surface)
        if want is None or not any(commit_control(e, _surface_of(e)) == want for e in self.trajectory
                                   if e.get("event_type") == "tool_call" and not _is_harness(e)):
            return "irreversible_commit_no_agent_attempt", want
        landed = self._landed(action)
        if landed is True:
            return "irreversible_commit_already_landed", want
        if landed is None:
            return "irreversible_commit_landing_unknown", want
        return None, want

    def _landed(self, action):
        try:
            from .obligation_gate import decision_landed
            return decision_landed(self.task, self.env, action, self.workspace)
        except Exception as e:
            self.errors.append("harness_action.decision_landed: %r" % (e,))
            return None

    def _provenance_refusal(self, tool, args, controls):
        """Write provenance: the value a harness write puts into a SEMANTIC slot (bindings.classify_field, default-deny) must be found
        in the shell's own view of a semantic source: agent_commitment (values the agent itself wrote), authoritative_state
        (persisted portal state and pages the environment rendered, option lists excluded) or bound_evidence (the task's
        bound_evidence and VALIDATED ledger evidence). Found only in system_metadata, or nowhere, it is refused."""
        from .bindings import classify_field, SEMANTIC
        ref = args.get("ref")
        ctl = next((c for c in (controls or []) if str(c.get("ref")) == str(ref)), None) or {}
        names = [n for n in (ctl.get("name"), ctl.get("label"), ref) if n]
        if any(classify_field(str(n)) != SEMANTIC for n in names):
            return None                                  # a whitelisted operational field
        value = _norm_value(_written_value(tool, args))
        if not value:
            return None
        if _found(value, self._semantic_sources()):
            return None
        if _found(value, _scalars(self.task.get("system_metadata"))):
            return "value_from_system_metadata_only"
        return "value_untraced"

    def _semantic_sources(self):
        texts = []
        for e in self.trajectory:                        # agent_commitment
            if e.get("event_type") == "tool_call" and not _is_harness(e) and e.get("status") == "ok" \
                    and e.get("tool") in _VALUE_TOOLS:
                texts.append(str(_written_value(e.get("tool"), e.get("args") or {}) or ""))
        st = self.read_recovery_state()                  # authoritative_state: persisted portal state ...
        texts.extend(_scalars({k: v for k, v in st.items() if k != "draft_ui_state"}))
        texts.extend(_scalars(self.task.get("authoritative_state")))
        for e in self.trajectory:                        # ... and the pages the environment rendered
            if e.get("event_type") == "tool_call":
                r = e.get("result") if isinstance(e.get("result"), dict) else {}
                texts.append(_without_options(str(r.get("observation") or "")))
        texts.extend(_scalars(self.task.get("bound_evidence")))   # bound_evidence
        for rec in (getattr(self.h.ledger, "evidence", None) or []):
            if rec.get("status") == "VALIDATED":
                texts.append(_without_options(str(rec.get("value_full") or rec.get("value") or "")))
        return texts

    def _current_surface(self):
        for e in reversed(self.trajectory):
            if e.get("event_type") == "tool_call":
                s = _surface_of(e)
                if s is not None:
                    return s
        return None

    # ---------------------------------------------------------------------------------------------- content guard
    _CONTENT_LEAF_RE = re.compile(r"(notes?|rationale|comment|narrative|justification|summary|explanation|reason)$",
                                  re.I)

    def _guards_content(self):
        """The environment holds agent-authored page content a harness action could clear (a portal page)."""
        return getattr(self.env, "page", None) is not None

    def _content_snapshot(self):
        """TEXT a harness action could destroy: persisted content leaves (triageNotes, rationale, ...) and non-empty
        content controls on the page (unsaved text). Whose text it is is decided when something was cleared
        (_authorship)."""
        try:
            st = self.read_recovery_state() or {}
        except Exception:
            st = {}
        st = {k: v for k, v in st.items() if k != "draft_ui_state"}
        persisted = {k: v for k, v in flat_leaves(st).items() if self._CONTENT_LEAF_RE.search(content_key_segment(k))}
        return {"persisted": persisted, "dom": self._dom_content(self.list_controls())}

    def _dom_content(self, controls):
        out = {}
        for c in (controls or []):
            v = str(c.get("value") or "").strip()
            role = str(c.get("role") or "").lower()
            lbl = "%s %s" % (c.get("label") or "", c.get("name") or "")
            if v and (role in ("textbox", "textarea") or self._CONTENT_LEAF_RE.search(lbl)):
                ref = c.get("ref")
                if ref is not None:
                    out[str(ref)] = v
        return out

    def _authorship(self):
        """Whose text is it. -> ({control: (author, text) of the last executed write naming it}, the normalized
        texts the agent wrote, the normalized texts the harness wrote). author is "agent" for the agent's own write,
        "harness" for a harness write, and "restore" for a content_guard write-back (it authors nothing: it puts back what
        was there)."""
        last, agent_texts, harness_texts = {}, set(), set()
        for e in self.trajectory:
            if e.get("event_type") != "tool_call" or e.get("tool") not in ("type", "fill") or e.get("status") != "ok":
                continue
            a = e.get("args") if isinstance(e.get("args"), dict) else {}
            text = _written_value(e.get("tool"), a)
            if text in (None, ""):
                continue
            if not _is_harness(e):
                author = "agent"
            elif e.get("requester") == _GUARD_REQUESTER:
                author = "restore"
            else:
                author = "harness"
            ref = _bare_ref(a.get("ref"))
            if ref:
                last[ref] = (author, text)
            if author == "agent" or e.get("content_author") == "agent":
                agent_texts.add(_norm_value(text))
            elif author == "harness":
                harness_texts.add(_norm_value(text))
        return last, agent_texts, harness_texts

    @staticmethod
    def _harness_authored(value, ref, last, agent_texts, harness_texts):
        """The harness wrote this text: the last write naming the control was the harness's with this text, or (for
        persisted content, or a control no write named) only the harness ever wrote it."""
        v = _norm_value(value)
        rec = last.get(_bare_ref(ref)) if ref is not None else None
        if rec is not None and _norm_value(rec[1]) == v:
            return rec[0] == "harness"
        return v in harness_texts and v not in agent_texts

    def _guard(self, before, after_action_id, step=None, kind=INVOCATION):
        """After a harness action: every content field that was non-empty and is now empty gets its prior value back,
        through the executor, recorded (never new content). Text the harness itself wrote is not written back: it is
        skipped and named (harness_authored). A restore passes the per-task caps like any harness action; one that no
        longer fits is refused and named (refused). A restore of the agent's own text carries content_author agent."""
        step = self.step if step is None else step
        dom_restored, dom_lost, dom_skipped, refused = [], [], [], []
        last, agent_texts, harness_texts = self._authorship()
        dom_before = before.get("dom") or {}
        if dom_before:
            controls = self.list_controls()
            dom_after = self._dom_content(controls)
            present = set(str(c.get("ref")) for c in controls)
            for ref, value in dom_before.items():
                if str(dom_after.get(ref) or "").strip():
                    continue
                if self._harness_authored(value, ref, last, agent_texts, harness_texts):
                    dom_skipped.append(ref)
                    continue
                if ref not in present:
                    dom_lost.append(ref)
                    continue
                action = {"type": "tool_call", "tool": "type", "args": {"ref": ref, "text": value}}
                why = self._cap_refusal(kind) or self._reconcile_refusal()
                if why:
                    self._refuse("type", action["args"], "content_guard", why, _GUARD_REQUESTER, step)
                    refused.append(ref)
                    continue
                out = self._restore_action(action, step, kind, agent=(_norm_value(value) in agent_texts))
                (dom_lost if out.err else dom_restored).append(ref)
        lost, skipped, restored = {}, [], None
        persisted_before = before.get("persisted") or {}
        if persisted_before:
            try:
                st = self.read_recovery_state() or {}
            except Exception:
                st = {}
            after = flat_leaves({k: v for k, v in st.items() if k != "draft_ui_state"})
            cleared = {k: v for k, v in persisted_before.items() if not str(after.get(k) or "").strip()}
            skipped = sorted(k for k, v in cleared.items()
                             if self._harness_authored(v, None, last, agent_texts, harness_texts))
            lost = {k: v for k, v in cleared.items() if k not in skipped}
            if lost:
                why = self._cap_refusal(kind) or self._reconcile_refusal()
                if why:
                    self._refuse(RESTORE_TOOL, {"fields": sorted(lost)}, "content_guard", why, _GUARD_REQUESTER, step)
                    refused.extend(sorted(lost))
                else:
                    out = self._restore_persisted(lost, step, kind,
                                                  agent=all(_norm_value(v) in agent_texts for v in lost.values()))
                    restored = (out.res or {}).get("restored") if isinstance(out.res, dict) else None
        if dom_restored or dom_lost or lost or dom_skipped or skipped or refused:
            rec = {"step": step, "event_type": "agent_content_guard", "action_origin": _ORIGIN,
                   "action_requester": _GUARD_REQUESTER,
                   "after_action_id": after_action_id, "dom_restored": sorted(dom_restored),
                   "dom_lost": sorted(dom_lost), "persisted_fields": sorted(lost),
                   "persisted_restored": restored, "status": "ok"}
            if dom_skipped or skipped:
                rec["harness_authored"] = {"dom": sorted(dom_skipped), "persisted": skipped}
            if refused:
                rec["refused"] = sorted(refused)
            self.trajectory.append(rec)

    def _reconcile_refusal(self):
        """A restore is a write, refused like any harness write while a commit is unconfirmed."""
        return "unconfirmed_commit_pending" if (self.reconcile is not None and self.reconcile.active) else None

    def _restore_action(self, action, step=None, kind=INVOCATION, agent=False):
        step = self.step if step is None else step
        outcome = self.ex.execute_and_normalize(action, self.env)
        self._charge(_GUARD_REQUESTER, kind)
        self._settle(action.get("tool"))
        ev, _obs = self.ex.build_event(action, outcome, step, origin=_ORIGIN, audience="harness",
                                       requester=_GUARD_REQUESTER)
        if agent:
            ev["content_author"] = "agent"
        self._emit(ev, step)
        return outcome

    def _restore_persisted(self, lost, step=None, kind=INVOCATION, agent=False):
        step = self.step if step is None else step
        outcome = self.ex.execute_restore(self.env, lost)
        action = {"type": "tool_call", "tool": RESTORE_TOOL, "args": {"fields": sorted(lost)}}
        self._charge(_GUARD_REQUESTER, kind)
        ev, _obs = self.ex.build_event(action, outcome, step, origin=_ORIGIN, audience="harness",
                                       requester=_GUARD_REQUESTER)
        if agent:
            ev["content_author"] = "agent"
        self._emit(ev, step)
        return outcome

    # ---------------------------------------------------------------------------------------------- helpers
    def _refuse(self, tool, args, stage, reason, requester=_REQUESTER, step=None):
        self.trajectory.append({"step": self.step if step is None else step, "event_type": "harness_action_refused",
                                "action_origin": _ORIGIN, "action_requester": requester,
                                "tool": tool, "args": args, "stage": stage, "reason": reason, "status": "refused"})
        return {"error": reason}

    def _emit(self, ev, step=None):
        step = self.step if step is None else step
        self._seq += 1
        ev["action_id"] = "rec-%d-%d" % (step, self._seq)
        ev["recovery_episode_id"] = self.episode_id
        ev["parent_agent_step"] = step
        self.trajectory.append(ev)

    def _extend(self, events):
        for e in (events or []):
            self.trajectory.append(e)

    def _vocabulary(self):
        try:
            return self.env.available_tools()
        except Exception:
            return None

    def _settle(self, tool):
        if tool not in _SETTLE_TOOLS:
            return
        page = getattr(self.env, "page", None)
        if page is None:
            return
        try:
            page.wait_for_load_state("networkidle", timeout=4000)
        except Exception:
            pass
        try:
            page.wait_for_timeout(STAGE_SETTLE_MS)
        except Exception:
            pass

    def _obligation_snapshot(self):
        if self._obligations is None:
            return None
        try:
            return self._obligations()
        except Exception as e:
            self.errors.append("harness_action.obligations_before: %r" % (e,))
            return None

    def _set_obligations(self, before):
        """The terminal lock's inputs, exactly as run.py sets them before an agent action's after_action."""
        L = self.h.ledger
        try:
            from .obligation_gate import lock_inputs
            after = self._obligations() if self._obligations is not None else None
            L.public_obligations_met, L.observed_submission_realized = lock_inputs(before, after)
        except Exception as e:
            L.public_obligations_met = None
            L.observed_submission_realized = None
            self.errors.append("harness_action.obligations_met: %r" % (e,))

    def _finalize(self, auth, outcome, hpost, after_ok):
        """The authorization finalization run.py applies to an agent's dispatched authorization."""
        L = self.h.ledger
        ver = getattr(self.h.ctx, "verification", None)
        aeff = hpost.type if hpost is not None else "ALLOW"
        cf = outcome.recon.get("confirmed") if outcome.recon else None
        if not after_ok:
            L.unknown_authorization(auth)
        elif outcome.result_status == "failed" and cf is False:
            L.fail_authorization(auth)
        elif outcome.result_status == "unknown" or aeff == "RECONCILE" or cf is None:
            L.unknown_authorization(auth)
        elif cf is True and ver is True and aeff == "ALLOW":
            L.verify_authorization(auth)
        elif ver is False:
            L.fail_authorization(auth)
        else:
            L.unknown_authorization(auth)


def _bare_ref(ref):
    """A control reference with surrounding brackets and blanks dropped ('[e12]' -> 'e12')."""
    return str(ref if ref is not None else "").strip().strip("[]").strip()


# ---------------------------------------------------------------------------------------------------- module helpers
_PAYLOAD_ROLES = frozenset(("input", "textbox", "textarea", "select", "combobox", "checkbox", "radio", "spinbutton"))


def payload_present(controls):
    """The non-empty payload check: a visible, enabled, editable payload control outside a lookup section that holds a value
    (a checked box, a non-placeholder selection, non-blank text)."""
    try:
        from adapters.hab.gui_control_model import _is_lookup_section
    except Exception:
        _is_lookup_section = lambda _s: False
    for c in (controls or []):
        if c.get("commit") or c.get("disabled") or c.get("readonly") or c.get("visible") is False:
            continue
        role = str(c.get("role") or "").lower()
        if role not in _PAYLOAD_ROLES or _is_lookup_section(c.get("section")):
            continue
        if role in ("checkbox", "radio"):
            if c.get("checked") is True:
                return True
            continue
        if role == "select" and c.get("placeholder_selected") is True:
            continue
        if str(c.get("value") or "").strip():
            return True
    return False


def flat_leaves(obj, prefix="", out=None, depth=0):
    """{dotted_key: non-empty string leaf}; a scalar list item is keyed by its index (native triageNotes is string[])."""
    if out is None:
        out = {}
    if depth > 6:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = "%s.%s" % (prefix, k) if prefix else str(k)
            if isinstance(v, (dict, list)):
                flat_leaves(v, key, out, depth + 1)
            elif isinstance(v, str) and v.strip():
                out[key] = v
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, (dict, list)):
                flat_leaves(v, "%s.%d" % (prefix, i), out, depth + 1)
            elif isinstance(v, str) and v.strip():
                out["%s.%d" % (prefix, i)] = v
    return out


def content_key_segment(dotted):
    """The last non-numeric segment of a flattened key ('full_state.triageNotes.0' -> 'triageNotes')."""
    segs = [s for s in str(dotted).split(".") if s != ""]
    for s in reversed(segs):
        if not s.isdigit():
            return s
    return segs[-1] if segs else ""


def _written_value(tool, args):
    args = args if isinstance(args, dict) else {}
    if tool in ("type", "fill"):
        return args.get("text")
    return args.get("option") if args.get("option") is not None else args.get("value")


def _norm_value(s):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(s if s is not None else "").lower()).split())


def _found(norm_value, texts):
    needle = " %s " % norm_value
    return any(needle in " %s " % _norm_value(t) for t in texts if t)


_OPTION_LINE = re.compile(r"^\s*(?:\[[^\]]*\]\s*)?option\b.*$", re.M)


def _without_options(text):
    """A rendered page minus its option lists: an option offered by a dropdown is not a recorded fact."""
    return _OPTION_LINE.sub("", text or "")


def _scalars(obj):
    """Every scalar leaf of a nested structure, one text per leaf (so a match never spans two values)."""
    vals = []

    def walk(x, depth=0):
        if depth > 8:
            return
        if isinstance(x, dict):
            for v in x.values():
                walk(v, depth + 1)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v, depth + 1)
        elif x is not None and not isinstance(x, bool):
            vals.append(str(x))
    walk(obj)
    return vals


def commit_control(action, surface):
    """A commit's identity for the irreversible-commit rule: (portal, control). portal is the first segment of the page path
    (emr, payer-a, payer-b, fax-portal), control the ref the action names with brackets dropped (click and submit on one
    control are one commit). None without either."""
    from .obligation_gate import action_control
    ref = action_control(action) if isinstance(action, dict) else ""
    if not ref or surface is None:
        return None
    return (str(surface).strip("/").split("/")[0], ref)


def _surface_of(ev):
    r = ev.get("result") if isinstance(ev.get("result"), dict) else {}
    url = r.get("url")
    if not url:
        return None
    m = re.match(r"^(?:[a-z]+://[^/]+)?([^?#]*)", str(url), re.I)
    path = (m.group(1) if m else "") or "/"
    return path.rstrip("/") or "/"
