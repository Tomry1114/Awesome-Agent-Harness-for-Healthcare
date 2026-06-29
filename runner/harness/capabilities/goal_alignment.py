"""Scoped Repair capability (replaces the old NL goal-alignment gate; kept under the registered name
'goal_alignment' so wiring is unchanged).

Two layers, both oracle-blind, substrate-agnostic (it speaks PROJECTIONS via repair_surface):
  L1 Deterministic guard  -- drop any 'missing' finding whose named target is in fact already non-empty
                             (so the harness never tells the agent to redo work it already did).
  L2 Semantic sufficiency -- the judge names localized defects (target_path + smallest repair + preserve).

A finding then enters the ledger lifecycle: delivered once, NOT re-nagged until the agent acts, and every
attempt is DELTA-VALIDATED (defect fixed AND protected content preserved) before it is accepted. A repair
that fixes the target but overwrites protected content is vetoed as `repair_regression` -- this is what
prevents the HAB-12/15 collapse. Active only when MH_REPAIR is soft/select/full (ablatable)."""
import os
from ..capability import Capability
from .. import decision as D
from ..risk import at_least, R2
from ..repair_surface import surface_for, is_present
from ..repair_delta import validate_repair


def _enabled():
    return os.environ.get("MH_REPAIR", "hard") in ("soft", "select", "full")


class GoalAlignment(Capability):
    # LAYER (see HARNESS_DESIGN.md): AMPLIFICATION -- localized goal-aware repair; the delta veto is INFRASTRUCTURE
    name = "goal_alignment"

    # ---- hooks ------------------------------------------------------------------------------------------
    def before_action(self, action, ctx):
        if not _enabled() or not at_least(ctx.risk or R2, R2):
            return None
        if not (ctx.sem and getattr(ctx.sem, "semantic_type", None) in ("create", "update", "submit")):
            return None
        candidate = getattr(ctx.sem, "raw", None) or action
        return self._run(ctx, state=ctx.current_state, candidate=candidate, stage="before_action")

    def before_final(self, answer, ctx):
        if not _enabled():
            return None
        # answer substrate: the candidate IS the answer; coarse text projection (no env state).
        cand = {"answer": answer if isinstance(answer, str) else str(answer)}
        return self._run(ctx, state=cand, candidate=cand, stage="before_final")

    # ---- engine -----------------------------------------------------------------------------------------
    def _goal_spec(self, ctx):
        return (ctx.contract.meta or {}).get("goal_spec") if (ctx.contract and ctx.contract.meta) else None

    def _task_id(self, ctx):
        return str((ctx.contract.meta or {}).get("task_id") or "t") if (ctx.contract and ctx.contract.meta) else "t"

    def _run(self, ctx, state, candidate, stage):
        surf = surface_for(ctx.env_type)
        led = ctx.ledger

        # 1) DELTA-VALIDATE delivered findings the agent may have acted on (deterministic; no judge spend).
        for fid, rec in list(led.repair_findings.items()):
            if rec.status not in ("delivered", "attempted"):
                continue
            after = surf.project(state, candidate, rec.finding)
            if after == rec.last_projection:
                continue                                   # agent has not touched this target -> later
            led.mark_attempted(fid, after, ctx.step)
            v = validate_repair(rec.finding, rec.baseline_projection, after)
            if v.accepted:
                led.resolve_finding(fid)                   # patch applied cleanly -> stop tracking
                continue
            # regression / not-yet-resolved -> re-scope the SAME finding (anti-churn: capped by attempts)
            return self._emit([rec.finding], v.reason, stage, ctx)

        # 2) If something is already delivered and unresolved, WAIT (don't pile on new findings).
        if any(r.status in ("delivered", "attempted") for r in led.repair_findings.values()):
            return None

        # 3) DISCOVER new localized findings (judge-gated by the semantic budget).
        gs = self._goal_spec(ctx)
        if not gs or not ctx.judge_fn or not ctx.spend_semantic():
            return None
        from ..engines.semantic import scoped_goal_findings
        findings = scoped_goal_findings(gs, state, candidate, ctx.judge_fn, self._task_id(ctx))

        fresh = []
        for f in findings:
            proj = surf.project(state, candidate, f)
            # L1 deterministic guard: never claim a 'missing' field that is actually present.
            if f.defect_type == "missing" and is_present(proj.get("target")):
                continue
            mode, _rec = led.repair_decision(f, proj)
            if mode == "suppress":
                continue
            if mode == "new":
                led.open_finding(f, proj, ctx.step)
            led.mark_delivered(f.finding_id, proj, ctx.step)
            fresh.append(f)
        if not fresh:
            return None
        return self._emit(fresh, "goal_misalignment", stage, ctx)

    # ---- decision rendering -----------------------------------------------------------------------------
    def _emit(self, findings, reason_code, stage, ctx):
        rf = [f.to_dict() for f in findings]
        miss = [f.required_change for f in findings]
        head = findings[0]
        reason = ("scoped repair (%s): %s at %s -- %s"
                  % (reason_code, head.defect_type, head.target_path, head.required_change))
        return self._decide(
            D.REVISE, rule_id="scoped_repair", reason_code=reason_code, deterministic=False,
            missing_obligations=miss, reason=reason,
            extra={"repair_findings": rf},
            feedback="Apply this localized patch to %s (operation %s): %s. Modify ONLY that target; preserve "
                     "all existing substantive content." % (head.target_path, head.operation.value, head.required_change))
