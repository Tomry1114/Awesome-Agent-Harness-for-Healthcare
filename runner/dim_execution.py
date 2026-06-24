#!/usr/bin/env python3
"""Execution dimension — MILESTONE-based (supersedes lifecycle_exec.execution's required_tool_groups path).

Construct: Execution = was the SELECTED action actually carried out to its intended target STATE? It is
NOT 'did you pick the right tool' (that is Tooling) and NOT 'did you manage the run' (Lifecycle). The old
required_operation_completion in lifecycle_exec.execution() keyed completion off required_tool_groups, i.e.
it matched on TOOL NAMES. That is benchmark-coupled (it must know the tool literals) and brittle: a tool
firing != the operation it represents actually completing. This module re-expresses completion as the
fraction of the policy's required_milestones that the semantic trace actually reached, so the scoring logic
consumes ONLY substrate structures (SemanticTrace milestones + DimensionPolicy.required_milestones) and
carries no tool/benchmark literal. The benchmark supplies which milestones matter — via the plugin/policy.

Consumes ONLY: SemanticTrace (substrate.map_trace output), DimensionPolicy (substrate.dimension_policy
output), and optionally a CapabilityManifest (substrate.capability_manifest). No raw trace, no benchmark
name, no tool name, no image/DOM/FHIR literal appears below.

Sub-metrics (applicable-only — a sub-metric with no opportunity is status=not_applicable and EXCLUDED from
the mean, so a clean run never gets a vacuous 1.0):
  action_validity            - fraction of the agent's actions that were WELL-FORMED (an agent-attributed
                               failure that produced no progress/state change is a malformed action).
  tool_invocation_success    - among invocations the agent OWNS (success or failure_attribution=='agent'),
                               the fraction that succeeded. env / external_service / harness / unknown
                               failures are NOT blamed on the agent: excluded from the denominator and
                               reported separately (unknown == evidence_insufficient, not auto-blame).
  required_operation_completion - |required_milestones ∩ milestones_reached(sem)| / |required_milestones|.
                               This is the milestone migration. N/A if the policy declares none.
  terminal_completion        - did a terminal final/escalate event get reached? Always applicable.

Tier: experimental (milestone migration; pending fault-injection + human audit, like lifecycle_exec).
"""

# Roles that represent an AGENT ACTION the agent is responsible for executing. Terminal roles
# (final/escalate) and pure book-keeping are not 'actions' for action_validity. These are substrate
# role tokens (substrate.ROLES), NOT benchmark tool names — benchmark-agnostic.
_ACTION_ROLES = ("acquire", "act", "verify", "commit")
_TERMINAL_TERMS = ("final", "escalate")
# Failure ownership taxonomy from the substrate (semantic_event.failure_attribution domain). Only 'agent'
# counts against the agent; env/external/harness are excluded; unknown is evidence_insufficient.
_AGENT_OWNED = ("agent", None)            # None == a successful event the agent performed
_EXCLUDED = ("environment", "external_service", "harness")
_EVIDENCE_INSUFFICIENT = ("unknown",)


def _sm(score, status="valid", opportunities=None, **kw):
    d = {"score": score, "status": status}
    if opportunities is not None:
        d["opportunities"] = opportunities
    d.update(kw)
    return d


def _aggregate(subs):
    """Average ONLY applicable (status=valid, numeric) sub-metrics. Never vacuous-1.0."""
    valid = {k: v for k, v in subs.items()
             if v.get("status") == "valid" and isinstance(v.get("score"), (int, float))}
    score = round(sum(v["score"] for v in valid.values()) / len(valid), 3) if valid else None
    vals = [v["score"] for v in valid.values()]
    return {"score": score, "submetrics": subs,
            "applicable_submetrics": sorted(valid), "n_applicable": len(valid),
            "zero_variance": (len(set(vals)) == 1) if vals else None}


def _is_action(s):
    return s.get("event_role") in _ACTION_ROLES


def _ok(s):
    return s.get("status") == "success"


def _attr(s):
    """failure_attribution for a failed action; None for a success."""
    return s.get("failure_attribution") if not _ok(s) else None


def execution(sem_trace, dimension_policy=None, manifest=None):
    """Score the Execution dimension off a SemanticTrace + DimensionPolicy (+ optional CapabilityManifest).

    sem_trace        : list[SemanticEvent]  (substrate.map_trace(raw_trace, plugin))
    dimension_policy : dict                 (substrate.dimension_policy(task, plugin)) — we read
                       required_milestones from here; nothing else benchmark-specific is consulted.
    manifest         : dict|None            (substrate.capability_manifest(provenance)) — used only to
                       OVERRIDE attribution: an invocation whose capability is reported healthy==False is
                       environmental (excluded) regardless of its per-event attribution.
    """
    sem_trace = list(sem_trace or [])
    dimension_policy = dimension_policy or {}
    manifest = manifest or {}

    actions = [s for s in sem_trace if _is_action(s)]
    sub = {}

    # ------------------------------------------------------------------ 1. action_validity
    # A well-formed action = the agent emitted a syntactically/semantically usable action. We treat an
    # action as MALFORMED only when it failed, the failure is the AGENT's (failure_attribution=='agent'),
    # AND it produced no state change (no progress) — i.e. a bad/invalid action, not an action that ran
    # but returned an unhelpful result for environmental reasons. env/harness/unknown failures are not
    # 'malformed actions' by the agent, so they do not lower action_validity.
    if actions:
        bad = sum(1 for s in actions
                  if (not _ok(s)) and _attr(s) == "agent" and not s.get("state_changed"))
        sub["action_validity"] = _sm(round((len(actions) - bad) / len(actions), 3),
                                     opportunities=len(actions), malformed=bad)
    else:
        sub["action_validity"] = _sm(None, "not_applicable", 0)

    # ------------------------------------------------------------------ attribution (manifest override)
    def _eff_attr(s):
        """Effective ownership of a FAILED action. CapabilityManifest is authoritative: a failure on a
        capability the env reports unhealthy is environmental; unauthorized -> harness. Else the
        substrate's per-event failure_attribution stands. (Manifest keyed by progress_token, which the
        mapper derives from the capability/tool — we never read a tool literal, just the token.)"""
        if _ok(s):
            return None
        tok = s.get("progress_token")
        cap = manifest.get(tok) if isinstance(manifest, dict) else None
        if isinstance(cap, dict):
            if cap.get("healthy") is False:
                return "environment"
            if cap.get("authorized") is False:
                return "harness"
        return s.get("failure_attribution")

    # ------------------------------------------------------------------ 2. tool_invocation_success
    # Denominator = invocations the agent OWNS: every success + every failure attributed to the agent.
    # Failures owned by environment/external_service/harness are excluded (not the agent's fault);
    # 'unknown' is evidence_insufficient and also excluded (never auto-blamed).
    owned, succ = 0, 0
    excluded = {"environment": 0, "external_service": 0, "harness": 0, "unknown": 0}
    for s in actions:
        if _ok(s):
            owned += 1
            succ += 1
            continue
        ea = _eff_attr(s)
        if ea == "agent":
            owned += 1                     # owned failure: counts against, not toward
        elif ea in _EXCLUDED:
            excluded[ea] += 1
        elif ea in _EVIDENCE_INSUFFICIENT or ea is None:
            excluded["unknown"] += 1
    if owned:
        sub["tool_invocation_success"] = _sm(round(succ / owned, 3), opportunities=owned,
                                             agent_failures=owned - succ)
    else:
        sub["tool_invocation_success"] = _sm(None, "not_applicable", 0)

    # ------------------------------------------------------------------ 3. required_operation_completion
    # THE MIGRATION: completion = fraction of the policy's required_milestones the trace actually reached.
    # required_milestones and milestones come from substrate (policy + sem trace) — no tool names.
    required = list(dimension_policy.get("required_milestones") or [])
    if required:
        reached = set()
        for s in sem_trace:
            reached.update(s.get("milestones_added") or [])      # == substrate.milestones_reached(sem)
        satisfied = [m for m in required if m in reached]
        sub["required_operation_completion"] = _sm(
            round(len(satisfied) / len(required), 3), opportunities=len(required),
            satisfied=sorted(satisfied), missing=sorted(set(required) - reached))
    else:
        sub["required_operation_completion"] = _sm(None, "not_applicable", 0,
                                                   note="policy declares no required_milestones")

    # ------------------------------------------------------------------ 4. terminal_completion
    # Did a terminal final/escalate event get reached? Always applicable (every run can terminate).
    has_terminal = any(s.get("terminal") in _TERMINAL_TERMS for s in sem_trace)
    sub["terminal_completion"] = _sm(1.0 if has_terminal else 0.0, opportunities=1)

    # ------------------------------------------------------------------ aggregate + report
    out = _aggregate(sub)
    out["error_attribution"] = {
        "agent_failures": sub["tool_invocation_success"].get("agent_failures", 0)
        if sub["tool_invocation_success"].get("status") == "valid" else 0,
        "env_or_harness_failures_excluded": excluded["environment"] + excluded["external_service"] + excluded["harness"],
        "unknown_failures_evidence_insufficient": excluded["unknown"]}
    out["degraded_tool_health"] = (excluded["environment"] + excluded["external_service"]) > 0
    out["attribution_source"] = "capability_manifest+semantic_trace" if manifest else "semantic_trace_only"
    out["completion_basis"] = "required_milestones"          # migrated from required_tool_groups
    out["terminal_reached"] = has_terminal
    out["tier"] = "experimental"
    return out


# --------------------------------------------------------------------------- self-verify
if __name__ == "__main__":
    import json, sys, os
    sys.path.insert(0, "runner")
    import substrate

    def _load(traj):
        return [json.loads(l) for l in open(traj) if l.strip()]

    BUNDLES = [
        ("MedCTA", "results_mctaGov/gpt5/MCTA-0"),
        ("PhysicianBench", "results_pb_chk3/gpt5/PB-aberrant_drug_screen"),
        ("HealthAdminBench", "results_hab10/gpt5/HAB-denial-easy-1"),
    ]
    for bench, d in BUNDLES:
        traj = os.path.join(d, "trajectory.jsonl")
        task = json.load(open(os.path.join(d, "task.json")))
        res = json.load(open(os.path.join(d, "result.json")))
        plugin = substrate.get_plugin(task.get("source_benchmark"))
        sem = substrate.map_trace(_load(traj), plugin)
        policy = substrate.dimension_policy(task, plugin)
        manifest = substrate.capability_manifest(res.get("provenance") or {})
        out = execution(sem, policy, manifest)
        print("=" * 78)
        print("%-16s  %s" % (bench, d))
        print("  score              :", out["score"])
        print("  applicable submets :", out["applicable_submetrics"], "(n=%d)" % out["n_applicable"])
        for k, v in out["submetrics"].items():
            print("    %-30s %-14s %s" % (k, v["status"], v.get("score")),
                  {kk: vv for kk, vv in v.items() if kk not in ("score", "status")})
        print("  error_attribution  :", out["error_attribution"])
        print("  completion_basis   :", out["completion_basis"], "| attribution_source:", out["attribution_source"])
    print("=" * 78)
    print("module imported & ran clean.")
