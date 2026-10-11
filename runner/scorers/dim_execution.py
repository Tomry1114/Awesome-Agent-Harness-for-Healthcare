#!/usr/bin/env python3
"""
Care Action Fulfillment dimension, milestone-based.

Construct: Care Action Fulfillment = was the SELECTED action actually carried out to its intended target STATE? It is
NOT 'did you pick the right tool' (that is Clinical Resource Competence) and NOT 'did you manage the run' (Clinical Workflow Continuity).
Completion is not keyed off tool names: matching on TOOL NAMES is benchmark-coupled (it must know the tool
literals) and brittle, because a tool firing != the operation it represents actually completing. This
module expresses completion as the fraction of the policy's required_milestones that the semantic trace
actually reached, so the scoring logic consumes ONLY substrate structures (SemanticTrace milestones +
DimensionPolicy.required_milestones) and carries no tool/benchmark literal. The benchmark supplies which
milestones matter, via the plugin/policy.

Consumes ONLY: SemanticTrace (substrate.map_trace output), DimensionPolicy (substrate.dimension_policy
output), and optionally a CapabilityManifest (substrate.capability_manifest). No raw trace, no benchmark
name, no tool name, no image/DOM/FHIR literal appears below.

Sub-metrics (applicable-only: a sub-metric with no opportunity is status=not_applicable and EXCLUDED from
the mean, so a clean run never gets a vacuous 1.0):
  action_validity            - PURE protocol/schema validity: fraction of the agent's action attempts that
                               were WELL-FORMED (a usable tool_call/final/etc.), read from
                               SemanticEvent.action_valid. A tool that RAN but failed at execution is still
                               action_valid=True (schema-valid); only a MALFORMED/unparseable action
                               (invalid_action / bad_action_type / truncated_tool_call) is penalized. It does
                               not inspect tool execution success; that is tool_invocation_success.
  tool_invocation_success    - among invocations the agent OWNS (success or failure_attribution=='agent'),
                               the fraction that succeeded. env / external_service / harness / unknown
                               failures are NOT blamed on the agent: excluded from the denominator and
                               reported separately (unknown == evidence_insufficient, not auto-blame).
  required_operation_completion - |required_milestones ∩ milestones_reached(sem)| / |required_milestones|.
                               N/A if the policy declares none.
  terminal_completion        - did a terminal final/escalate event get reached? Always applicable.

Tier: experimental (pending fault-injection + human audit, like lifecycle_exec).
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


try:                                                   # flat sys.path (runner/scorers on the path)
    from dimension_core import submetric as _sm, aggregate_applicable as _aggregate
except ImportError:                                    # pragma: no cover - packaged import
    from .dimension_core import submetric as _sm, aggregate_applicable as _aggregate


def _is_action(s):
    return s.get("event_role") in _ACTION_ROLES


def _ok(s):
    return s.get("status") == "success"


def _invocation_ok(s):
    """INVOCATION succeeded = the call ran without a true failure. 'partial' (ran, but the semantic effect is
    unproven) is a SUCCESSFUL invocation -- effect is judged separately by milestones/completion, so the
    agent is not penalized in tool_invocation_success for a technically-successful call with no evidence."""
    return s.get("status") in ("success", "partial")


def _attr(s):
    """failure_attribution for a failed action; None for a success."""
    return s.get("failure_attribution") if not _ok(s) else None


def execution(sem_trace, dimension_policy=None, manifest=None, actor_scope="agent"):
    """Score the Care Action Fulfillment dimension off a SemanticTrace + DimensionPolicy (+ optional CapabilityManifest).

    sem_trace        : list[SemanticEvent]  (substrate.map_trace(raw_trace, plugin))
    dimension_policy : dict                 (substrate.dimension_policy(task, plugin)) — we read
                       required_milestones from here; nothing else benchmark-specific is consulted.
    manifest         : dict|None            (substrate.capability_manifest(provenance)) — used only to
                       OVERRIDE attribution: an invocation whose capability is reported healthy==False is
                       environmental (excluded) regardless of its per-event attribution.
    """
    sem_trace = list(sem_trace or [])
    # ACTOR-AWARE (agent-process profile by default): score the AGENT's own execution. A harness/recovery
    # re-acquire/commit must not count as the agent's action_validity / tool_invocation_success / operation
    # completion. Consistent with dim_lifecycle/tooling/observability/verification. actor_scope="all" =
    # end-to-end profile. (This has no effect until injected actions carry origin=recovery -> actor=harness.)
    if actor_scope == "agent":
        _f = [s for s in sem_trace if s.get("actor", "agent") != "harness"]
        if _f:
            sem_trace = _f
    elif actor_scope == "harness":
        _f = [s for s in sem_trace if s.get("actor") == "harness"]
        if _f:
            sem_trace = _f
    dimension_policy = dimension_policy or {}
    manifest = manifest or {}

    actions = [s for s in sem_trace if _is_action(s)]
    sub = {}

    # ------------------------------------------------------------------ attribution (manifest override)
    def _eff_attr(s):
        """Effective ownership of a FAILED action. CapabilityManifest is authoritative: a failure on a
        capability the env reports unhealthy is environmental; unauthorized -> harness. Else the
        substrate's per-event failure_attribution stands. (Manifest keyed by capability_id, which the
        mapper sets on every event including failures — never a tool literal at scoring time.)"""
        if _ok(s):
            return None
        cap_id = s.get("capability_id")
        cap = manifest.get(cap_id) if (isinstance(manifest, dict) and cap_id) else None
        if isinstance(cap, dict):
            if cap.get("healthy") is False:
                return "environment"
            if cap.get("authorized") is False:
                return "harness"
            if cap.get("available") is False:
                return "environment"
            if cap.get("implemented") is False:
                return "harness"
        return s.get("failure_attribution")

    # ------------------------------------------------------------------ 1. action_validity (SCHEMA-ONLY)
    # PURE protocol/schema validity: fraction of the agent's action attempts that were WELL-FORMED (a
    # usable tool_call/final/control/etc.), reading ONLY SemanticEvent.action_valid. This does NOT look
    # at tool execution success / failure_attribution / state_changed (that is solely
    # tool_invocation_success's job): a well-formed tool_call that RAN and returned an error is still a
    # VALID action and does not lower this score. Only a MALFORMED/unparseable action (action_valid=False:
    # the mapper's invalid_action / bad_action_type / truncated_tool_call agent_error events) is penalized.
    # action_valid defaults True (.get(..., True)) so a tool event that simply omits the field is well-formed.
    if actions:
        bad = sum(1 for s in actions if s.get("action_valid", True) is False)
        sub["action_validity"] = _sm(round((len(actions) - bad) / len(actions), 3),
                                     opportunities=len(actions), malformed=bad)
    else:
        sub["action_validity"] = _sm(None, "not_applicable", 0)

    # ------------------------------------------------------------------ 2. tool_invocation_success
    # Denominator = invocations the agent OWNS: every success + every failure attributed to the agent.
    # Failures owned by environment/external_service/harness are excluded (not the agent's fault);
    # 'unknown' is evidence_insufficient and also excluded (never auto-blamed). A MALFORMED action
    # (action_valid=False) is NOT a tool invocation at all (no tool ran) -> excluded here so it is scored
    # by action_validity ONLY, never double-counted as an execution failure.
    owned, succ = 0, 0
    excluded = {"environment": 0, "external_service": 0, "harness": 0, "unknown": 0}
    for s in actions:
        if s.get("action_valid", True) is False:
            continue                       # malformed action: not an invocation -> action_validity's concern
        if _invocation_ok(s):              # success OR partial = the invocation technically worked
            owned += 1
            succ += 1
            continue
        ea = _eff_attr(s)                  # only TRUE failures reach here
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
    # Completion = fraction of the policy's required_milestones the trace actually reached.
    # required_milestones and milestones come from substrate (policy + sem trace), no tool names.
    # ALTERNATIVE PATHS: required_milestone_groups are any_of acceptable paths; completion = the BEST
    # (max) fraction over them, so legally finishing the SHORT path is full completion, not "incomplete".
    groups = dimension_policy.get("required_milestone_groups") or (
        [list(dimension_policy.get("required_milestones"))] if dimension_policy.get("required_milestones") else [])
    groups = [g for g in groups if g]
    if groups:
        reached = set()
        for s in sem_trace:
            reached.update(s.get("milestones_added") or [])      # == substrate.milestones_reached(sem)
        best = max(groups, key=lambda g: len(set(g) & reached) / len(g))
        frac = len(set(best) & reached) / len(best)
        sub["required_operation_completion"] = _sm(
            round(frac, 3), opportunities=len(best), n_paths=len(groups),
            satisfied=sorted(set(best) & reached), missing=sorted(set(best) - reached))
    else:
        sub["required_operation_completion"] = _sm(None, "not_applicable", 0,
                                                   note="policy declares no required_milestones")

    # ------------------------------------------------------------------ 4. terminal_completion
    # Did a terminal final/escalate event get reached? Always applicable (every run can terminate).
    has_terminal = any(s.get("terminal") in _TERMINAL_TERMS for s in sem_trace)
    sub["terminal_completion"] = _sm(1.0 if has_terminal else 0.0, opportunities=1)

    # ------------------------------------------------------------------ aggregate + report
    out = _aggregate(sub)
    # COMPLETION-GATED MECHANICS.
    #
    #     E = C * (LAMBDA + (1 - LAMBDA) * M),  LAMBDA = 0.8
    #     C = required_operation_completion        (the gating axis)
    #     M = mean of the mechanics submetrics in _MECHANICS   (mechanics modifier)
    #
    # WHY GATING RATHER THAN RE-WEIGHTING. The construct of Care Action Fulfillment is "was the selected action actually
    # carried out to its intended target state?". Under ANY additive form a run that completed NOTHING still
    # earns credit for calling tools tidily: 0.6C+0.4M would give C=0 a score of 0.4, and even
    # 0.8C+0.2M would give it 0.2. That contradicts the construct. Multiplying makes mechanics able only to
    # DISCOUNT a completion, never to manufacture one -- C=0 -> E=0. Worked examples:
    #     C=0.333, M=1.0  ->  additive 0.6C+0.4M 0.60  ->  gated 0.333   (flawless calls do not turn 1/3 done into 0.6)
    #     C=1.0,   M=0.8  ->  additive 0.6C+0.4M 0.92  ->  gated 0.96    (it WAS completed, with some mechanical friction)
    # LAMBDA=0.8 is a DESIGN CHOICE, not a derived constant: it caps the mechanics modifier's influence at
    # 20%. It is recorded here so the number is auditable rather than looking principled.
    #
    # terminal_completion is DELIBERATELY NOT in M. It asks "did a terminal event occur", which is almost always
    # true by construction of the runner -- a property of the harness, not of the
    # agent -- and whether the episode ended SENSIBLY is already Clinical Workflow Continuity's termination_quality. Keeping it
    # out of the headline improves orthogonality between E (was the action carried out), T (were the tools
    # used correctly) and L (how the episode progressed and ended). It is still computed and reported as a
    # diagnostic submetric.
    #
    # Partial-invocation contract: a partial still counts as a successful INVOCATION
    # (so it is not double-penalised, see test_partial_counts_as_successful_invocation) while its milestone
    # is withheld -> C drops -> E drops.
    _LAMBDA = 0.8
    # tool_invocation_success is the TOOL LAYER's signal and is scored by dim_tooling: Clinical Resource Competence also
    # grades whether the invocations actually worked, alongside arguments, path, authorization and waste.
    # It stays computed here as a diagnostic, but it must not drive TWO dimension headlines, because a
    # submetric shared between two headlines makes those dimensions strongly correlated.
    _MECHANICS = ("action_validity",)
    _comp = sub.get("required_operation_completion", {})
    _mech = [sub[k]["score"] for k in _MECHANICS
             if sub.get(k, {}).get("status") == "valid" and isinstance(sub.get(k, {}).get("score"), (int, float))]
    _om = (sum(_mech) / len(_mech)) if _mech else None
    if _comp.get("status") == "valid" and isinstance(_comp.get("score"), (int, float)):
        _c = _comp["score"]
        _modifier = (_LAMBDA + (1 - _LAMBDA) * _om) if _om is not None else 1.0
        out["score"] = round(_c * _modifier, 3)
        out["score_basis"] = "completion_gated_mechanics_lambda0.8"
        out["gated"] = True
        out["mechanics_mean"] = round(_om, 3) if _om is not None else None
        out["completion"] = _c
    elif _om is not None:
        # no required operations declared for this task -> nothing to gate on; degrade to pure mechanics
        # rather than treating "no requirement" as "completed nothing".
        out["score"] = round(_om, 3)
        out["score_basis"] = "mechanics_only_no_completion_axis"
        out["gated"] = False
        out["mechanics_mean"] = round(_om, 3)
        out["completion"] = None
    out["error_attribution"] = {
        "agent_failures": sub["tool_invocation_success"].get("agent_failures", 0)
        if sub["tool_invocation_success"].get("status") == "valid" else 0,
        "env_or_harness_failures_excluded": excluded["environment"] + excluded["external_service"] + excluded["harness"],
        "unknown_failures_evidence_insufficient": excluded["unknown"]}
    out["degraded_tool_health"] = (excluded["environment"] + excluded["external_service"]) > 0
    out["attribution_source"] = "capability_manifest+semantic_trace" if manifest else "semantic_trace_only"
    out["completion_basis"] = "required_milestones"          # completion is keyed off milestones, not tool names
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
        ("MedCTA", "examples/medcta/MCTA-0"),
        ("PhysicianBench", "examples/pb/PB-adrenal_incidentaloma"),
        ("HealthAdminBench", "examples/hab/HAB-denial-hard-11"),
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
