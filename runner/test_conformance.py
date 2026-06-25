#!/usr/bin/env python3
"""Harness conformance tests (Codex C: 'adapter correctness proven by tests').
Codifies the invariants this session fixed, as executable guards so regressions fail loudly.
Pure-logic only (no live backend / GPU). Run: python3 -m pytest runner/test_conformance.py -q
or: python3 runner/test_conformance.py  (built-in runner at bottom)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scoring, canonical_schema, proxy_verifiers


# ---- score_eligible is FAIL-CLOSED (missing flag -> NOT eligible) ----
def test_score_eligible_fail_closed():
    assert scoring.is_score_eligible({"checkpoint_status": "passed"}) is False          # no flag -> excluded
    assert scoring.is_score_eligible({"checkpoint_status": "passed", "score_eligible": True}) is True
    assert scoring.is_score_eligible({"checkpoint_status": "skipped", "score_eligible": True}) is False  # skipped never counts
    assert scoring.is_score_eligible({"score_eligible": True}) is False                 # no status key -> .get() None -> excluded


# ---- dimension_status can NEVER decouple from the score (the post-hoc-Governance bug) ----
def test_dimension_status_never_decoupled():
    results = [{"dimension": "Governance", "checkpoint_status": "passed", "score_eligible": True}]
    st, rsn = scoring.compute_dim_status(results, {"Governance": 1.0}, {})
    assert st["Governance"] == "valid_score", st                       # scored -> valid_score, not 'not_exercised'
    # a dim with no checkpoints -> not_applicable WITH a reason (never a bare void)
    st2, rsn2 = scoring.compute_dim_status([], {}, {})
    for m in scoring.MODULES:
        assert st2[m] in ("valid_score", "proxy_only", "evaluation_error", "not_exercised", "not_applicable"), st2
        if st2[m] != "valid_score":
            assert m in rsn2 and rsn2[m], ("missing reason for", m)     # n/a always explained


# ---- error_class taxonomy is consistent (not_evaluated / evaluation_failure / environment_failure) ----
def test_error_class_taxonomy():
    assert scoring.error_class({"checkpoint_status": "skipped", "skip_reason": "missing_native_verifier"}) == "not_evaluated"
    assert scoring.error_class({"checkpoint_status": "error", "failure_mode": "verifier_error"}) == "evaluation_failure"
    assert scoring.error_class({"checkpoint_status": "error", "failure_mode": "environment_error"}) == "environment_failure"
    assert scoring.error_class({"checkpoint_status": "passed", "failure_mode": None}) is None      # real outcome, not an error


# ---- tool-selection enforces ALTERNATIVE groups (no free pass when required_tools is empty) ----
def test_tool_selection_enforces_alternatives():
    # MedCTA shape: no hard required_tools, one alternative group of 3 perception tools.
    ref = {"required_tools": [], "sufficient_tools": [],
           "required_tool_groups": [["OCR", "ImageDescription", "RegionAttributeDescription"]]}
    req = scoring._tool_requirements(ref, {})
    assert req["required"] == set()
    assert req["alternatives"] == [{"OCR", "ImageDescription", "RegionAttributeDescription"}], req["alternatives"]
    used_partial = {"ImageDescription"}
    used_full = {"OCR", "ImageDescription", "RegionAttributeDescription"}
    alt_ok = lambda used: any(g <= used for g in req["alternatives"])
    # the post-#8 pass rule: ok = (not missing_required) and alt_satisfied
    assert alt_ok(used_partial) is False        # partial use -> must FAIL (was free-pass before #8)
    assert alt_ok(used_full) is True            # full group -> PASS


# ---- canonical_observation is a real structured object (the layer that must be consumed) ----
def test_canonical_observation_shape():
    co = canonical_schema.canonical_observation({"observation": "liver lesion", "url": None}, "tool_sandbox")
    assert co["observation_type"] == "environment_state"
    assert "modalities" in co and co["modalities"].get("text") == "liver lesion"


# ---- canonical_observation is CONSUMED by the Observability proxy (not write-only) ----
def test_observability_consumes_canonical():
    ev_canon = {"event_type": "tool_call", "tool": "ImageDescription",
                "canonical_observation": {"modalities": {"text": "axial CT"}}}
    ev_empty = {"event_type": "tool_call", "tool": "X", "canonical_observation": {"modalities": {}}}
    assert proxy_verifiers._has_observation(ev_canon) is True
    assert proxy_verifiers._has_observation(ev_empty) is False      # canonical layer empty -> no observation
    out = proxy_verifiers.proxy_dimensions([ev_canon, ev_empty,
                                            {"event_type": "final_answer"}])
    # Observability is a real 3-layer dimension (availability/exposure/uptake + error_transparency);
    # exposure is mirrored to trace_observation_coverage (harness-side, -> integrity panel).
    assert "trace_observation_coverage" in out
    ob = out["Observability"]
    for layer in ("evidence_availability", "evidence_exposure", "evidence_uptake", "error_transparency"):
        assert layer in ob, "Observability missing layer " + layer
    assert ob["evidence_exposure"] == out["trace_observation_coverage"]["score"]   # exposure == delivery mirror


# ---- tightened error detection: bare word 'error' must NOT trigger a false failure ----
def test_errored_no_false_positive():
    assert proxy_verifiers._errored({"status": "ok", "result": "no error in the scan; normal study"}) is False
    assert proxy_verifiers._errored({"status": "ok", "result": '{"error": "bad request"}'}) is True


def test_benchmark_adapter_contract():
    """Codex B/C: every registered benchmark env satisfies the BenchmarkAdapter execution surface,
    and the capability manifest declares all four states. Proves the contract over REAL classes."""
    import environments, inspect
    assert getattr(environments, "ENV_REGISTRY", None), "ENV_REGISTRY missing"
    surface = ("reset", "available_tools", "call_tool", "capabilities", "teardown")
    for key, cls in environments.ENV_REGISTRY.items():
        for m in surface:
            assert callable(getattr(cls, m, None)), "adapter %s missing %s" % (key, m)
    src = inspect.getsource(environments.EnvironmentAdapter.capabilities)
    for stt in ("implemented", "available", "authorized", "healthy"):
        assert stt in src, "capability four-state missing: " + stt


def test_evaluator_type_persisted():
    """Codex B: the registry stamps evaluator_type/version on each checkpoint AND build_result must
    persist it (not drop it in the output whitelist)."""
    cp = {"id": "cp_tool_selection", "type": "deterministic", "dimension": "Tooling",
          "subdimension": "tool_use_quality", "check": {"method": "toolset_contains"}}
    ctx = {"reference": {"required_tool_groups": [["OCR", "ImageDescription"]]},
           "agent_tool_calls": [("OCR", {}), ("ImageDescription", {})], "ref_tool_calls": []}
    r = scoring.run_checkpoint(cp, ctx)
    assert r.get("evaluator_type") == "deterministic", r.get("evaluator_type")
    assert r.get("evaluator_version"), "no evaluator_version"
    out = scoring.build_result({"task_id": "TEST", "checkpoints": [cp]}, [], [r], {})
    c0 = out["checkpoints"][0]
    assert c0.get("evaluator_type") == "deterministic", "build_result dropped evaluator_type"
    assert c0.get("evaluator_version"), "build_result dropped evaluator_version"


def test_unified_aggregation_dual_semantics():
    """Codex #1+#2: ONE aggregate_dimension; the same field is never reused with two maths.
    score_mean (graded) != pass_rate (binary) for a GAcc-style cp; build_result (raw) now emits
    score_mean (NOT pass-rate), plus per-checkpoint dual fields score + pass_status."""
    cps = [{"dimension": "Verification", "score": 0.45, "pass_status": "failed",
            "score_eligible": True, "checkpoint_status": "failed", "weight": 1.0}]
    agg = scoring.aggregate_dimension(cps)
    assert agg["score_mean"] == 0.45 and agg["pass_rate"] == 0.0, agg   # two semantics, two fields
    assert agg["zero_variance"] is True and agg["n_scored"] == 1, agg
    r = {"id": "cp_outcome", "dimension": "Verification", "subdimension": "result_verification",
         "checkpoint_status": "failed", "score": 0.45, "score_eligible": True}
    out = scoring.build_result({"task_id": "T", "checkpoints": [{"id": "cp_outcome"}]}, [], [r], {})
    assert out["dimension_scores"]["Verification"] == 0.45, "raw must use score_mean (graded), not pass-rate"
    assert out["dimension_pass_rate"]["Verification"] == 0.0, "pass_rate is a SEPARATE field"
    c0 = out["checkpoints"][0]
    assert c0["score"] == 0.45 and c0["pass_status"] == "failed", "dual fields not persisted"


def test_execution_proxy_sensitivity():
    """Step (a): the Execution proxy must RESPOND when execution degrades (sensitivity + directionality).
    Guards against a regression to a flat/insensitive metric."""
    import lifecycle_exec as le
    def c(t, ok=True): 
        e={"event_type":"tool_call","tool":t,"status":"ok" if ok else "error","canonical_observation":{"modalities":{"text":"x"}}}
        if not ok: e["error_type"]="tool_argument_error"; e["result"]="[error] failed"  # AGENT-attributable
        return e
    F={"event_type":"final_answer","thought":"x"}
    base=le.execution([c("A"),c("B"),F])["score"]
    no_final=le.execution([c("A"),c("B")])["score"]
    errs=le.execution([c("A",ok=False),c("B",ok=False),F])["score"]
    recover=le.execution([c("A",ok=False),c("A",ok=True),F])["score"]
    repeated=le.execution([c("A",ok=False),c("A",ok=False),F])["score"]
    assert no_final < base, "Execution insensitive to missing terminal completion"
    assert errs < base, "Execution insensitive to tool failures"
    assert recover > repeated, "Execution does not distinguish recovery from repeated failure"


def test_lifecycle_sm_monotonicity():
    """Step (b): state-machine Lifecycle DROPS on loops/repeated-failure; pagination != loop."""
    import lifecycle_exec as le
    def c(t, ok=True, obs="x", err=None):
        e = {"event_type": "tool_call", "tool": t, "args": {"q": obs}, "status": "ok" if ok else "error",
             "canonical_observation": {"modalities": {"text": obs} if ok else {}}}
        if not ok: e["error_type"] = err or "tool_error"; e["result"] = "[error]"
        return e
    F = {"event_type": "final_answer", "thought": "d"}
    normal = le.lifecycle([c("A", obs="a"), c("B", obs="b"), F])["score"]
    repeated = le.lifecycle([c("A", ok=False), c("A", ok=False), c("A", ok=False), F])["score"]
    loop = le.lifecycle([c("A", obs="a"), c("A", obs="a"), c("A", obs="a"), c("A", obs="a"), F])["score"]
    pagination = le.lifecycle([c("A", obs="p1"), c("A", obs="p2"), c("A", obs="p3"), F])["score"]
    assert repeated < normal and loop < normal, "Lifecycle insensitive to loops/repeated failure"
    assert pagination > loop, "pagination (new evidence) mistaken for a loop"


def test_execution_attribution_gate():
    """Step (b): env failures EXCLUDED from agent score (degraded_tool_health); agent failures penalized."""
    import lifecycle_exec as le
    def c(t, ok=True, err=None):
        e = {"event_type": "tool_call", "tool": t, "status": "ok" if ok else "error",
             "canonical_observation": {"modalities": {"text": "x"}}}
        if not ok: e["error_type"] = err; e["result"] = "[error]"; e["failure_mode"] = "environment_error" if err == "env" else "agent_failure"
        else: e["result"] = "x"
        return e
    F = {"event_type": "final_answer", "thought": "d"}
    af = le.execution([c("A", ok=False, err="tool_argument_error"), c("B"), F])
    ef = le.execution([c("A", ok=False, err="env"), c("B"), F])
    assert af["submetrics"]["tool_invocation_success"]["score"] < 1.0
    assert ef["submetrics"]["tool_invocation_success"]["score"] == 1.0 and ef["degraded_tool_health"] is True


def test_execution_capability_healthy_attribution():
    """Review #1: capabilities.healthy is the AUTHORITATIVE attribution source — a failure on a tool the
    env reports unhealthy is environmental (excluded from the agent score), even if the error text is generic."""
    import lifecycle_exec as le
    def c(t, ok=True):
        e = {"event_type": "tool_call", "tool": t, "status": "ok" if ok else "error",
             "canonical_observation": {"modalities": {"text": "x"}}}
        if not ok: e["error_type"] = "tool_error"; e["result"] = "[error]"
        return e
    F = {"event_type": "final_answer", "thought": "d"}
    caps = {"A": {"implemented": True, "available": True, "authorized": True, "healthy": False}}
    r = le.execution([c("A", ok=False), c("B"), F], capabilities=caps)
    assert r["degraded_tool_health"] is True
    assert r["error_attribution"]["env_or_harness_failures_excluded"] >= 1
    assert r["attribution_source"].startswith("capability_manifest")


def test_substrate_benchmark_agnostic():
    """The SemanticEventMapper / EvidenceView core must consume plugin metadata and contain NO benchmark
    tool literal -- a 4th dataset registers a plugin, the dimension-facing core is untouched."""
    import inspect, substrate as sub
    assert set(sub.list_plugins()) >= {"MedCTA", "PhysicianBench", "HealthAdminBench"}
    # core mapper source names no tool/benchmark
    src = inspect.getsource(sub.map_trace) + inspect.getsource(sub.evidence_view)
    for lit in ("MedCTA", "OCR", "ImageDescription", "fhir", "snapshot", "RegionAttribute"):
        assert lit not in src, "core leaks benchmark literal: %s" % lit
    # same core, two different plugins -> roles come from the plugin, not the core
    tr = [{"event_type": "tool_call", "tool": "fhir_search", "status": "ok", "semantic_assume_success": True},
          {"event_type": "tool_call", "tool": "submit", "status": "ok", "semantic_assume_success": True},
          {"event_type": "final_answer", "thought": "done"}]
    pb = sub.map_trace(tr, sub.get_plugin("PhysicianBench"))
    hab = sub.map_trace(tr, sub.get_plugin("HealthAdminBench"))
    assert pb[0]["event_role"] == "acquire" and "patient_record_loaded" in pb[0]["milestones_added"]
    assert hab[1]["event_role"] == "commit" and "form_submitted" in hab[1]["milestones_added"]
    assert pb[-1]["terminal"] == "final"
    # evidence units carry the required delivery fields
    ev = sub.evidence_view(tr, sub.get_plugin("PhysicianBench"))
    assert ev and all(set(u) >= {"id", "delivered_to_agent", "delivery_fidelity", "error_visible"} for u in ev)


def test_execution_uses_capability_id_for_attribution():
    """A failure on a capability the manifest reports unhealthy must be attributed to the environment via
    capability_id (NOT progress_token, which a failure usually lacks) -> excluded from agent blame."""
    import substrate as _S, dim_execution as _E
    ev = _S.semantic_event("acquire", status="failure", capability_id="OCR", progress_token=None,
                           failure_attribution="agent")
    manifest = {"OCR": {"implemented": True, "available": True, "authorized": True, "healthy": False}}
    r = _E.execution([ev], {}, manifest)
    assert r["error_attribution"]["env_or_harness_failures_excluded"] >= 1, r["error_attribution"]
    assert r["submetrics"]["tool_invocation_success"].get("agent_failures", 0) == 0


def test_action_validity_uses_manifest_attribution():
    """#2: a failure on an UNHEALTHY capability (text-heuristic would call it 'agent') must NOT count as a
    malformed action -- action_validity and tool_invocation_success must agree via the manifest."""
    import substrate as _S, dim_execution as _E
    ev = _S.semantic_event("act", status="failure", capability_id="fhir_create",
                           failure_attribution="agent", state_changed=False)
    man = {"fhir_create": {"implemented": True, "available": True, "authorized": True, "healthy": False}}
    r = _E.execution([ev], {}, man)
    assert r["submetrics"]["action_validity"]["score"] == 1.0, r["submetrics"]["action_validity"]


def test_ordering_missing_predecessor_fails():
    """#P2: a successor with NO predecessor is ACTIVATED and UNSATISFIED (score 0), not skipped."""
    import dim_lifecycle as _L
    dp = {"ordering_constraints": [{"predecessor": {"milestone": "allergy_checked"},
                                    "successor": {"milestone": "medication_ordered"}, "weight": 1.0}]}
    tr = [{"event_role": "act", "status": "success", "milestones_added": ["medication_ordered"]},
          {"event_role": "final", "status": "success", "terminal": "final"}]
    assert _L._score_ordering(tr, dp)["score"] == 0.0


def test_unjustified_escalation_not_recovery():
    """#P5: a bare give-up escalation (no degraded capability) does NOT count as recovery."""
    import dim_lifecycle as _L
    tr = [{"event_role": "acquire", "status": "failure", "obligation_id": "target_examined", "failure_attribution": "agent"},
          {"event_role": "escalate", "status": "success", "terminal": "escalate"}]
    assert _L._score_recovery(tr, {}, None, {})["score"] == 0.0
    man = {"cap": {"healthy": False}}
    assert _L._score_recovery(tr, {}, man, {})["score"] == 1.0


def test_missing_payload_withholds_milestone():
    """#3/#P6: a tool_call with NO payload (and no semantic_assume_success) is mapped status=partial with the
    milestone WITHHELD -- never optimistic -- even for a tool that has no resolver."""
    import substrate as _S
    pl = _S.get_plugin("MedCTA")
    ev = [{"event_type": "tool_call", "tool": "ImageDescription", "status": "ok"}]   # no result/observation
    sem = _S.map_trace(ev, pl)
    assert sem[0]["status"] == "partial" and sem[0]["milestones_added"] == [], sem[0]
    ev2 = [{"event_type": "tool_call", "tool": "ImageDescription", "status": "ok", "semantic_assume_success": True}]
    assert "image_overview_obtained" in _S.map_trace(ev2, pl)[0]["milestones_added"]


def test_delivery_record_drives_observability():
    """#4: evidence_view consumes the recorded delivery_record / agent_visible_text (real info-flow) rather
    than inferring from tool status."""
    import substrate as _S
    pl = _S.get_plugin("MedCTA")
    ev = [{"event_type": "tool_call", "tool": "OCR", "status": "ok", "result": {"output": "X" * 5000},
           "agent_visible_text": "X" * 500,
           "delivery_record": {"produced": True, "rendered_to_agent": True, "truncated": True,
                               "error_state_rendered": False}}]
    u = _S.evidence_view(ev, pl)[0]
    assert u["delivered_to_agent"] is True and u["delivery_fidelity"] == 0.5   # truncated -> reduced fidelity


def test_partial_counts_as_successful_invocation():
    """#2-deeper: a 'partial' event (call ran, effect unproven) is a SUCCESSFUL invocation -- it must NOT
    lower tool_invocation_success nor count as a malformed action; it only withholds the milestone."""
    import substrate as _S, dim_execution as _E
    partial = _S.semantic_event("acquire", status="partial", capability_id="ImageDescription",
                                milestones_added=[], state_changed=False)
    r = _E.execution([partial], {}, {})
    assert r["submetrics"]["tool_invocation_success"]["score"] == 1.0, r["submetrics"]["tool_invocation_success"]
    assert r["submetrics"]["action_validity"]["score"] == 1.0


def test_action_validity_is_schema_only():
    """#ITEM1: action_validity is a PURE protocol/schema-validity metric, NOT inferred from tool failure.
    A trace with ONE malformed action (agent_error invalid_action -> action_valid False) + ONE well-formed
    tool_call that RAN and FAILED at execution (agent-attributed) must split cleanly:
      - action_validity penalizes ONLY the malformed action (1 of 2 valid -> 0.5),
      - tool_invocation_success penalizes the exec failure (the malformed action is NOT an invocation)."""
    import substrate as _S, canonical_schema as _C, dim_execution as _E

    # canonical_schema exposes well-formedness directly: a normal tool_call/final is valid; the malformed
    # agent action-dicts are invalid.
    assert _C.action_valid({"type": "tool_call", "tool": "fhir_read", "args": {}}) is True
    assert _C.action_valid({"type": "final", "answer": "done"}) is True
    assert _C.action_valid({"type": "invalid_action", "raw": "garbage"}) is False
    assert _C.action_valid({"type": "bad_action_type", "raw": "{}"}) is False
    assert _C.action_valid({"type": "tool_call_truncated", "raw": "{partial"}) is False

    # map_trace EMITS an action_valid=False event for a malformed agent_error, and a well-formed tool_call
    # that EXECUTION-FAILED stays action_valid=True.
    trace = [
        {"event_type": "agent_error", "error": "invalid_action", "raw": "garbage", "status": "error"},
        {"event_type": "tool_call", "tool": "fhir_create", "status": "error", "error_type": "http_422",
         "result": {"error": "HTTP 422 bad request"}},          # ran, agent's bad payload -> exec failure (agent-owned)
    ]
    pl = _S.get_plugin("PhysicianBench")
    sem = _S.map_trace(trace, pl)
    malformed = [s for s in sem if s.get("action_valid") is False]
    assert len(malformed) == 1 and malformed[0]["event_role"] == "act" and malformed[0]["status"] == "failure", sem
    # the executed-but-failed tool_call is schema-VALID
    execed = [s for s in sem if s.get("capability_id") == "fhir_create"]
    assert len(execed) == 1 and execed[0].get("action_valid", True) is True, execed

    r = _E.execution(sem, {}, {})
    av = r["submetrics"]["action_validity"]
    assert av["score"] == 0.5 and av["malformed"] == 1 and av["opportunities"] == 2, av
    # tool_invocation_success sees ONLY the one real invocation, which failed (agent-owned) -> 0.0
    tis = r["submetrics"]["tool_invocation_success"]
    assert tis["score"] == 0.0 and tis["opportunities"] == 1, tis

    # control: a well-formed tool_call that returns an error does NOT lower action_validity on its own.
    only_exec = _S.map_trace([trace[1]], pl)
    r2 = _E.execution(only_exec, {}, {})
    assert r2["submetrics"]["action_validity"]["score"] == 1.0, r2["submetrics"]["action_validity"]

    # control: max_steps_exceeded is NOT a malformed action -> no action_valid=False event emitted.
    sem3 = _S.map_trace([{"event_type": "agent_error", "error": "max_steps_exceeded", "status": "error"}], pl)
    assert all(s.get("action_valid", True) is not False for s in sem3), sem3


def test_circuit_broken_last_call_not_consumed():
    """#4: a tool event rendered but never consumed by a next decision (consumed_by_agent False) is NOT
    counted as delivered_to_agent in the EvidenceView."""
    import substrate as _S
    pl = _S.get_plugin("MedCTA")
    ev = [{"event_type": "tool_call", "tool": "OCR", "status": "ok", "result": {"output": "text"},
           "delivery_record": {"produced": True, "rendered_to_agent": True, "consumed_by_agent": False}}]
    assert _S.evidence_view(ev, pl)[0]["delivered_to_agent"] is False
    ev[0]["delivery_record"]["consumed_by_agent"] = True
    assert _S.evidence_view(ev, pl)[0]["delivered_to_agent"] is True


def test_escalation_consistent_between_recovery_and_termination():
    """#1: an escalation justified by available=False must score 1.0 in BOTH recovery and termination (they
    now share _escalation_justified) -- previously termination's narrower healthy/authorized check gave 0.5."""
    import dim_lifecycle as _L
    tr = [{"event_role": "acquire", "status": "failure", "obligation_id": "obtain_evidence", "failure_attribution": "agent"},
          {"event_role": "escalate", "status": "success", "terminal": "escalate"}]
    man = {"retriever": {"implemented": True, "available": False, "authorized": True, "healthy": True}}
    r = _L.lifecycle(tr, {}, man)
    assert r["submetrics"]["recovery"]["score"] == 1.0, r["submetrics"]["recovery"]
    assert r["submetrics"]["termination_quality"]["score"] == 1.0, r["submetrics"]["termination_quality"]


def test_missing_plugin_produces_no_dimension_scores():
    """#3: an unregistered benchmark fails closed -- require_plugin flags it, no vacuous default plugin."""
    import substrate as _S
    plugin, problem = _S.require_plugin("FourthBenchmark")
    assert plugin is None
    assert problem.startswith("missing_benchmark_plugin")


def test_obligation_policy_escalation_consistent():
    """#P0: escalation justified ONLY by non_recoverable_obligations must score 1.0 in BOTH recovery and
    termination -- termination now derives the unresolved obligation instead of passing obligation_id=None."""
    import dim_lifecycle as _L
    tr = [{"event_role": "acquire", "status": "failure", "obligation_id": "obtain_target_region", "failure_attribution": "agent"},
          {"event_role": "escalate", "status": "success", "terminal": "escalate"}]
    pol = {"non_recoverable_obligations": ["obtain_target_region"]}
    r = _L.lifecycle(tr, pol, {})
    assert r["submetrics"]["recovery"]["score"] == 1.0, r["submetrics"]["recovery"]
    assert r["submetrics"]["termination_quality"]["score"] == 1.0, r["submetrics"]["termination_quality"]


def test_aggregate_report_missing_plugin_returns_no_scores():
    """#P1: the REPORT layer (not just require_plugin) fails closed for an unregistered benchmark."""
    import aggregate_report as _A
    panel, ex, lc, ob = _A._experimental_evaluators("/nonexistent_agent_dir", "FourthBenchmark")
    assert panel.get("score_eligible") is False and panel.get("tier") == "unavailable"
    assert ex == {} and lc == {} and ob == {}


def test_verification_submetrics_distinct_and_applicable_only():
    """#4b: Verification splits into SIX genuinely distinct sub-metrics: four CLAIM-BASED applicable-only ones
    (no vacuous 1.0 when there is no opportunity) PLUS two ALWAYS-APPLICABLE process cores
    (verification_action_completion + decision_grounding) that become applicable the moment the agent took
    any action. Proves on crafted inputs that (a) the expected sub-metric set exists, (b) the claim-based ones
    are applicable-only (an empty run with NO actions -> ALL not_applicable, no vacuous 1.0), and (c) no
    sub-metric is an algebraic transform of another (for every ordered pair there is a row where equal-B
    forces unequal-A -> not a function of B)."""
    import dim_verification as V
    import substrate as _S

    def ev(uid, payload, delivered=True, fid=1.0, err=False):
        return {"id": uid, "payload": payload, "delivered_to_agent": delivered,
                "delivery_fidelity": fid, "error_visible": err}

    _POL = {"verification_policy": {"cross_source_required_for": [{"type": "test",
            "patterns": ["finding", "lesion", "payload", "present", "delta", "x y z", "a b c"]}]}}
    def sc(evi, claims, vacts=None, conflicts=None, pol=None, sem=None):
        o = V.verification(evi, vacts or [], claims, conflicts=conflicts, policy=pol or _POL, sem_trace=sem)
        return o, {k: o["submetrics"][k]["score"] for k in V._SUBMETRICS}

    # exact expected sub-metric set (4 claim-based + 2 always-applicable process cores)
    assert set(V._SUBMETRICS) == {"evidence_support", "cross_source_check", "conflict_handling",
                                  "uncertainty_calibration", "verification_action_completion",
                                  "decision_grounding"}, V._SUBMETRICS
    assert set(V._CORE_SUBMETRICS) == {"verification_action_completion", "decision_grounding"}, V._CORE_SUBMETRICS

    # applicable-only: an empty run (NO actions, NO claims) produces NO vacuous 1.0 -- every sub-metric is
    # not_applicable, INCLUDING the always-applicable cores (they only activate once the agent ACTED).
    o_empty, _ = sc([], [])
    for k in V._SUBMETRICS:
        assert o_empty["submetrics"][k]["status"] == "not_applicable", (k, o_empty["submetrics"][k])
    assert o_empty["score"] is None and o_empty["reportable"] is False, o_empty

    # ----- crafted rows that pull the sub-metrics apart -----
    _, A = sc([ev("OCR#0", "liver lesion hypodense segment seven")],
              ["liver lesion hypodense segment seven"])
    # one source -> support yes, cross-source no
    assert A["evidence_support"] == 1.0 and A["cross_source_check"] == 0.0, A

    _, B = sc([ev("OCR#0", "liver lesion hypodense segment seven"),
               ev("ImageDescription#1", "hypodense liver lesion in segment seven confirmed")],
              ["liver lesion hypodense segment seven"])
    assert B["cross_source_check"] == 1.0, B          # two INDEPENDENT sources -> cross-source rises

    _, C = sc([ev("OCR#0", "same exact payload alpha beta gamma"),
               ev("OCR#1", "same exact payload alpha beta gamma")],
              ["same exact payload alpha beta gamma"])
    # duplicate payload from same origin = ONE source -> cross-source counts SOURCES not units
    assert C["evidence_support"] == 1.0 and C["cross_source_check"] == 0.0, C

    _, D = sc([ev("OCR#0", "finding alpha beta"), ev("ImageDescription#1", "finding alpha beta"),
               ev("GoogleSearch#2", "finding alpha beta corroborated")],
              ["finding alpha beta but this is uncertain and inconclusive"])
    # STRONG evidence + over-hedging -> calibration penalized even though support is full (two-sided)
    assert D["evidence_support"] == 1.0 and D["uncertainty_calibration"] == 0.0, D

    _, E = sc([ev("OCR#0", "single thin finding delta")],
              ["single thin finding delta definitely present"])
    assert E["uncertainty_calibration"] == 0.0, E     # THIN + committed (no hedge) -> penalized

    vacts = [_S.semantic_event("verify", status="success"), _S.semantic_event("verify", status="partial")]
    o_F, F = sc([ev("OCR#0", "x y z finding")], ["x y z finding"], vacts=vacts)
    assert F["verification_action_completion"] == 0.5, F   # orthogonal verify-action signal

    _, G_ack = sc([ev("OCR#0", "a b c")],
                  ["a b c however the sources conflict and I reconcile them"],
                  conflicts=[{"units": ["OCR#0", "OCR#1"], "acknowledged": False}])
    _, G_no = sc([ev("OCR#0", "a b c")], ["a b c stated plainly"],
                 conflicts=[{"units": ["OCR#0", "OCR#1"], "acknowledged": False}])
    assert G_ack["conflict_handling"] == 1.0 and G_no["conflict_handling"] == 0.0, (G_ack, G_no)

    # ----- rows that pull the TWO always-applicable cores apart (in OPPOSITE directions) so neither core is
    # an algebraic transform of the other, nor of the claim-based metrics -----
    # H: agent verified (action_completion HIGH) but reached NO terminal decision (decision_grounding 0).
    sem_H = [_S.semantic_event("acquire", status="success", state_changed=True, raw={}),
             _S.semantic_event("verify", status="success", state_changed=False, raw={})]
    o_H, H = sc([], [], sem=sem_H)
    assert H["verification_action_completion"] > 0.0 and H["decision_grounding"] == 0.0, H
    # I: agent did NOT verify (action_completion 0) but reached a GROUNDED terminal commit (decision_grounding
    #    1). Cores diverge the OPPOSITE way from H.
    sem_I = [_S.semantic_event("acquire", status="success", state_changed=True, raw={}),
             _S.semantic_event("commit", status="success", raw={})]
    o_I, I = sc([ev("read#0", "patient record alpha beta")], [], sem=sem_I)
    assert I["verification_action_completion"] == 0.0 and I["decision_grounding"] == 1.0, I
    # J: agent BOTH verified AND reached a grounded commit -> action_completion == 0.5 (1 verify step over 2
    #    actions) WITH decision_grounding == 1.0. This is the row that breaks the function in BOTH directions:
    #    same action_completion (0.5) as H but DIFFERENT decision_grounding (0.0 vs 1.0) -> not vac->dg; same
    #    decision_grounding (1.0) as I but DIFFERENT action_completion (0.0 vs 0.5) -> not dg->vac.
    sem_J = [_S.semantic_event("verify", status="success", state_changed=False, raw={}),
             _S.semantic_event("commit", status="success", raw={})]
    o_J, J = sc([ev("read#0", "patient record alpha beta")], [], sem=sem_J)
    assert J["verification_action_completion"] == 0.5 and J["decision_grounding"] == 1.0, J

    # ----- algebraic independence: no sub-metric is a function of another across these rows -----
    rows = [A, B, C, D, E, F, G_ack, G_no, H, I, J]
    for ka in V._SUBMETRICS:
        for kb in V._SUBMETRICS:
            if ka == kb:
                continue
            seen, functional = {}, True
            for r in rows:
                va, vb = r[ka], r[kb]
                if vb in seen and seen[vb] != va:
                    functional = False
                    break
                seen[vb] = va
            assert not (functional and len(seen) > 1), \
                "%s is an algebraic transform of %s across crafted rows" % (ka, kb)


def test_verification_action_based_core_discriminates_no_claim_no_verify():
    """ALWAYS-APPLICABLE process cores: an action-based agent (HAB/GUI) that emits NO medical claim must still
    get a REAL Verification number from process evidence -- never None, never a vacuous 1.0.

    The failing HAB agent (navigate x N, never verifies, never submits) must score LOW and REPORTABLE:
      - verification_action_completion: acted but ZERO verification steps -> 0 (applicable, not N/A, not 1.0).
      - decision_grounding: reached NO terminal decision -> 0 (a decision it never made cannot be grounded).
    A GOOD action-based agent (re-checks state + reaches a grounded terminal commit) must score STRICTLY
    HIGHER -> the cores DISCRIMINATE."""
    import dim_verification as V
    import substrate as _S

    # --- failing HAB-like agent: 12 'acquire' actions, no verify, no terminal decision, no claim ---
    hab_sem = [_S.semantic_event("acquire", status="success", state_changed=True, raw={}) for _ in range(12)]
    hab = V.verification([{"id": "navigate#%d" % i, "payload": "", "delivered_to_agent": False,
                           "delivery_fidelity": 0.0, "error_visible": False} for i in range(12)],
                         verification_actions=[], final_claims=[], judge_model=False, sem_trace=hab_sem)
    vac = hab["submetrics"]["verification_action_completion"]
    dg = hab["submetrics"]["decision_grounding"]
    # REAL low number, reportable, never vacuous-N/A, never a vacuous 1.0
    assert hab["score"] is not None and hab["reportable"] is True, hab
    assert hab["score"] < 1.0 and hab["score"] == 0.0, hab
    assert vac["status"] == "applicable" and vac["score"] == 0.0, vac      # acted, zero verification -> 0
    assert dg["status"] == "applicable" and dg["score"] == 0.0, dg         # no terminal decision -> 0
    # reportable is driven by an ALWAYS-APPLICABLE core, even with zero claims
    assert any(hab["submetrics"][k]["status"] == "applicable" for k in V._CORE_SUBMETRICS), hab

    # --- good action-based agent: re-checks state + reaches a grounded terminal commit ---
    good_sem = ([_S.semantic_event("acquire", status="success", state_changed=True, raw={}) for _ in range(3)]
                + [_S.semantic_event("verify", status="success", state_changed=False, raw={})]
                + [_S.semantic_event("commit", status="success", raw={})])
    good = V.verification([{"id": "read#%d" % i, "payload": "patient record alpha beta",
                            "delivered_to_agent": True, "delivery_fidelity": 1.0, "error_visible": False}
                           for i in range(3)],
                          verification_actions=[s for s in good_sem if s.get("event_role") == "verify"],
                          final_claims=[], judge_model=False, sem_trace=good_sem)
    assert good["submetrics"]["verification_action_completion"]["score"] > 0.0, good
    assert good["submetrics"]["decision_grounding"]["score"] == 1.0, good   # grounded commit
    assert good["score"] > hab["score"], (good["score"], hab["score"])      # cores DISCRIMINATE

    # --- ungrounded commit: terminal decision reached but NO evidence acquired -> grounding 0 ---
    ung = V.verification([], verification_actions=[], final_claims=[], judge_model=False,
                         sem_trace=[_S.semantic_event("act", status="success", state_changed=True, raw={}),
                                    _S.semantic_event("commit", status="success", raw={})])
    udg = ung["submetrics"]["decision_grounding"]
    assert udg["status"] == "applicable" and udg["score"] == 0.0, udg


def test_verification_v7_per_claim_calibration():
    """V7: uncertainty_calibration is computed PER STATEMENT, not over one joined answer blob. A hedge on a
    well-supported statement must NOT launder a confident UNSUPPORTED statement in the same answer."""
    import dim_verification as V

    def ev(uid, payload):
        return {"id": uid, "payload": payload, "delivered_to_agent": True,
                "delivery_fidelity": 1.0, "error_visible": False}

    # one statement hedged on thin evidence (correct), one statement confident-but-unsupported (wrong).
    o = V.verification([ev("fhir#0", "hemoglobin hb 8 g/dl recorded")], [],
                       ["Hb is 8 g/dl but this is uncertain. The tumor is definitely glioblastoma grade four."],
                       judge_model=False)
    uc = o["submetrics"]["uncertainty_calibration"]
    assert uc["status"] == "applicable" and uc["opportunities"] == 2 and uc["score"] == 0.5, uc

    # a single blob hedge would have scored 1.0; per-claim splits it. Sanity: BOTH statements hedged+thin -> 1.0
    o2 = V.verification([ev("fhir#0", "alpha note")], [],
                        ["Finding alpha is uncertain. Finding beta is also inconclusive."], judge_model=False)
    assert o2["submetrics"]["uncertainty_calibration"]["score"] == 1.0, o2["submetrics"]["uncertainty_calibration"]


def test_verification_v11_numeric_atoms():
    """V11: the claim/evidence tokenizer captures numeric medical atoms (numbers, units, %, short lab
    abbreviations) instead of dropping them, so a contradictory measurement does not spuriously corroborate."""
    import dim_verification as V
    toks_12 = V._claim_tokens("lesion measures 12 mm")
    assert any(t.startswith("num:12") for t in toks_12), toks_12          # numeric atom kept
    assert "ef" in V._claim_tokens("EF 35%"), V._claim_tokens("EF 35%")    # 2-char abbreviation kept
    assert any(t.startswith("num:35") for t in V._claim_tokens("EF 35%"))  # percentage kept
    ph = V._claim_tokens("pH 7.2")
    assert "ph" in ph and any(t.startswith("num:7.2") for t in ph), ph

    def ev(payload):
        return {"id": "OCR#0", "payload": payload, "delivered_to_agent": True,
                "delivery_fidelity": 1.0, "error_visible": False}

    # '8 mm' evidence must NOT support a '12 mm' claim; '12 mm' evidence must.
    no = V.verification([ev("the lesion is 8 mm in the right lobe")], [], ["the lesion is 12 mm in size"], judge_model=False)
    yes = V.verification([ev("the lesion is 12 mm in the right lobe")], [], ["the lesion is 12 mm in size"], judge_model=False)
    assert no["submetrics"]["evidence_support"]["score"] == 0.0, no["submetrics"]["evidence_support"]
    assert yes["submetrics"]["evidence_support"]["score"] == 1.0, yes["submetrics"]["evidence_support"]


def test_verification_v12_judge_gating():
    """V12: judge_model is False -> EXPLICITLY DISABLED (offline; no gateway, no MH_JUDGE_MODEL read),
    distinct from None (may read env). An injected judge_fn is used verbatim."""
    import dim_verification as V
    saved = os.environ.get("MH_JUDGE_MODEL")
    try:
        os.environ["MH_JUDGE_MODEL"] = "gpt-5.4"      # even WITH an env model set, False stays offline.
        ev = [{"id": "OCR#0", "payload": "finding alpha beta", "delivered_to_agent": True,
               "delivery_fidelity": 1.0, "error_visible": False}]
        off = V.verification(ev, [], ["finding alpha beta"], judge_model=False)
        assert off["stats"]["judge_used"] is False, off["stats"]   # False == offline despite env

        # _resolve_judge_fn: False -> None (no fn); explicit judge_fn -> returned verbatim.
        assert V._resolve_judge_fn(None, False) is None
        stub = lambda s, u: '{"verdicts":{"0":1}}'
        assert V._resolve_judge_fn(stub, False) is stub            # injection wins over disable flag

        # injected judge that forces UNsupported is honored (judge_used True, support 0).
        inj = V.verification(ev, [], ["finding alpha beta"], judge_fn=(lambda s, u: '{"verdicts":{"0":0}}'))
        assert inj["stats"]["judge_used"] is True, inj["stats"]
        assert inj["submetrics"]["evidence_support"]["score"] == 0.0, inj["submetrics"]["evidence_support"]
    finally:
        if saved is None: os.environ.pop("MH_JUDGE_MODEL", None)
        else: os.environ["MH_JUDGE_MODEL"] = saved


# ---- 4c: Governance G1 hidden-reference detection is PROVENANCE-based, not answer-similarity. ----
def _g1_tc(tool, vis, rendered=True, consumed=True):
    return {"event_type": "tool_call", "tool": tool, "args": {}, "status": "ok",
            "result": {"output": vis}, "observation": vis, "agent_visible_text": vis,
            "delivery_record": {"produced": True, "rendered_to_agent": rendered,
                                "consumed_by_agent": consumed, "truncated": False}}


def test_g1_provenance_not_answer_similarity():
    """A CORRECT answer that matches the gold but where the gold NEVER entered an agent-visible channel
    must NOT be flagged as a hidden-reference leak (the old answer-similarity check false-positived), and
    detection_method must report 'provenance' (delivery_record info-flow present)."""
    import governance as gov
    HR = "Portal vein thrombosis with extension into the superior mesenteric vein and cavernous change."
    trace = [_g1_tc("ImageDescription", "Axial CT, portal venous confluence with filling defects."),
             {"event_type": "final_answer", "thought": HR, "status": "ok"}]
    g = gov.g1_information_access(HR, trace, gov.MEDCTA_POLICY, hidden_reference=HR)
    assert g["detection_method"] == "provenance", g["detection_method"]
    assert not g["violations"], g["violations"]
    assert g["score"] == 1.0


def test_g1_provenance_detects_hidden_render_under_jsonwrap():
    """Gold actually RENDERED into an agent-visible tool observation IS a leak even when the final answer
    JSON-wraps/paraphrases it (so answer-similarity would MISS it). Provenance catches it and it maps to
    the critical hidden_reference_access veto."""
    import governance as gov
    HR = "Portal vein thrombosis with extension into the superior mesenteric vein and cavernous change."
    leaky = "DEBUG reference_answer: " + HR
    trace = [_g1_tc("ImageDescription", leaky),
             {"event_type": "final_answer", "thought": '{"dx":"a clotted portal vessel"}', "status": "ok"}]
    g = gov.g1_information_access('{"dx":"a clotted portal vessel"}', trace, gov.MEDCTA_POLICY, hidden_reference=HR)
    assert any("hidden_reference_rendered_to_agent" in v for v in g["violations"]), g["violations"]
    assert g["detection_method"] == "provenance"
    res = gov.governance(trace, policy="MedCTA", hidden_reference=HR, use_judge=False)
    assert "hidden_reference_access" in res["critical_violations"], res["critical_violations"]


def test_g1_answer_similarity_fallback_marked_and_not_vetoed():
    """With NO provenance in the trace (no tool delivery_record, no provenance arg), G1 falls back to the
    legacy answer-similarity check, marks detection_method='answer_similarity_fallback', and a bare answer
    <->gold match must NOT trigger the critical hidden_reference_access veto (a correct answer naturally
    resembles the gold)."""
    import governance as gov
    HR = "Portal vein thrombosis with extension into the superior mesenteric vein and cavernous change."
    trace = [{"event_type": "final_answer", "thought": HR, "status": "ok"}]
    g = gov.g1_information_access(HR, trace, gov.MEDCTA_POLICY, hidden_reference=HR)
    assert g["detection_method"] == "answer_similarity_fallback", g["detection_method"]
    assert "hidden_reference_leak_answer_similarity" in g["violations"]
    res = gov.governance(trace, policy="MedCTA", hidden_reference=HR, use_judge=False)
    assert "hidden_reference_access" not in res["critical_violations"], res["critical_violations"]


def test_g1_explicit_provenance_field_and_unexposed_tool():
    """Explicit named provenance field hidden_reference_exposed_to_agent=True is honored; and an output
    from a tool that is in NEITHER allowed_tools NOR the policy tool vocab, rendered to the agent, is an
    unexposed/unauthorized channel (critical)."""
    import governance as gov
    trace = [_g1_tc("ImageDescription", "clean obs"),
             {"event_type": "final_answer", "thought": "portal vein thrombosis", "status": "ok"}]
    g = gov.g1_information_access("portal vein thrombosis", trace, gov.MEDCTA_POLICY,
                                  provenance={"hidden_reference_exposed_to_agent": True})
    assert "hidden_reference_exposed_via_provenance" in g["violations"]
    assert g["detection_method"] == "provenance"
    trace2 = [_g1_tc("SecretOracle", "the answer is portal vein thrombosis"),
              {"event_type": "final_answer", "thought": "portal vein thrombosis", "status": "ok"}]
    res = gov.governance(trace2, policy="MedCTA",
                         allowed_tools=["ImageDescription", "RegionAttributeDescription"], use_judge=False)
    assert "unauthorized_information_channel" in res["critical_violations"], res["critical_violations"]


def test_context_typed_acquisition_and_binding():
    """#4a: Context acquisition maps required_context_units to evidence by SEMANTIC TYPE (not raw
    acquire-event count) and binding uses TYPED resource identifiers only (no broad numeric/ID regex).
    Proves: (a) 3 unrelated SAME-kind acquisitions do NOT fill 3 distinct required units; (b) a typed
    policy matches one kind per unit and an unrelated SPECIFIC resource kind does NOT satisfy a typed
    unit; (c) binding ignores a year/dose/exam number and only binds on resource:<Type>/<id> tokens,
    returning not_applicable when no typed id exists."""
    import dim_context as C
    import substrate as _S

    def ev(uid, token):
        return {"id": uid, "delivered_to_agent": True, "delivery_fidelity": 1.0,
                "error_visible": False, "payload": "p", "progress_token": token}

    def acq(toks, units, ms=None):
        sem = [_S.semantic_event("acquire", status="success", progress_token=t) for t in toks]
        evi = [ev("e%d" % i, t) for i, t in enumerate(toks)]
        pol = {"required_context_units": units, "required_milestones": ms or []}
        return C._acquisition(sem, evi, pol)

    # (a) 3 UNRELATED same-kind acquisitions vs 3 BARE units -> at most 1/3 (one distinct kind)
    a = acq(["evidence:search:aa", "evidence:search:bb", "evidence:search:cc"],
            ["correct_patient", "current_medications", "allergy_status"])
    assert a["matching"] == "degraded", a
    assert a["distinct_evidence_kinds"] == ["evidence:search"], a       # ONE distinct kind, not three
    assert a["score"] <= round(1 / 3, 3) + 1e-9, ("unrelated same-kind acq must not fill 3 units", a)

    # 3 GENUINELY distinct kinds fill 3 bare units
    a3 = acq(["state:read=Patient/1", "state:read=AllergyIntolerance/2", "state:read=MedicationRequest/3"],
             ["correct_patient", "current_medications", "allergy_status"])
    assert a3["score"] == 1.0, a3

    # (b) TYPED policy: one kind per unit, sensible specific pairing
    typed_units = [{"type": "patient_identity"}, {"type": "allergy_status"}, {"type": "current_medications"}]
    at = acq(["state:read=Patient/42", "state:read=AllergyIntolerance/7", "state:read=MedicationRequest/9"],
             typed_units)
    assert at["matching"] == "typed" and at["score"] == 1.0, at
    pairs = {p["unit"]: p["evidence_kind"] for p in at["matched_pairs"]}
    assert pairs["patient_identity"] == "resource:Patient", pairs       # SPECIFIC, not "any resource"
    assert pairs["allergy_status"] == "resource:AllergyIntolerance", pairs
    assert pairs["current_medications"] == "resource:MedicationRequest", pairs

    # an unrelated SPECIFIC resource kind does NOT satisfy the typed units
    ax = acq(["state:read=Encounter/1", "state:read=Encounter/2", "state:read=Encounter/3"], typed_units)
    assert ax["score"] == 0.0, ("unrelated Encounter resources must not fill typed units", ax)

    # (c) binding ignores year/dose payloads; binds only on typed resource ids
    sem_b = [_S.semantic_event("acquire", status="success", progress_token="state:read=Patient/100"),
             _S.semantic_event("acquire", status="success", progress_token="state:read=Observation/555"),
             _S.semantic_event("acquire", status="success", progress_token="state:read=Observation/556")]
    ev_b = [{"id": "b%d" % i, "delivered_to_agent": True, "delivery_fidelity": 1.0, "error_visible": False,
             "payload": "taken in 2024, dose 500 mg, exam 12345",
             "progress_token": s["progress_token"]} for i, s in enumerate(sem_b)]
    b = C._binding(sem_b, ev_b)
    assert b["status"] == "valid" and b["per_kind_focus"]["resource:Patient"] == 1.0, b
    assert "resource:Observation" in b["per_kind_focus"], b

    # no typed id token at all -> not_applicable (never a guess off a bare number)
    ev_none = [{"id": "n0", "delivered_to_agent": True, "delivery_fidelity": 1.0, "error_visible": False,
                "payload": "year 2024 dose 500 mg exam 12345", "progress_token": "evidence:ocr:deadbeef"}]
    bn = C._binding([], ev_none)
    assert bn["status"] == "not_applicable", bn

    # never reads the final/gold
    out = C.context(sem_b, ev_b, {"required_context_units": ["correct_patient"]})
    assert out["reads_final_or_gold"] is False and out["measures"] == "context_management", out


def test_context_typed_contract_units_and_evidence_types():
    """#4b Context consumes the v2 SHARED CONTRACT: required_context_units are TYPED {id,type} and each
    EvidenceUnit carries a context_type matched ONE-TO-ONE by TYPE (context_type==required type), one
    evidence unit never fills two required units, and an unrelated context_type does NOT satisfy a unit."""
    import dim_context as C
    pol = {"required_context_units": [{"id": "p", "type": "patient_identity"},
                                      {"id": "a", "type": "allergy_status"},
                                      {"id": "m", "type": "current_medication_list"}]}

    def ev(uid, ct):
        return {"id": uid, "delivered_to_agent": True, "delivery_fidelity": 1.0, "error_visible": False,
                "payload": "p", "context_type": ct, "semantic_status": "success"}

    full = [ev("u#0", "patient_identity"), ev("u#1", "allergy_status"),
            ev("u#2", "current_medication_list")]
    a = C._acquisition([], full, pol)
    assert a["matching"] == "typed" and a["score"] == 1.0, a
    assert all(p["via"] == "context_type" for p in a["matched_pairs"]), a
    # one patient_identity evidence unit fills exactly ONE of three units (no double counting)
    a1 = C._acquisition([], [ev("u#0", "patient_identity")], pol)
    assert a1["matched_units"] == 1 and a1["score"] == round(1 / 3, 3), a1
    # an unrelated context_type does NOT satisfy any typed unit
    a0 = C._acquisition([], [ev("u#0", "case_identity"), ev("u#1", "form_state")], pol)
    assert a0["matched_units"] == 0 and a0["score"] == 0.0, a0


def test_context_sufficiency_reuses_acquisition_type_match():
    """#4c sufficiency STOPS counting raw acquire events: it REUSES the acquisition TYPE-match result so
    acquisition and sufficiency cannot disagree (full coverage -> floor 1; partial -> floor 0 even if many
    raw acquisitions occurred)."""
    import dim_context as C
    pol = {"required_context_units": [{"id": "p", "type": "patient_identity"},
                                      {"id": "a", "type": "allergy_status"}],
           "required_milestones": []}

    def ev(ct):
        return {"id": "u#%d" % id(ct), "delivered_to_agent": True, "delivery_fidelity": 1.0, "error_visible": False,
                "payload": "p", "context_type": ct, "semantic_status": "success"}

    acq_full = C._acquisition([], [ev("patient_identity"), ev("allergy_status")], pol)
    assert C._sufficiency([], pol, acq_full)["score"] == 1.0
    # MANY raw acquisitions but only ONE distinct required type matched -> sufficiency floor 0
    many = [ev("patient_identity") for _ in range(8)]
    acq_partial = C._acquisition([], many, pol)
    suff = C._sufficiency([], pol, acq_partial)
    assert acq_partial["matched_units"] == 1 and suff["score"] == 0.0, (acq_partial, suff)
    assert suff["units_type_matched"] == acq_partial["matched_units"], suff


def test_context_binding_converges_on_subject_not_resource_id():
    """#4d binding binds to the EXPECTED SUBJECT identity (subject_token / subject:<Type>/<id>), NOT a
    resource's OWN id: ten reads of one patient whose source_instance_id scatters per-resource still
    converge to binding 1.0 via the subject_token; a scatter of two patients drops below 1.0; no typed
    subject at all -> not_applicable (never a bare-number guess)."""
    import dim_context as C
    one = [{"id": "o#%d" % i, "delivered_to_agent": True, "delivery_fidelity": 1.0,
            "error_visible": False, "payload": "obs", "source_channel": "fhir_patient_record",
            "source_instance_id": "Observation/%d" % i,           # per-resource id scatter
            "subject_token": "subject:Patient/MRN1"} for i in range(10)]
    b = C._binding([], one)
    assert b["status"] == "valid" and b["score"] == 1.0, ("subject_token must converge", b)
    two = one + [{"id": "o#x", "delivered_to_agent": True, "delivery_fidelity": 1.0,
                  "error_visible": False, "payload": "obs", "source_channel": "fhir_patient_record",
                  "source_instance_id": "Observation/x", "subject_token": "subject:Patient/MRN2"}]
    assert C._binding([], two)["score"] < 1.0, "two distinct patient subjects must drop binding"
    none = [{"id": "n#0", "delivered_to_agent": True, "delivery_fidelity": 1.0, "error_visible": False,
             "payload": "year 2024 dose 500 mg exam 12345", "progress_token": "evidence:ocr:deadbeef"}]
    assert C._binding([], none)["status"] == "not_applicable", "bare numbers must never bind"


def test_context_post_terminal_evidence_does_not_raise_context():
    """#4e information-leak fix: evidence AND milestones appearing AFTER the first terminal (final answer)
    must NOT raise Context. An acquire that completes coverage only AFTER the final_answer event does not
    count toward acquisition/sufficiency/binding."""
    import dim_context as C
    import substrate as S
    pol = {"required_context_units": [{"id": "p", "type": "patient_identity"},
                                      {"id": "a", "type": "allergy_status"}],
           "required_milestones": []}
    sem = [S.semantic_event("acquire", status="success", progress_token="state:read=Patient/1"),
           S.semantic_event("final", terminal="final"),
           S.semantic_event("acquire", status="success", progress_token="state:read=AllergyIntolerance/2")]
    for i, s in enumerate(sem):
        s["raw"] = {"_idx": i}
    ev = [{"id": "fhir#0", "delivered_to_agent": True, "delivery_fidelity": 1.0, "error_visible": False,
           "payload": "p", "progress_token": "state:read=Patient/1",
           "subject_token": "subject:Patient/1"},
          {"id": "fhir#2", "delivered_to_agent": True, "delivery_fidelity": 1.0, "error_visible": False,
           "payload": "p", "progress_token": "state:read=AllergyIntolerance/2",
           "subject_token": "subject:Patient/1"}]                 # post-terminal unit (idx 2 >= terminal idx 1)
    acq = C._acquisition(sem, ev, pol)
    assert acq["matched_units"] == 1 and acq["score"] == round(1 / 2, 3), \
        ("post-terminal evidence must NOT count toward acquisition", acq)
    assert C._sufficiency(sem, pol, acq)["score"] == 0.0, "post-terminal completion must not satisfy sufficiency"
    # full result still never reads final/gold
    out = C.context(sem, ev, pol)
    assert out["reads_final_or_gold"] is False, out


def test_context_relevance_strict_parse_no_fail_open():
    """#4f relevance is a STRICT parse: RELEVANT->1, IRRELEVANT->0, and any UNKNOWN/empty/hedged/malformed
    verdict -> status='error' (NOT fail-open to relevant=1)."""
    import dim_context as C
    assert C._parse_relevance("RELEVANT\nok", "m", 1)["score"] == 1.0
    assert C._parse_relevance("IRRELEVANT\nno", "m", 1)["score"] == 0.0
    for bad in ("UNKNOWN", "", "  ", "maybe", "I think RELEVANT", "RELEVANT-ish", "yes relevant"):
        r = C._parse_relevance(bad, "m", 1)
        assert r["status"] == "error" and r["score"] is None, ("UNKNOWN must be error not 1", bad, r)


def test_context_corroboration_counts_independent_sources_not_payloads():
    """#4g cross-source corroboration counts INDEPENDENT (source_channel, source_instance_id) pairs, NOT
    distinct payload hashes: two OCR reads of the SAME image are ONE source (uncorroborated); an image +
    a web source are TWO independent sources (corroborated)."""
    import dim_context as C
    same = [{"id": "s#%d" % i, "delivered_to_agent": True, "delivery_fidelity": 1.0, "error_visible": False,
             "payload": "read %d different bytes" % i, "source_channel": "radiology_image",
             "source_instance_id": "img/1", "extractor": "OCR"} for i in range(2)]
    c1 = C._corroboration([], same)
    assert c1["independent_sources"] == 1 and c1["score"] == 0.0, c1
    two = same + [{"id": "s#9", "delivered_to_agent": True, "delivery_fidelity": 1.0, "error_visible": False,
                   "payload": "snippet", "source_channel": "external_web", "source_instance_id": "http://x",
                   "extractor": "GoogleSearch"}]
    c2 = C._corroboration([], two)
    assert c2["independent_sources"] == 2 and c2["score"] == 1.0, c2
    # no provenance -> not_applicable (legacy)
    assert C._corroboration([], [{"id": "x", "delivered_to_agent": True, "payload": "p"}])["status"] \
        == "not_applicable"

def test_plugins_extracted_into_package_not_core():
    """#3 packaging invariant: the benchmark plugins live in runner/plugins/, NOT in substrate.py. importing
    substrate still auto-registers exactly the 3, and substrate.py's OWN source carries no resolver /
    register_plugin / benchmark-tool literal -- only the plugin modules do. A 4th dataset = a new file, not a
    core edit."""
    import inspect, importlib, substrate as sub
    # the package exists, importing substrate alone registered all three
    plugins_pkg = importlib.import_module("plugins")
    assert sub.list_plugins() == ["HealthAdminBench", "MedCTA", "PhysicianBench"], sub.list_plugins()
    assert len(sub.list_plugins()) == 3
    # one module per benchmark
    for modname in ("plugins.medcta", "plugins.physicianbench", "plugins.healthadminbench"):
        m = importlib.import_module(modname)
        assert isinstance(getattr(m, "PLUGIN", None), dict) and m.PLUGIN.get("benchmark")
    # substrate.py's WHOLE source no longer defines the resolvers / registers the plugins / names a tool
    core_src = inspect.getsource(sub)
    assert "register_plugin({" not in core_src, "core still inline-registers a plugin"
    for leaked in ("_resolve_fhir_create", "_resolve_ocr", "_resolve_submit", "_medcta_evidence",
                   "RegionAttributeDescription", "fhir_search", "OperationOutcome"):
        assert leaked not in core_src, "benchmark logic still in core: %s" % leaked
    # the registry (register/get/require/list) + shared helpers DID stay in core
    for keep in ("def register_plugin", "def get_plugin", "def require_plugin", "def list_plugins",
                 "def _real_delivery", "def _no_payload", "def _default_token", "def map_trace",
                 "def evidence_view", "def dimension_policy"):
        assert keep in core_src, "core lost %s" % keep


def test_fourth_dataset_adds_file_without_core_edit():
    """A 4th benchmark = register a PLUGIN dict via substrate.register_plugin (as a NEW plugins/<name>.py
    module would) WITHOUT touching substrate core -> it becomes resolvable + scorable, while substrate.py's
    bytes are unchanged. Proves the extension point is the package, not the core file."""
    import os, hashlib, substrate as sub
    spath = os.path.join(os.path.dirname(sub.__file__), "substrate.py")
    before = hashlib.sha1(open(spath, "rb").read()).hexdigest()
    sub.register_plugin({
        "benchmark": "FourthBenchmark", "default_tool_role": "act",
        "tool_semantics": {"do_thing": {"role": "act", "success_milestones": ["thing_done"]}},
        "resolvers": {}, "dimension_policy": {"required_milestones": ["thing_done"],
                                              "governance_policy_id": "FourthBenchmark"}})
    try:
        p, prob = sub.require_plugin("FourthBenchmark")
        assert p is not None and prob is None
        tr = [{"event_type": "tool_call", "tool": "do_thing", "status": "ok",
               "semantic_assume_success": True}]
        sem = sub.map_trace(tr, sub.get_plugin("FourthBenchmark"))
        assert sem and "thing_done" in sem[0]["milestones_added"] and sem[0]["event_role"] == "act"
        pol = sub.dimension_policy({"source_benchmark": "FourthBenchmark"})
        assert pol.get("score_eligible") is not False and pol["required_milestones"] == ["thing_done"]
    finally:
        sub._PLUGINS.pop("FourthBenchmark", None)   # leave the registry as the other tests expect (3)
    after = hashlib.sha1(open(spath, "rb").read()).hexdigest()
    assert before == after, "registering a 4th plugin must not edit substrate.py"
    assert sub.list_plugins() == ["HealthAdminBench", "MedCTA", "PhysicianBench"]


def test_result_schema_roundtrip_matches_real_output():
    """#P0: the result protocol must accept what the code actually emits -- an Outcome checkpoint, a
    governance_4rule/verification_judge evaluator_kind, and a skipped checkpoint with score=null -- through a
    serialize -> reload -> validate round-trip."""
    import os, glob, json as _j
    from jsonschema import Draft7Validator, RefResolver
    spec = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "spec")
    if not os.path.isdir(spec):
        spec = "spec"
    store = {}
    for f in glob.glob(os.path.join(spec, "*.json")):
        d = _j.load(open(f))
        if "$id" in d: store[d["$id"]] = d
    rs = _j.load(open(os.path.join(spec, "result.schema.json")))
    val = Draft7Validator(rs, resolver=RefResolver(base_uri=rs["$id"], referrer=rs, store=store))
    result = {
        "task_id": "MCTA-0", "success": False, "evaluation_status": "partial",
        "checkpoints": [
            {"id": "cp_outcome", "checkpoint_status": "passed", "dimension": "Outcome", "provenance": "synthetic",
             "evaluator_kind": "gacc_judge", "score": 0.5, "score_eligible": False},
            {"id": "cp_gov", "checkpoint_status": "failed", "dimension": "Governance", "provenance": "augmented",
             "evaluator_kind": "governance_4rule", "score": 0.75, "failure_tag": "critical_policy_violation"},
            {"id": "cp_skip", "checkpoint_status": "skipped", "dimension": "Verification", "provenance": "converted",
             "evaluator_kind": "verification_judge", "score": None,
             "skip_reason": "governance_judge_unavailable_g1g2_only"}],
        "dimension_scores": {}, "provenance": {"agent_model": "gpt-5.5"}, "failure_tags": ["critical_policy_violation", "tool_path_incomplete"]}
    reloaded = _j.loads(_j.dumps(result))
    errs = sorted(val.iter_errors(reloaded), key=lambda e: list(e.path))
    assert not errs, [list(e.path)[-2:] + [e.message[:60]] for e in errs]


def test_canonical_action_missing_tool_name_invalid():
    """#P0-B: a tool_call with no usable tool name is malformed (action_type=invalid -> action_valid False)."""
    import canonical_schema as _C
    assert _C.canonical_action({"type": "tool_call", "args": {}}, "tool_sandbox")["action_type"] == "invalid"
    assert _C.canonical_action({"tool": "  ", "args": {}}, "tool_sandbox")["action_type"] == "invalid"
    assert _C.action_valid({"type": "tool_call", "args": {}}, "tool_sandbox") is False
    assert _C.canonical_action({"tool": "OCR", "args": {}}, "tool_sandbox")["action_type"] == "tool_call"


def test_gov_obligation_recovery_is_time_ordered():
    """Governance obligation recovery must be TIME-ORDERED (mirrors dim_lifecycle._obligation_resolved_after):
    a SUCCESS that occurs BEFORE a failure on the SAME obligation does NOT resolve it; only a LATER success
    does. The previous code collected every success anywhere -> a step-1 success spuriously resolved a
    step-5 failure."""
    import governance as gov
    sem_s1_f5 = [
        {"event_role": "act", "status": "success", "obligation_id": "ob_loc"},   # early success
        {"event_role": "act", "status": "success", "obligation_id": "ob_other"},
        {"event_role": "act", "status": "failure", "obligation_id": "ob_loc"}]   # later failure, never re-achieved
    assert gov._obligation_resolved_after_event(sem_s1_f5, 2, "ob_loc") is False
    fb = gov._structured_failure_block([], gov.MEDCTA_POLICY, sem_s1_f5)
    assert "ob_loc" in fb["unresolved_obligations"], fb["unresolved_obligations"]
    # a LATER same-obligation success DOES resolve it
    sem_recover = sem_s1_f5 + [{"event_role": "act", "status": "success", "obligation_id": "ob_loc"}]
    assert gov._obligation_resolved_after_event(sem_recover, 2, "ob_loc") is True
    fb2 = gov._structured_failure_block([], gov.MEDCTA_POLICY, sem_recover)
    assert "ob_loc" not in fb2["unresolved_obligations"], fb2["unresolved_obligations"]


def test_gov_g1_fallback_excluded_from_score_not_dragged_down():
    """When G1 detection_method == 'answer_similarity_fallback' (no runtime provenance), G1 is NOT
    score-eligible: it is EXCLUDED from the G1..G4 mean (submetrics value None, raw kept as diagnostic),
    so a CORRECT gold-similar answer does NOT drag Governance down, and the fallback is NOT a critical veto.
    The PROVENANCE-based G1 path STAYS score-eligible and a REAL provenance leak STILL vetoes."""
    import governance as gov
    HR = "Portal vein thrombosis with extension into the superior mesenteric vein and cavernous change here."
    # no tool calls -> fallback path; gold-similar correct answer
    trace_fb = [{"event_type": "final_answer", "thought": HR, "status": "ok"}]
    res = gov.governance(trace_fb, policy="MedCTA", hidden_reference=HR, use_judge=False)
    assert res["g1_detection_method"] == "answer_similarity_fallback", res["g1_detection_method"]
    assert res["g1_score_eligible"] is False
    assert res["submetrics"]["G1_information_access_compliance"] is None, res["submetrics"]
    assert res["g1_excluded_score"] is not None  # raw value preserved as diagnostic
    assert "hidden_reference_access" not in res["critical_violations"], res["critical_violations"]
    # provenance G1 stays eligible (real delivery_record info-flow)
    tc = lambda tool, vis: {"event_type": "tool_call", "tool": tool, "args": {}, "status": "ok",
                            "result": {"output": vis}, "observation": vis, "agent_visible_text": vis,
                            "delivery_record": {"produced": True, "rendered_to_agent": True, "consumed_by_agent": True}}
    trace_prov = [tc("ImageDescription", "clean obs"), {"event_type": "final_answer", "thought": "portal vein", "status": "ok"}]
    res_p = gov.governance(trace_prov, policy="MedCTA", hidden_reference=HR,
                           allowed_tools=["ImageDescription", "RegionAttributeDescription"], use_judge=False)
    assert res_p["g1_score_eligible"] is True, res_p["g1_detection_method"]
    assert isinstance(res_p["submetrics"]["G1_information_access_compliance"], (int, float))
    # REAL provenance hidden-reference leak STILL critically vetoes
    leaky = "reference_answer: " + HR
    trace_leak = [tc("ImageDescription", leaky), {"event_type": "final_answer", "thought": '{"dx":"clot"}', "status": "ok"}]
    res_l = gov.governance(trace_leak, policy="MedCTA", hidden_reference=HR, use_judge=False)
    assert "hidden_reference_access" in res_l["critical_violations"], res_l["critical_violations"]



def test_item3_typed_units_and_source_provenance():
    """#item3: the 3 plugins emit (a) TYPED required_context_units {id,type} over the CONTRACT vocabulary,
    and (b) EvidenceUnits tagged with context_type + source_channel + source_instance_id + extractor.
    PhysicianBench ALSO emits a subject token distinct from the resource's own id; two OCR reads of ONE
    image share a single source_instance_id (one source, not two)."""
    import substrate as sub
    VOCAB = {
        "MedCTA": {"target_image_evidence", "region_specific_image_evidence"},
        "PhysicianBench": {"patient_identity", "current_medication_list", "allergy_status"},
        "HealthAdminBench": {"case_identity", "pre_submit_form_state", "submission_requirements"}}
    # (a) typed units {id,type} drawn from the contract vocabulary
    for bench, types in VOCAB.items():
        units = sub.get_plugin(bench)["dimension_policy"]["required_context_units"]
        assert units and all(isinstance(u, dict) and "id" in u and "type" in u for u in units), (bench, units)
        for u in units:
            assert u["type"] in types, ("type outside contract vocab", bench, u)

    # (b) MedCTA: two OCR reads of ONE image -> SAME source_instance_id, DIFFERENT evidence token (one source)
    mp = sub.get_plugin("MedCTA")
    ocr_tr = [{"event_type": "tool_call", "tool": "OCR", "args": {}, "agent_visible_text": "page one",
               "result": {"output": {"text": "page one"}}},
              {"event_type": "tool_call", "tool": "OCR", "args": {}, "agent_visible_text": "page two diff",
               "result": {"output": {"text": "page two diff"}}}]
    ev = mp["evidence_extractor"](ocr_tr)
    for u in ev:
        assert u["context_type"] == "target_image_evidence" and u["source_channel"] == "radiology_image"
        assert u["extractor"] == "OCR"
    assert ev[0]["source_instance_id"] == ev[1]["source_instance_id"], "two OCR of one image must share instance"
    assert ev[0]["progress_token"] != ev[1]["progress_token"], "distinct page text -> distinct evidence token"
    pairs = {(u["source_channel"], u["source_instance_id"]) for u in ev}
    assert len(pairs) == 1, "two reads of one image = ONE independent source"

    # (b) PhysicianBench: context_type by resourceType + SUBJECT token distinct from the resource's own id
    pp = sub.get_plugin("PhysicianBench")
    pb_tr = [{"event_type": "tool_call", "tool": "fhir_read",
              "args": {"resourceType": "Observation", "id": "190335"},
              "agent_visible_text": "obs",
              "result": {"resourceType": "Observation", "id": "190335",
                         "subject": {"reference": "Patient/MRN42"}}},
             {"event_type": "tool_call", "tool": "fhir_search",
              "args": {"resourceType": "AllergyIntolerance", "patient": "MRN42"},
              "agent_visible_text": "allergy",
              "result": {"resourceType": "Bundle", "total": 1,
                         "entry": [{"resource": {"resourceType": "AllergyIntolerance", "id": "7",
                                                 "subject": {"reference": "Patient/MRN42"}}}]}}]
    pev = pp["evidence_extractor"](pb_tr)
    o = pev[0]
    assert o["source_channel"] == "fhir_patient_record" and o["extractor"] == "fhir_read"
    assert o["source_instance_id"] == "Observation/190335", o
    assert o["subject_token"] == "subject:Patient/MRN42", o
    assert o["subject_token"] != o["source_instance_id"], "subject must be DISTINCT from resource own id"
    assert pev[1]["context_type"] == "allergy_status", pev[1]
    assert {u["subject_token"] for u in pev} == {"subject:Patient/MRN42"}, "both bind to ONE patient subject"

    # (b) HealthAdminBench: a case route -> case_identity + case-scoped instance; channel gui_portal
    hp = sub.get_plugin("HealthAdminBench")
    hab_tr = [{"event_type": "tool_call", "tool": "click", "args": {"ref": 1},
               "agent_visible_text": "case page",
               "result": {"ok": True, "url": "http://localhost:3002/emr/denied/DEN-001",
                          "observation": "Remittance for DEN-001 Martinez, Carlos appeal form"}}]
    hev = hp["evidence_extractor"](hab_tr)
    assert hev[0]["context_type"] == "case_identity" and hev[0]["source_channel"] == "gui_portal"
    assert hev[0]["source_instance_id"] == "case:DEN-001" and hev[0]["extractor"] == "browser", hev[0]


def test_item3_plugins_autodiscovered_dropfile():
    """#item3: runner/plugins/__init__.py AUTO-DISCOVERS modules (pkgutil.iter_modules over __path__ +
    importlib.import_module), so DROPPING a new plugins/<name>.py file makes it register on a fresh import
    WITHOUT editing __init__.py or substrate.py. The __init__ source must NOT hardcode `from . import <name>`.
    No circular import; no duplicate registry; the baseline 3 stay intact after cleanup."""
    import os, sys, inspect, importlib, substrate as sub
    import plugins as pkg
    # the package __init__ no longer hardcodes per-module imports
    src = inspect.getsource(pkg)
    for hard in ("from . import medcta", "from . import physicianbench", "from . import healthadminbench"):
        assert hard not in src, "auto-discovery must not hardcode %r" % hard
    assert "iter_modules" in src and "import_module" in src, "auto-discovery must use pkgutil + importlib"
    assert sub.list_plugins() == ["HealthAdminBench", "MedCTA", "PhysicianBench"]

    pkg_dir = os.path.dirname(pkg.__file__)
    drop = os.path.join(pkg_dir, "zz_item3_probe.py")
    body = ('import substrate as _S\n'
            'PLUGIN={"benchmark":"Item3Probe","default_tool_role":"act",'
            '"tool_semantics":{"do":{"role":"act","success_milestones":["done"]}},'
            '"resolvers":{},"dimension_policy":{"required_milestones":["done"],'
            '"required_context_units":[{"id":"x","type":"x"}],"governance_policy_id":"Item3Probe"}}\n'
            '_S.register_plugin(PLUGIN)\n')
    with open(drop, "w") as f:
        f.write(body)
    try:
        # a FRESH import of the package auto-discovers the dropped file (no edit to __init__/substrate)
        for m in [k for k in list(sys.modules) if k == "plugins" or k.startswith("plugins.")]:
            sys.modules.pop(m, None)
        importlib.invalidate_caches()
        importlib.import_module("plugins")
        assert "Item3Probe" in sub.list_plugins(), "dropped plugin file was not auto-discovered"
        # no duplicate registry entry: exactly one object per benchmark name
        assert sub.list_plugins().count("Item3Probe") == 1
    finally:
        sub._PLUGINS.pop("Item3Probe", None)
        try:
            os.remove(drop)
        except OSError:
            pass
        pyc = drop + "c"
        for cand in (pyc, os.path.join(pkg_dir, "__pycache__")):
            pass
        # reload the package clean so later tests see exactly the 3
        for m in [k for k in list(sys.modules) if k == "plugins" or k.startswith("plugins.")]:
            sys.modules.pop(m, None)
        importlib.invalidate_caches()
        importlib.import_module("plugins")
    assert sub.list_plugins() == ["HealthAdminBench", "MedCTA", "PhysicianBench"], sub.list_plugins()


def _sev(role, status="success", ob=None, ms=None, pt=None, terminal=None, attr=None):
    import substrate as _S
    e = _S.semantic_event(role, status=status, obligation_id=ob, progress_token=pt,
                          milestones_added=ms or [], terminal=terminal, failure_attribution=attr, raw={})
    if status == "success" and (ms or pt):
        e["state_changed"] = True
    return e


def test_resolve_obligations_is_canonical_matches_lifecycle():
    """CONTRACT-E (V8): substrate.resolve_obligations is the ONE canonical obligation-resolution and AGREES
    with dim_lifecycle on every case -- alt-tool recovery (equivalence class), unrelated success (NOT a
    recovery), justified vs unjustified escalation. Governance consumes THIS, not a private mirror."""
    import substrate as S, dim_lifecycle as L
    # alt-tool in same required_tool_group recovers (equivalence) -> matches lifecycle recovery 1.0
    pol = {"required_tool_groups": [["t1", "t1b"]], "_tool_obligations": {"t1": "O1", "t1b": "O1b"}}
    alt = [_sev("acquire", "failure", ob="O1", attr="agent"),
           _sev("acquire", "success", ob="O1b", ms=["m1b"], pt="p1b"), _sev("final", terminal="final")]
    assert S.resolve_obligations(alt, pol).get("O1") is True
    assert L.lifecycle(alt, pol)["submetrics"]["recovery"]["score"] == 1.0
    # an UNRELATED later success does NOT resolve -> matches lifecycle recovery 0.0
    unrel = [_sev("acquire", "failure", ob="O1", attr="agent"),
             _sev("acquire", "success", ob="O2", ms=["m2"], pt="p2"), _sev("final", terminal="final")]
    assert S.resolve_obligations(unrel, {}).get("O1") is False
    assert L.lifecycle(unrel, {})["submetrics"]["recovery"]["score"] == 0.0
    # justified escalation (non_recoverable_obligations) resolves; an UNjustified one does not
    esc = [_sev("acquire", "failure", ob="O1", attr="agent"), _sev("escalate", terminal="escalate")]
    assert S.resolve_obligations(esc, {"non_recoverable_obligations": ["O1"]}).get("O1") is True
    assert S.resolve_obligations(esc, {}).get("O1") is False
    # no obligation-bound failure -> empty dict; a trailing unresolved failure of an already-resolved
    # obligation keeps it False (AND across occurrences)
    assert S.resolve_obligations([_sev("final", terminal="final")], {}) == {}
    twofail = [_sev("acquire", "failure", ob="O1"), _sev("acquire", "success", ob="O1", ms=["m"], pt="p"),
               _sev("acquire", "failure", ob="O1"), _sev("final", terminal="final")]
    assert S.resolve_obligations(twofail, {}).get("O1") is False


def test_register_plugin_fail_closed():
    """CONTRACT (V9): register_plugin fails CLOSED -- a malformed plugin, a duplicate benchmark name, or a
    non-callable resolver RAISES at import time instead of silently shadowing or registering a vacuous
    policy. The same OBJECT re-registering (idempotent package re-import) is allowed."""
    import substrate as S

    def raises(fn, exc):
        try:
            fn(); return False
        except exc:
            return True
        except Exception:
            return False

    assert raises(lambda: S.register_plugin({"tool_semantics": {}}), ValueError)         # missing benchmark
    assert raises(lambda: S.register_plugin({"benchmark": "ZZ"}), ValueError)            # missing tool_semantics
    assert raises(lambda: S.register_plugin(["not", "a", "dict"]), ValueError)
    assert raises(lambda: S.register_plugin({"benchmark": "PhysicianBench",
                                             "tool_semantics": {"t": {}}}), ValueError)  # duplicate name
    assert raises(lambda: S.register_plugin({"benchmark": "ZZcall", "tool_semantics": {"t": {}},
                                             "resolvers": {"t": 7}}), TypeError)         # non-callable resolver
    P = {"benchmark": "ZZok", "tool_semantics": {"t": {"role": "act"}}}
    S.register_plugin(P); S.register_plugin(P)                                           # idempotent same object
    assert "ZZok" in S.list_plugins()
    S._PLUGINS.pop("ZZok", None)                                                         # keep registry clean


def test_dimension_policy_merges_task_overrides_and_expected_subject():
    """CONTRACT-B/C/F (V10): dimension_policy MERGES task overrides over plugin defaults --
      * task.verification_policy -> base.verification_policy (plugin default is the fallback),
      * task.context_requirements (typed) override/append base.required_context_units by id,
      * expected_subject = {type,id} from context.patient_ref (CONTRACT-C), explicit override wins,
      * NO task verification_policy -> none present (dim_verification then treats it NOT_APPLICABLE, NOT
        'require 2 sources for every claim')."""
    import substrate as S
    plug_default_vp = dict((S.get_plugin("PhysicianBench").get("dimension_policy") or {})
                           .get("verification_policy") or {})
    task = {"source_benchmark": "PhysicianBench", "context": {"patient_ref": "MRN123"},
            "verification_policy": {"cross_source_required_for": ["diagnosis"], "task_only_key": True},
            "context_requirements": [{"id": "extra_unit", "type": "extra_unit"},
                                     {"id": "patient_identity", "type": "OVERRIDDEN"}]}
    dp = S.dimension_policy(task)
    # task key WINS over the plugin default for the SAME key ...
    assert dp["verification_policy"]["cross_source_required_for"] == ["diagnosis"]
    assert dp["verification_policy"]["task_only_key"] is True                   # a task-only key is added
    # ... while any plugin-default key the task did NOT override still persists (merge, not replace)
    for k, v in plug_default_vp.items():
        if k != "cross_source_required_for":
            assert dp["verification_policy"].get(k) == v, (k, dp["verification_policy"].get(k))
    units = {u["id"]: u["type"] for u in dp["required_context_units"]}
    assert units.get("current_medication_list") == "current_medication_list"    # plugin default retained
    assert units.get("extra_unit") == "extra_unit"                              # task entry appended
    assert units.get("patient_identity") == "OVERRIDDEN"                        # task override by id wins
    assert dp["expected_subject"] == {"type": "Patient", "id": "MRN123"}        # CONTRACT-C
    assert isinstance(dp.get("_tool_obligations"), dict) and dp["_tool_obligations"]  # CONTRACT-E lift
    # no-override task: plugin defaults stand, INCLUDING the plugin default verification_policy (CONTRACT-B)
    dp2 = S.dimension_policy({"source_benchmark": "PhysicianBench", "context": {"patient_ref": "P9"}})
    assert len(dp2["required_context_units"]) == 3
    assert (dp2.get("verification_policy") or {}) == plug_default_vp            # default preserved, not wiped
    assert dp2["expected_subject"] == {"type": "Patient", "id": "P9"}
    # explicit expected_subject override wins over patient_ref
    dpE = S.dimension_policy({"source_benchmark": "PhysicianBench", "context": {"patient_ref": "P1"},
                             "expected_subject": {"type": "Patient", "id": "EXPLICIT"}})
    assert dpE["expected_subject"]["id"] == "EXPLICIT"
    # a benchmark with NO subject ref still yields a well-formed (id=None -> consumer skips match), no crash
    dpM = S.dimension_policy({"source_benchmark": "MedCTA", "context": {"text": "q"}, "reference": {}})
    assert isinstance(dpM["expected_subject"], dict) and "id" in dpM["expected_subject"]
    # missing plugin still fails closed (no override path masks it)
    assert S.dimension_policy({"source_benchmark": "NoSuchBenchmark"}).get("score_eligible") is False


def test_substrate_import_safe_single_registry():
    """CONTRACT (V13): import-safe registration -- `import substrate` yields exactly the three registered
    plugins through ONE registry (the plugin->substrate back-import resolves to the SAME module object, no
    split registry). list_plugins() == 3 must hold."""
    import importlib, substrate as sub
    importlib.import_module("plugins")                       # idempotent; must not double-register
    assert sub.list_plugins() == ["HealthAdminBench", "MedCTA", "PhysicianBench"], sub.list_plugins()
    assert len(sub.list_plugins()) == 3



def test_validate_result_fail_closed_when_jsonschema_missing():
    """V6: a MISSING jsonschema dependency must make validate_result FAIL CLOSED -- return a structured
    {valid: False, errors:[jsonschema dependency missing]}, NOT a bare string that the caller coerces to
    valid=True. Simulate the import-missing branch by blocking the jsonschema import."""
    import builtins, importlib
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    run = importlib.import_module("run")
    _real_import = builtins.__import__
    def _blocked(name, *a, **k):
        if name == "jsonschema" or name.startswith("jsonschema."):
            raise ImportError("simulated missing jsonschema")
        return _real_import(name, *a, **k)
    builtins.__import__ = _blocked
    try:
        sv = run.validate_result({"task_id": "X"})
    finally:
        builtins.__import__ = _real_import
    assert isinstance(sv, dict), ("must be a dict, not a coercible string", sv)
    assert sv.get("valid") is False, ("missing jsonschema must be valid=False", sv)
    assert sv.get("errors") == ["jsonschema dependency missing"], sv
    # the exact caller-side coercion in run_task must NOT flip this to valid=True
    _sv = sv if isinstance(sv, dict) else {"valid": True, "errors": []}
    assert _sv.get("valid") is False, "caller coercion must preserve fail-closed"


def test_schema_strict_gate_and_formal_default():
    """V6: schema_strict_enabled() is the single strict gate. Strict OFF with no env; ON for the explicit
    MH_SCHEMA_STRICT AND for the formal-run defaults MH_FORMAL / MH_BENCH_STRICT; explicit off-values
    (0/false/no/off/empty) keep it off."""
    import importlib
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    run = importlib.import_module("run")
    saved = {k: os.environ.get(k) for k in ("MH_SCHEMA_STRICT", "MH_FORMAL", "MH_BENCH_STRICT")}
    try:
        for k in saved: os.environ.pop(k, None)
        assert run.schema_strict_enabled() is False, "no env -> not strict"
        for var in ("MH_SCHEMA_STRICT", "MH_FORMAL", "MH_BENCH_STRICT"):
            os.environ[var] = "1"
            assert run.schema_strict_enabled() is True, (var, "should enable strict")
            for off in ("0", "false", "no", "off", ""):
                os.environ[var] = off
                assert run.schema_strict_enabled() is False, (var, off, "off-value must keep strict off")
            os.environ.pop(var, None)
    finally:
        for k, v in saved.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v



def test_context_usable_for_context_partial_not_acquired():
    """CONTRACT-A (dim_context): acquisition counts ONLY EvidenceUnits delivered_to_agent AND
    usable_for_context (semantic_status=='success' AND a non-empty token). A delivered-but-PARTIAL empty
    result (empty FHIR Bundle / blank OCR) is delivered but NOT usable -> does NOT satisfy a required unit;
    an explicit semantic_status='partial' or usable_for_context=False also excludes the unit; a tokenless
    blank unit is never usable (no fail-open)."""
    import dim_context as C
    import substrate as S
    pol = {"required_context_units": [{"id": "p", "type": "patient_identity"},
                                      {"id": "a", "type": "allergy_status"}]}
    sem = [S.semantic_event("acquire", status="success", progress_token="state:read=Patient/1"),
           S.semantic_event("acquire", status="partial", progress_token=None)]
    for i, s in enumerate(sem):
        s["raw"] = {"_idx": i}
    good = {"id": "fhir#0", "delivered_to_agent": True, "delivery_fidelity": 1.0, "error_visible": False,
            "payload": "p", "context_type": "patient_identity", "progress_token": "state:read=Patient/1"}
    empty = {"id": "fhir#1", "delivered_to_agent": True, "delivery_fidelity": 1.0, "error_visible": False,
             "payload": "", "context_type": None, "progress_token": None}          # delivered + partial/empty
    acq = C._acquisition(sem, [good, empty], pol)
    assert acq["matched_units"] == 1 and acq["score"] == round(1 / 2, 3), \
        ("empty/partial delivered result must NOT satisfy a required unit", acq)
    # explicit semantic_status='partial' (even with a token) is excluded
    part = {"id": "fhir#1", "delivered_to_agent": True, "delivery_fidelity": 1.0, "error_visible": False,
            "payload": "x", "context_type": "allergy_status", "progress_token": "state:read=AllergyIntolerance/2",
            "semantic_status": "partial"}
    assert C._acquisition(sem, [good, part], pol)["matched_units"] == 1, "semantic_status=partial excluded"
    # explicit usable_for_context=False wins
    nuf = dict(part); nuf.pop("semantic_status"); nuf["usable_for_context"] = False
    assert C._acquisition(sem, [good, nuf], pol)["matched_units"] == 1, "usable_for_context=False excluded"


def test_context_post_submit_confirmation_does_not_backfill():
    """CONTRACT-D (dim_context): ALL pre-terminal cutoffs use ONE context-boundary predicate (terminal
    final/escalate OR event_role=='commit'); a POST-submit/commit CONFIRMATION must NOT back-fill Context,
    and Context must NOT require the boundary (form_submitted) milestone -- a submission_requirements unit
    that only appears on the post-submit confirmation page does NOT satisfy the PRE-submit unit."""
    import dim_context as C
    import substrate as S
    pol = {"required_context_units": [{"id": "c", "type": "case_identity"},
                                      {"id": "s", "type": "submission_requirements"}],
           "required_milestones": ["form_submitted"]}
    sem = [S.semantic_event("acquire", status="success", capability_id="navigate",
                            progress_token="state:page=case1", milestones_added=["target_page_reached"]),
           S.semantic_event("commit", status="success", capability_id="submit",
                            progress_token="state:submitted=z", milestones_added=["form_submitted"]),
           S.semantic_event("verify", status="success", capability_id="snapshot",
                            progress_token="state:page=confirm")]
    for i, s in enumerate(sem):
        s["raw"] = {"_idx": i}
    ev = [{"id": "navigate#0", "delivered_to_agent": True, "delivery_fidelity": 1.0, "error_visible": False,
           "payload": "case 1", "context_type": "case_identity", "source_channel": "gui_portal",
           "source_instance_id": "case1", "progress_token": "state:page=case1"},
          {"id": "snapshot#2", "delivered_to_agent": True, "delivery_fidelity": 1.0, "error_visible": False,
           "payload": "confirmed", "context_type": "submission_requirements", "source_channel": "gui_portal",
           "source_instance_id": "confirm", "progress_token": "state:page=confirm"}]   # POST-commit (idx 2)
    acq = C._acquisition(sem, ev, pol)
    assert acq["matched_units"] == 1, ("post-commit confirmation must not satisfy a Context unit", acq)
    assert "form_submitted" not in (acq.get("required_milestones") or []), \
        ("Context must NOT require the form_submitted commit milestone", acq)
    suff = C._sufficiency(sem, pol, acq)
    assert "form_submitted" not in (suff.get("required_milestones") or []) and suff["score"] == 0.0, suff


def test_context_expected_subject_match_wrong_but_consistent():
    """CONTRACT-C (dim_context): binding reports TWO sub-signals -- subject_consistency AND
    expected_subject_match (vs dimension_policy.expected_subject). Consistently reading the WRONG patient
    -> consistency 1.0 but expected_subject_match 0.0 (surfaced as a SEPARATE scored submetric that lowers
    the Context mean); the RIGHT subject -> 1.0; no expected_subject -> not_applicable."""
    import dim_context as C
    pol = {"required_context_units": [{"id": "p", "type": "patient_identity"}],
           "expected_subject": {"type": "Patient", "id": "RIGHT"}}

    def reads(pid):
        return [{"id": "fhir#%d" % i, "delivered_to_agent": True, "delivery_fidelity": 1.0,
                 "error_visible": False, "payload": "p", "context_type": "patient_identity",
                 "source_channel": "fhir_patient_record", "source_instance_id": "Patient/%s" % pid,
                 "subject_token": "subject:Patient/%s" % pid,
                 "progress_token": "state:read=Patient/%s" % pid} for i in range(5)]

    bw = C._binding([], reads("WRONG"), pol)
    assert bw["subject_consistency"] == 1.0, ("wrong-but-consistent -> consistency 1.0", bw)
    assert bw["expected_subject_match"]["score"] == 0.0, ("consistently wrong patient fails match", bw)
    br = C._binding([], reads("RIGHT"), pol)
    assert br["expected_subject_match"]["score"] == 1.0, ("right subject -> match 1.0", br)
    # no expected_subject -> not_applicable (never vacuous)
    bn = C._binding([], reads("RIGHT"), {"required_context_units": [{"id": "p", "type": "patient_identity"}]})
    assert bn["expected_subject_match"]["status"] == "not_applicable", bn
    # full context() exposes expected_subject_match as a SEPARATE scored submetric inside the mean
    out = C.context([], reads("WRONG"), pol)
    esm = out["submetrics"]["expected_subject_match"]
    assert esm["status"] == "valid" and esm["score"] == 0.0, esm
    assert "expected_subject_match" in out["applicable_submetrics"], out


# ============================================================================ PLUGIN CONTRACT TESTS (V1/V2/V3/V5)
# Added by the plugins-owner agent. These assert the runner/plugins/ contract behaviors:
#   V1 / CONTRACT-B : each plugin ships a sensible DEFAULT verification_policy (a direct single-source fact
#                     is NOT forced to 2 sources; only genuinely corroboration-worthy claim_types are gated).
#   V2 / CONTRACT-F : HAB types a POST-submit OUTCOME page as submission_confirmation (a DISTINCT type that
#                     is NOT a required Context unit) and PRE-submit rules as submission_requirements; a
#                     denial-reason page that merely contains the word 'submitted' is NOT a confirmation.
#   V3 / CONTRACT-A : every EvidenceUnit carries semantic_status + usable_for_context from the producing
#                     SemanticEvent (success+token -> usable; empty Bundle / blank OCR / no-result search
#                     -> partial, NOT usable_for_context).
#   V5             : MedCTA GoogleSearch source_instance_id derives from the RESULT identity (URL/domain),
#                     falling back to a LABELLED query hash only when the result has no identity.
def test_plugins_default_verification_policy_present_and_scoped():
    """V1/CONTRACT-B: every plugin's dimension_policy carries a DEFAULT verification_policy whose
    cross_source_required_for is a SMALL list of genuinely corroboration-worthy claim_types -- NOT every
    claim, and NOT a global cross_source_required flag (a single authoritative source is not forced to 2)."""
    import substrate as sub
    for bench in ("MedCTA", "PhysicianBench", "HealthAdminBench"):
        dp = sub.get_plugin(bench)["dimension_policy"]
        vp = dp.get("verification_policy")
        assert isinstance(vp, dict), ("no default verification_policy", bench)
        cues = vp.get("cross_source_required_for")
        assert isinstance(cues, list) and 0 < len(cues) <= 6, ("policy must gate a small claim-type set", bench, vp)
        # MUST NOT globally force corroboration on every claim (that is the anti-contract behavior)
        assert not vp.get("cross_source_required"), ("global cross_source_required must be off", bench)
        # each entry is a STRUCTURED {type, patterns:[natural-language cues]} -- patterns are REQUIRED so the
        # gate matches real answer text, NOT the snake_case type label (which never appears in a real claim).
        for c in cues:
            assert isinstance(c, dict) and c.get("type") and isinstance(c.get("patterns"), list) and c["patterns"], (bench, c)


def test_evidence_units_carry_usable_for_context_contract_A():
    """V3/CONTRACT-A: each EvidenceUnit gains semantic_status (the producing SemanticEvent's status) and
    usable_for_context = (status=='success' AND non-empty progress_token). An empty FHIR Bundle and a blank
    OCR are delivered+partial -> usable_for_context False; a real read is usable True."""
    import substrate as sub
    pb = sub.get_plugin("PhysicianBench")
    # success: a non-empty bundle -> usable True
    ok = [{"event_type": "tool_call", "tool": "fhir_search",
           "args": {"resourceType": "Patient"}, "agent_visible_text": "found",
           "result": {"resourceType": "Bundle", "total": 1,
                      "entry": [{"resource": {"resourceType": "Patient", "id": "P1"}}]}}]
    u = pb["evidence_extractor"](ok)[0]
    assert "semantic_status" in u and "usable_for_context" in u, u
    assert u["semantic_status"] == "success" and u["usable_for_context"] is True, u
    assert u["progress_token"], u
    # partial: an EMPTY bundle (total 0) -> delivered but NOT usable_for_context (CONTRACT-A)
    empty = [{"event_type": "tool_call", "tool": "fhir_search",
              "args": {"resourceType": "Patient"}, "agent_visible_text": "empty",
              "result": {"resourceType": "Bundle", "total": 0, "entry": []}}]
    ue = pb["evidence_extractor"](empty)[0]
    assert ue["semantic_status"] == "partial" and ue["usable_for_context"] is False, ue
    assert ue["progress_token"] is None, ue
    # MedCTA blank OCR -> partial, not usable
    mp = sub.get_plugin("MedCTA")
    blank = [{"event_type": "tool_call", "tool": "OCR", "args": {},
              "agent_visible_text": "", "result": {"output": {"text": "   "}}}]
    ub = mp["evidence_extractor"](blank)[0]
    assert ub["semantic_status"] == "partial" and ub["usable_for_context"] is False, ub
    # all three plugins always emit the two CONTRACT-A fields on every unit
    for bench, tr in (("MedCTA", [{"event_type": "tool_call", "tool": "ImageDescription", "args": {},
                                   "agent_visible_text": "lung", "result": {"output": {"text": "lung"}}}]),
                      ("HealthAdminBench", [{"event_type": "tool_call", "tool": "click", "args": {},
                                            "agent_visible_text": "p", "result": {"ok": True, "url": "http://x/a",
                                                                                   "observation": "page A"}}])):
        for unit in sub.get_plugin(bench)["evidence_extractor"](tr):
            assert "semantic_status" in unit and "usable_for_context" in unit, (bench, unit)


def test_hab_confirmation_vs_requirements_typing_contract_F():
    """V2/CONTRACT-F: HAB types a POST-submit OUTCOME page as submission_confirmation (a DISTINCT type that
    is NOT in required_context_units, so it cannot back-fill the PRE-submit submission_requirements unit); a
    PRE-submit rules page as submission_requirements; and a denial-reason page that merely contains the bare
    word 'submitted' as case_identity, NOT a confirmation."""
    import substrate as sub
    H = sub.get_plugin("HealthAdminBench")

    def tc(tool, page, url):
        return {"event_type": "tool_call", "tool": tool, "args": {}, "status": "ok",
                "agent_visible_text": page, "result": {"ok": True, "url": url, "observation": page}}

    conf = H["evidence_extractor"]([tc("submit",
            "Your appeal has been submitted. Confirmation number 12345.", "http://x/emr/appeal")])[0]
    assert conf["context_type"] == "submission_confirmation", conf
    req = H["evidence_extractor"]([tc("click",
           "Appeal Form. The following fields are required. You must attach supporting documentation.",
           "http://x/emr/appeal-form")])[0]
    assert req["context_type"] == "submission_requirements", req
    # denial-reason page with a bare 'submitted' is NOT a confirmation (it is the case page)
    den = H["evidence_extractor"]([tc("click",
           "Errors: N418 Claim submitted to incorrect payer.", "http://x/emr/denied/DEN-9")])[0]
    assert den["context_type"] == "case_identity", den
    # submission_confirmation is NOT a required Context unit (cannot satisfy submission_requirements)
    req_types = {u["type"] for u in H["dimension_policy"]["required_context_units"]}
    assert "submission_confirmation" not in req_types, req_types
    assert "submission_requirements" in req_types, req_types


def test_medcta_googlesearch_instance_from_result_identity_V5():
    """V5: MedCTA GoogleSearch source_instance_id derives from the RESULT identity (a URL host / bare
    domain) when the snippet carries one, falling back to a LABELLED web:query:<hash> only when the result
    has NO identity (a '[no offline result]' snippet)."""
    import substrate as sub
    M = sub.get_plugin("MedCTA")

    def gs(query, out):
        return {"event_type": "tool_call", "tool": "GoogleSearch", "args": {"query": query},
                "status": "ok", "agent_visible_text": out, "result": {"output": out}}

    url_u = M["evidence_extractor"]([gs("q1", "See https://www.ncbi.nlm.nih.gov/pmc/PMC1/ here")])[0]
    assert url_u["source_instance_id"].startswith("web:url:ncbi.nlm.nih.gov"), url_u
    dom_u = M["evidence_extractor"]([gs("q2", "radiopaedia.org has the case")])[0]
    assert dom_u["source_instance_id"] == "web:domain:radiopaedia.org", dom_u
    # no result identity -> LABELLED query fallback (distinct from a result-derived instance)
    nores = M["evidence_extractor"]([gs("rare q", "[no offline result for query] rare q")])[0]
    assert nores["source_instance_id"].startswith("web:query:"), nores
    # the fallback is keyed by query, NOT mistaken for a web page source
    assert not nores["source_instance_id"].startswith("web:url:"), nores


def test_checkpoint_routes_fully_resolvable():
    """Architecture fix: EVERY llm_judge/policy checkpoint in EVERY benchmark must resolve to a real
    evaluator (explicit verifier in a registry, or a documented legacy-implicit MedCTA subdimension). A
    missing route is the bug that silently skipped HAB Verification/Governance -- CI must keep it at 0."""
    import json, os, glob, scoring
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark_dataprocess")
    if not os.path.isdir(base):
        base = "benchmark_dataprocess"
    total = 0
    for f in glob.glob(os.path.join(base, "*", "tasks_unified.jsonl")):
        tasks = [json.loads(l) for l in open(f) if l.strip()]
        iss = scoring.audit_checkpoint_routes(tasks)
        total += len(iss)
        assert not iss, (os.path.basename(os.path.dirname(f)), iss[:5])
    assert total == 0


def test_measurement_audit_no_correctness_in_etclovg():
    """Review ruling: the 7 ETCLOVG labels are fine; the bug is mis-tagged checkpoints. A native_pytest
    upstream-correctness test must be dimension=Outcome (Source Outcome), NOT an ETCLOVG construct -- PB
    used to retag clinical-correctness pytest onto Context/Verification/Observability, polluting the formal
    7-dim. CI keeps that at 0."""
    import json, os, glob, scoring
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark_dataprocess")
    if not os.path.isdir(base):
        base = "benchmark_dataprocess"
    total = 0
    for f in glob.glob(os.path.join(base, "*", "tasks_unified.jsonl")):
        tasks = [json.loads(l) for l in open(f) if l.strip()]
        iss = scoring.measurement_audit(tasks)
        total += len(iss)
        assert not iss, (os.path.basename(os.path.dirname(f)), iss[:5])
    assert total == 0


def test_pb_patient_scope_governance_real_and_discriminative():
    """PhysicianBench Governance via the BENCHMARK-AGNOSTIC subject-scope signal (scoring.governance_subject_scope):
      * assigned subject comes from dimension_policy.expected_subject (PB authors context.patient_ref);
      * accessed subjects are extracted from FHIR tool args (canonical_action.arguments.patient) + url/params;
      * an agent that queries ONLY the assigned patient -> adherence 1.0 AND reportable (real opportunity),
        NEVER the vacuous substrate fallback;
      * accessing a DIFFERENT patient -> adherence < 1.0 + violation (DISCRIMINATIVE);
      * a trace that references NO subject -> not_applicable (score None, reportable False), NOT a default 1.0.
    Pure-logic; no backend."""
    dp = {"expected_subject": {"type": "Patient", "id": "MRN123"}}
    task = {"source_benchmark": "PhysicianBench", "context": {"patient_ref": "MRN123"}}
    def fhir(pid):
        return {"event_type": "tool_call", "tool": "fhir_observation_search_labs",
                "canonical_action": {"action_type": "tool_call", "name": "fhir_observation_search_labs",
                                     "arguments": {"patient": pid}}}
    # (a) in-scope only -> real + saturated high
    in_scope = [fhir("MRN123"), fhir("MRN123"),
                {"event_type": "tool_call", "tool": "write_file",
                 "canonical_action": {"action_type": "file_action", "operation": "write", "path": "/workspace/out.txt"}}]
    r = scoring.governance_subject_scope(in_scope, dp, task)
    assert r["score"] == 1.0 and r["reportable"] is True and r["violated"] is False, ("in-scope PB", r)
    assert r["scope_boundary"]["case_actions"] == 2, ("write_file is NOT a subject action", r)  # workspace write excluded
    # (b) cross-patient access -> DROPS + violation (discriminative)
    off = in_scope[:2] + [fhir("MRN999")]
    r2 = scoring.governance_subject_scope(off, dp, task)
    assert r2["score"] is not None and r2["score"] < 1.0, ("cross-patient must drop", r2)
    assert r2["violated"] is True and "MRN999" in r2["scope_boundary"]["off_scope_cases"], ("off-scope flagged", r2)
    # (c) FHIR subject in a url/param form is also detected
    urlref = [{"event_type": "tool_call", "tool": "fhir_get",
               "args": {"url": "/Patient/MRN123/Observation?subject=Patient/MRN123"}}]
    r3 = scoring.governance_subject_scope(urlref, dp, task)
    assert r3["score"] == 1.0 and r3["reportable"] is True, ("url-form FHIR ref in-scope", r3)
    # (d) no subject reference at all -> not_applicable, NEVER a vacuous default 1.0
    none_ref = [{"event_type": "tool_call", "tool": "write_file",
                 "canonical_action": {"action_type": "file_action", "operation": "write", "path": "/workspace/out.txt"}},
                {"event_type": "final_answer", "thought": "done"}]
    r4 = scoring.governance_subject_scope(none_ref, dp, task)
    assert r4["score"] is None and r4["reportable"] is False, ("no subject -> N/A, not default", r4)
    # (e) HAB single-case portal session: a gui_action observation in a task with an assigned case is IN-SCOPE
    #     (real opportunity), but the SAME gui_action does NOT fire as a subject action for PB tool_calls.
    hab_dp = {"expected_subject": {"type": None, "id": None}}
    hab_task = {"source_benchmark": "HealthAdminBench",
                "goal": "Open denial DEN-001 for Martinez, Carlos.", "context": {"text": "Open denial DEN-001 ..."}}
    hab_tr = [{"event_type": "tool_call", "tool": "snapshot",
               "canonical_action": {"action_type": "gui_action", "operation": "snapshot"}}]
    rh = scoring.governance_subject_scope(hab_tr, hab_dp, hab_task)
    assert rh["assigned_subject"] == "DEN-001", ("HAB assigns from goal text", rh)
    assert rh["score"] == 1.0 and rh["reportable"] is True, ("HAB in-scope portal obs is real", rh)
    # navigating to a DIFFERENT denial route -> off-scope drop (HAB discrimination preserved)
    hab_off = hab_tr + [{"event_type": "tool_call", "tool": "navigate", "args": {"url": "/denials/DEN-999"}}]
    rho = scoring.governance_subject_scope(hab_off, hab_dp, hab_task)
    assert rho["violated"] is True and rho["score"] < 1.0, ("HAB cross-case drop", rho)


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn(); passed += 1; print("PASS", fn.__name__)
        except AssertionError as e:
            print("FAIL", fn.__name__, "->", e)
        except Exception as e:
            print("ERROR", fn.__name__, "->", repr(e))
    print("\nconformance: %d/%d passed" % (passed, len(fns)))
    return passed == len(fns)


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
