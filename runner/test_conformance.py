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
    """#4b: Verification splits into FIVE genuinely distinct, applicable-only sub-metrics. Proves on
    crafted inputs that (a) the expected sub-metrics exist, (b) each is applicable-only (no vacuous 1.0
    when there is no opportunity), and (c) no sub-metric is an algebraic transform of another (for every
    ordered pair there is a row where equal-B forces unequal-A -> not a function of B)."""
    import dim_verification as V
    import substrate as _S

    def ev(uid, payload, delivered=True, fid=1.0, err=False):
        return {"id": uid, "payload": payload, "delivered_to_agent": delivered,
                "delivery_fidelity": fid, "error_visible": err}

    def sc(evi, claims, vacts=None, conflicts=None, pol=None):
        o = V.verification(evi, vacts or [], claims, conflicts=conflicts, policy=pol or {})
        return o, {k: o["submetrics"][k]["score"] for k in V._SUBMETRICS}

    # exact expected sub-metric set
    assert set(V._SUBMETRICS) == {"evidence_support", "cross_source_check", "conflict_handling",
                                  "uncertainty_calibration", "verification_action_completion"}, V._SUBMETRICS

    # applicable-only: an empty run produces NO vacuous 1.0 -- every sub-metric is not_applicable.
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

    # ----- algebraic independence: no sub-metric is a function of another across these rows -----
    rows = [A, B, C, D, E, F, G_ack, G_no]
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
        return {"id": uid, "delivered_to_agent": True, "delivery_fidelity": 1.0,
                "error_visible": False, "payload": "p", "context_type": ct}

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
        return {"id": "u#%d" % id(ct), "delivered_to_agent": True, "delivery_fidelity": 1.0,
                "error_visible": False, "payload": "p", "context_type": ct}

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
        "HealthAdminBench": {"case_identity", "form_state", "submission_requirements"}}
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
