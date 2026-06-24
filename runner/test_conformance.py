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
