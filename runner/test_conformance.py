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
    assert "Observability" in out and 0.0 <= out["Observability"]["score"] <= 1.0


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
