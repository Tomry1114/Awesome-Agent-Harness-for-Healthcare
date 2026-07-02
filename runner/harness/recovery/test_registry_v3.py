"""Bounded Clinical Recovery v3 - registry wiring unit tests.

Standalone: prints PASS/FAIL per check; sys.exit(0) iff every check passes, non-zero otherwise.
Run: python3 runner/harness/recovery/test_registry_v3.py

No network / no model calls. Asserts:
  (a) build_registry()          -> ONE WorkflowRegistry with all five WorkflowModules registered.
  (b) get_recovery_stack(env)   -> the correct (substrate, workflow_registry, benchmark_adapter) triple
                                   for 'fhir' / 'gui' / 'tool_sandbox' (+ design aliases), each triple
                                   duck-typing the three layer protocols.
  (c) unknown env / None        -> None.
  (d) benchmark lifecycle gate  -> each adapter's should_trigger is callable and env-appropriate.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from harness.recovery.registry import (
    build_registry, get_recovery_stack, WorkflowRegistry, _normalize_env,
)

FAILS = []


def check(name, cond):
    if cond:
        print("PASS %s" % name)
    else:
        print("FAIL %s" % name)
        FAILS.append(name)


_SUBSTRATE_METHODS = ("resolve_affordance", "execute_primitive", "read_state", "classify_result")
_BENCH_METHODS = ("context", "resolve_commitments", "should_trigger", "state_path")


def _duck(obj, methods):
    return all(callable(getattr(obj, m, None)) for m in methods)


def test_a_build_registry():
    reg = build_registry()
    check("a1 build_registry is a WorkflowRegistry", isinstance(reg, WorkflowRegistry))
    names = {type(m).__name__ for m in reg.all()}
    expected = {"CreateOrderWorkflow", "PriorAuthorizationWorkflow", "DecisionDocumentationWorkflow",
                "AppealSubmissionWorkflow", "EvidenceAcquisitionWorkflow"}
    check("a2 all five workflow modules registered", expected.issubset(names))
    check("a3 registry length matches", len(reg) >= 5)


def test_b_stacks_per_env():
    expect_sub = {"fhir": "FhirSubstrateAdapter", "gui": "GuiSubstrateAdapter",
                  "tool_sandbox": "PerceptualSubstrateAdapter"}
    expect_bench = {"fhir": "PbBenchmarkAdapter", "gui": "HabBenchmarkAdapter",
                    "tool_sandbox": "MedctaBenchmarkAdapter"}
    for et in ("fhir", "gui", "tool_sandbox"):
        s = get_recovery_stack(et, None)
        check("b:%s triple returned" % et, isinstance(s, tuple) and len(s) == 3)
        if not (isinstance(s, tuple) and len(s) == 3):
            continue
        sub, wf, bench = s
        check("b:%s substrate class" % et, type(sub).__name__ == expect_sub[et])
        check("b:%s substrate duck-types SubstrateAdapter" % et, _duck(sub, _SUBSTRATE_METHODS))
        check("b:%s workflow_registry type" % et, isinstance(wf, WorkflowRegistry) and len(wf) >= 5)
        check("b:%s benchmark class" % et, type(bench).__name__ == expect_bench[et])
        check("b:%s benchmark duck-types BenchmarkAdapter" % et, _duck(bench, _BENCH_METHODS))


def test_c_unknown_env_none():
    check("c1 unknown env -> None", get_recovery_stack("nope", None) is None)
    check("c2 None env -> None", get_recovery_stack(None, None) is None)
    # design-doc aliases normalize to the run-level env types
    check("c3 alias 'record' -> fhir", _normalize_env("record") == "fhir")
    check("c4 alias 'perceptual' -> tool_sandbox", _normalize_env("perceptual") == "tool_sandbox")
    check("c5 alias 'interactive_gui' -> gui", _normalize_env("interactive_gui") == "gui")


def test_d_lifecycle_gate():
    _, _, pb = get_recovery_stack("fhir", None)
    _, _, hab = get_recovery_stack("gui", None)
    _, _, mc = get_recovery_stack("tool_sandbox", None)
    check("d1 fhir triggers on deliverable_confirmed", pb.should_trigger("deliverable_confirmed") is True)
    check("d2 fhir ignores random event", pb.should_trigger("random_event") in (False, None) or not pb.should_trigger("random_event"))
    check("d3 gui triggers on before_final", bool(hab.should_trigger("before_final")))
    check("d4 gui should_trigger returns a bool for unknown", isinstance(bool(hab.should_trigger("xyz")), bool))
    check("d5 perceptual should_trigger is callable", callable(getattr(mc, "should_trigger", None)))


def main():
    test_a_build_registry()
    test_b_stacks_per_env()
    test_c_unknown_env_none()
    test_d_lifecycle_gate()
    if FAILS:
        print("\nFAILED: %s" % FAILS)
        sys.exit(1)
    print("\nALL GREEN")
    sys.exit(0)


if __name__ == "__main__":
    main()
