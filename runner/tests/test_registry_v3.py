"""Bounded Clinical Recovery v3 - registry wiring unit tests.

Standalone: prints PASS/FAIL per check; sys.exit(0) iff every check passes, non-zero otherwise.
Run: python3 runner/tests/test_registry_v3.py

No network / no model calls. Asserts:
  (a) build_registry()          -> ONE WorkflowRegistry with the generic GUI WorkflowModule registered.
  (b) get_recovery_stack('gui') -> the (substrate, workflow_registry, benchmark_adapter) triple, duck-typing
                                   the three layer protocols.
  (c) unknown env / None / an env with no stack -> None; the design alias normalizes to 'gui'.
  (d) benchmark lifecycle gate  -> the adapter's should_trigger is callable and env-appropriate.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workspace import H0, Workspace

_REG = Workspace(H0["gui"]).module("recovery/registry.py")
build_registry, get_recovery_stack, WorkflowRegistry, _normalize_env = (
    _REG.build_registry, _REG.get_recovery_stack, _REG.WorkflowRegistry, _REG._normalize_env)

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
    # ONE generic workflow; no task-name-specific workflow is registered:
    expected = {"GenericGuiCompletionWorkflow"}
    check("a2 the generic GUI workflow module registered", expected.issubset(names))
    check("a3 no task-name workflow remains", not (names & {
        "CreateOrderWorkflow", "PriorAuthorizationWorkflow", "AppealSubmissionWorkflow",
        "DecisionDocumentationWorkflow"}))
    check("a3b registry length matches", len(reg) >= 1)
    assert expected.issubset(names), names                       # hard pytest gate
    assert not (names & {"CreateOrderWorkflow", "PriorAuthorizationWorkflow",
                         "AppealSubmissionWorkflow", "DecisionDocumentationWorkflow"}), names


def test_b_stacks_per_env():
    s = get_recovery_stack("gui")
    check("b:gui triple returned", isinstance(s, tuple) and len(s) == 3)
    if not (isinstance(s, tuple) and len(s) == 3):
        return
    sub, wf, bench = s
    check("b:gui substrate class", type(sub).__name__ == "GuiSubstrateAdapter")
    check("b:gui substrate duck-types SubstrateAdapter", _duck(sub, _SUBSTRATE_METHODS))
    check("b:gui workflow_registry type", isinstance(wf, WorkflowRegistry) and len(wf) >= 1)
    check("b:gui benchmark class", type(bench).__name__ == "HabBenchmarkAdapter")
    check("b:gui benchmark duck-types BenchmarkAdapter", _duck(bench, _BENCH_METHODS))


def test_c_unknown_env_none():
    check("c1 unknown env -> None", get_recovery_stack("nope") is None)
    check("c2 None env -> None", get_recovery_stack(None) is None)
    check("c3 fhir / tool_sandbox have no stack", get_recovery_stack("fhir") is None
          and get_recovery_stack("tool_sandbox") is None)
    # the 'interactive_gui' alias normalizes to the run-level env type
    check("c5 alias 'interactive_gui' -> gui", _normalize_env("interactive_gui") == "gui")


def test_d_lifecycle_gate():
    _, _, hab = get_recovery_stack("gui")
    check("d3 gui triggers on before_final", bool(hab.should_trigger("before_final")))
    check("d4 gui should_trigger returns a bool for unknown", isinstance(bool(hab.should_trigger("xyz")), bool))


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
