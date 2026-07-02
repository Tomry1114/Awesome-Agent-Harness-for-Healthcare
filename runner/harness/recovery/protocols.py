"""Bounded Clinical Recovery v3 - the three layer protocols.

Four-layer separation (the anti-specialization rule):

  1. Recovery Kernel   - state machine / gap routing / auth / budget / provenance / metrics.
                         Depends ONLY on these Protocols (never on a concrete adapter).
  2. SubstrateAdapter  - HOW to observe/act in an environment (structured record read/create/update;
                         GUI snapshot/resolve/click/type/upload; perceptual describe/region/ocr).
                         Knows NOTHING about any clinical/admin process or benchmark field.
  3. WorkflowModule    - WHAT a clinical/admin process is (its steps, required semantic bindings,
                         affordance targets by role/label, postcondition, transaction contract).
                         Substrate-agnostic in intent; calls the substrate for primitives.
  4. BenchmarkAdapter  - normalizes a task's fields / lifecycle events / state-paths. Does NOT change
                         kernel or workflow rules.

These are typing.Protocol (structural) so stacks may implement them by duck-typing without importing a
base class; runtime_checkable is provided for defensive isinstance checks. Python 3.8 compatible.
"""
from typing import Any, Dict, List, Optional, Union, runtime_checkable, Protocol

from .contracts import (
    AffordanceBinding, Outcome, Plan, CommittedGoal,
)


# resolve_affordance returns EITHER a located AffordanceBinding OR a BLOCKED_* terminal string constant
# (contracts.BLOCKED_UNRESOLVED_AFFORDANCE / BLOCKED_AMBIGUOUS_TARGET).
AffordanceResult = Union[AffordanceBinding, str]


@runtime_checkable
class SubstrateAdapter(Protocol):
    """Environment mechanics only. No workflow, no benchmark fields."""

    def resolve_affordance(self, target_spec: Any, observation: Any) -> AffordanceResult:
        """Locate the control named by target_spec in the LIVE observation. Return an AffordanceBinding
        when uniquely located, else a BLOCKED_* terminal string (0 candidates / ordering violation ->
        BLOCKED_UNRESOLVED_AFFORDANCE; >1 candidate with no disambiguating bound id ->
        BLOCKED_AMBIGUOUS_TARGET)."""
        ...

    def execute_primitive(self, kind: str, action: Dict[str, Any],
                          auth: Optional[Any]) -> Outcome:
        """Execute one primitive (read/navigate/acquire/staged_write/irreversible_commit/verify). Read-like
        kinds receive auth=None (strict reader). Mutations receive a single-use MutationAuthorization."""
        ...

    def read_state(self, paths: List[Any]) -> Dict[str, Any]:
        """Authoritative read-back of the given concrete state paths -> a state_view dict."""
        ...

    def classify_result(self, result: Any) -> str:
        """Map a primitive's result/Outcome to an outcome-vocabulary string
        (contracts.RESULT_OK / RESULT_UNKNOWN / RESULT_FAILED / RESULT_ALREADY_REALIZED)."""
        ...


@runtime_checkable
class WorkflowModule(Protocol):
    """Process knowledge; substrate-agnostic. Declares steps, bindings, targets, postconditions."""

    def match_goal(self, goal: CommittedGoal, ctx: Dict[str, Any]) -> bool:
        """Does THIS workflow realize this committed goal?"""
        ...

    def required_bindings(self, goal: CommittedGoal, ctx: Dict[str, Any]) -> List[Any]:
        """The semantic args (BindingSpec / names) this workflow needs to complete the goal."""
        ...

    def compile_plan(self, goal: CommittedGoal, ctx: Dict[str, Any]) -> Plan:
        """Produce the bounded Plan (steps with kind/affordance_target/arg_specs, postcondition,
        optional transaction_contract). Must respect the <=1 irreversible_commit rule."""
        ...

    def verify_effect(self, goal: CommittedGoal, state_view: Dict[str, Any]) -> Optional[bool]:
        """True if the goal's postcondition is realized in state_view, False if refuted, None if the
        read-back is ambiguous (-> idempotent reconciliation / UNKNOWN)."""
        ...


@runtime_checkable
class BenchmarkAdapter(Protocol):
    """Task/field/lifecycle/state-path normalization. Does NOT change kernel or workflow rules."""

    def context(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a task into a ctx dict the kernel/workflow consume (may carry a binding 'schema',
        'observation', 'system_metadata', 'bound_evidence')."""
        ...

    def resolve_commitments(self, root: Any, trajectory: Any, goal: Any, judge: Any,
                            ctx: Dict[str, Any]) -> List[CommittedGoal]:
        """Oracle-blind: derive the goals the ROOT AGENT committed to (from its own output/trajectory),
        one CommittedGoal per independent episode. Empty list => no commitment (DECLINED)."""
        ...

    def should_trigger(self, lifecycle_event: Any) -> bool:
        """Whether recovery should engage on this lifecycle event (deliverable_confirmed / before_final)."""
        ...

    def state_path(self, logical_name: str) -> str:
        """Map a logical state name to the concrete environment state path for read-back."""
        ...
