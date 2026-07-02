"""Bounded Clinical Recovery (BCR) v3 - recovery kernel package.

A NEW, additive package. It does NOT touch the existing recovery_adapter / recovery_orchestrator /
run_driver / run.py path. Four-layer separation is mandatory and enforced:

    Recovery Kernel  (kernel.py)        - state machine, gap routing, auth, metrics; depends only on
                                          contracts/bindings/metrics + the injected protocols.
    Substrate Adapter (protocols.py)    - environment mechanics only.
    Workflow Module   (protocols.py)    - clinical/admin process knowledge.
    Benchmark Adapter (protocols.py)    - task/field/lifecycle/state-path normalization.

Kernel + Substrate + Workflow files are oracle-blind and carry NO benchmark proper names.
"""
from . import contracts
from . import bindings
from . import protocols
from . import metrics
from . import registry
from . import kernel

from .contracts import (
    # states
    NOT_STARTED, ACQUIRING, PLANNING, EXECUTING_STAGED, READY_TO_COMMIT, COMMITTING, VERIFYING, VERIFIED,
    AGENT_REENTRY, ACCEPTED, KEPT_ORIGINAL, ALREADY_REALIZED,
    BLOCKED_NEEDS_DECISION, BLOCKED_MISSING_EVIDENCE, BLOCKED_AMBIGUOUS_TARGET,
    BLOCKED_UNRESOLVED_AFFORDANCE, FAILED, UNKNOWN, NOT_APPLICABLE, DECLINED_NO_COMMITMENT,
    TERMINAL_STATES, BLOCKED_STATES, DECLINED_STATES,
    # paths
    PATH_EVIDENCE, PATH_EXECUTION, PATH_VERIFICATION, PATH_DECISION,
    # step kinds
    READ, NAVIGATE, ACQUIRE, STAGED_WRITE, IRREVERSIBLE_COMMIT, VERIFY, STEP_KINDS,
    # gap classes
    GAP_EVIDENCE, GAP_EXECUTION, GAP_VERIFICATION, GAP_DECISION,
    # auth tiers + result vocab
    AUTH_TIER_READ, AUTH_TIER_SCOPED, AUTH_TIER_IRREVERSIBLE,
    RESULT_OK, RESULT_UNKNOWN, RESULT_FAILED, RESULT_ALREADY_REALIZED,
    # data types
    ArgumentBinding, AffordanceBinding, ProtocolConstant, GapSignal,
    MutationAuthorization, Outcome, CommittedGoal, RecoveryStep, Plan, EpisodeResult,
    PlanCompileError, validate_plan,
)
from .bindings import (
    classify_field, is_semantic, resolve_argument, decision_boundary,
    SEMANTIC, OPERATIONAL, SEMANTIC_SOURCES, OPERATIONAL_SOURCES,
    AGENT_COMMITMENT, AUTHORITATIVE_STATE, BOUND_EVIDENCE, SYSTEM_METADATA,
)
from .protocols import SubstrateAdapter, WorkflowModule, BenchmarkAdapter, AffordanceResult
from .metrics import (
    classify as classify_metric, tally, is_eligible, is_engaged,
    ELIGIBLE, ENGAGED, VERIFIED_RECOVERY, CORRECTLY_BLOCKED, FAILED_RECOVERY, UNKNOWN_RECOVERY,
    OUTCOME_BUCKETS, ALL_BUCKETS,
)
from .registry import WorkflowRegistry, RecoveryStack, get_recovery_stack
from .kernel import RecoveryKernel, run_episode

__all__ = [
    "contracts", "bindings", "protocols", "metrics", "registry", "kernel",
    "RecoveryKernel", "run_episode",
    "SubstrateAdapter", "WorkflowModule", "BenchmarkAdapter",
    "WorkflowRegistry", "RecoveryStack", "get_recovery_stack",
    "ArgumentBinding", "AffordanceBinding", "ProtocolConstant", "GapSignal",
    "MutationAuthorization", "Outcome", "CommittedGoal", "RecoveryStep", "Plan", "EpisodeResult",
    "PlanCompileError", "validate_plan",
    "classify_field", "is_semantic", "resolve_argument", "decision_boundary",
    "classify_metric", "tally", "is_eligible", "is_engaged",
]

# --- v3 wiring exports (additive) ---
from .registry import build_registry  # noqa: E402
from . import entry  # noqa: E402
from .entry import run_recovery_v3  # noqa: E402
