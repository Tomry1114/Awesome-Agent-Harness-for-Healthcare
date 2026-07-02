"""Bounded Clinical Recovery v3 - contracts.

Frozen data types + string state constants shared by every layer. This module knows NOTHING about any
specific benchmark, environment mechanic, or clinical process: it only declares the vocabulary the Kernel,
the Substrate Adapter, the Workflow Modules and the Benchmark Adapters agree on. Substrate-agnostic and
oracle-blind by construction.

Python 3.8 compatible (dataclasses + typing only; no match statements).
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------------------------------
# STATES (string constants). Progress states, agent-re-entry states, and terminals.
# --------------------------------------------------------------------------------------------------
# Execution-path progress
NOT_STARTED = "NOT_STARTED"
ACQUIRING = "ACQUIRING"
PLANNING = "PLANNING"
EXECUTING_STAGED = "EXECUTING_STAGED"
READY_TO_COMMIT = "READY_TO_COMMIT"
COMMITTING = "COMMITTING"
VERIFYING = "VERIFYING"
VERIFIED = "VERIFIED"
# Evidence-path (acquire -> hand control back to the root agent -> non-regression acceptance)
AGENT_REENTRY = "AGENT_REENTRY"
ACCEPTED = "ACCEPTED"
KEPT_ORIGINAL = "KEPT_ORIGINAL"
# Effect already present before we acted
ALREADY_REALIZED = "ALREADY_REALIZED"
# Terminals (blocks = correct refusals)
BLOCKED_NEEDS_DECISION = "BLOCKED_NEEDS_DECISION"
BLOCKED_MISSING_EVIDENCE = "BLOCKED_MISSING_EVIDENCE"
BLOCKED_AMBIGUOUS_TARGET = "BLOCKED_AMBIGUOUS_TARGET"
BLOCKED_UNRESOLVED_AFFORDANCE = "BLOCKED_UNRESOLVED_AFFORDANCE"
FAILED = "FAILED"
UNKNOWN = "UNKNOWN"
# Declined / not-applicable (NEVER failures)
NOT_APPLICABLE = "NOT_APPLICABLE"
DECLINED_NO_COMMITMENT = "DECLINED_NO_COMMITMENT"

TERMINAL_STATES = frozenset({
    VERIFIED, ACCEPTED, KEPT_ORIGINAL, ALREADY_REALIZED,
    BLOCKED_NEEDS_DECISION, BLOCKED_MISSING_EVIDENCE, BLOCKED_AMBIGUOUS_TARGET,
    BLOCKED_UNRESOLVED_AFFORDANCE, FAILED, UNKNOWN, NOT_APPLICABLE, DECLINED_NO_COMMITMENT,
})
BLOCKED_STATES = frozenset({
    BLOCKED_NEEDS_DECISION, BLOCKED_MISSING_EVIDENCE, BLOCKED_AMBIGUOUS_TARGET,
    BLOCKED_UNRESOLVED_AFFORDANCE,
})
DECLINED_STATES = frozenset({NOT_APPLICABLE, DECLINED_NO_COMMITMENT})


# --------------------------------------------------------------------------------------------------
# Recovery paths (the Gap Router picks exactly one per episode)
# --------------------------------------------------------------------------------------------------
PATH_EVIDENCE = "evidence"
PATH_EXECUTION = "execution"
PATH_VERIFICATION = "verification"
PATH_DECISION = "decision"
RECOVERY_PATHS = frozenset({PATH_EVIDENCE, PATH_EXECUTION, PATH_VERIFICATION, PATH_DECISION})


# --------------------------------------------------------------------------------------------------
# Step kinds
# --------------------------------------------------------------------------------------------------
READ = "read"
NAVIGATE = "navigate"
ACQUIRE = "acquire"
STAGED_WRITE = "staged_write"
IRREVERSIBLE_COMMIT = "irreversible_commit"
VERIFY = "verify"
STEP_KINDS = frozenset({READ, NAVIGATE, ACQUIRE, STAGED_WRITE, IRREVERSIBLE_COMMIT, VERIFY})
# Read-like kinds mint NO mutation authorization (strict reader gate).
READ_LIKE_KINDS = frozenset({READ, NAVIGATE, ACQUIRE, VERIFY})
MUTATION_KINDS = frozenset({STAGED_WRITE, IRREVERSIBLE_COMMIT})


def is_commit_kind(kind):
    return kind == IRREVERSIBLE_COMMIT


# --------------------------------------------------------------------------------------------------
# Gap classes
# --------------------------------------------------------------------------------------------------
GAP_EVIDENCE = "evidence"
GAP_EXECUTION = "execution"
GAP_VERIFICATION = "verification"
GAP_DECISION = "decision"
GAP_CLASSES = frozenset({GAP_EVIDENCE, GAP_EXECUTION, GAP_VERIFICATION, GAP_DECISION})


# --------------------------------------------------------------------------------------------------
# Authorization tiers (see MutationAuthorization). The AVAILABLE->...->CANCELLED machine lives in the
# existing substrate; here the Kernel only needs the tier + single-use idempotency semantics.
# --------------------------------------------------------------------------------------------------
AUTH_TIER_READ = "read"
AUTH_TIER_SCOPED = "scoped"
AUTH_TIER_IRREVERSIBLE = "irreversible"


# --------------------------------------------------------------------------------------------------
# classify_result outcome vocabulary (SubstrateAdapter.classify_result returns one of these strings)
# --------------------------------------------------------------------------------------------------
RESULT_OK = "ok"
RESULT_UNKNOWN = "unknown"
RESULT_FAILED = "failed"
RESULT_ALREADY_REALIZED = "already_realized"


class PlanCompileError(Exception):
    """Raised when a Plan violates a Kernel invariant at compile time (e.g. >1 irreversible_commit
    without a transaction_contract)."""


# --------------------------------------------------------------------------------------------------
# Bindings (three distinct kinds - do not conflate)
# --------------------------------------------------------------------------------------------------
@dataclass
class ArgumentBinding:
    """WHERE a parameter VALUE comes from."""
    name: str
    value: Any
    source: str                 # bindings.SEMANTIC_SOURCES or bindings.SYSTEM_METADATA
    provenance: str = ""


@dataclass
class AffordanceBinding:
    """WHERE the environment control is (a located ref in a live observation)."""
    target_spec: Any
    ref: Any
    observation_hash: str = ""


@dataclass
class ProtocolConstant:
    """A schema-FIXED value (status='active', intent='order'). Proves nothing about a clinical value."""
    name: str
    value: Any


@dataclass
class GapSignal:
    gap_class: str              # GAP_EVIDENCE | GAP_EXECUTION | GAP_VERIFICATION | GAP_DECISION
    detail: str = ""


# --------------------------------------------------------------------------------------------------
# Single-use mutation authorization (tiered)
# --------------------------------------------------------------------------------------------------
@dataclass
class MutationAuthorization:
    tier: str                                   # AUTH_TIER_SCOPED | AUTH_TIER_IRREVERSIBLE
    idempotency_key: Optional[str] = None
    allowed_kind: Optional[str] = None          # the RecoveryStep.kind it authorizes
    side_effect_scope: Optional[str] = None
    max_uses: int = 1
    uses: int = 0

    @property
    def matchable(self):
        return self.uses < self.max_uses

    def consume(self):
        if self.uses >= self.max_uses:
            return False
        self.uses += 1
        return True


@dataclass
class Outcome:
    """Result of SubstrateAdapter.execute_primitive."""
    status: str = RESULT_OK                      # ok | unknown | failed | already_realized
    result: Any = None
    created_id: Optional[str] = None
    state_view: Optional[Dict[str, Any]] = None
    reason: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------------------------------
# Committed goal + plan + result
# --------------------------------------------------------------------------------------------------
@dataclass
class CommittedGoal:
    """A goal the ROOT AGENT already decided (oracle-blind: derived from the agent's own trajectory,
    never from gold/reference). One CommittedGoal == one Recovery Episode."""
    goal_id: str
    goal_type: str = ""                                     # logical process key (e.g. "create_order")
    committed_fields: Dict[str, Any] = field(default_factory=dict)  # agent-committed SEMANTIC values
    dedup_key: Optional[str] = None
    provenance: str = "agent_commitment"
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryStep:
    kind: str                                               # one of STEP_KINDS
    name: str = ""
    action: Optional[Dict[str, Any]] = None                # substrate primitive descriptor
    affordance_target: Any = None                          # target_spec resolved via the substrate
    arg_specs: List[Any] = field(default_factory=list)     # arg names (str) or {"name": ...} specs
    manifest: Dict[str, Any] = field(default_factory=dict) # side-effect declaration (drives auth tier)
    probe: bool = False                                    # existing-effect probe (read that may short to ALREADY_REALIZED)


@dataclass
class Plan:
    steps: List[RecoveryStep]
    required_bindings: List[Any] = field(default_factory=list)
    stop_conditions: List[Any] = field(default_factory=list)
    expected_postcondition: Optional[Dict[str, Any]] = None
    transaction_contract: Optional[Dict[str, Any]] = None   # {commits, compensation, partial_landing_policy}


@dataclass
class EpisodeResult:
    state: str
    path: Optional[str] = None
    goal_id: Optional[str] = None
    completed_steps: List[int] = field(default_factory=list)
    blocked_step_index: Optional[int] = None
    blocked_argument: Optional[str] = None
    created_ids: List[Any] = field(default_factory=list)
    auth_status: Optional[str] = None
    reason: str = ""
    metrics_bucket: Optional[str] = None
    events: List[Dict[str, Any]] = field(default_factory=list)


def validate_plan(plan):
    """Enforce the <=1 irreversible_commit rule. Returns the plan; raises PlanCompileError on violation."""
    if plan is None or plan.steps is None:
        raise PlanCompileError("empty_plan")
    bad_kinds = [s.kind for s in plan.steps if s.kind not in STEP_KINDS]
    if bad_kinds:
        raise PlanCompileError("illegal_step_kind:%s" % bad_kinds)
    n_commit = sum(1 for s in plan.steps if s.kind == IRREVERSIBLE_COMMIT)
    if n_commit > 1 and not plan.transaction_contract:
        raise PlanCompileError(
            "multi_irreversible_commit_without_transaction_contract:%d" % n_commit)
    return plan
