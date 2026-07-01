"""Mutation authorization (Operational Non-Degradation).

Core invariant: an UNCERTAIN/semantic harness finding NEVER grants write permission. It may trigger
read/inspect/ACQUIRE/candidate/advisory only. While a `mutation_hold` is active, any state mutation
(create/update/submit) must match an explicit, scoped, SINGLE-USE MutationAuthorization minted from exactly
one provenance: user_goal | deterministic_gap | evidence_supported_plan. The check unit is "does THIS action
match an unconsumed authorization", NOT "was the last finding deterministic". Substrate-agnostic: matches on
semantic_type/tool/target_path/effect declared by the adapter, never on tool-name guesses.
"""
from dataclasses import dataclass, field

VALID_SOURCES = ("user_goal", "deterministic_gap", "evidence_supported_plan")
MUTATION_TYPES = ("create", "update", "submit")

# Authorization lifecycle (Commit C1). Matchable ONLY while AVAILABLE. RESERVED = claimed for one
# pending action (combined ALLOW not yet confirmed). DISPATCHED = set IMMEDIATELY before the env
# call; from then the mutation may have landed even on a transport error, so it can NEVER
# re-authorize another mutation. VERIFIED = read-back confirmed. UNKNOWN = read-back ambiguous
# (reconcile only, never reuse). FAILED = definitely not executed. CANCELLED = reservation released.
AUTH_AVAILABLE = "AVAILABLE"
AUTH_RESERVED = "RESERVED"
AUTH_DISPATCHED = "DISPATCHED"
AUTH_VERIFIED = "VERIFIED"
AUTH_FAILED = "FAILED"
AUTH_UNKNOWN = "UNKNOWN"
AUTH_CANCELLED = "CANCELLED"
AUTH_STATES = (AUTH_AVAILABLE, AUTH_RESERVED, AUTH_DISPATCHED, AUTH_VERIFIED, AUTH_FAILED, AUTH_UNKNOWN, AUTH_CANCELLED)


@dataclass
class MutationAuthorization:
    authorization_id: str
    intervention_id: str
    source: str                                  # user_goal | deterministic_gap | evidence_supported_plan
    allowed_semantic_type: str                   # create | update | submit
    allowed_tool: str = None
    target_path: str = None
    allowed_effect: str = None
    expected_postcondition: dict = field(default_factory=dict)
    baseline_state_version: int = 0
    evidence_version: int = 0
    max_uses: int = 1
    status: str = AUTH_AVAILABLE       # AVAILABLE|RESERVED|DISPATCHED|VERIFIED|FAILED|UNKNOWN|CANCELLED

    @property
    def matchable(self):
        return self.status == AUTH_AVAILABLE

    @property
    def consumed(self):
        """Back-compat: 'consumed' == no longer matchable (anything past AVAILABLE)."""
        return self.status != AUTH_AVAILABLE


def action_target_path(sem, action):
    """The scope key of an action: explicit args target, else adapter-declared resource/target_entity."""
    args = (action or {}).get("args") or {}
    for k in ("target_path", "path", "field"):
        if args.get(k) is not None:
            return str(args.get(k))
    res = getattr(sem, "resource", None); ent = getattr(sem, "target_entity", None)
    if res and ent:
        return "%s/%s" % (res, ent)
    return res or ent


def exact_scope_match(auth, sem, action):
    """True ONLY on an EXACT scope match (never a superset). A mismatch on semantic_type / tool / effect /
    target_path -> no match -> the mutation is unauthorized."""
    if auth is None:
        return False   # pure scope match; matchability (status == AVAILABLE) is checked by the caller
    if getattr(sem, "semantic_type", None) != auth.allowed_semantic_type:
        return False
    if auth.allowed_tool is not None and (action or {}).get("tool") != auth.allowed_tool:
        return False
    if auth.allowed_effect is not None and getattr(sem, "effect", None) != auth.allowed_effect:
        return False
    if auth.target_path is not None and action_target_path(sem, action) != auth.target_path:
        return False
    return True


def has_verifiable_postcondition(auth):
    return bool(auth and auth.expected_postcondition)


def read_only_equivalent(manifest, action):
    """Adapter-DECLARED read-only equivalent of a write tool (rule 6: core never guesses substitutes)."""
    if not isinstance(manifest, dict):
        return None
    tool = (action or {}).get("tool")
    return ((manifest.get("read_only_equivalents") or {}).get(tool)) if tool else None


def should_set_mutation_hold(decision_type, deterministic):
    """A NON-deterministic semantic finding that emits feedback (REVISE / ACQUIRE) puts writes under a hold.
    CHANNEL-INDEPENDENT: inline and external feedback both activate it (the decision, not its renderer, is
    what matters). A DETERMINISTic finding does NOT hold -- it MINTS a scoped authorization instead."""
    return (deterministic is False) and (decision_type in ("REVISE", "ACQUIRE"))


def authorize_evidence_supported_plan(ledger, scope, new_evidence, support_passed, intervention_id=None):
    """Mint an `evidence_supported_plan` write authorization ONLY when (a) genuinely NEW validated evidence
    exists AND (b) the change-vs-delta gate passed (the plan change is supported by that new evidence). No new
    evidence -> None (ACQUIRE that found nothing cannot authorize a write). `scope` carries allowed_semantic_type
    / allowed_tool / target_path / allowed_effect / expected_postcondition."""
    if not new_evidence or not support_passed:
        return None
    return ledger.mint_authorization(
        source="evidence_supported_plan",
        allowed_semantic_type=scope.get("allowed_semantic_type"),
        allowed_tool=scope.get("allowed_tool"), target_path=scope.get("target_path"),
        allowed_effect=scope.get("allowed_effect"),
        expected_postcondition=scope.get("expected_postcondition"),
        intervention_id=intervention_id)


def authorize_deterministic_gap(ledger, scope, intervention_id=None):
    """Mint a `deterministic_gap` write authorization from an EXACT, adapter-resolved scope (required field
    empty + goal requires it + repair_surface exact target_path). Requires a verifiable postcondition and an
    exact target_path -- otherwise None (no blanket 'fix the form' authority)."""
    if not scope.get("target_path") or not scope.get("expected_postcondition"):
        return None
    return ledger.mint_authorization(
        source="deterministic_gap",
        allowed_semantic_type=scope.get("allowed_semantic_type", "update"),
        allowed_tool=scope.get("allowed_tool"), target_path=scope.get("target_path"),
        allowed_effect=scope.get("allowed_effect"),
        expected_postcondition=scope.get("expected_postcondition"),
        intervention_id=intervention_id)
