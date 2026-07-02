"""Bounded Clinical Recovery v3 - CreateOrderWorkflow (Layer 3, Committed Workflow Completion).

The process knowledge for realizing a committed executable order (imaging/lab/referral/procedure/medication)
that the ROOT AGENT decided but never PERFORMED. Substrate-agnostic in intent: it declares the bounded 4-step
plan and the required bindings, and it reuses the REAL payload logic from effect_completion.build_order_resource
(imported, never copied) so the resource shape stays identical to the proven cp3 completion path.

The 4 steps (design sec.6b):
    0. read  (governance / prerequisite evidence)   - read-only; optional per policy, harmless if unused.
    1. read  (existing-effect PROBE)                 - probe=True; if the effect is already present -> the
                                                       kernel short-circuits to ALREADY_REALIZED (fail-closed).
    2. irreversible_commit (create the record)       - the ONE durable mutation; server_persisted manifest ->
                                                       irreversible auth tier + single-use idempotency key.
    3. verify (server read-back)                     - read-only confirmation; the kernel also runs read_state
                                                       + verify_effect over expected_postcondition.

Binding declaration (required_bindings + the Decision-Boundary gate the kernel runs at step 0):
    SEMANTIC (a clinical decision -> only agent_commitment / authoritative_state / bound_evidence):
        code_text  <- agent_commitment      (the agent's verbatim order phrase = the decision)
        subject    <- authoritative_state    (WHO the order is for)
        requester  <- authoritative_state    (WHO is ordering)
    OPERATIONAL (never a decision -> system_metadata):
        authoredOn, idempotency_key, recovery_tag <- system_metadata

An unbound SEMANTIC arg (e.g. no subject) -> BLOCKED_NEEDS_DECISION (a correct refusal, never FAILED).
"""
from ..contracts import (
    RecoveryStep, Plan,
    READ, IRREVERSIBLE_COMMIT, VERIFY,
)
from ...effect_completion import (
    build_order_resource, resource_type_for_category, classify_effect_inspection,
)

# logical postcondition read-back key (opaque to the substrate; the substrate just returns {key: raw}).
_READBACK_KEY = "_bcr_order_readback"
# the process-fixed governance read (a create pre-read); the substrate treats it as a generic search.
_GOVERNANCE_RT = "AllergyIntolerance"

GOAL_TYPE = "create_order"

# The semantic + operational arguments this workflow must have bound before it may commit. Order matters:
# the Decision-Boundary gate stops at the FIRST unbound arg, so the most decision-critical come first.
_REQUIRED = ["code_text", "subject", "requester", "authoredOn", "idempotency_key", "recovery_tag"]


class CreateOrderWorkflow(object):
    """Realizes a committed order as a structured record via an injected SubstrateAdapter."""

    goal_type = GOAL_TYPE

    def match_goal(self, goal, ctx):
        gt = getattr(goal, "goal_type", "") or ""
        if gt == GOAL_TYPE:
            return True
        # tolerate a raw hint (benchmark adapters set goal_type, but be forgiving)
        return bool((getattr(goal, "raw", None) or {}).get("order"))

    def required_bindings(self, goal, ctx):
        return list(_REQUIRED)

    def compile_plan(self, goal, ctx):
        fields = getattr(goal, "committed_fields", None) or {}
        code_text = str(fields.get("code_text") or "").strip()
        category = str(fields.get("category") or "other")

        auth_state = (ctx or {}).get("authoritative_state") or {}
        sys_meta = (ctx or {}).get("system_metadata") or {}
        subject = auth_state.get("subject")
        requester = auth_state.get("requester")
        authored_on = sys_meta.get("authoredOn")

        order = {"text": code_text, "category": category}
        refs = {"subject": subject, "requester": requester, "authoredOn": authored_on}
        rt, resource = build_order_resource(order, refs)
        if rt is None:
            # subject/text unresolved -> keep a well-typed plan; the kernel's step-0 gate will BLOCK
            # on the missing SEMANTIC arg (never a FAILED create with a None payload).
            rt = resource_type_for_category(category)
            resource = None

        steps = [
            RecoveryStep(
                kind=READ, name="governance_read",
                action={"op": "search", "resourceType": _GOVERNANCE_RT, "subject": subject}),
            RecoveryStep(
                kind=READ, name="existing_effect_probe", probe=True,
                action={"op": "search", "resourceType": rt, "subject": subject,
                        "match_text": code_text}),
            RecoveryStep(
                kind=IRREVERSIBLE_COMMIT, name="create_record",
                action={"op": "create", "resourceType": rt, "resource": resource},
                manifest={"side_effect_scope": "create_%s" % rt, "server_persisted": True,
                          "rollback_available": False, "autosave_possible": False}),
            RecoveryStep(
                kind=VERIFY, name="server_read_back",
                action={"op": "search", "resourceType": rt, "subject": subject,
                        "match_text": code_text}),
        ]
        postcondition = {"paths": [{"key": _READBACK_KEY, "resourceType": rt,
                                    "subject": subject, "match_text": code_text}]}
        return Plan(steps=steps, required_bindings=[], expected_postcondition=postcondition)

    def verify_effect(self, goal, state_view):
        """True if the committed order's record is present in the read-back, False if refuted, None if the
        read-back is ambiguous (-> idempotent reconciliation / UNKNOWN)."""
        state_view = state_view or {}
        raw = state_view.get(_READBACK_KEY)
        if raw is None:
            return None
        insp = classify_effect_inspection(raw)
        st = insp.get("state")
        if st == "UNKNOWN":
            return None
        code_text = str((getattr(goal, "committed_fields", None) or {}).get("code_text") or "").strip().lower()
        if not code_text:
            return None
        for t in (insp.get("texts") or []):
            tt = str(t or "").strip().lower()
            if not tt:
                continue
            if code_text[:40] in tt or tt[:40] in code_text:
                return True
        # ABSENT, or PRESENT with no matching record -> the specific committed effect is not realized.
        return False
