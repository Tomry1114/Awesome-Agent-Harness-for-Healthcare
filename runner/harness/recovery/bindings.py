"""Bounded Clinical Recovery v3 - binding system.

Splits value provenance into two classes so `system_metadata` can never launder a clinical decision:

  SEMANTIC parameters  (drug, dose, diagnosis, rationale, disposition, claim, patient, order_type ...)
      -> legal sources: agent_commitment | authoritative_state | bound_evidence
      -> system_metadata is FORBIDDEN; an untraceable semantic arg resolves to None => caller BLOCKS.

  OPERATIONAL metadata (timestamp/authoredOn, request_id, idempotency_key, recovery_tag, episode_id,
                        client-generated resource id ...)
      -> may bind from system_metadata, but ONLY for fields a schema explicitly whitelists as operational.

Default-deny: an unrecognized field is treated as SEMANTIC (must be traced to a semantic source).

This module is substrate-agnostic and oracle-blind: it never reads gold/reference material.
"""
from .contracts import ArgumentBinding, GapSignal, GAP_DECISION, GAP_EVIDENCE


# Value-source classes.
AGENT_COMMITMENT = "agent_commitment"
AUTHORITATIVE_STATE = "authoritative_state"
BOUND_EVIDENCE = "bound_evidence"
SYSTEM_METADATA = "system_metadata"

SEMANTIC_SOURCES = (AGENT_COMMITMENT, AUTHORITATIVE_STATE, BOUND_EVIDENCE)
OPERATIONAL_SOURCES = (SYSTEM_METADATA,)

# Resolution priority (first hit wins).
SEMANTIC_SOURCE_PRIORITY = (AGENT_COMMITMENT, AUTHORITATIVE_STATE, BOUND_EVIDENCE)
# Operational fields prefer system_metadata, but may also be carried by authoritative/agent context.
OPERATIONAL_SOURCE_PRIORITY = (SYSTEM_METADATA, AUTHORITATIVE_STATE, AGENT_COMMITMENT)

SEMANTIC = "semantic"
OPERATIONAL = "operational"

# Conservative default whitelist of purely-operational fields (never clinical payload). A schema may
# extend this via schema["operational_fields"]; nothing here changes a decision's meaning.
DEFAULT_OPERATIONAL_WHITELIST = frozenset({
    "timestamp", "authoredOn", "authored_on", "recorded_on", "lastUpdated", "last_updated",
    "request_id", "idempotency_key", "recovery_tag", "episode_id", "correlation_id", "trace_id",
    "client_id", "client_request_id", "resource_id", "id", "meta",
})


def _operational_whitelist(schema):
    schema = schema or {}
    extra = schema.get("operational_fields") or ()
    return set(DEFAULT_OPERATIONAL_WHITELIST) | set(extra)


def classify_field(name, schema=None):
    """Return SEMANTIC or OPERATIONAL. Default-deny: unknown => SEMANTIC.

    A schema may force a field semantic by listing it in schema['semantic_fields'] (overrides the
    operational whitelist), so an adapter can never accidentally mark clinical payload operational."""
    schema = schema or {}
    if name in set(schema.get("semantic_fields") or ()):
        return SEMANTIC
    if name in _operational_whitelist(schema):
        return OPERATIONAL
    return SEMANTIC


def is_semantic(name, schema=None):
    return classify_field(name, schema) == SEMANTIC


def resolve_argument(name, sources, schema=None, priority=None):
    """Resolve one argument to an ArgumentBinding, or None if unbindable.

    `sources` maps a source-class name -> a dict of {field: value} available under that class, e.g.
        {agent_commitment: {...}, authoritative_state: {...}, bound_evidence: {...}, system_metadata: {...}}

    SEMANTIC fields may bind only from the 3 semantic sources (system_metadata is skipped even if it
    carries the field). OPERATIONAL fields may bind from system_metadata (and, if present, from
    authoritative/agent context). An unbindable SEMANTIC arg returns None -> the caller must BLOCK.
    """
    sources = sources or {}
    kind = classify_field(name, schema)
    if priority is None:
        allowed = SEMANTIC_SOURCE_PRIORITY if kind == SEMANTIC else OPERATIONAL_SOURCE_PRIORITY
    else:
        allowed = priority
    for src in allowed:
        if kind == SEMANTIC and src == SYSTEM_METADATA:
            # HARD GUARD: system_metadata may never satisfy a semantic (decision-bearing) argument.
            continue
        bag = sources.get(src) or {}
        if name in bag and bag[name] is not None:
            return ArgumentBinding(
                name=name, value=bag[name], source=src,
                provenance="%s:%s" % (src, name))
    return None


def _arg_name(spec):
    if isinstance(spec, str):
        return spec
    if isinstance(spec, dict):
        return spec.get("name")
    return getattr(spec, "name", None)


def decision_boundary(required_args, sources, schema=None):
    """Decision-Boundary gate for a set of required arguments.

    Returns a dict:
        {"bindings": {name: ArgumentBinding}, "blocked_argument": name|None, "gap": GapSignal|None}

    Stops at the FIRST unbound argument. An unbound SEMANTIC arg -> GAP_DECISION (the harness must not
    decide -> BLOCKED_NEEDS_DECISION). An unbound OPERATIONAL arg -> GAP_EVIDENCE (missing metadata the
    system was supposed to supply -> BLOCKED_MISSING_EVIDENCE)."""
    bindings = {}
    for spec in (required_args or []):
        name = _arg_name(spec)
        if name is None:
            continue
        b = resolve_argument(name, sources, schema)
        if b is None:
            gc = GAP_DECISION if classify_field(name, schema) == SEMANTIC else GAP_EVIDENCE
            return {"bindings": bindings, "blocked_argument": name,
                    "gap": GapSignal(gap_class=gc, detail="unbound:%s" % name)}
        bindings[name] = b
    return {"bindings": bindings, "blocked_argument": None, "gap": None}
