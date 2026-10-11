"""Value provenance classes (fixed shell): the write-provenance check of the harness action service
(runner/harness/action_service.py) and the recovery strategy's own binding search both classify slots with it.

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
