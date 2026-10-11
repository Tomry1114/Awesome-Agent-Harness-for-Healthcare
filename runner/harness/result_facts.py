# -*- coding: utf-8 -*-
"""THE definition of what an action result CONTAINS. One fact, one reader.

"This result is empty" can be read two ways, and the two readings must agree on every action.

  The BASE capability (fhir_empty_targeted_search_broad_reacquire) reads the RAW result -- `entries`,
  `total`, `result_status == "empty_result"`.

  The OBSERVATION layer reads `canonical_observation(res)["modalities"]`. A PB FHIR result is
  `{"entries": [...], "pages": ..., "total": ...}` and carries neither `observation` nor `output`; a
  reader that only looks at those keys sees empty modalities and would flag EVERY PB read as empty --
  including a demographics search that returned a full Patient record -- so the semantic judge would be
  told "the last read returned nothing" after successful reads, and any check that turns on emptiness
  would be judged against a constant.

A constant is not a fact. So:

  * ONE function answers the question, for the raw result AND for the canonical envelope.
  * It returns None when it does not RECOGNISE the shape. That is the whole mechanism: a reader must not
    turn "I do not know this shape" into "it was empty". An environment with a new payload shape reports
    `unknown`, which is visible, instead of silently reporting emptiness.

What this deliberately does NOT decide: whether an empty result should be ACTED on. "The query returned
nothing" is a fact; "a malformed-filter error is not evidence of absence" is a policy, and policy stays
with the capability that holds it.
"""

_EMPTY_TEXT = ("", "[]", "{}", "0", "none", "null")

# Keys whose presence means the environment produced something, even with no readable body.
_ARTIFACT_KEYS = ("written", "downloaded", "uploaded", "image_ref")


def _text_is_empty(value):
    if value is None:
        return True
    body = value if isinstance(value, str) else str(value)
    stripped = body.strip().lower()
    if stripped in _EMPTY_TEXT:
        return True
    # A result that reached us SERIALISED still states its own emptiness. This matches the base
    # capability's reader exactly, so using the shared definition changes none of the base harness's firing
    # decisions.
    if '"entries": []' in stripped and ('"total": 0' in stripped or "'total': 0" in stripped):
        return True
    return "empty_result" in stripped


def _channel_is_empty(value, _depth=0):
    """A content channel may itself hold a structured payload. canonical_observation carries an
    unrecognised body through as `structured`, so a PB search's {"entries": [], "total": 0} arrives here
    as a dict -- and str()-ing it would read as content, which is how "0 results" would look like data.
    """
    if isinstance(value, dict) and _depth < 3:
        inner = result_is_empty(value, _depth + 1)
        if inner is not None:
            return inner
    if isinstance(value, (list, tuple)):
        return len(value) == 0
    return _text_is_empty(value)


def result_is_empty(payload, _depth=0):
    """Did this action result carry any content?  True / False / None (shape not recognised).

    Accepts the raw tool result, the canonical observation, or the {result_ok, result_status,
    observation} envelope the view hands out -- the three shapes this fact is asked about.
    """
    if payload is None:
        return None
    if not isinstance(payload, dict):
        return _text_is_empty(payload)

    # -- the view's envelope: the declared status is authoritative, then recurse into the observation.
    if "observation" in payload and ("result_ok" in payload or "result_status" in payload):
        if _declared_empty(payload):
            return True
        return result_is_empty(payload.get("observation"), _depth + 1)

    if _declared_empty(payload):                      # explicit: the environment SAID it was empty
        return True

    for k in _ARTIFACT_KEYS:                          # something was produced, even if unreadable
        if payload.get(k):
            return False

    # -- the canonical observation: modalities carry the body, artifacts count as content.
    mods = payload.get("modalities")
    if isinstance(mods, dict):
        if payload.get("artifacts"):
            return False
        for k in ("text", "structured"):
            if mods.get(k) is not None:
                return _channel_is_empty(mods[k], _depth)
        # modalities present but carrying nothing is only meaningful if this really is the canonical
        # envelope; otherwise fall through to the raw-result channels below.
        if payload.get("observation_type") is not None:
            return True

    # -- FHIR search results: `entries` / `total` ARE the body.
    total, entries = payload.get("total"), payload.get("entries")
    if isinstance(entries, list):
        return len(entries) == 0
    if total is not None:
        try:
            return int(total) == 0
        except (TypeError, ValueError):
            pass

    # -- a FHIR resource returned directly: `create` answers with the resource it made (e.g. a ServiceRequest
    # or MedicationRequest). Checked AFTER entries/total so a search result is never mistaken for one.
    if payload.get("resourceType") and payload.get("id"):
        return False

    # -- generic tool results.
    for k in ("observation", "output"):
        if payload.get(k) is not None:
            return _channel_is_empty(payload[k], _depth)

    if payload.get("error") is not None:              # an error is a failure, NOT an empty result;
        return False                                  # result_status already carries that at higher resolution

    return None                                       # shape not recognised -- say so, never assume empty


def _declared_empty(payload):
    return (payload.get("result_status") == "empty_result"
            or payload.get("status") == "empty_result"
            or payload.get("empty_result") is True)
