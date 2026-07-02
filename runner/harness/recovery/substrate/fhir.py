"""Bounded Clinical Recovery v3 - structured-record (FHIR) Substrate Adapter (Layer 2).

FhirSubstrateAdapter implements the SubstrateAdapter protocol for a structured-record environment. It only
knows the environment mechanics of a FHIR-shaped record store:

    resolve_affordance  - a structured record has NO DOM/visual affordances, so this is a trivial pass-through
                          (the "control" is the primitive descriptor itself; there is nothing to disambiguate).
    execute_primitive   - read-like kinds  -> a record SEARCH; irreversible_commit -> a record CREATE. Both run
                          through an injected `backend` (the same search/create/read-back mechanics the existing
                          RunDriver.execute / execute_recovery_read route through the ActionExecutor + env). The
                          backend is dependency-injected so this adapter is testable with an in-memory stub and,
                          in production, wraps the real driver/executor/env.
    read_state          - authoritative read-back SEARCH of the given concrete state paths.
    classify_result     - reuse the generic evidence-state / effect-inspection classifiers.

It knows NOTHING about "order", "appeal", "prior-auth", or any benchmark: it matches generic record
SIGNATURES (resourceType + subject + an optional match token), never clinical concepts. The clinical payload
(what an order resource looks like) is built by the WorkflowModule and handed here as an opaque `resource` dict.
"""
from ..contracts import (
    Outcome, AffordanceBinding,
    READ, NAVIGATE, ACQUIRE, VERIFY, STAGED_WRITE, IRREVERSIBLE_COMMIT,
    RESULT_OK, RESULT_UNKNOWN, RESULT_FAILED, RESULT_ALREADY_REALIZED,
)
from ...evidence_state import classify_evidence_state, PRESENT, ABSENT, UNKNOWN, FAILED
from ...effect_completion import classify_effect_inspection

# Adapter result-semantics for a record SEARCH returned as a flat entries list (matches effect_completion).
_SEARCH_SEM = {"collection_paths": ["entries"], "absence_when_empty": True}

_READ_LIKE = (READ, NAVIGATE, ACQUIRE, VERIFY)


def _token_match(match_text, texts):
    """Generic signature match: does any record's comparable text overlap the requested token? (No clinical
    knowledge -- pure string containment, tolerant of the 180-char code.text truncation build applies.)"""
    mt = str(match_text or "").strip().lower()
    if not mt:
        return False
    for t in (texts or []):
        tt = str(t or "").strip().lower()
        if not tt:
            continue
        if mt[:40] in tt or tt[:40] in mt:
            return True
    return False


class FhirSubstrateAdapter(object):
    """Structured-record substrate mechanics. Inject a `backend` exposing:
        backend.search(resource_type, subject) -> raw result dict (flat 'entries' list, per _SEARCH_SEM)
        backend.create(resource)               -> raw result dict carrying the new record id ('id')
    The real backend wraps RunDriver/ActionExecutor/env; the test backend is in-memory. Both share this
    adapter's mechanics so they cannot diverge.
    """

    def __init__(self, backend=None):
        self.backend = backend

    # -- Layer-2 protocol ------------------------------------------------------------------------
    def resolve_affordance(self, target_spec, observation):
        """A structured record has no located visual control to disambiguate -> pass-through located ref.
        (Kept protocol-complete; create_order steps set affordance_target=None so this is rarely called.)"""
        return AffordanceBinding(target_spec=target_spec, ref=target_spec, observation_hash="")

    def execute_primitive(self, kind, action, auth):
        desc = (action or {}).get("action") or {}
        op = desc.get("op")
        if kind in _READ_LIKE:
            return self._do_search(desc)
        if kind == IRREVERSIBLE_COMMIT:
            return self._do_create(desc, auth)
        if kind == STAGED_WRITE:
            # No staged (reversible) primitive in the create-order path; support it generically so a future
            # workflow can use it. Requires the scoped/irreversible auth the kernel minted from the manifest.
            if auth is None or not auth.consume():
                return Outcome(status=RESULT_FAILED, reason="staged_write_unauthorized")
            return self._do_create(desc, None, _pre_consumed=True) if op == "create" else Outcome(
                status=RESULT_OK, result={"staged": True})
        return Outcome(status=RESULT_FAILED, reason="unsupported_kind:%s" % kind)

    def read_state(self, paths):
        """Authoritative read-back: one SEARCH per concrete state-path descriptor -> {key: raw_result}."""
        view = {}
        for p in (paths or []):
            if isinstance(p, dict):
                key = p.get("key") or p.get("resourceType") or "state"
                try:
                    ids = p.get("ids")
                    if ids and hasattr(self.backend, "read"):
                        # authoritative, immediate read-back by created id (avoids HAPI search-index lag)
                        entries = []
                        for rid in ids:
                            rec = self.backend.read(p.get("resourceType"), rid)
                            if isinstance(rec, dict) and rec.get("resourceType"):
                                entries.append({"resource": rec})
                        view[key] = {"entries": entries} if entries else self.backend.search(
                            p.get("resourceType"), p.get("subject"))
                    else:
                        view[key] = self.backend.search(p.get("resourceType"), p.get("subject"))
                except Exception as ex:  # read-back failure is non-fatal (kernel reconciles) -> UNKNOWN shape
                    view[key] = {"error": repr(ex), "status": "failed"}
            elif isinstance(p, str):
                view[p] = None
        return view

    def classify_result(self, result):
        """Map an Outcome (or raw) to the outcome vocabulary. Prefer the status we already computed."""
        st = getattr(result, "status", None)
        if st in (RESULT_OK, RESULT_UNKNOWN, RESULT_FAILED, RESULT_ALREADY_REALIZED):
            return st
        raw = getattr(result, "raw", None) or getattr(result, "result", None)
        ev = classify_evidence_state(raw if isinstance(raw, dict) else None, _SEARCH_SEM)
        if ev == FAILED:
            return RESULT_FAILED
        if ev == UNKNOWN:
            return RESULT_UNKNOWN
        return RESULT_OK

    # -- mechanics -------------------------------------------------------------------------------
    def _do_search(self, desc):
        rt = desc.get("resourceType")
        subj = desc.get("subject")
        try:
            raw = self.backend.search(rt, subj)
        except Exception as ex:
            return Outcome(status=RESULT_FAILED, reason="search_error:%r" % ex, raw={"error": repr(ex)})
        ev = classify_evidence_state(raw, _SEARCH_SEM)
        if ev == FAILED:
            return Outcome(status=RESULT_FAILED, result=raw, raw={"search": raw})
        if ev == UNKNOWN:
            return Outcome(status=RESULT_UNKNOWN, result=raw, raw={"search": raw})
        status = RESULT_OK
        match_text = desc.get("match_text")
        if match_text and ev in (PRESENT, ABSENT):
            insp = classify_effect_inspection(raw)
            if insp.get("state") == UNKNOWN:
                # present-but-no-comparable-representation -> cannot claim the effect is already there
                status = RESULT_UNKNOWN
            elif _token_match(match_text, insp.get("texts")):
                status = RESULT_ALREADY_REALIZED
        return Outcome(status=status, result=raw, raw={"search": raw})

    def _do_create(self, desc, auth, _pre_consumed=False):
        if not _pre_consumed:
            if auth is None or not auth.consume():
                return Outcome(status=RESULT_FAILED, reason="create_unauthorized")
        res = desc.get("resource")
        if not res:
            return Outcome(status=RESULT_FAILED, reason="no_resource_payload")
        try:
            raw = self.backend.create(res)
        except Exception as ex:
            # a create that may or may not have landed is UNKNOWN, never FAILED (no re-commit after UNKNOWN)
            return Outcome(status=RESULT_UNKNOWN, reason="create_error:%r" % ex, raw={"error": repr(ex)})
        cid = raw.get("id") if isinstance(raw, dict) else None
        if not cid:
            return Outcome(status=RESULT_UNKNOWN, reason="create_no_id", result=raw, raw=raw if isinstance(raw, dict) else {})
        return Outcome(status=RESULT_OK, created_id=cid, result=raw,
                       raw=raw if isinstance(raw, dict) else {})
