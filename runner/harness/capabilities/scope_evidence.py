"""Module A — Scope & Evidence Binding. Operates on the CANONICAL SemanticAction (ctx.sem), never tool
names. "Did the right thing to the WRONG subject / from the WRONG evidence."

The operated-on subject is sem.target_entity (the substrate manifest declares how to extract it — a
structured arg for record systems, the displayed subject for GUIs). Read actions (sem.source_class set)
produce evidence tagged with source_class / modality / resource, bound to the active subject. The
algorithm is identical for patients, cases, images — they are all just entity ids.
"""
from ..capability import Capability
from .. import decision as D


class ScopeEvidenceBinding(Capability):
    # LAYER (see HARNESS_DESIGN.md): INFRASTRUCTURE -- subject binding + evidence ledger (the interface a clinical agent needs; environment-facing, durable)
    name = "scope_evidence"

    def before_action(self, action, ctx):
        sem = ctx.sem
        target = sem.target_entity if sem else None
        binding = sem.subject_binding if sem else "implicit_active"
        # this action operates on a subject (-> a subject_bearing opportunity, deduped per action) if it has
        # a target arg, OR it is an implicit_active commit on a displayed subject (GUI submit).
        if target is not None or (binding == "implicit_active" and sem and sem.is_commit() and ctx.displayed_subject is not None):
            ctx.ledger.bump_opportunity("subject_bearing_action", ctx.step)
        active = ctx.ledger.subject_id()
        # a subject-bound COMMIT under `required` binding with NO resolved target: "operating on nobody"
        # must NOT pass as "operating on the active subject" -> REVISE (name the subject explicitly).
        # `required` binding applies to EVERY subject-bearing action of that rule (a commit OR a read that
        # produces evidence) — not only commits. "Search all patients" / a subject-less write both fail.
        if binding == "required" and target is None and sem and (sem.is_commit() or sem.source_class):
            return self._decide(
                D.REVISE, rule_id="subject_unspecified", reason_code="subject_unspecified", deterministic=True,
                reason="action does not specify which subject it operates on",
                feedback="This action does not name a subject; specify the subject it applies to.")
        # PROSPECTIVE guard for implicit_active substrates (GUI): a COMMIT (e.g. submit) while the page
        # shows a DIFFERENT subject than assigned is blocked BEFORE it executes — post-hoc is too late for
        # an irreversible submit. (Non-commit navigation toward the right subject is still allowed.)
        if binding == "implicit_active" and active and sem and sem.is_commit():
            shown = ctx.displayed_subject            # last known displayed subject (manifest-projected)
            if shown is not None and not _same_subject(shown, active):
                return self._decide(
                    D.BLOCK, rule_id="subject_scope_mismatch", reason_code="wrong_scope", deterministic=True,
                    reason="committing while the page shows %s but the assigned subject is %s" % (shown, active),
                    feedback="The page shows %s; do not submit until you are on %s." % (shown, active),
                    extra={"shown": shown, "active_subject": active})
        if not active or target is None:
            return None
        if not _same_subject(target, active):
            return self._decide(
                D.BLOCK, rule_id="subject_scope_mismatch", reason_code="wrong_scope", deterministic=True,
                reason="action operates on %s but the active subject is %s" % (target, active),
                feedback="This action targets %s; the assigned subject is %s. Operate only on %s."
                         % (target, active, active),
                extra={"target": target, "active_subject": active})
        return None

    def after_action(self, action, result, before_state, after_state, ctx):
        sem = ctx.sem
        active = ctx.ledger.subject_id()
        # subject displayed by THIS action's observation (manifest-projected from the raw env result).
        shown = ctx.observed_subject if ctx.observed_subject is not None else (sem.target_entity if sem else None)
        # observation-derived subject (GUIs): if this action's subject was NOT a structured arg (so
        # before_action didn't count it), count the opportunity here and check the displayed subject.
        if active and shown is not None and not _arg_subject(action, ctx.manifest):
            ctx.ledger.bump_opportunity("subject_bearing_action", ctx.step)
            if not _same_subject(shown, active):
                return self._decide(
                    D.REVISE, rule_id="subject_scope_mismatch", reason_code="wrong_scope", deterministic=True,
                    reason="page now shows %s but the assigned subject is %s" % (shown, active),
                    feedback="You are viewing %s; the assigned subject is %s — return to it." % (shown, active),
                    extra={"shown": shown, "active_subject": active})
        # evidence binding: a read action that yields evidence (source_class declared by the manifest).
        # Evidence is tagged with VALIDITY (only success + non-empty -> VALIDATED) and bound to the
        # action's OWN subject (sem.target_entity), with a scope_relation to the active subject. A failed/
        # empty result, or a foreign-subject read, therefore does NOT satisfy an obligation.
        if sem and sem.source_class:
            binding = sem.subject_binding
            # bind evidence to the action's OWN subject; only fall back to the active subject when the
            # adapter guarantees it (implicit_active). Under `required` binding a read with no named subject
            # is NOT assumed to be about the active subject -> subject None + NOT validated (cannot satisfy).
            if sem.target_entity is not None:
                subj = sem.target_entity
            elif binding == "implicit_active":
                subj = active
            else:
                subj = None
            # STRICT: evidence is VALIDATED only on an explicit success signal (result_ok is True). A missing
            # signal (None) is UNKNOWN -> ATTEMPTED, never VALIDATED (a future adapter that forgets the field
            # must fail safe, not pass).
            valid = ((ctx.result_ok is True) and _has_payload(result)
                     and (binding != "required" or sem.target_entity is not None))
            rel = ("matched" if (subj is not None and active is not None and _same_subject(subj, active))
                   else ("foreign" if (subj is not None and active is not None) else "unknown"))
            # EvidenceState (Commit A): classify via the adapter's declared result-semantics for this resource,
            # so a confirmed-EMPTY read is ABSENT (obligation CHECKED, not decision-changing) rather than a
            # non-resolving ATTEMPTED. A failed/uncertain read is FAILED/UNKNOWN -> still unresolved.
            from ..evidence_state import classify_evidence_state
            from ..adapter_compiler import result_semantics_for
            _es = None
            if sem.resource:
                _es = classify_evidence_state(result, result_semantics_for(ctx.manifest, sem.resource))
            ctx.ledger.add_evidence(type=(sem.resource or sem.capability), value=_summarize(result),
                                    subject_id=subj, source_event="step-%d" % ctx.step,
                                    source_type=sem.source_class,
                                    extra={"modality": sem.modality, "resource": sem.resource,
                                           "value_full": _full(result),
                                           "source_class": sem.source_class,
                                           "status": ("VALIDATED" if valid else "ATTEMPTED"),
                                           "evidence_state": _es,
                                           "scope_relation": rel})
        return None


def _arg_subject(action, manifest):
    args = (action or {}).get("args") or {}
    if not isinstance(args, dict):
        return None
    for k in ((manifest.get("subject") or {}).get("from_args") or []):
        if args.get(k):
            return str(args[k])
    return None


def _ref(x):
    """Parse a subject ref into (type, id). 'Patient/123' -> ('patient','123'); '123' -> (None,'123')."""
    t = str(x or "").strip().lower()
    if "/" in t:
        a, b = t.rsplit("/", 1)
        return (a, b)
    return (None, t)


def _same_subject(a, b):
    """Typed identity: ids must match; if BOTH carry a type, the types must match too (so Patient/123 !=
    Encounter/123, while Patient/123 == 123 when one side is untyped)."""
    (ta, ia), (tb, ib) = _ref(a), _ref(b)
    if ia != ib:
        return False
    return not (ta and tb and ta != tb)


_ENVELOPE_KEYS = {"ok", "status", "mode", "tool", "args", "url", "title", "state_changed", "surface_changed"}


def _has_payload(result):
    """Does the result carry actual evidence content? {} / [] / {"output": ""} / {"ok": true} are NOT
    evidence (str({}) == '{}' would wrongly look non-empty); only meaningful payload counts."""
    if result is None:
        return False
    if isinstance(result, str):
        return bool(result.strip())
    if isinstance(result, (list, tuple, set)):
        return any(_has_payload(x) for x in result)
    if isinstance(result, dict):
        return any(_has_payload(v) for k, v in result.items() if k not in _ENVELOPE_KEYS)
    return True


def _summarize(result):
    # short, human-readable PREVIEW for the audit ledger (transparency / compactness).
    s = result if isinstance(result, str) else str(result)
    return s[:200]


def _full(result):
    # the VERIFICATION payload the grounding judge actually reads. The 200-char preview above is far too
    # short to verify a clinical finding (a CT region description, a lab panel) — judging against it makes
    # the judge see a truncated stub and reject well-grounded answers. Bounded (not unbounded) so the
    # context stays sane; stripped from the persisted audit (state.to_dict) to keep result.json compact.
    s = result if isinstance(result, str) else str(result)
    return s[:4000]
