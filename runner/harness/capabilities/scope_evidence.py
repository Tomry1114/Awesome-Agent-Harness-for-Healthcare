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
    name = "scope_evidence"

    def before_action(self, action, ctx):
        sem = ctx.sem
        target = sem.target_entity if sem else None
        if target is not None:
            ctx.ledger.bump_opportunity("subject_bearing_action")
        active = ctx.ledger.subject_id()
        if not active or target is None:
            return None
        if _norm(target) != _norm(active):
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
        shown = sem.target_entity if sem else None
        # observation-derived subject (GUIs): if this action's subject was NOT a structured arg (so
        # before_action didn't count it), count the opportunity here and check the displayed subject.
        if active and shown is not None and not _arg_subject(action, ctx.manifest):
            ctx.ledger.bump_opportunity("subject_bearing_action")
            if _norm(shown) != _norm(active):
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
            valid = (ctx.result_ok is not False) and _nonempty(result)
            subj = sem.target_entity if sem.target_entity is not None else active
            rel = ("matched" if (subj is not None and active is not None and _norm(subj) == _norm(active))
                   else ("foreign" if (subj is not None and active is not None) else "unknown"))
            ctx.ledger.add_evidence(type=(sem.resource or sem.capability), value=_summarize(result),
                                    subject_id=subj, source_event="step-%d" % ctx.step,
                                    source_type=sem.source_class,
                                    extra={"modality": sem.modality, "resource": sem.resource,
                                           "source_class": sem.source_class,
                                           "status": ("VALIDATED" if valid else "ATTEMPTED"),
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


def _norm(x):
    return str(x or "").strip().lower().split("/")[-1]


def _nonempty(result):
    if result is None:
        return False
    s = result if isinstance(result, str) else str(result)
    return bool(s.strip())


def _summarize(result):
    s = result if isinstance(result, str) else str(result)
    return s[:200]
