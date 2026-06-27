"""Module B — Obligation Lifecycle. Operates on the canonical SemanticAction + the evidence ledger.

A commit point (matched by sem: effect/semantic_type) requires obligations be SATISFIED first; if not,
REVISE with the actionable leaf obligations. Evidence obligations are satisfied when evidence of the
required CLASS / RESOURCE / MODALITY exists in the ledger — not when a particular TOOL ran. So "check
allergies" is satisfied by record-evidence about AllergyIntolerance regardless of which tool produced it,
and "image grounding" by any perception-class evidence. No tool names here.
"""
from ..capability import Capability
from .. import decision as D
from ..state import SATISFIED, WAIVED


class ObligationLifecycle(Capability):
    name = "obligation_lifecycle"

    def on_contract(self, ctx):
        c = ctx.contract
        if not c:
            return None
        for o in c.evidence_obligations:
            if o.get("id"):
                ctx.ledger.declare_obligation(o["id"], kind="evidence", satisfied_by=o.get("satisfied_by"))
        for o in c.workflow_obligations:
            if o.get("id"):
                ctx.ledger.declare_obligation(o["id"], kind="workflow", requires=o.get("requires"))
        return None

    def before_action(self, action, ctx):
        return self._gate(ctx, "commit_requires_obligations")

    def before_final(self, answer, ctx):
        return self._gate(ctx, "final_requires_obligations")

    def _gate(self, ctx, default_rule):
        cp = ctx.contract.commit_point_for(ctx.sem) if (ctx.contract and ctx.sem) else None
        if not cp:
            return None
        missing = ctx.ledger.pending_prerequisites(cp.get("requires", []))
        if not missing:
            return None
        leaves, sugg = _expand_missing(ctx.ledger, missing)
        return self._decide(
            D.REVISE, rule_id=cp.get("requires_rule", default_rule), reason_code="missing_prerequisite",
            deterministic=True,
            missing_obligations=leaves, suggested_capabilities=sugg,
            reason="commit requires unmet obligations: %s" % ", ".join(leaves),
            feedback="Before this commit, complete: %s." % ", ".join(leaves))

    def after_action(self, action, result, before_state, after_state, ctx):
        # 1) satisfy evidence obligations whose required evidence now EXISTS in the ledger
        active = ctx.ledger.subject_id()
        for oid, ob in ctx.ledger.obligations.items():
            if ob.get("state") in (SATISFIED, WAIVED):
                continue
            if ob.get("kind") == "evidence" and _evidence_satisfies(ctx.ledger, ob.get("satisfied_by"), active):
                ctx.ledger.set_obligation(oid, SATISFIED, note="valid evidence present",
                                          event_id="step-%d" % ctx.step)
        # 2) propagate workflow obligations whose prerequisites are all satisfied
        for oid, ob in ctx.ledger.obligations.items():
            if ob.get("kind") == "workflow" and ob.get("state") not in (SATISFIED, WAIVED):
                if not ctx.ledger.pending_prerequisites(ob.get("requires", [])):
                    ctx.ledger.set_obligation(oid, SATISFIED, note="prerequisites met")
        return None


def _evidence_satisfies(ledger, req, active=None):
    """True iff some VALIDATED, SUBJECT-CONSISTENT ledger evidence matches every declared field of `req`
    (source_class / resource / modality). A failed/empty (ATTEMPTED) read, or evidence about a foreign
    subject, does NOT satisfy an obligation."""
    req = req or {}
    want_sc, want_res, want_mod = req.get("source_class"), req.get("resource"), req.get("modality")
    for e in ledger.evidence:
        if e.get("status") != "VALIDATED":       # only VALIDATED evidence counts (no legacy status-None fail-open)
            continue
        if active is not None and e.get("subject_id") is not None and _eq(e["subject_id"], active) is False:
            continue                                          # foreign-subject evidence never satisfies
        if want_sc and (e.get("source_class") or e.get("source_type")) != want_sc:
            continue
        if want_res and e.get("resource") != want_res:
            continue
        if want_mod and e.get("modality") != want_mod:
            continue
        return True
    return False


def _eq(a, b):
    """Typed identity: ids must match; if both carry a type, types must match too."""
    def _ref(x):
        t = str(x or "").strip().lower()
        return tuple(t.rsplit("/", 1)) if "/" in t else (None, t)
    (ta, ia), (tb, ib) = _ref(a), _ref(b)
    return ia == ib and not (ta and tb and ta != tb)


def _expand_missing(ledger, missing):
    leaves, seen = [], set()
    for oid in missing:
        ob = ledger.obligations.get(oid) or {}
        sub = (ledger.pending_prerequisites(ob["requires"]) or [oid]) if (ob.get("kind") == "workflow"
                                                                          and ob.get("requires")) else [oid]
        for s in sub:
            if s not in seen:
                seen.add(s); leaves.append(s)
    sugg = []
    for oid in leaves:
        sb = (ledger.obligations.get(oid) or {}).get("satisfied_by") or {}
        label = sb.get("resource") or sb.get("modality") or sb.get("source_class")
        if label:
            sugg.append(str(label))
    return leaves, sugg
