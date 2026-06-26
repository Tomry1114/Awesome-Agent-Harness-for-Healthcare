"""Module B — Obligation Lifecycle Controller.

Maintains "what still must be done" as an obligation DAG (not a fixed reference path). Before a commit
point it checks that the commit's required obligations are SATISFIED; if not, it REVISEs with the
missing obligations (it does NOT produce the answer — the agent re-plans). After each action it marks
obligations satisfied when their `satisfied_by` matches the executed action, and propagates workflow
obligations whose prerequisites are all met. P0: structure + deterministic satisfaction-by-tool;
per-dataset obligation graphs live in the policy packs.
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
        c = ctx.contract
        if not c:
            return None
        name = _name(action)
        cp = c.commit_point_for(name) or (_final_cp(c) if action.get("type") == "final" else None)
        if not cp:
            return None
        missing = ctx.ledger.pending_prerequisites(cp.get("requires", []))
        if missing:
            sugg = []
            for oid in missing:
                ob = ctx.ledger.obligations.get(oid) or {}
                sb = ob.get("satisfied_by") or {}
                if sb.get("tool"):
                    sugg.append(sb["tool"] + ("." + sb["resource_type"] if sb.get("resource_type") else ""))
            return self._decide(
                D.REVISE, rule_id=cp.get("requires_rule", "commit_requires_obligations"),
                deterministic=True, missing_obligations=missing, suggested_capabilities=sugg,
                reason="commit '%s' requires unmet obligations: %s" % (name, ", ".join(missing)),
                feedback="Before '%s', complete: %s." % (name, ", ".join(missing)))
        return None

    def before_final(self, answer, ctx):
        """The final answer is a commit point — gate it on the same obligation prerequisites."""
        c = ctx.contract
        cp = _final_cp(c) if c else None
        if not cp:
            return None
        missing = ctx.ledger.pending_prerequisites(cp.get("requires", []))
        if missing:
            return self._decide(
                D.REVISE, rule_id=cp.get("requires_rule", "final_requires_obligations"),
                deterministic=True, missing_obligations=missing,
                reason="final answer requires unmet obligations: %s" % ", ".join(missing),
                feedback="Before answering, complete: %s." % ", ".join(missing))
        return None

    def after_action(self, action, result, before_state, after_state, ctx):
        name = _name(action)
        # 1) satisfy evidence obligations whose satisfied_by.tool matches this action
        for oid, ob in ctx.ledger.obligations.items():
            if ob.get("state") in (SATISFIED, WAIVED):
                continue
            sb = ob.get("satisfied_by") or {}
            if sb.get("tool") and sb["tool"] == name and _ok(result):
                rt = sb.get("resource_type")
                if rt is None or rt.lower() in str(result).lower():
                    ctx.ledger.set_obligation(oid, SATISFIED, note="matched %s" % name,
                                              event_id="step-%d" % ctx.step)
        # 2) propagate workflow obligations whose requires are all satisfied
        for oid, ob in ctx.ledger.obligations.items():
            if ob.get("kind") == "workflow" and ob.get("state") not in (SATISFIED, WAIVED):
                if not ctx.ledger.pending_prerequisites(ob.get("requires", [])):
                    ctx.ledger.set_obligation(oid, SATISFIED, note="prerequisites met")
        return None


def _name(action):
    if not isinstance(action, dict):
        return ""
    if action.get("type") == "final":
        return "final"
    return action.get("tool") or action.get("action") or ""


def _final_cp(contract):
    for cp in contract.commit_points:
        if cp.get("action") in ("final", "final_answer", "final_clinical_decision"):
            return cp
    return None


def _ok(result):
    s = str(result).lower()
    return "error" not in s and "[vlm_api_error]" not in s
