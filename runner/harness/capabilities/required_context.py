"""RequiredContext (Selective Capability Amplification -- the MISSING_CONTEXT -> ACQUIRE primitive).

Before a COMMIT (create/update/submit), the harness checks the GoalContract: the commit point's requires-closure
down to leaf evidence obligations, minus what the EvidenceLedger has VALIDATED. A missing required-context unit
with a read affordance -> ACQUIRE that record read-only, so the agent decides WITH the evidence it must have.
Oracle-blind + substrate-agnostic: reads only contract obligations + ledger evidence; the resource->read mapping
comes from the obligation's own `satisfied_by`, never a benchmark name. PB is the first adapter; the same
primitive serves any stateful substrate that declares evidence obligations.
"""
import os
from ..capability import Capability
from .. import decision as D
from ..gap import detect_missing_context, missing_context_proposal, MISSING_CONTEXT


def _enabled():
    return os.environ.get("MH_REPAIR", "hard") in ("soft", "select", "full")


def _tool_names(tools):
    out = []
    for t in (tools or []):
        out.append(t if isinstance(t, str) else (t.get("name") if isinstance(t, dict) else None))
    return [x for x in out if x]


# read affordances a stateful substrate may expose (adapter-declared verbs; core does not invent substitutes)
_READ_TOOLS = ("fhir_search", "fhir_read", "read_record", "get_record")   # search (by resourceType) before read (needs an id)


class RequiredContext(Capability):
    # LAYER: AMPLIFICATION (read-only context acquisition) -- never decides the answer; raises outcome by
    # guaranteeing the decision-relevant record is gathered before an irreversible clinical commit.
    name = "required_context"

    def _missing_obligation_acquire(self, ctx, required):
        """required = [(oid,resource)]. Return an ACQUIRE decision for the first unsatisfied one, or None."""
        led = ctx.ledger
        if getattr(led, "acquire_count", 0) >= 2:
            return None
        meta = (ctx.contract.meta or {})
        read_tool = next((t for t in _READ_TOOLS if t in _tool_names(meta.get("available_tools"))), None)
        if not read_tool:
            return None
        validated = self._validated_resources(led)
        missing = [(oid, res) for (oid, res) in required if res and res not in validated]
        prop = missing_context_proposal(
            [oid for (oid, _r) in missing],
            affordance_for=lambda oid: self._affordance(oid, dict(missing).get(oid), read_tool))
        if not prop:
            return None
        na = dict(prop.affordance); na["read_only"] = True
        return self._decide(D.ACQUIRE, rule_id="required_context", reason_code="missing_required_context",
                            reason="acquire required context %r (%s) before committing"
                                   % (prop.missing_unit, na.get("args", {}).get("resourceType")),
                            extra={"next_action": na, "gap": {"type": MISSING_CONTEXT, "unit": prop.missing_unit}})

    def _all_evidence_obligations(self, c):
        out = []
        for o in (getattr(c, "evidence_obligations", []) or []):
            res = (o.get("satisfied_by") or {}).get("resource")
            if o.get("id") and res:
                out.append((o["id"], res))
        return out

    def before_final(self, answer, ctx):
        # DISABLED: deliverable acquisition is driven by the INTENT-SCOPED before_action hook on the deliverable
        # write (which the forced-deliverable path also traverses). A before_final _all_evidence_obligations
        # trigger over-fires (ignores relevance), lands AFTER the artifact is written (too late to shape it), and
        # cannot be content-scoped -- so it is not used here.
        return None

    def before_action(self, action, ctx):
        # EARLY trigger: before a COMMIT or a DELIVERABLE WRITE (so the deliverable is produced WITH the required
        # context, and read-before-action ordering is satisfied). Checks ALL task-level evidence obligations,
        # not a single commit point's requires -- the deliverable may be a FHIR create OR a written plan.
        if not _enabled():
            return None
        sem = ctx.sem
        is_commit = bool(sem and (getattr(sem, "semantic_type", None) in ("create", "update", "submit")
                                  or (hasattr(sem, "is_commit") and sem.is_commit())))
        if not is_commit or not ctx.contract:
            return None
        obligations = self._all_evidence_obligations(ctx.contract)
        if not obligations:
            return None
        # P2 INTENT-SCOPE: for a content-bearing deliverable write, restrict required context to the obligations
        # the deliverable's own clinical actions depend on (don't acquire allergies/meds for a non-medication
        # plan). Oracle-blind: reads the agent's proposed content + public goal.
        content = ((action or {}).get("args") or {}).get("content")
        if content:
            from ..engines.semantic import select_relevant_obligations
            obligations = select_relevant_obligations((ctx.contract.meta or {}).get("goal"), content,
                                                      obligations, getattr(ctx, "judge_fn", None))
        return self._missing_obligation_acquire(ctx, obligations)

    # -- helpers (pure; over contract obligations + ledger evidence) --
    def _required_evidence_obligations(self, c, sem):
        cp = c.commit_point_for(sem) or {}
        top = list(cp.get("requires") or [])
        if not top:
            return []
        idx = {}
        for o in list(getattr(c, "evidence_obligations", []) or []) + list(getattr(c, "workflow_obligations", []) or []):
            if o.get("id"):
                idx[o["id"]] = o
        seen, leaves, queue = set(), [], list(top)
        while queue:
            oid = queue.pop(0)
            if oid in seen:
                continue
            seen.add(oid)
            o = idx.get(oid)
            if not o:
                continue
            res = (o.get("satisfied_by") or {}).get("resource")
            if res:
                leaves.append((oid, res))
            for r in (o.get("requires") or []):
                queue.append(r)
        return leaves

    def _validated_resources(self, led):
        out = set()
        for e in getattr(led, "evidence", []):
            if e.get("status") == "VALIDATED" and e.get("scope_relation") != "foreign" and e.get("resource"):
                out.add(e.get("resource"))
        return out

    def _affordance(self, oid, resource, read_tool):
        if not resource:
            return None
        return {"capability": "acquire_record", "tool": read_tool,
                "args": {"resourceType": resource}, "risk": "R0"}
