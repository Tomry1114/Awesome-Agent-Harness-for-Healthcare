"""RequiredContext (Commit A) -- MISSING_CONTEXT -> ACQUIRE, driven by the ADAPTER-COMPILED evidence affordance.

Before a COMMIT / deliverable write, the harness checks the task's evidence obligations against the ledger's
RESOLVED evidence (an obligation is resolved once its resource has been CHECKED for the active subject --
PRESENT *or* ABSENT; a confirmed empty allergy list resolves the allergy-check just as a positive one does).
An unresolved obligation is acquired through `compile_evidence_request` so the query uses the adapter's exact
tool + parameter (e.g. AllergyIntolerance?patient=Patient/<id>, never a hard-coded `subject`). Oracle-blind +
substrate-agnostic: reads only contract obligations + ledger evidence + the adapter manifest.
"""
import os
from ..capability import Capability
from .. import decision as D
from ..gap import MISSING_CONTEXT
from ..evidence_state import is_resolved
from ..adapter_compiler import compile_evidence_request


def _enabled():
    return os.environ.get("MH_REPAIR", "hard") in ("soft", "select", "full")


class RequiredContext(Capability):
    # LAYER: AMPLIFICATION (read-only context acquisition) -- never decides the answer; raises outcome by
    # guaranteeing the decision-relevant record is CHECKED before an irreversible clinical commit.
    name = "required_context"

    def _resolved_units(self, led):
        """Resource evidence_units the ledger has CHECKED for the ACTIVE subject (PRESENT or ABSENT). C3.1
        STRICT: the read must be scope_relation == "matched" AND its subject_id must equal the active subject
        -- a not-foreign-but-unresolved-subject empty read must NOT close a required-patient obligation."""
        active = led.subject_id()
        out = set()
        for e in getattr(led, "evidence", []):
            if (e.get("resource") and is_resolved(e.get("evidence_state"))
                    and e.get("scope_relation") == "matched" and e.get("subject_id") == active):
                out.add(e.get("resource"))
        return out

    def _tool_available(self, ctx, tool):
        """Admission: the adapter-compiled tool MUST exist in the task's available tools. Unknown tool list ->
        do not over-block (return True). Prevents RequiredContext emitting an ACQUIRE the env cannot run."""
        meta = (ctx.contract.meta if getattr(ctx, "contract", None) is not None else None) or {}
        tools = meta.get("available_tools")
        if not tools:
            return True
        names = set()
        for t in tools:
            names.add(t if isinstance(t, str) else (t.get("name") if isinstance(t, dict) else None))
        return tool in names

    def _missing_obligation_acquire(self, ctx, required):
        """required = [(oid, resource)]. ACQUIRE the first UNRESOLVED one via the adapter-compiled affordance."""
        led = ctx.ledger
        if getattr(led, "acquire_count", 0) >= 2:      # bounded acquisition budget
            return None
        resolved = self._resolved_units(led)
        active = led.subject_id()
        if not active:
            return None
        for (oid, res) in required:
            if not res or res in resolved:
                continue
            req = compile_evidence_request(ctx.manifest, res, active, obligation_id=oid)
            if not req or not (req.affordance or {}).get("tool"):
                continue                               # adapter declares no affordance -> not acquirable here
            if not self._tool_available(ctx, (req.affordance or {}).get("tool")):
                continue                               # C3.1 fix 6: adapter tool absent from available_tools -> a dead ACQUIRE; skip (never emit an unexecutable acquisition)
            na = dict(req.affordance); na["read_only"] = True
            return self._decide(D.ACQUIRE, rule_id="required_context", reason_code="missing_required_context",
                                reason="acquire required context %s (%s) before committing" % (oid, res),
                                extra={"next_action": na, "gap": {"type": MISSING_CONTEXT, "unit": oid},
                                       "evidence_unit": res, "target_entity": active})
        return None

    def _all_evidence_obligations(self, c):
        out = []
        for o in (getattr(c, "evidence_obligations", []) or []):
            res = (o.get("satisfied_by") or {}).get("resource")
            if o.get("id") and res:
                out.append((o["id"], res))
        return out

    def before_final(self, answer, ctx):
        # DISABLED (see history): before_final over-fires (not intent-scoped, lands after the artifact is
        # written). Deliverable acquisition is driven by the intent-scoped before_action hook below.
        return None

    def before_action(self, action, ctx):
        # EARLY trigger: before a COMMIT or a DELIVERABLE WRITE, acquire the required context the deliverable's
        # own clinical actions depend on, so it is produced WITH that context.
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
        # INTENT-SCOPE: for a content-bearing deliverable write, restrict to the obligations the deliverable's
        # own clinical actions depend on. Oracle-blind: reads the agent's proposed content + public goal.
        content = ((action or {}).get("args") or {}).get("content")
        if content:
            from ..engines.semantic import select_relevant_obligations
            obligations = select_relevant_obligations((ctx.contract.meta or {}).get("goal"), content,
                                                      obligations, getattr(ctx, "judge_fn", None))
        return self._missing_obligation_acquire(ctx, obligations)
