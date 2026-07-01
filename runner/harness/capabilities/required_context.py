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
        """required = [(oid, resource)]. Resolve/ACQUIRE required context before a commit.

        FAIL-CLOSED is SCOPED to IRREVERSIBLE state mutations (fhir_create/submit): an unresolved required
        obligation that CANNOT be acquired (no affordance / tool unavailable / budget spent) ESCALATEs rather
        than committing an irreversible patient change without the decision-relevant record.

        A REVERSIBLE deliverable write (a scratchpad note the agent can still revise) stays BEST-EFFORT: ACQUIRE
        the context WHEN it is gatherable, but if it is genuinely ungatherable, fall through to None (allow) --
        aborting the whole run over ungatherable context for a reversible artifact is worse than producing it.
        When acquirable, ACQUIRE fires for BOTH kinds (so the deliverable is still produced WITH the context)."""
        led = ctx.ledger
        resolved = self._resolved_units(led)
        active = led.subject_id()
        unresolved = [(oid, res) for (oid, res) in required if res and res not in resolved]
        if not unresolved:
            return None                                # all required context CHECKED -> no opinion (allow)
        _sem = getattr(ctx, "sem", None)
        irreversible = bool(_sem and getattr(_sem, "effect", None) == "irreversible")

        def _fail_closed(reason_code, reason, feedback):
            # irreversible -> ESCALATE (never commit without required context); reversible -> None (best-effort)
            if irreversible:
                return self._decide(D.ESCALATE, rule_id="required_context", reason_code=reason_code,
                                    deterministic=True, reason=reason, feedback=feedback)
            return None

        if not active:
            return _fail_closed("required_context_no_subject",
                                "required context is needed but there is no active subject to bind the query to",
                                "Required patient context is missing and no subject is resolved to gather it; escalating instead of committing.")
        if getattr(led, "acquire_count", 0) >= 2:      # bounded acquisition budget, still unresolved
            return _fail_closed("required_context_budget_exhausted",
                                "required context still unresolved after the acquisition budget",
                                "Required patient context could not be gathered within budget; escalating rather than committing without it.")
        for (oid, res) in unresolved:
            req = compile_evidence_request(ctx.manifest, res, active, obligation_id=oid)
            if not req or not (req.affordance or {}).get("tool"):
                continue
            if not self._tool_available(ctx, (req.affordance or {}).get("tool")):
                continue
            na = dict(req.affordance); na["read_only"] = True
            return self._decide(D.ACQUIRE, rule_id="required_context", reason_code="missing_required_context",
                                reason="acquire required context %s (%s) before committing" % (oid, res),
                                extra={"next_action": na, "gap": {"type": MISSING_CONTEXT, "unit": oid},
                                       "evidence_unit": res, "target_entity": active})
        # unresolved required obligations exist but NONE is acquirable
        return _fail_closed("required_context_unavailable",
                            "a required evidence obligation is unresolved and no executable affordance exists to acquire it",
                            "Required patient context cannot be gathered (no available tool for it); escalating instead of committing without it.")

    def _all_evidence_obligations(self, c):
        out = []
        for o in (getattr(c, "evidence_obligations", []) or []):
            res = (o.get("satisfied_by") or {}).get("resource")
            if o.get("id") and res:
                out.append((o["id"], res))
        return out

    def _scoped_evidence_obligations(self, ctx):
        """Only the EVIDENCE obligations the MATCHED commit-point actually requires (transitively through
        workflow obligations) -- NOT every declared obligation. A ServiceRequest create matches only the
        generic irreversible-write invariant (requires []), so it does not demand medication-safety evidence;
        a MedicationRequest create matches medication_safety and DOES. Mirrors obligation_lifecycle's commit
        resolution so amplification and enforcement AGREE on what THIS specific commit requires."""
        c = ctx.contract
        cp = c.commit_point_for(ctx.sem) if (c and ctx.sem) else None
        if not cp:
            return []
        ev_map = {o["id"]: (o.get("satisfied_by") or {}).get("resource")
                  for o in (getattr(c, "evidence_obligations", []) or []) if o.get("id")}
        wf_map = {o["id"]: list(o.get("requires") or [])
                  for o in (getattr(c, "workflow_obligations", []) or []) if o.get("id")}
        seen, stack, leaves = set(), list(cp.get("requires") or []), []
        while stack:                                   # flatten required ids -> evidence-obligation leaves
            rid = stack.pop()
            if rid in seen:
                continue
            seen.add(rid)
            if rid in ev_map:
                leaves.append(rid)
            elif rid in wf_map:
                stack.extend(wf_map[rid])
        return [(rid, ev_map[rid]) for rid in leaves if ev_map.get(rid)]

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
        # SCOPE to what THIS commit-point requires (not every declared obligation) -- so an unrelated create
        # (e.g. a ServiceRequest imaging order) is not blocked demanding medication-safety evidence.
        obligations = self._scoped_evidence_obligations(ctx)
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
