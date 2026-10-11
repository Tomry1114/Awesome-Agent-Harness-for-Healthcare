"""Capability — Reference Integrity (self-opened gap kind: fabricated_reference).

An agent's COMMIT (create/submit) carries resource references (order.encounter, message.recipient, requester,
...). When a reference points at a resource the agent NEVER retrieved / that does not exist, the whole commit
fails at the substrate (e.g. FHIR HAPI-1094 "Resource X not found") and the order/message never lands. That is
a PROCESS/plumbing gap the harness may close ORACLE-BLIND: a reference is legitimate only if it is GROUNDED in
the agent's OWN retrieved evidence or the PUBLIC task spec (goal/context) -- never gold. CONTENT-NEUTRAL: it
never changes WHAT to order/say, only fixes the reference plumbing (which encounter; whether to keep an
invented recipient). GENERAL: the core is substrate-agnostic; the adapter declares which arg paths are
references (`reference_paths`), each with resource type, required?, and on_ungrounded strategy.

An optional top-level reference whose spec says on_ungrounded=strip is removed from the args in place
(auto-repair, recorded on the ledger); every other ungrounded reference is DETECTED and answered with REVISE
carrying the exact grounded fix in the feedback. Fires only when the substrate declares `reference_paths`.
Deterministic; no judge.
"""
import re
from ..capability import Capability
from .. import decision as D

_REF_RE = re.compile(r"^([A-Z][A-Za-z]+)/([A-Za-z0-9\-.]+)$")
# ids that are obviously placeholders, never a real retrieved resource
_PLACEHOLDERS = {"current", "unknown", "new", "temp", "tbd", "example", "none", "null", "self", "me",
                 "active", "latest", "recent", "default", "placeholder"}


def _get_path(obj, dotted):
    cur = obj
    for k in (dotted or "").split("."):
        if isinstance(cur, dict):
            cur = cur.get(k)
        else:
            return None
    return cur


class ReferenceIntegrity(Capability):
    name = "reference_integrity"

    def before_action(self, action, ctx):
        sem = ctx.sem
        if not sem or not sem.is_commit():
            return None
        specs = (ctx.manifest or {}).get("reference_paths") or []
        if not specs:
            return None
        args = (action or {}).get("args") or {}
        if not isinstance(args, dict):
            return None
        gtext, active = self._grounding(ctx)
        repaired, revise = [], []                     # strip-able optional refs are AUTO-REPAIRED in place
        for spec in specs:
            path = spec.get("path", "")
            val = str(_get_path(args, path) or "").strip()
            if not val:
                continue
            m = _REF_RE.match(val)
            if not m:
                continue                              # not a typed reference -> ignore
            rtype, rid = m.group(1), m.group(2)
            if self._grounded(val, rid, gtext, active):
                continue
            # ungrounded reference. Optional + on_ungrounded=strip -> AUTO-REPAIR: remove it so the commit lands.
            if spec.get("on_ungrounded") == "strip" and not spec.get("required", False) and "." not in path:
                args.pop(path, None)                  # mutate the shared action -> dispatched payload is clean
                repaired.append({"path": path, "value": val})
            else:
                revise.append((path, val, self._fix_hint(spec, rtype)))
        # record auto-repairs on the ledger so run.py can log a `reference_repaired` audit event
        if repaired:
            try:
                cur = getattr(ctx.ledger, "_reference_repairs_pending", None) or []
                cur.extend(repaired)
                setattr(ctx.ledger, "_reference_repairs_pending", cur)
            except Exception:
                pass
        if not revise:
            return None                               # all bad refs auto-stripped -> proceed with clean args
        fb = "; ".join("`%s` = %s is not a resource you retrieved -> %s" % (p, v, h) for p, v, h in revise)
        return self._decide(
            D.REVISE, rule_id="fabricated_reference", reason_code="fabricated_reference", deterministic=True,
            reason="commit references ungrounded resource(s): %s" % ", ".join(v for _, v, _ in revise),
            feedback=("This commit will FAIL to land: %s. Fix the reference(s) first -- one invalid reference "
                      "makes the whole action bounce." % fb),
            extra={"ungrounded_refs": [v for _, v, _ in revise],
                   "auto_stripped": [r["value"] for r in repaired]})

    def _grounded(self, ref, rid, gtext, active):
        if active and (ref == str(active) or rid == str(active).split("/")[-1]):
            return True
        if rid.lower() in _PLACEHOLDERS:
            return False                              # placeholder id -> never grounded
        return rid in gtext                           # the bare id appears in public input or retrieved evidence

    def _grounding(self, ctx):
        """Concatenated ORACLE-BLIND grounding text: the PUBLIC task spec (goal/context) + the agent's OWN
        retrieved evidence + the raw output of its reads. The active subject id is returned separately."""
        active = None
        try:
            active = ctx.ledger.subject_id()
        except Exception:
            pass
        parts = [self._public_text(ctx)]
        try:
            # ALL retrieved evidence, not only VALIDATED: a resource id the agent read ANYWHERE is grounded
            # (VALIDATED-only would flag genuinely-retrieved practitioners as fabricated).
            for e in ctx.ledger.evidence:
                parts.append(str(e.get("value_full") or e.get("value") or ""))
                if e.get("subject_id"):
                    parts.append(str(e.get("subject_id")))
        except Exception:
            pass
        try:
            for o in getattr(ctx.ledger, "observations", None) or []:   # raw tool-output text of every read act
                if o.get("content"):
                    parts.append(str(o.get("content")))
        except Exception:
            pass
        return ("\n".join(parts), active)

    def _public_text(self, ctx):
        parts = []
        pol = getattr(ctx, "policy", None)
        task = pol.get("_task") if isinstance(pol, dict) else None
        if isinstance(task, dict):
            parts += [str(task.get("goal") or ""), str(task.get("context") or "")]
        c = getattr(ctx, "contract", None)
        if c is not None:
            parts.append(str(getattr(c, "goal", "") or ""))
        return "\n".join(parts)

    def _fix_hint(self, spec, rtype):
        rule = spec.get("on_ungrounded", "revise")
        if rule == "strip" and not spec.get("required", False):
            return "REMOVE this reference (it is optional; do not invent a %s)" % (rtype or "target")
        return "use a %s you actually retrieved, or remove this reference" % (rtype or "resource")
