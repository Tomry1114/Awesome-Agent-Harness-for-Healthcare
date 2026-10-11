"""Capability — Commitment Care Action Fulfillment (self-opened gap kind: committed_not_done).

The agent's OWN deliverable states it is performing an environment-supported action (placing an order,
submitting a form, sending a message) but it NEVER executed a corresponding commit. This is a PROCESS gap the
harness may close: the reference is the agent's own output + its own executed actions -- ORACLE-BLIND (never
gold/checkpoint/reference). CONTENT-NEUTRAL: it never says WHICH action to take or whether it is clinically
right -- only that a stated intent-to-act was not carried out. GENERAL: works on any substrate that declares
commit affordances (create/update/submit actions in its manifest).

Complements verify_commit.unresolved_operational_commit (a write ATTEMPTED but failed): this catches a commit
that was NEVER attempted. Fail-safe OFF when no semantic judge is configured (like the other semantic checks).
"""
import json
import re
from workspace_api import Capability, decision as D
from workspace_api import canonicalize

_PROMPT = (
    "You audit whether an agent DID what its own deliverable says it is doing. You are given the agent's "
    "DELIVERABLE, the ENVIRONMENT ACTION TYPES it can execute here, and how many of each it ACTUALLY "
    "executed.\n\n"
    "Return every action the deliverable FIRMLY and UNCONDITIONALLY states the agent IS performing NOW "
    "(e.g. 'ordering X', 'placing Y', 'submitting Z', 'sending W') that maps to an environment action type "
    "whose EXECUTED COUNT is 0 (i.e. it was never carried out).\n\n"
    "STRICT EXCLUSIONS -- never flag: conditional or hypothetical statements ('if...', 'consider', 'may "
    "need', 'could', 'should be considered'); future or EXTERNAL follow-up ('follow up with cardiology in 2 "
    "weeks', patient instructions, monitoring plans); any action with no matching environment action type; "
    "and any action type whose executed count is >= 1. NEVER flag CONTINUING/MAINTAINING an existing item "
    "('continue X', 'maintain Y', 'hold Z') -- that is not a NEW action; and NEVER flag SAVING/DOCUMENTING the "
    "note ('saved to disk', 'documented', 'management plan saved', 'assessment completed') -- that IS the "
    "deliverable, not an executable order. Only a NEW order/prescription/referral/appointment/message counts. "
    "Do NOT judge whether an action is clinically correct "
    "or advisable -- ONLY whether a firm stated intent-to-act was actually executed.\n\n"
    "DELIVERABLE:\n{deliverable}\n\n"
    "ENVIRONMENT ACTION TYPES (executed count):\n{executed}\n\n"
    "Reply with JSON only: {{\"unmet\": [\"<short action phrase>\", ...]}}. Empty list if every firmly "
    "stated action was executed, or there are no firm action commitments."
)

_MUT = ("create", "update", "submit")


class CommitmentExecution(Capability):
    name = "commitment_execution"

    def __init__(self):
        self._cache = {}          # answer-hash -> decision (bound judge calls to DISTINCT deliverables)

    def before_final(self, answer, ctx):
        # Budgeted draw: ctx.budgeted_judge() counts against the max_semantic_checks budget.
        # self._cache only deduplicates per deliverable; it is not a budget.
        _bj = ctx.budgeted_judge()
        if _bj is None or not answer:
            return None
        manifest = ctx.manifest or {}
        # environment commit affordances (content-free action-type names) declared by the substrate manifest
        affs = sorted({(a.get("resource") or a.get("semantic_type"))
                       for a in (manifest.get("actions") or [])
                       if a.get("semantic_type") in _MUT and (a.get("resource") or a.get("semantic_type"))})
        if not affs:
            return None            # substrate has no executable commits -> nothing to enforce
        key = (hash(str(answer)), tuple(affs))
        if key in self._cache:
            return self._cache[key]
        # executed count per affordance type (only VERIFIED create/update/submit commits count as "done")
        counts = {a: 0 for a in affs}
        for c in ctx.ledger.commit_history:
            if c.get("verified") is True and c.get("semantic_type") in _MUT:
                try:
                    sem = canonicalize(c.get("action"), manifest)
                    r = getattr(sem, "resource", None) or c.get("semantic_type")
                except Exception:
                    r = c.get("semantic_type")
                if r in counts:
                    counts[r] += 1
        executed = "\n".join("- %s: %d executed" % (a, counts[a]) for a in affs)
        prompt = _PROMPT.format(deliverable=str(answer)[:4000], executed=executed)
        try:
            raw = _bj(prompt)
        except Exception:
            self._cache[key] = None
            return None
        if raw is None:                      # budget exhausted: same path as a failed judge call -- no intervention.
            self._cache[key] = None          # This is a completeness check, not hard safety (hard safety is in verify_commit,
            return None                      # where an exhausted budget leads to ESCALATE).
        unmet = [u for u in _parse_unmet(raw) if _executable(u)]
        # DETERMINISTIC reconciliation: drop any commitment the agent ACTUALLY executed (content/type
        # match against its OWN commit ledger). committed_not_done means NEVER attempted; a matching
        # executed commit (verified or not) is not this gap. Kills the LLM's already-done false positives.
        descriptors, executed_types = _executed_commits(ctx.ledger)
        unmet = [u for u in unmet if not _already_executed(u, descriptors, executed_types)]
        if not unmet:
            self._cache[key] = None
            return None
        lst = "; ".join(unmet[:5])
        dec = self._decide(
            D.REVISE, rule_id="committed_not_done", reason_code="committed_not_done", deterministic=False,
            reason="deliverable commits to action(s) never executed: %s" % lst,
            feedback=("Your deliverable states you are performing: %s -- but no corresponding action was "
                      "executed in this environment. Either perform the action(s) now, or revise the "
                      "deliverable so it does not report doing something that was not done." % lst),
            extra={"unmet_commitments": unmet[:10]})
        self._cache[key] = dec
        return dec



_EXEC = ("order", "plac", "prescrib", "submit", "send", "refer", "schedul", "book", "administer")
_NON_EXEC = ("continue", "maintain", "keep ", "hold ", "monitor", "saved", "save ", "document", " note",
             "to disk", "management plan", "assessment", "complete", "review", "ensure", "recommend")


def _executable(phrase):
    """A flagged commitment must describe a NEW environment-executable action (create/order/submit/send/...),
    not a continuation/documentation/plan -- this removes committed_not_done false positives."""
    p = " " + str(phrase or "").lower() + " "
    if any(x in p for x in _NON_EXEC):
        return False
    return any(x in p for x in _EXEC)

def _parse_unmet(raw):
    if not raw:
        return []
    data = raw if isinstance(raw, dict) else None
    if data is None:
        m = re.search(r"\{.*\}", str(raw), re.DOTALL)
        if not m:
            return []
        try:
            data = json.loads(m.group())
        except Exception:
            return []
    u = data.get("unmet") if isinstance(data, dict) else None
    if not isinstance(u, list):
        return []
    return [str(x).strip() for x in u if str(x).strip()]


# --- deterministic already-executed reconciliation (kills committed_not_done "already done" false positives) --
_STOP = {"the", "for", "and", "with", "a", "an", "of", "in", "to", "new", "order", "orders", "ordering",
         "place", "placing", "placed", "schedule", "scheduling", "scheduled", "submit", "submitting",
         "send", "sending", "refer", "referral", "service", "request", "requests", "appointment", "appt",
         "booking", "book", "follow", "followup", "follow-up", "up", "consultation", "consult", "test",
         "testing", "panel", "screen", "screening", "level", "levels", "assessment", "plan", "management",
         "patient", "gi", "interim", "urgent", "routine", "care", "visit", "recheck"}


def _tokens(text):
    return {t for t in re.findall(r"[a-z0-9][a-z0-9\-]+", str(text or "").lower()) if t not in _STOP}


def _infer_type(text):
    t = str(text or "").lower().replace("_", " ")
    if "appointment" in t:
        return "appointment"
    if "communication" in t or "message" in t or "notify" in t:
        return "communication"
    if "medication" in t or "prescri" in t:
        return "medication"
    if "service request" in t or "order" in t or "refer" in t or "lab" in t or "imaging" in t \
            or "biopsy" in t or "panel" in t or "screen" in t:
        return "service_request"
    return None


def _executed_commits(ledger):
    """(descriptor token-sets, executed resource-type set) for every create/update/submit the agent ATTEMPTED
    (verified or not -- an attempted commit is not committed_not_done). Read from the agent's own ledger."""
    descriptors, types = [], set()
    for c in (getattr(ledger, "commit_history", None) or []):
        if c.get("semantic_type") not in _MUT:
            continue
        a = c.get("descriptor")                       # the ACTUAL action dict {tool,args}; commit_history["action"]
        if not isinstance(a, dict):                   # is the capability NAME string, not the action -> use descriptor
            a = c.get("action") if isinstance(c.get("action"), dict) else {}
        tool = (a.get("tool") or "") if isinstance(a, dict) else ""
        args = a.get("args") if isinstance(a, dict) else {}
        parts = [tool]
        if isinstance(args, dict):
            for k in ("code_display", "code_text", "display", "text", "reason_display", "reason_text",
                      "description", "name", "type", "service", "medication", "code"):
                if args.get(k):
                    parts.append(str(args[k]))
        descriptors.append(_tokens(" ".join(str(p) for p in parts)))
        rt = _infer_type(tool)
        if rt:
            types.add(rt)
    return descriptors, types


def _already_executed(phrase, descriptors, executed_types):
    """True iff the committed action was actually carried out: (a) a DISTINCTIVE token (digit-bearing or
    len>=6) or >=2 tokens overlap an executed commit's descriptor (precise, per-order), OR (b) a purely
    GENERIC process commit (appointment/message with no specific clinical content) whose type was executed."""
    pt = _tokens(phrase)
    if not pt:
        rt = _infer_type(phrase)
        return bool(rt in ("appointment", "communication") and rt in executed_types)
    distinctive = {t for t in pt if any(ch.isdigit() for ch in t) or len(t) >= 6}
    for d in descriptors:
        shared = pt & d
        if (shared & distinctive) or len(shared) >= 2:
            return True
    if not distinctive:
        rt = _infer_type(phrase)
        if rt in ("appointment", "communication") and rt in executed_types:
            return True
    return False
