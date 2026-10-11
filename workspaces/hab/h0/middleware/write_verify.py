# -*- coding: utf-8 -*-
"""HealthAdminBench H0 middleware: write verification after a form write.

After every fill/select the shell reads the written control back and puts the receipt in the action's result
(`write_receipt`, runner/environments.py). Two nudges read that receipt; neither ever supplies a value:
  write_not_landed  the value did not persist in the target control;
  write_misroute    it landed, but in a control whose field concept is not the one the agent declared
                    (args.field_intent), while a required (or the only) empty control of that concept exists.
They share one bound, MAX_WRITEFIX, and both unlock the repair budget (PostRevision). Both stay silent once the tool
budget is spent and on harness reads. The field-concept ontology below is this
middleware's own content.
"""
import re

from workspace_api import is_control_complete, norm_value
from workspace_api import PostRevision

MAX_WRITEFIX = 4


# ---- field ontology + intent-vs-control-semantics misroute reconciler (generic, no gold) ----
# GENERIC UI field aliases (canonical concept -> public synonyms). NOT dataset heuristics: never "value looks
# like AET\\d+ -> Member ID". Only accessible-name semantics of controls / agent-declared field intents.
FIELD_ALIASES = {
    "member_id": ["member id", "member number", "member no", "subscriber id", "subscriber number",
                  "insurance id", "policy number", "member"],
    "date_of_birth": ["date of birth", "dob", "birth date", "birthdate"],
    "patient_search": ["patient search", "search patient", "find patient", "member search", "lookup patient"],
    "request_type": ["request type", "authorization type", "service type", "auth type", "review type"],
    "diagnosis": ["diagnosis", "diagnosis code", "icd", "dx code"],
    "cpt": ["cpt", "procedure code", "hcpcs", "service code", "cpt/hcpcs"],
    "provider": ["provider", "npi", "servicing provider", "requesting provider", "rendering provider"],
    "clinical_justification": ["clinical justification", "medical necessity", "clinical indication",
                               "justification", "clinical note", "clinical rationale"],
    "fax_number": ["fax number", "fax", "recipient fax"],
    "patient_name": ["patient name", "member name", "last name", "first name"],
}


def _field_concept(text):
    """Map a control accessible-name / agent field intent to a canonical concept, or None if unknown. Longest
    matching synonym wins so 'member search' -> patient_search (not member_id)."""
    n = norm_value(text)
    if not n:
        return None
    best, best_len = None, 0
    for concept, syns in FIELD_ALIASES.items():
        for syn in syns:
            if syn in n and len(syn) > best_len:
                best, best_len = concept, len(syn)
    return best


def semantic_field_match(a, b):
    """Do two field labels/intents refer to the SAME field concept? Concept match if both resolve; else a
    conservative token-subset fallback."""
    ca, cb = _field_concept(a), _field_concept(b)
    if ca and cb:
        return ca == cb
    ta = set(norm_value(a).split()) - {"the", "a", "of", "id", "no", "number"}
    tb = set(norm_value(b).split()) - {"the", "a", "of", "id", "no", "number"}
    return bool(ta) and bool(tb) and (ta <= tb or tb <= ta)


_LOOKUP_SECTION_RE = re.compile(r"\b(search|lookup|look ?up|find (?:a )?(?:patient|member)|eligibility|member search|patient search)\b", re.I)


def _is_lookup_section(section):
    """A form section that is a LOOKUP/search surface (patient/member search, eligibility) -- distinct from
    the payload form where the checkpoint reads the value. Public DOM section semantics; no gold."""
    return bool(_LOOKUP_SECTION_RE.search(str(section or "")))


def find_misroute(field_intent, target_label, controls, target_section=None):
    """The agent typed a value (intent=field_intent) and it LANDED in target_label. If the target does not
    match the intent semantically AND there is a REQUIRED, EMPTY, VISIBLE control that DOES match the intent,
    return that control (a misroute the agent should correct). Conservative: needs an identifiable intent
    concept and at least one matching empty target; returns None otherwise (never guesses). No gold, no value."""
    ci = _field_concept(field_intent)
    if ci is None:
        return None                                  # unknown intent -> do not reason
    _landed_lookup = _is_lookup_section(target_section)
    if semantic_field_match(field_intent, target_label) and not _landed_lookup:
        return None                                  # landed in the right kind of control -> not a misroute
    cands = [c for c in (controls or [])
             if isinstance(c, dict) and not is_control_complete(c) and c.get("visible", True)
             and not c.get("disabled") and _field_concept(c.get("label")) == ci]
    if _landed_lookup:
        # SAME-NAME field problem: the value landed in a LOOKUP/search section; the PAYLOAD field is the same
        # concept but OUTSIDE a lookup section. Prefer a matching empty control not in a lookup section.
        payload = [c for c in cands if not _is_lookup_section(c.get("section"))]
        if payload:
            cands = payload
    if not cands:
        return None                                  # no matching empty control -> stay silent
    # prefer a REQUIRED same-concept field. A mis-route must NOT be asserted toward an OPTIONAL same-name field
    # while a required one matches: required matches are selected first (not merely sorted ahead), shortest label wins.
    req = sorted([c for c in cands if c.get("required")], key=lambda c: len(str(c.get("label") or "")))
    if req:
        return req[0]
    # no required match -> act ONLY on a UNIQUE candidate (high confidence); ambiguous (>1) -> stay silent.
    return cands[0] if len(cands) == 1 else None


class WriteVerify(object):
    def __init__(self, task):
        self.rounds = 0          # write_not_landed + write_misroute refusals so far (MAX_WRITEFIX)

    def after_tool(self, ctx, action, result):
        if ctx.budget_spent_at_turn_start:
            return None
        if self.rounds >= MAX_WRITEFIX:
            return None
        if not (isinstance(result, dict) and result.get("write_receipt")) or action.get("_harness_acquire"):
            return None
        wr = result["write_receipt"]
        intent = (action.get("args") or {}).get("field_intent")
        if wr.get("landed") is False:                 # the write did not persist
            self.rounds += 1
            fld = wr.get("field") or wr.get("control_label") or wr.get("ref")
            ctx.emit({"step": ctx.step, "event_type": "write_not_landed", "round": self.rounds,
                      "ref": wr.get("ref"), "field": wr.get("field"),
                      "attempted": wr.get("attempted"), "observed": wr.get("observed"), "status": "ok"})
            return PostRevision("write_verify",
                                "Your last entry did not persist in the target control (%s). Re-select the intended "
                                "control by its exact ref and re-enter the value, or inspect the form for the correct "
                                "control." % fld, "write_not_landed")
        if intent:                                    # landed, but maybe in the WRONG control
            try:
                cand = find_misroute(intent, wr.get("control_label") or "", ctx.controls(),
                                     target_section=wr.get("control_section"))
            except Exception:
                cand = None
            if cand is not None:
                self.rounds += 1
                ctx.emit({"step": ctx.step, "event_type": "write_misroute", "round": self.rounds,
                          "field_intent": intent, "landed_control": wr.get("control_label"),
                          "landed_ref": wr.get("ref"), "correct_ref": cand.get("ref"),
                          "correct_label": cand.get("label"), "status": "ok"})
                return PostRevision("misroute",
                                    "Your value landed in the '%s' control, but your stated field intent was '%s'. The "
                                    "required '%s' control (ref=%s) is still empty -- enter the value there instead. "
                                    "The harness does not supply the value; you already have it."
                                    % (wr.get("control_label") or wr.get("ref"), intent,
                                       cand.get("label"), cand.get("ref")), "write_misroute")
        return None
