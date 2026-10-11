# -*- coding: utf-8 -*-
"""HealthAdminBench H0 middleware: the collection coverage nudge (host before_tool, after the attachment check).

At a terminal commit on a page that has a diagnosis or CPT field, if the agent OBSERVED more distinct codes of that
concept (in what it was shown, ctx.observed_history) than the form holds (eval_state's diagnosisCodes / cptCodes),
remind it to go back and add the ones it saw. NEVER supplies a code and NEVER uses a gold count; the trigger is
oracle-blind (observed count > present count) and the text names only the concept and the counts. At most
MAX_COLLECTION times; the refusal does not unlock the repair budget (Nudge). Harness reads are not checked.
"""
import re

from workspace_api import CPT_HCPCS_RE, ICD10_RE
from workspace_api import is_terminal_action
from workspace_api import Nudge

MAX_COLLECTION = 2

# billing/clinical code patterns shared with GUI recovery's collection executor (adapters/hab/common.py)
_CONCEPT_PAT = {"diagnosis": ICD10_RE, "cpt": CPT_HCPCS_RE, "procedure_codes": CPT_HCPCS_RE}

# concept -> substrings that identify a control able to RECEIVE that collection.
CONCEPT_FIELD_HINTS = {"diagnosis": ("diagnos", "icd"),
                       "cpt": ("cpt", "hcpcs", "procedure", "service"),
                       "procedure_codes": ("cpt", "hcpcs", "procedure", "service"),
                       "supporting_documents": ("attach", "document", "upload")}


def terminal_commit(ctx, action):
    """(is the action a terminal commit, the live controls it was classified against: ctx.controls(), the shell's
    read-only control model of the current page)."""
    if action.get("tool") not in ("submit", "click"):
        return False, None
    try:
        controls = ctx.controls()
    except Exception:
        controls = []
    try:
        return bool(is_terminal_action(action.get("tool"), (action.get("args") or {}).get("ref"), controls)), controls
    except Exception:
        return False, controls


def _norm(v):
    return re.sub(r"[\s\-\.]+", "", str(v or "")).lower()


def observed_codes(text, concept):
    """The distinct codes of a concept present in OBSERVED text (ICD for diagnosis, 5-digit CPT). Oracle-blind."""
    pat = _CONCEPT_PAT.get(concept)
    if not pat:
        return set()
    return {m.strip() for m in pat.findall(str(text or "")) if m and m.strip()}


def concept_affordance(controls, concept):
    """Can the CURRENT page receive this collection? True when some live control's ref/label/name/testid
    names the concept. Used to keep an 'add the codes you saw' reminder from firing on a page with no such
    field (e.g. the payer login screen, where the agent would spend steps on absent controls)."""
    hints = CONCEPT_FIELD_HINTS.get(concept) or ()
    if not hints:
        return True                      # unknown concept -> do not silence on a guess
    for c in (controls or []):
        if not isinstance(c, dict):
            continue
        blob = " ".join(str(c.get(k) or "") for k in ("ref", "label", "name", "testid", "id")).lower()
        if any(h in blob for h in hints):
            return True
    return False


def coverage_nudge(observed_text, present_values, concept):
    """Coverage reminder. Fires only when the agent OBSERVED strictly more distinct codes than are in the form
    (oracle-blind; no gold count). Returns a nudge that names the CONCEPT and the counts -- never a specific
    code (the agent saw them; it must go back and add them itself). None when the form already covers what was
    seen."""
    obs = observed_codes(observed_text, concept)
    pres = {_norm(p) for p in (present_values or []) if str(p or "").strip()}
    seen_missing = [c for c in obs if _norm(c) not in pres]
    if not seen_missing or len(obs) <= len(pres):
        return None
    return ("The %s section of the authorization form has %d code(s), but you saw %d %s code(s) in the "
            "records. Go back and add the %s codes you already observed -- do not invent any." %
            (concept, len(pres), len(obs), concept, concept))


def _dig_codes(o, key):
    """Every code listed under a key named `key` (any case, any depth) of a portal state."""
    acc = []
    if isinstance(o, dict):
        for k, v in o.items():
            if str(k).lower() == key and isinstance(v, list):
                for it in v:
                    acc.append(str(it.get("code") if isinstance(it, dict) else it))
            else:
                acc += _dig_codes(v, key)
    elif isinstance(o, list):
        for it in o:
            acc += _dig_codes(it, key)
    return acc


class CollectionNudge(object):
    def __init__(self, task):
        self.rounds = 0

    def before_tool(self, ctx, action):
        if self.rounds >= MAX_COLLECTION or action.get("_harness_acquire"):
            return None
        terminal, controls = terminal_commit(ctx, action)
        if not terminal:
            return None
        try:
            state = ctx.persisted_state()
            state = {} if state is None else state
            observed = " ".join(ctx.observed_history)[-20000:]
            # A reminder to ADD codes is only actionable on a page that HAS the field. Elsewhere (e.g. the payer login
            # screen) it would send the agent after controls that do not exist there. The gap persists; the nudge simply waits
            # for the right page.
            msg = None
            for concept, key in (("diagnosis", "diagnosiscodes"), ("cpt", "cptcodes")):
                if not concept_affordance(controls, concept):
                    continue
                msg = coverage_nudge(observed, _dig_codes(state, key), concept)
                if msg:
                    break
        except Exception:
            msg = None
        if not msg:
            return None
        self.rounds += 1
        ctx.emit({"event_type": "collection_nudge", "status": "ok", "step": ctx.step, "round": self.rounds})
        return Nudge("collection", msg, "collection_nudge", "REVISE")
