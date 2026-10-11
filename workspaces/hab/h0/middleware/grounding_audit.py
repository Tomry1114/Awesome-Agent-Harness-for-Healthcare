"""HAB grounding / public-requirement COVERAGE audit.

The harness may AUDIT whether the agent's authored note references the salient facts the agent ITSELF observed
(denial/remark codes, ICD, CPT, auth/confirmation/referral IDs, fax numbers) and NUDGE it to supplement -- it
REPLAYS agent-observed values (with provenance) as a reminder. This does surface the concrete value, which is sound
only because the agent already observed it; it may NEVER judge the note's clinical quality (that stays the rubric's
job). Oracle-blind: salient facts come ONLY from text the agent OBSERVED in its own trajectory, never from gold.
HAB-scoped.

Generic mechanism (extract + gap + feedback); no gold, no benchmark names in the logic. GroundingAudit at the bottom
is how the shell (runner/middleware_host.py) calls it: host before_final, after ObligationNudge in the manifest.
"""
import re
import workspace_api as _W
from workspace_api import Nudge
# ICD-10 + CPT/HCPCS patterns shared with collection (runner/adapters/hab/common.py) so they can't drift
from workspace_api import ICD10_RE, CPT_HCPCS_RE

# Structured PUBLIC identifiers/codes. Deliberately conservative patterns so free text does not create noise.
# CLINICAL / BILLING codes a triage / appeal note is expected to reference (the 'note' slot).
_NOTE_CODE_PATTERNS = [
    re.compile(r"\b(?:CO|PR|OA|CARC|RARC)[- ]?\d{1,3}\b", re.I),   # adjustment reason code (CO-50)
    re.compile(r"\b[NM]\d{2,3}\b"),                                # remark code (N386, M-codes)
    ICD10_RE,                                                      # ICD-10 (H35.32)
    CPT_HCPCS_RE,                                                  # CPT (67028) OR HCPCS (J2778); standalone
]
# a confirmation / authorization number is note-relevant ONLY when a keyword anchors it (never a bare CLM/DOC id)
_NOTE_CONF_RE = re.compile(
    r"\b(?:confirmation|authorization|auth|tracking)\b\s*(?:number|no\.?|#|id)?\s*[:#]?\s*([A-Z0-9][A-Z0-9\-]{4,})", re.I)
# generic RECORD ids (claim / doc / member / referral) -- NOT required in a clinical note; only for slot='all'
_RECORD_ID_RE = re.compile(r"\b[A-Z]{2,4}-[A-Z0-9]{2,}(?:-[A-Z0-9]+)*\b")


def _norm(s):
    return re.sub(r"[\s\-]+", "", str(s or "")).lower()


_NOTE_KEY_RE = re.compile(r"(note|triage|communication|rationale|justification|indication|narrative)", re.I)
_CONTENT_KEY_RE = re.compile(r"content|text|body|note|rationale|justification|indication|narrative", re.I)


def note_text_from_state(state):
    """The PERSISTED authored note / rationale / communication text from the NATIVE state -- the surface the
    rubric actually scores (full_state.triageNotes / communications[].content / submitted rationale / clinical
    indication), NOT the final chat answer. Generic dig; no gold."""
    out = []

    def _collect(o):
        if isinstance(o, str):
            if o.strip():
                out.append(o)
        elif isinstance(o, dict):
            for k, v in o.items():
                if _CONTENT_KEY_RE.search(str(k)):
                    _collect(v)
        elif isinstance(o, list):
            for it in o:
                _collect(it)

    def _walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if _NOTE_KEY_RE.search(str(k)):
                    _collect(v)
                else:
                    _walk(v)
        elif isinstance(o, list):
            for it in o:
                _walk(it)

    _walk(state or {})
    return "\n".join(out)


# The ACTUAL persisted surface each family's native rubric scores (from the checkpoint student_answer
# templates): denial/appeal triage note = full_state.triageNotes; prior_auth auth note + dme progress note =
# full_state.communications[-1].content (the LAST communication) -- NOT an authNote/progressNote field (those
# are just agentActions booleans). So the grounding audit reads the SAME text the rubric does, per family.
_FAMILY_SURFACE = {
    "denial_triage": "triage", "electronic_appeals": "triage", "appeal": "triage", "fax_appeal": "triage",
    "submit_auth_aetna": "last_comm", "submit_auth_anthem": "last_comm", "prior_auth": "last_comm",
    "dme": "last_comm", "dme_fax": "last_comm",
}


def note_surfaces(state):
    """{normalized-surface-key: text} for each DISTINCT persisted note surface (triage note / progress note /
    auth note / appeal rationale / communication / clinical indication) -- NOT merged."""
    out = {}

    def _text_of(o):
        acc = []
        if isinstance(o, str):
            if o.strip():
                acc.append(o)
        elif isinstance(o, dict):
            for k, v in o.items():
                if _CONTENT_KEY_RE.search(str(k)):
                    acc += _text_of(v)
        elif isinstance(o, list):
            for it in o:
                acc += _text_of(it)
        return acc

    def _walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if _NOTE_KEY_RE.search(str(k)):
                    t = "\n".join(_text_of(v)).strip()
                    if t:
                        kk = re.sub(r"[\s_]+", "", str(k)).lower()
                        out[kk] = (out.get(kk, "") + "\n" + t).strip()
                else:
                    _walk(v)
        elif isinstance(o, list):
            for it in o:
                _walk(it)

    _walk(state or {})
    return out


def note_text_for(state, family=None):
    """Text of the FAMILY's ACTUAL rubric surface: denial/appeal -> full_state.triageNotes; prior_auth/dme
    -> full_state.communications[-1].content (the last communication). Falls back to all note surfaces merged
    only when the family/surface is unknown or absent."""
    fs = (state or {}).get("full_state") if isinstance(state, dict) and "full_state" in state else (state or {})
    surf = _FAMILY_SURFACE.get(str(family or ""))
    if surf == "triage" and isinstance(fs, dict):
        t = fs.get("triageNotes")
        if isinstance(t, str) and t.strip():
            return t
        if isinstance(t, list) and t:            # native triageNotes is string[] -> the LATEST note
            _last = t[-1]
            _last = _last.get("content") if isinstance(_last, dict) else _last
            if isinstance(_last, str) and _last.strip():
                return _last
    if surf == "last_comm" and isinstance(fs, dict):
        comms = fs.get("communications")
        if isinstance(comms, list) and comms:
            last = comms[-1]
            c = last.get("content") if isinstance(last, dict) else last
            if isinstance(c, str) and c.strip():
                return c
    return "\n".join(note_surfaces(state).values())


def extract_salient(text, slot="note"):
    """Structured PUBLIC facts in OBSERVED text, filtered by the OUTPUT SLOT. slot='note' (default) returns only
    the CLINICAL/BILLING codes a clinical note is expected to reference (adjustment / remark / ICD / CPT) plus a
    keyword-anchored confirmation/auth number -- NOT generic record ids (claim/doc/member) and NOT 5-digit
    fragments of a longer id ('observed' != 'required in THIS note'). slot='all' also includes record ids.
    Oracle-blind."""
    out = set()
    t = str(text or "")
    for pat in _NOTE_CODE_PATTERNS:
        for m in pat.findall(t):
            tok = m if isinstance(m, str) else (m[0] if m else "")
            if tok and len(_norm(tok)) >= 2:
                out.add(tok.strip())
    for m in _NOTE_CONF_RE.findall(t):
        # a confirmation/auth id must contain a digit or hyphen -> rejects a plain word captured after the
        # keyword (Authorization Queue / Authorization Number / authorizations-link UI labels).
        if m and len(_norm(m)) >= 4 and re.search(r"\d", m):
            out.add(m.strip())
    if slot == "all":
        for m in _RECORD_ID_RE.findall(t):
            if m and len(_norm(m)) >= 4:
                out.add(m.strip())
    return out


def grounding_gap(note_text, observed_facts):
    """Salient OBSERVED facts NOT referenced in the authored note -> what the agent should supplement. The
    harness supplies the LIST-OF-MISSING (coverage), never the value (the agent already observed it)."""
    note = _norm(note_text)
    if not note:
        return sorted(observed_facts, key=str)
    return sorted([f for f in observed_facts if _norm(f) not in note], key=str)


def feedback(missing):
    """A NUDGE naming the un-referenced items -- never their meaning or a substitute value; the agent must
    write the note itself from the records it already read."""
    if not missing:
        return None
    return ("Your note does not reference these items you already saw in the records: %s. "
            "Supplement the note from what you have read -- do not invent or omit them."
            % ", ".join(str(m) for m in missing))


# ------------------------------------------------------------------ middleware hook (runner/middleware_host.py)
MAX_ROUNDS = 2        # grounding nudges per task


class GroundingAudit(object):
    """The persisted note surface the rubric scores must reference the salient facts the agent observed
    (ctx.observed_history). Otherwise the final is turned back with a Nudge naming what is missing. The shell applies
    it without unlocking repair or counting a repair turn; a re-answer costs no env budget, so it also runs once the
    budget is spent."""

    def __init__(self, task):
        self.rounds = 0

    def before_final(self, ctx, answer):
        if self.rounds >= MAX_ROUNDS:
            return None
        try:
            family = _W.load_relative(__file__, "../recovery/hab.py")._family(ctx.task)   # PUBLIC challenge type
            state = ctx.persisted_state()
            note = note_text_for(state, family) if state is not None else ""
            gap = grounding_gap(note, extract_salient(" ".join(ctx.observed_history)[-20000:]))
        except Exception:
            gap = []
        if not gap:
            return None
        self.rounds += 1
        ctx.emit({"event_type": "grounding_gap", "missing": gap[:12], "status": "ok", "step": ctx.step,
                  "round": self.rounds})
        return Nudge("grounding", feedback(gap), "grounding_gap", "REVISE")
