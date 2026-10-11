"""Bounded Clinical Recovery - GenericGuiCompletionWorkflow (Layer 3) : dynamic closed-loop (workspace strategy,
HealthAdminBench default workspace).

ONE generic GUI-completion state machine. It knows NO business names (no prior-auth / appeal / denial /
claim / CPT / rationale). It drives the environment as a closed loop -- observe -> decide the ONE next
deterministic action -> (kernel executes) -> re-observe -> decide again -- until the target effect lands
or a genuinely new decision/content is required.

Generic phases (data-driven, re-evaluated every step from the LIVE control model + persisted state):
    DONE       target effect already realized (verify_spec path satisfied)
    FILL       a required+empty control has a bindable value -> type/select/upload it
    COMMIT      every fillable field done + the control of a commit the agent itself attempted that did not land
               (goal.raw agent_commit_refs) -> the ONE irreversible submit, never a resubmit
    CONFIRM    a post-commit confirm/acknowledge control appeared -> click it
    NAVIGATE   not on the target surface yet -> click the single best control toward the goal
    BLOCK      a required field needs NEW content or a NEW decision the harness must not author

Binding is robust (not naive top-level string match): sources are flattened, keys aliased, values format-
normalized (dates / codes / names), select options semantically matched, already-filled fields preserved.

next_action() returns a Decision(kind, step|reason). Python 3.8 compatible.
"""
import json
import re

import workspace_api as _W
from workspace_api import (classify_field, SEMANTIC, SEMANTIC_SOURCE_PRIORITY, OPERATIONAL_SOURCE_PRIORITY,
                              AGENT_COMMITMENT, AUTHORITATIVE_STATE, BOUND_EVIDENCE, SYSTEM_METADATA)
from workspace_api import conjunct_met, obligation_view, readback_portals

_C = _W.load_relative(__file__, "contracts.py")
RecoveryStep = _C.RecoveryStep
ACQUIRE, STAGED_WRITE, IRREVERSIBLE_COMMIT = _C.ACQUIRE, _C.STAGED_WRITE, _C.IRREVERSIBLE_COMMIT

GOAL_TYPE = "complete_committed_gui_effect"

# generic control-role -> gui op.
_SELECTISH = ("select", "combobox", "listbox", "dropdown")
_FILEISH = ("file", "upload")
# confirm/acknowledge controls that appear AFTER a submit (generic verbs, no business names).
_CONFIRM_RE = re.compile(r"\b(confirm|acknowledge|ok|continue|yes|proceed|done|finish|close)\b", re.I)
# A control that DOWNLOADS a source document (EHR side of the download->upload chain). Only these get an
# op=download; a plain link/button is still a click.
_DOWNLOAD_RE = re.compile(r"\b(download|export|save\s+(?:as\s+)?pdf|get\s+document|retrieve\s+document)\b", re.I)
_CLEAR_WORKLIST_RE = re.compile(r"clear\s+(?:from\s+)?(?:the\s+)?worklist|remove\s+from\s+worklist|clear\s+(?:the\s+)?referral", re.I)
# Conjunction reachability: after the portal submit, HAB requires writing the result back to the EHR.
# These labels navigate toward / perform that documentation (incl. a legitimate return to the EHR/chart).
_DOC_NAV_RE = re.compile(r"\b(document|documentation|add\s+note|save\s+to\s+(?:epic|ehr|emr|chart)|write.?back|"
                         r"record\s+(?:result|outcome|decision)|return\s+to\s+(?:ehr|emr|chart|epic)|note)\b", re.I)
_COMMIT_RE = re.compile(r"\b(submit|send|transmit|finaliz|file\s+(?:claim|appeal|authorization|the))\b", re.I)
# a NAVIGATION control that merely LOOKS like a commit: 'Submit to <the payer / insurance / portal / plan /
# carrier>' opens that system's page (reversible) -- it is NOT the irreversible terminal submit ('Submit
# Authorization', 'Send Fax'). Exempted from the stage-nav commit guard so recovery can traverse to the portal;
# the real terminal submit on that system stays commit-only.
_NAV_SUBMIT_RE = re.compile(r"submit\s+to\s+(?:the\s+)?(?:payer|insurance|portal|plan|carrier|health\s*plan)", re.I)
# Labels that read like a commit verb but are INTERMEDIATE page actions, never the task terminal.
_INTERMEDIATE_RE = re.compile(r"\b(add|save\s*draft|save\s+and|update\s+search|apply\s+filter|search|"
                              r"next|previous|edit|attach|browse|choose\s+file|select\s+file)\b", re.I)
_NAV_STOP_RE = re.compile(r"\b(cancel|back|logout|sign out|delete|remove|clear|reset)\b", re.I)


try:                                                     # shared field-concept ontology (generic)
    from workspace_api import FIELD_ALIASES as _FIELD_ALIASES, field_concept as _concept_of
except Exception:                                        # pragma: no cover - keeps the workflow importable
    _FIELD_ALIASES, _concept_of = {}, (lambda _x: None)


def _norm(s):
    s = str(s or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def _is_filled(value):
    """A control value counts as FILLED only if it carries real content. A placeholder glyph (a bare
    dropdown arrow, '...'), an empty string, or a 'Select...'/'Choose' prompt is NOT filled -- treating it
    as filled would SKIP a required, still-empty control (e.g. the native fax-number picker renders just
    a glyph) so recovery would never set it."""
    s = str(value or "").strip()
    if not s:
        return False
    n = _norm(s)                      # lower; glyphs/punctuation -> stripped
    if not n:                         # pure glyph / punctuation value
        return False
    if n in ("select", "select one", "selectone", "choose", "please select"):
        return False
    return True


def _norm_code(s):
    """Normalize an identifier/code for comparison: strip spaces/punct, upper."""
    return re.sub(r"[^A-Za-z0-9]+", "", str(s or "")).upper()


def _flatten(obj, prefix="", out=None, depth=0):
    """Flatten a nested dict/list into {dotted_key: scalar} and also index every leaf by its LAST segment,
    so an EMR value buried at authoritative_state.patient.dob is reachable by 'dob'."""
    if out is None:
        out = {}
    if depth > 6:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = "%s.%s" % (prefix, k) if prefix else str(k)
            if isinstance(v, (dict, list)):
                _flatten(v, key, out, depth + 1)
            elif v not in (None, ""):
                out[key] = v
                out.setdefault(str(k), v)                    # last-segment alias
    elif isinstance(obj, list):
        # keep whole small scalar lists joinable (e.g. diagnosisCodes: ["H35.32"])
        scal = [x for x in obj if isinstance(x, (str, int, float))]
        if scal and prefix:
            out.setdefault(prefix, scal[0] if len(scal) == 1 else ", ".join(str(x) for x in scal))
            out.setdefault(prefix.split(".")[-1], out[prefix])
        for i, v in enumerate(obj):
            if isinstance(v, (dict, list)):
                _flatten(v, "%s.%d" % (prefix, i), out, depth + 1)
    return out


# label/name aliases: map common form-field concepts to EMR/state key fragments (generic clinical-admin
# vocabulary, NOT task-specific; extend freely -- this is alignment data, not business logic).
_ALIASES = {
    "diagnosis": ["diagnosiscodes", "diagnosis", "icd", "dx"],
    "procedure": ["cptcodes", "cpt", "procedure", "hcpcs"],
    "date of birth": ["patientdob", "dob", "birthdate", "dateofbirth"],
    "member id": ["subscriberid", "memberid", "insuranceid", "policyid"],
    "patient last name": ["patientlastname", "lastname", "familyname"],
    "patient first name": ["patientfirstname", "firstname", "givenname"],
    "patient name": ["patientname", "name", "patient"],
    "provider": ["providername", "provider", "orderingprovider", "physician"],
    "request type": ["requesttype", "authtype", "type"],
    "claim": ["claimid", "claim", "claimnumber"],
}


def _alias_targets(label, name):
    keys = set()
    for src in (label, name):
        n = _norm(src)
        if not n:
            continue
        keys.add(n.replace(" ", ""))
        keys.add(n)
        for canon, al in _ALIASES.items():
            if canon in n or n in canon or any(a in n.replace(" ", "") for a in al):
                keys.update(al)
    return keys



_COLLECTION_KEY_HINT = {
    "diagnosis": ("diagnos", "icd"),
    "cpt": ("cpt", "hcpcs", "procedure", "service"),
    "procedure_codes": ("cpt", "hcpcs", "procedure", "service"),
    "supporting_documents": ("document", "attachment", "supporting"),
}


def _concept_hint(concept, text):
    return any(h in str(text or "").lower() for h in _COLLECTION_KEY_HINT.get(concept, ()))


# RESULT_OK (kernel.py) means the primitive DISPATCHED, not that the write LANDED. A React / custom
# control can accept a type/click as RESULT_OK yet leave the field empty or form no chip. So a tag in
# `history` proves dispatch, NOT postcondition. The operational-field step and the collection executor
# therefore bound RE-DISPATCH by count and let the live check (is_control_complete / chip-present) decide
# 'done' -- the same attempt closure the read-evidence acquisition uses.
_EXEC_MAX_ATTEMPTS = 2


def _present_chips(state_view, concept):
    """Best-effort read of values already ADDED for a concept (chips landed in state): a list value under a key
    naming the concept, anywhere in the state tree. Generic; no gold."""
    out = []

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, list) and _concept_hint(concept, k):
                    for it in v:
                        sv = it.get("code") if isinstance(it, dict) else it
                        if isinstance(sv, str) and sv.strip():
                            out.append(sv.strip())
                else:
                    walk(v)
        elif isinstance(node, list):
            for it in node:
                walk(it)

    walk(state_view or {})
    return out


class GenericGuiCompletionWorkflow(object):
    goal_type = GOAL_TYPE

    def match_goal(self, goal, ctx):
        return (getattr(goal, "goal_type", "") or "") == GOAL_TYPE

    # ---- robust binding ---------------------------------------------------------------------------
    def _sources(self, goal, ctx):
        """Source classes kept SEPARATE (provenance preserved) so _bind_value can enforce the Decision-
        Boundary contract (bindings.py). _scratch_memory (facts READ from earlier surfaces, e.g. the EHR) is
        AUTHORITATIVE_STATE class -- a copyable fact, a legitimate semantic source. system_metadata is kept
        apart so it can NEVER satisfy a semantic slot. Each class carries a normalized-key variant so an
        exact/alias key match still hits."""
        ctx = ctx or {}
        def _normed(d):
            out = {}
            for k, v in _flatten(d).items():
                out.setdefault(k, v)
                out.setdefault(_norm(k).replace(" ", ""), v)
            return out
        auth = _normed(ctx.get("authoritative_state") or {})
        cur_subj = ctx.get("_mem_subject")
        for k, e in (ctx.get("_scratch_memory") or {}).items():   # trusted read-facts fold into authoritative
            if not isinstance(e, dict) or e.get("ambiguous") or e.get("rank", 0) < 2:
                continue                                 # draft / conflicting -> never auto-binds
            es = e.get("subject")
            if es and cur_subj and es != cur_subj:
                continue                                 # a different subject -> do not carry the fact across
            auth.setdefault(_norm(k), e.get("value"))
            auth.setdefault(_norm(k).replace(" ", ""), e.get("value"))
        return {
            AGENT_COMMITMENT: _normed(getattr(goal, "committed_fields", None) or {}),
            AUTHORITATIVE_STATE: auth,
            BOUND_EVIDENCE: _normed(ctx.get("bound_evidence") or {}),
            SYSTEM_METADATA: _normed(ctx.get("system_metadata") or {}),
        }

    def _submit_subgoal_realized(self, goal, state_view):
        """Is THIS task's SUBMIT subgoal already realized? The submit subgoals are the composed submission conjuncts (the
        effects of the commits the agent itself attempted, marked "decision" by the shell's completion_conjunction),
        payer-correct (the shell resolves the payer); all of them must hold. None composed -> the agent submitted nothing,
        so nothing is realized (no post-submit documentation hop, and no commit).
        The check is strict: only the current task's own effect counts. Accepting ANY payer / family / fax would let a
        Payer A submission satisfy a Payer B task, or a fax satisfy a prior-auth -> recovery would SKIP the real submit
        and the native submit checkpoint would never pass. Empty-form / wrong-surface safety is handled separately by
        the empty-form guard in next_action (recovery never submits a blank surface)."""
        raw = getattr(goal, "raw", None) or {}
        subs = [s for s in (raw.get("verify_all") or []) if isinstance(s, dict) and s.get("decision")]
        return bool(subs) and all(bool(self._check_one(s, state_view)) for s in subs)

    _RANK = {"persisted": 3, "read_only_record": 2}   # only trusted provenance (>=2) may bind; drafts never stored

    @staticmethod
    def _subject_sig(observation):
        """The CURRENT subject id from the live ROUTE/page (the referral/denial/claim/case id in the URL),
        NOT a scan of the whole persisted EMR/payer/fax DB snapshot -- that would yield a global-DB fingerprint,
        not 'the current subject'. None when indeterminate. Generic."""
        o = str(observation or "")
        m = re.search(r"/(?:referral|denied|claim|case|patient|auth)/([A-Za-z0-9][A-Za-z0-9\-]{2,})", o, re.I)
        if m:
            return _norm(m.group(1))
        m2 = re.search(r"\b((?:REF|DEN|CLM|CASE|MRN|PAT)-[A-Za-z0-9\-]{2,})\b", o, re.I)
        return _norm(m2.group(1)) if m2 else None

    @staticmethod
    def _is_readonly_record(c):
        """A control that DISPLAYS a persisted value (read-only / disabled record) -- a trusted fact. An
        EDITABLE input / search box / draft is NOT (its value may be an agent draft, wrong entry, placeholder,
        prior-subject residue, or recovery-filled content) and must never auto-propagate."""
        if not isinstance(c, dict):
            return False
        role = str(c.get("role") or "").lower()
        if role in ("button", "link", "a", "tab", "menuitem", "checkbox", "radio"):
            return False
        return bool(c.get("readonly") or c.get("readOnly") or c.get("disabled"))

    def _remember(self, ctx, controls, state_view, observation=""):
        """Long-term memory with PROVENANCE + SUBJECT binding. Only TRUSTED facts enter as bindable: the
        persisted state (rank 3) and read-only record controls (rank 2). Editable inputs / search boxes / drafts
        are SKIPPED (they may be agent-authored / wrong / residue). Each concept resolves to ONE trusted value;
        two DIFFERENT equally-trusted values -> AMBIGUOUS (auto-fill stops); a higher-provenance value replaces a
        lower one (never first-write-wins). Facts are tagged with the current subject so they never cross a
        patient/case switch. entry = {value, rank, subject, ambiguous}. Generic; no gold."""
        if not isinstance(ctx, dict):
            return
        mem = ctx.setdefault("_scratch_memory", {})
        subj = self._subject_sig(observation)
        if subj:
            ctx["_mem_subject"] = subj
        cur_subj = ctx.get("_mem_subject")

        def _put(key, value, rank):
            if not key:
                return
            e = mem.get(key)
            if e is None or rank > e.get("rank", 0):
                mem[key] = {"value": value, "rank": rank, "subject": cur_subj, "ambiguous": False}
                return
            if e.get("ambiguous") or rank < e.get("rank", 0):
                return                                   # keep higher provenance / stay ambiguous
            if _norm(e.get("value")) != _norm(value):    # same rank, different value -> ambiguous
                e["ambiguous"] = True

        def _store_fact(label, value, rank):
            if not label or value in (None, "", [], {}) or isinstance(value, (dict, list)):
                return
            _put(_norm(label), value, rank)
            _put(_norm(label).replace(" ", ""), value, rank)
            concept = _concept_of(label)
            if concept:                                  # concept + aliases (cross-synonym recall)
                _put(_norm(concept), value, rank)
                for alias in _FIELD_ALIASES.get(concept, ()):
                    _put(_norm(alias), value, rank)
                    _put(_norm(alias).replace(" ", ""), value, rank)

        # The state_view may be the WHOLE EMR/payer/fax DB. _flatten also emits LEAF-only
        # keys (memberId) that lose the parent case id, so a FOREIGN case's value would concept-bind to the
        # current form. Pre-scan every value that appears under a NON-current-subject id path; a fact carrying
        # such a value is stored under its RAW key only, never concept-expanded to bind to a clean concept.
        _flat = _flatten(state_view or {})
        _foreign_id = re.compile(r"\b(?:REF|DEN|CLM|CASE|MRN|PAT)-[A-Za-z0-9\-]{2,}\b", re.I)
        _foreign_vals = set()
        if cur_subj:
            for _fk, _fv in _flat.items():
                if isinstance(_fv, (dict, list)):
                    continue
                for _m in _foreign_id.findall(str(_fk)):
                    if _norm(_m) != cur_subj:
                        _foreign_vals.add(str(_fv))
                        break
        for k, v in _flat.items():  # persisted -> trusted (rank 3)
            if v in (None, "", [], {}) or isinstance(v, (dict, list)):
                continue
            if str(v) in _foreign_vals:
                _put(_norm(k), v, 3)                      # raw key only -- never concept-expanded to bind
                _put(_norm(k).replace(" ", ""), v, 3)
            else:
                _store_fact(k, v, 3)
        for c in controls or []:                         # read-only RECORDS -> trusted (rank 2); drafts skipped
            v = str(c.get("value") or "").strip()
            if not v or not self._is_readonly_record(c):
                continue
            for k in (c.get("name"), c.get("label")):
                _store_fact(k, v, 2)


    def _collection_exec_step(self, goal, controls, state_view, history):
        """Collection executor: drive ONE type->Add step to add the next value of the agent's OWN committed
        collection (diagnosis/CPT/...) that is not yet present as a chip. Values come ONLY from
        goal.raw['committed_collections'] -- the agent's own typed codes -- never invented (that would be
        content authoring = a trick). Returns a fill step (type the next value), a click step (Add the value
        already typed), or None. Re-observation happens naturally: the kernel calls next_action again after
        each action, the new chip shrinks the delta."""
        raw = getattr(goal, "raw", None) or {}
        committed = raw.get("committed_collections") or {}
        if not committed:
            return None
        from workspace_api import next_collection_action, present_values
        for concept, vals in committed.items():
            if not vals:
                continue
            inp = add = None
            for c in (controls or []):
                lab = "%s %s %s" % (c.get("label") or "", c.get("ref") or "", c.get("name") or "")
                role = str(c.get("role") or "").lower()
                if inp is None and role in ("input", "textbox", "combobox", "select") and _concept_hint(concept, lab) \
                        and not c.get("readonly") and not c.get("disabled"):
                    inp = c
                elif add is None and role == "button" and re.search(r"\badd\b", lab, re.I):
                    add = c
            if inp is None:
                continue
            present = _present_chips(state_view, concept) or present_values(controls)
            act = next_collection_action(vals, present, inp.get("ref"), (add or {}).get("ref"))
            if act is None:
                continue
            _cur = _norm(inp.get("value"))
            # the value is already typed into the input but not yet Added -> click Add
            if _cur and _norm(act["value"]) == _cur and act.get("add_ref"):
                tag = "coll_add:%s:%s" % (concept, _norm(act["value"]))
                if history.count(tag) >= _EXEC_MAX_ATTEMPTS:
                    continue                                 # chip STILL missing after N Add clicks that landed
                                                             # (Add formed no chip) -> give up. next_collection_action
                                                             # returns None once present -- that is the real check.
                return {"kind": "step", "tag": tag, "step": RecoveryStep(
                    kind=STAGED_WRITE, name="collection_add",
                    action={"op": "click", "ref": act["add_ref"]},
                    affordance_target={"role": (add or {}).get("role"), "label": (add or {}).get("label"),
                                       "ref": act["add_ref"]},
                    manifest={"side_effect_scope": "form_field", "rollback_available": True,
                              "server_persisted": False, "collection_value": act["value"]})}
            # otherwise type the next committed value into the collection input
            tag = "coll_fill:%s:%s" % (concept, _norm(act["value"]))
            if history.count(tag) >= _EXEC_MAX_ATTEMPTS:
                continue                                     # bounded retry (see coll_add); present-value check
                                                             # via next_collection_action is the real postcondition.
            return {"kind": "step", "tag": tag, "step": RecoveryStep(
                kind=STAGED_WRITE, name="collection_fill",
                action={"op": self._op_for(inp), "arg": "_direct", "value": act["value"]},
                affordance_target={"role": inp.get("role"), "label": inp.get("label"), "ref": inp.get("ref")},
                manifest={"side_effect_scope": "form_field", "rollback_available": True,
                          "server_persisted": False, "collection_value": act["value"]})}
        return None

    # Surface-local operational completion (_of1_step): the CLEAN operational subset split out of the broad fill --
    # fill ONLY a non-clinical operational field whose value is reliably grounded, one at a time, NO auto-submit.
    # EXCLUDES clinical-decision / collection fields (request_type=agent choice, diagnosis/cpt=collection
    # executor, clinical_justification=agent authors). Reuses the safe _bind_value (semantic default-deny +
    # reliable match or None). Attributable + does not cause false completion (never submits).
    _OF1_OPERATIONAL = frozenset({"member_id", "date_of_birth", "provider", "fax_number", "patient_name"})

    def _of1_step(self, goal, controls, sources, history, schema=None):
        """Return a fill step for ONE required+empty OPERATIONAL field on the current surface bound from a
        reliable grounded source, or None. Single-field; never submits; never a clinical/collection field."""
        try:
            from workspace_api import field_concept as _field_concept, is_control_complete
        except Exception:
            return None
        for c in (controls or []):
            if not isinstance(c, dict):
                continue
            if not c.get("required") or c.get("visible") is False or c.get("disabled"):
                continue
            if is_control_complete(c):
                continue
            concept = _field_concept(c.get("label") or c.get("name") or c.get("ref"))
            if concept not in self._OF1_OPERATIONAL:
                continue                                     # not a clean operational field -> leave it
            val = self._bind_value(c, sources, schema)
            if val is None:
                continue                                     # no reliable grounded source -> do NOT author
            tag = "of1_fill:%s" % _norm(c.get("name") or c.get("label") or c.get("ref"))
            if history.count(tag) >= _EXEC_MAX_ATTEMPTS:
                continue                                     # field STILL incomplete after N landed writes
                                                             # (control didn't take) -> give up. is_control_complete
                                                             # above is the real postcondition; this only caps retries.
            return {"kind": "step", "tag": tag, "step": RecoveryStep(
                kind=STAGED_WRITE, name="of1_fill",
                action={"op": self._op_for(c), "arg": "_direct", "value": val},
                affordance_target={"role": c.get("role"), "label": c.get("label"), "ref": c.get("ref")},
                manifest={"side_effect_scope": "form_field", "rollback_available": True,
                          "server_persisted": False, "operational_field": concept})}
        return None

    def _bind_value(self, control, source_classes, schema=None):
        """Resolve a control's value HONORING the Decision-Boundary contract (bindings.py). The slot is
        classified SEMANTIC (default-deny) vs OPERATIONAL. A SEMANTIC slot (clinical / payload / decision
        field) may bind ONLY from a semantic source (agent_commitment / authoritative_state incl. read-fact
        memory / bound_evidence) via a RELIABLE key match (exact control-ref key, or an exact alias/name
        key) -- NEVER from system_metadata and NEVER from a loose cross-field substring guess (that is where
        a fabricated / wrong binding = a trick would enter). An OPERATIONAL slot (whitelisted id/timestamp/
        ...) may also bind from system_metadata and may use the looser substring match. An unbindable
        SEMANTIC slot -> None -> the caller BLOCKs (needs_new_content): only the agent may author it.
        Oracle-blind (the agent's own values + read facts, never gold)."""
        kind = SEMANTIC
        for n in (control.get("name"), control.get("label"), control.get("ref")):
            if n and classify_field(str(n), schema) != SEMANTIC:
                kind = "operational"                     # a KNOWN operational field (id/timestamp/...)
                break
        allowed = SEMANTIC_SOURCE_PRIORITY if kind == SEMANTIC else OPERATIONAL_SOURCE_PRIORITY
        nref = _norm(control.get("ref")).replace(" ", "")
        targets = _alias_targets(control.get("label"), control.get("name"))
        if nref:
            targets.add(nref)
        def _scalar(v):
            return v not in (None, "") and not isinstance(v, (dict, list))
        # RELIABLE match ONLY (exact control-ref key, then exact alias/name key), within the ALLOWED source
        # classes in priority order. This is the only path a SEMANTIC value may take.
        for cls in allowed:
            bag = source_classes.get(cls) or {}
            if nref:
                for sk, sv in bag.items():
                    if _scalar(sv) and _norm(sk).replace(" ", "") == nref:
                        return self._format(control, sv)
            for t in targets:
                if t in bag and _scalar(bag.get(t)):
                    return self._format(control, bag[t])
        # LOOSE substring match: OPERATIONAL slots only (never fabricate a semantic / payload value).
        if kind != SEMANTIC:
            for cls in allowed:
                bag = source_classes.get(cls) or {}
                for sk, sv in bag.items():
                    if not _scalar(sv):
                        continue
                    nsk = _norm(sk).replace(" ", "")
                    if any(t and (t in nsk or nsk in t) for t in targets):
                        return self._format(control, sv)
        return None

    def _format(self, control, value):
        """Format a bound value to the control (select-option semantic match; codes/dates left as-is str)."""
        val = value if isinstance(value, str) else str(value)
        opts = control.get("options") or []
        if opts:
            nv = _norm_code(val)
            for o in opts:
                if _norm_code(o) == nv or _norm(o) == _norm(val):
                    return o
            for o in opts:                                   # substring fallback
                if nv and (nv in _norm_code(o) or _norm_code(o) in nv):
                    return o
        return val

    def _op_for(self, control):
        r = str(control.get("role") or "").lower()
        if _DOWNLOAD_RE.search(str(control.get("label") or "")):
            return "download"                              # real download, not a plain click
        if r in ("checkbox", "radio", "switch", "toggle"):
            return "check"                                  # toggle, never page.fill(checkbox, text)
        if r in _SELECTISH or (control.get("options")):
            return "select"
        if r in _FILEISH or r == "file":
            return "upload"
        return "type"

    # ---- dynamic closed loop ----------------------------------------------------------------------
    def _check_one(self, spec, state_view):
        if not isinstance(spec, dict):
            return None
        # one conjunct, evaluated as the shell evaluates every completion conjunct (path, then alt_paths: the first
        # truthy value is read; truthy / nonempty / equals) against the shell's one obligation view of this read-back
        # (the view the terminal lock and ctx.obligations() read)
        return conjunct_met(spec, obligation_view(readback_portals(state_view)))

    def is_realized(self, goal, state_view):
        raw = getattr(goal, "raw", None) or {}
        # A complete HAB task is a CONJUNCTION of subgoals (submit AND confirmation AND EHR documentation
        # AND work-item cleared ...), not a single submit. verify_all = every required subgoal -> done only when
        # ALL hold. A single verify_spec is still honored.
        specs = raw.get("verify_all")
        if isinstance(specs, list) and specs:
            return all(bool(self._check_one(sp, state_view)) for sp in specs if isinstance(sp, dict))
        spec = raw.get("verify_spec")
        if not isinstance(spec, dict):
            return None
        return self._check_one(spec, state_view)

    def next_action(self, goal, ctx, observation, controls, state_view, history):
        """Decide the ONE next deterministic action from the LIVE page. Returns a dict:
            {"kind": "step", "step": RecoveryStep}  |  {"kind": "done"}  |  {"kind": "block", "reason": ...}
        observation = the LIVE page observation THIS round (passed explicitly -- never read stale page state
        from the long-term ctx). history = prior action tags this episode (avoid loops / know we committed)."""
        history = history or []
        self._remember(ctx, controls, state_view, observation)   # fold this surface's facts into cross-page memory
        sources = self._sources(goal, ctx)

        # 0) already there?
        if self.is_realized(goal, state_view):
            return {"kind": "done", "reason": "effect_realized"}

        # 0.5) DOWNLOAD phase: the download->upload chain needs the SOURCE document in the workspace.
        # Only when the GOAL involves document handling (avoid over-eager downloads), and only a control not yet
        # fetched this episode -> download it (the env records _last_download for the later upload("last")).
        _goal_blob = (json.dumps((getattr(goal, "raw", None) or {})).lower()
                      + " " + str(getattr(goal, "goal_type", "")).lower())
        _needs_doc = any(k in _goal_blob for k in ("document", "attach", "fax", "upload", " pdf", "file"))
        for c in (controls or []) if _needs_doc else []:
            if _DOWNLOAD_RE.search(str(c.get("label") or "")):
                dtag = "download:%s" % _norm(c.get("label") or c.get("ref"))
                if dtag in (history or []):
                    continue
                return {"kind": "step", "tag": dtag, "step": RecoveryStep(
                    kind=STAGED_WRITE, name="download",
                    action={"op": "download"},
                    affordance_target={"role": c.get("role"), "label": c.get("label"), "ref": c.get("ref")},
                    manifest={"side_effect_scope": "read_only_download", "rollback_available": True,
                              "server_persisted": False})}

        # 0.6) Prior-auth READ-evidence acquisition (read-only), driven by goal.raw acquire_read_sources. For the
        # SUBMISSION workflow: open each UNMET public read surface (coverages/services/diagnoses tab, clinical
        # note, auth letter -- each reveals real coverage/clinical data, i.e. reading evidence NOT a
        # signal-check), flipping clicked*Tab / readClinicalNote / downloadedAuthLetter F->T, then FALL THROUGH
        # to fill/commit (NO re-entry -- these are workflow reads, not a decision).
        _ars = (getattr(goal, "raw", None) or {}).get("acquire_read_sources") or []
        if _ars:
            _aa = state_view.get("agentActions") if isinstance(state_view, dict) else None
            if not isinstance(_aa, dict):
                for _w in ("_live_full_state", "full_state"):
                    _wv = (state_view or {}).get(_w) if isinstance(state_view, dict) else None
                    if isinstance(_wv, dict) and isinstance(_wv.get("agentActions"), dict):
                        _aa = _wv["agentActions"]
                        break
            _aa = _aa if isinstance(_aa, dict) else {}
            for _src in _ars:
                _flag = _src.get("flag")
                if _aa.get(_flag):
                    continue                             # already read -> flag met
                _atag = "acquire_evidence:%s" % _flag
                if _atag in history:
                    continue                             # already acquired this episode
                _kws = [_norm(k) for k in (_src.get("nav_keywords") or []) if k]
                _rctrl = None
                for c in (controls or []):
                    if str(c.get("role") or "") not in ("button", "link", "a", "tab", "menuitem"):
                        continue
                    _lab = _norm(c.get("label") or "")
                    if _lab and _kws and any(k in _lab for k in _kws):
                        _rctrl = c
                        break
                if _rctrl is not None:                    # ACQUIRE: open + read the surface (read-only)
                    _rop = "download" if _DOWNLOAD_RE.search(str(_rctrl.get("label") or "")) else "click"
                    return {"kind": "step", "tag": _atag, "step": RecoveryStep(
                        kind=ACQUIRE, name="acquire_evidence",
                        action={"op": _rop, "provenance": "harness-acquired-evidence",
                                "evidence_flag": _flag, "ref": _rctrl.get("ref")},
                        affordance_target={"role": _rctrl.get("role"), "label": _rctrl.get("label"),
                                           "ref": _rctrl.get("ref")},
                        manifest={"side_effect_scope": "none", "rollback_available": True,
                                  "server_persisted": False, "read_only": True,
                                  "provenance": "harness-acquired-evidence"})}

        # 0.7) DECISION-READINESS EVIDENCE ACQUISITION. The adapter flagged that the root agent produced a decision
        # WITHOUT observing a decision-input source the portal TRACKS (e.g. a remittance image / denial detail / a
        # *viewed flag's page). Recovery does NOT author the disposition; instead it ACQUIRES the missing evidence
        # READ-ONLY -- opening the PUBLIC source control the adapter named -- recorded as a first-class
        # harness-acquired-evidence event, and once every named source is acquired it hands control BACK to the root
        # agent to re-deliberate (re_entry_requested). This preempts FILL/COMMIT so the harness never makes an
        # under-evidenced decision irreversible. The adapter maps gap-flag -> PUBLIC source; this workflow stays
        # benchmark-name-free (reads only goal.raw). Re-invoking the root agent on the updated state is the caller's
        # job: the re_entry_requested block is a CLEARLY-MARKED handoff carrying the acquired-evidence flags, not a
        # dead end.
        _raw = getattr(goal, "raw", None) or {}
        _dr_gap = _raw.get("decision_readiness_gap") or []
        # SCOPE: preempt fill/commit for a readiness gap ONLY on disposition-CHOICE tasks (authoring_disposition);
        # submission tasks (prior-auth, appeal submission, DME) fall through to completion. Otherwise
        # recovery would abandon a nearly-complete agent submission instead of finishing it.
        if _dr_gap and _raw.get("authoring_disposition", False):
            _dr_sources = _raw.get("decision_readiness_sources") or []
            _acq_done = [h for h in history if str(h).startswith("acquire_evidence:")]
            for _src in _dr_sources:
                _flag = _src.get("flag")
                _atag = "acquire_evidence:%s" % _flag
                if _atag in history:
                    continue                                 # this source already acquired this episode
                # locate the PUBLIC source control by the adapter-named nav keywords (generic label match).
                _kws = [_norm(k) for k in (_src.get("nav_keywords") or []) if k]
                _ctrl = None
                for c in (controls or []):
                    if str(c.get("role") or "") not in ("button", "link", "a", "tab", "menuitem"):
                        continue
                    _lab = _norm(c.get("label") or "")
                    if _lab and _kws and any(k in _lab for k in _kws):
                        _ctrl = c
                        break
                if _ctrl is not None:                         # ACQUIRE: open + read the evidence surface (read-only)
                    return {"kind": "step", "tag": _atag, "step": RecoveryStep(
                        kind=ACQUIRE, name="acquire_evidence",
                        action={"op": "click", "provenance": "harness-acquired-evidence",
                                "evidence_flag": _flag, "ref": _ctrl.get("ref")},
                        affordance_target={"role": _ctrl.get("role"), "label": _ctrl.get("label"),
                                           "ref": _ctrl.get("ref")},
                        manifest={"side_effect_scope": "none", "rollback_available": True,
                                  "server_persisted": False, "read_only": True,
                                  "provenance": "harness-acquired-evidence"})}
                # source NOT visible on the current surface: a semantic route (remittance_image, ...) is
                # NOT a navigable URL and GuiEnvReal.navigate ignores route/ref -> navigate(route) would land on
                # '/' (harmful). Without a route->case-URL resolver and a postcondition check on the acquisition
                # (viewed*==true), no nav is emitted; the source stays un-acquired and the loop falls through to the
                # re-entry handoff.
                continue
            # every named source acquired (or has no locator) -> return control to the ROOT AGENT to re-decide
            # on the acquired evidence. The harness NEVER authors the disposition; this is the re-entry hand-off.
            if _acq_done or not _dr_sources:
                return {"kind": "block", "reason": "re_entry_requested:decision_readiness",
                        "re_entry_requested": True,
                        "acquired_evidence_flags": [h.split(":", 1)[1] for h in _acq_done]}
            # sources named but none reachable on any surface seen yet -> surface the gap (defensive).
            return {"kind": "block", "reason": "decision_readiness_gap_unresolved"}

        # a commit control is eligible only when the agent itself attempted that commit and it did not land
        _agent_refs = set(str(r) for r in ((getattr(goal, "raw", None) or {}).get("agent_commit_refs") or []))
        strong_commit = None                 # label carries a real terminal verb (Send/Submit/Transmit/File X)
        weak_commit = None                   # only the backend's loose commit=True heuristic (e.g. a 'File' menu)
        fillable = []                        # (control, value)
        required_unbound = None
        confirm_ctrl = None
        for c in controls or []:
            label = c.get("label", "")
            _role = str(c.get("role") or "").lower()
            # Even a backend commit=True is NOT the primary terminal commit if it is an
            # INTERMEDIATE action (Add / Save Draft / Update Search) or a DOCUMENTATION action (Save to Epic /
            # Add Note) -- the latter is driven by the post-commit doc-nav, not taken as the payer submit.
            # A NAVIGATION-submit ('Submit to Insurance/Payer/Portal') is a hop to the payer system, NOT the
            # terminal prior-auth submit -- it must NOT be taken as a commit here (the commit phase runs BEFORE
            # _nav_toward, so otherwise stage-nav never gets it).
            _looks_commit = ((c.get("commit") or (_role == "button" and _COMMIT_RE.search(label or "")))
                             and not _NAV_SUBMIT_RE.search(label or "")
                             and not _INTERMEDIATE_RE.search(label or "")
                             and not _DOC_NAV_RE.search(label or ""))
            if _looks_commit:
                if str(c.get("ref")) not in _agent_refs:
                    continue                         # not a commit the agent attempted: never the harness's to make
                # A control the backend flagged commit=True purely via its LOOSE heuristic (list_controls marks
                # a 'File'/'Edit' MENU button through /\bfile\b/) but whose label is NOT a terminal-commit verb
                # is only a WEAK match. A label with a real terminal verb (Send/Submit/Transmit/File X) or a
                # type=submit control is STRONG. Prefer the STRONG terminal so recovery drives the real
                # Send/Submit, never a menu-bar item that merely matched the heuristic (which would waste the
                # whole bounded budget on a no-op submit and land recovery_env_actions with no effect).
                if _COMMIT_RE.search(label or "") or _role == "submit":
                    if strong_commit is None:
                        strong_commit = c
                elif weak_commit is None:
                    weak_commit = c
                continue
            if c.get("role") == "button" and _CONFIRM_RE.search(label or "") and not _NAV_STOP_RE.search(label or ""):
                confirm_ctrl = confirm_ctrl or c
                continue
            role = str(c.get("role") or "").lower()
            is_field = (role in ("input", "textbox", "textarea", "select", "checkbox", "radio", "switch", "toggle")
                        or role in _SELECTISH or c.get("options"))
            if not is_field:
                continue
            if _is_filled(c.get("value")):
                continue                                     # preserve already-filled fields (glyph/placeholder != filled)
            val = self._bind_value(c, sources, (ctx or {}).get("schema"))
            if val is None:
                if c.get("required"):
                    required_unbound = required_unbound or c
                continue
            fillable.append((c, val))

        commit_ctrl = strong_commit or weak_commit           # prefer a real terminal verb over a heuristic hit
        already_committed = any(h.startswith("commit:") for h in history)

        # Before FILL/COMMIT: ACQUIRE unmet EVIDENCE first -- BEFORE payload FILL and the irreversible COMMIT
        # (prior_auth/fax/appeal fields are not purely mechanical; acquiring the required pre-commit evidence must
        # precede filling/submitting an under-evidenced payload). verify_effect names the gap; open the matching
        # control READ-ONLY. Never a commit control (guarded); never authors content -- only the READ the agent skipped.
        # Generic MECHANICAL resolver dispatch over the structured unmet nodes. Only workflow steps the
        # agent SKIPPED that carry no content decision are harness-executed: acquire (read evidence), download
        # (fetch a document), clear (clear the worklist -- ONLY once everything else is satisfied, the mechanical
        # terminal step of an already-complete task). document/attach/fill involve content/binding and stay the
        # agent's (fill/block below). Never a commit control (guarded).
        _eff = self._verify_effect(goal, state_view)
        _non_clear_unmet = sum(1 for u in _eff["unmet"] if u["resolver_type"] != "clear")

        def _find(hints, extra=None):
            for c in controls or []:
                if str(c.get("role") or "") not in ("button", "link", "a", "tab", "menuitem"):
                    continue
                lbl = c.get("label") or ""
                if not lbl or _NAV_STOP_RE.search(lbl) or c.get("commit") or _COMMIT_RE.search(lbl):
                    continue
                nl = _norm(lbl)
                if not nl or not any(h in nl for h in hints):
                    continue
                if extra is not None and not extra(lbl):
                    continue
                return c
            return None

        for _node in _eff["unmet"]:
            _rt = _node["resolver_type"]; _hints = _node["target_hints"]
            if not _hints:
                continue
            if _rt == "acquire":
                c = _find(_hints)
                if c is not None and ("acquire_effect:%s" % c.get("ref")) not in history:
                    return {"kind": "step", "tag": "acquire_effect:%s" % c.get("ref"), "step": RecoveryStep(
                        kind=ACQUIRE, name="acquire_evidence",
                        action={"op": "click", "provenance": "harness-acquired-evidence"},
                        affordance_target={"role": c.get("role"), "label": c.get("label"), "ref": c.get("ref")},
                        manifest={"side_effect_scope": "navigation", "rollback_available": True,
                                  "server_persisted": False, "provenance": "harness-acquired-evidence"})}
            elif _rt == "download":
                c = _find(_hints, extra=lambda l: bool(_DOWNLOAD_RE.search(l))) or _find([], extra=lambda l: bool(_DOWNLOAD_RE.search(l)))
                if c is not None and ("download_effect:%s" % c.get("ref")) not in history:
                    return {"kind": "step", "tag": "download_effect:%s" % c.get("ref"), "step": RecoveryStep(
                        kind=STAGED_WRITE, name="download_document",
                        action={"op": "download", "ref": c.get("ref")},
                        affordance_target={"role": c.get("role"), "label": c.get("label"), "ref": c.get("ref")},
                        manifest={"side_effect_scope": "download", "rollback_available": True, "server_persisted": False})}
            elif _rt == "clear" and _non_clear_unmet == 0:
                c = _find([""], extra=lambda l: bool(_CLEAR_WORKLIST_RE.search(l)))
                if c is None:
                    c = next((x for x in (controls or []) if _CLEAR_WORKLIST_RE.search(str(x.get("label") or ""))
                              and not x.get("commit")), None)
                if c is not None and ("clear_effect:%s" % c.get("ref")) not in history:
                    return {"kind": "step", "tag": "clear_effect:%s" % c.get("ref"), "step": RecoveryStep(
                        kind=STAGED_WRITE, name="clear_worklist",
                        action={"op": "click", "ref": c.get("ref")},
                        affordance_target={"role": c.get("role"), "label": c.get("label"), "ref": c.get("ref")},
                        manifest={"side_effect_scope": "worklist", "rollback_available": True, "server_persisted": True})}

        # COLLECTION EXECUTOR. Add the agent's OWN committed multi-value set BEFORE the single-field fill and
        # the irreversible commit, so the set is complete at submit. Unlike a nudge to the agent, this lands the
        # outcome without agent cooperation; values are strictly the agent's own codes (never invented).
        _ce = self._collection_exec_step(goal, controls, state_view, history)
        if _ce is not None:
            return _ce

        # Operational-field step: complete the clean operational fields first, attributably, no submit.
        _of1 = self._of1_step(goal, controls, sources, history, (ctx or {}).get("schema"))
        if _of1 is not None:
            return _of1

        # 1) FILL the next unfilled bindable field (one at a time -> re-observe).
        for c, val in fillable:
            tag = "fill:%s" % _norm(c.get("name") or c.get("label") or c.get("ref"))
            if tag in history:
                continue                                     # already filled this field; skip
            return {"kind": "step", "tag": tag, "step": RecoveryStep(
                kind=STAGED_WRITE, name="fill",
                action={"op": self._op_for(c), "arg": "_direct", "value": val},
                affordance_target={"role": c.get("role"), "label": c.get("label"), "ref": c.get("ref")},
                manifest={"side_effect_scope": "form_field", "rollback_available": True,
                          "server_persisted": False})}

        # 2) a required field with no bindable value -> the agent must author it -> BLOCK.
        if required_unbound is not None and not already_committed:
            return {"kind": "block", "reason": "needs_new_content:%s" % (required_unbound.get("label") or "")}

        # 3) post-commit CONFIRM step (multi-stage submit).
        if already_committed and confirm_ctrl is not None and ("confirm:%s" % confirm_ctrl.get("ref")) not in history:
            return {"kind": "step", "tag": "confirm:%s" % confirm_ctrl.get("ref"), "step": RecoveryStep(
                kind=STAGED_WRITE, name="confirm",
                action={"op": "click"},
                affordance_target={"role": "button", "label": confirm_ctrl.get("label"), "ref": confirm_ctrl.get("ref")},
                manifest={"side_effect_scope": "confirm", "rollback_available": False, "server_persisted": False})}

        # 4) COMMIT (first submit) OR RE-submit after a validation-revealed fill. The prior submit may have
        #    failed validation and re-rendered a still-empty required field; the FILL phase above refilled it
        #    (a bindable value only, never new content), so submit again. Bounded to avoid duplicate submits:
        #    resubmit ONLY when a fill happened AFTER the last commit, and never more than twice. (step 0 has
        #    already returned done if the effect actually landed, so a resubmit never duplicates a real effect.)
        _commit_idxs = [i for i, h in enumerate(history) if str(h).startswith("commit:")]
        _fill_idxs = [i for i, h in enumerate(history) if str(h).startswith("fill:")]
        _commit_n = len(_commit_idxs)
        _last_commit = _commit_idxs[-1] if _commit_idxs else -1
        _fill_since_commit = bool(_fill_idxs) and _fill_idxs[-1] > _last_commit
        _submit_realized = self._submit_subgoal_realized(goal, state_view)
        # NEVER submit a fresh/EMPTY form. If the CURRENT surface carries no filled payload field, the
        # agent's filled form is elsewhere (recovery navigated to a blank auth queue/form) and submitting would
        # create an EMPTY submission that overwrites the real one (e.g. clicking a 'submit authorizations' nav
        # link on a blank surface). Only commit when the current form actually holds payload; else refuse.
        # A REAL filled payload field is not a hidden input / CSRF token / search query / sidebar filter /
        # read-only patient header: restrict to visible, enabled, editable payload roles OUTSIDE a
        # lookup/search section, with a real value.
        _PAYLOAD_ROLES = {"input", "textbox", "textarea", "select", "combobox", "checkbox", "radio", "spinbutton"}

        def _is_filled_payload(c):
            if c.get("commit") or c.get("disabled") or c.get("readonly") or c.get("visible") is False:
                return False
            if str(c.get("role") or "").lower() not in _PAYLOAD_ROLES:
                return False
            try:
                from workspace_api import is_lookup_section as _lk
                if _lk(c.get("section")):
                    return False
            except Exception:
                pass
            return bool(str(c.get("value") or "").strip())

        _payload_filled = any(_is_filled_payload(c) for c in (controls or []))
        _has_required = any(c.get("required") for c in (controls or []))
        _commit_is_nav = commit_ctrl is not None and (
            "link" in str(commit_ctrl.get("ref") or "").lower()
            or re.search(r"authorizations\b", str(commit_ctrl.get("label") or ""), re.I) is not None)
        if (commit_ctrl is not None and not _submit_realized and not _payload_filled
                and (_has_required or _commit_is_nav)
                and _commit_n == 0):
            # the current surface is a BLANK payload form or a queue/nav link -> submitting would create an
            # empty submission overwriting the agent's real one. Refuse.
            return {"kind": "block", "reason": "empty_form_no_submit"}
        if (commit_ctrl is not None and not _submit_realized
                and _commit_n == 0):    # one completion attempt (the shell refuses a repeat after an unconfirmed one)
            # A NEW irreversible submit is authored ONLY when the ROOT agent formed a verifiable operational
            # commitment (it began this submission). A cold start (task_objective only) must not have the
            # harness submit from scratch -> BLOCK. (When the submit is ALREADY realized we never reach here;
            # the residual is documentation, driven below -- so this gate does not block the write-back.)
            if (getattr(goal, "raw", None) or {}).get("operational_commitment") is False:
                return {"kind": "block", "reason": "no_operational_commitment_cold_start"}
            # SAFETY: do NOT drive an irreversible submit of a decision the PUBLIC policy flagged as formed
            # WITHOUT the required available evidence -- persisting an under-evidenced decision is a harm, not a
            # recovery. BLOCK and surface it; the harness never makes an under-evidenced decision irreversible.
            _graw = getattr(goal, "raw", None) or {}
            if _graw.get("decision_readiness_gap") and _graw.get("authoring_disposition", False):
                return {"kind": "block", "reason": "decision_readiness_gap_unresolved"}
            return {"kind": "step", "tag": "commit:%s:%d" % (commit_ctrl.get("ref"), _commit_n), "step": RecoveryStep(
                kind=IRREVERSIBLE_COMMIT, name="submit",
                action={"op": "submit"},
                affordance_target={"role": "button", "label": commit_ctrl.get("label"), "ref": commit_ctrl.get("ref")},
                manifest={"side_effect_scope": "gui_submission", "rollback_available": False,
                          "autosave_possible": False, "server_persisted": True})}

        # 5) NAVIGATE toward the target when we are not on a fillable surface yet.
        nav = self._nav_toward(goal, controls, history, observation=observation or "")
        if nav is not None:
            return nav

        # 5.5) SCROLL recovery: the target control may not be in the DOM yet (lazy-loading / virtual
        # list). Before declaring no_actionable_control, scroll to reveal more controls, then re-observe.
        # Bounded so it cannot loop; a scroll that revealed nothing new is not repeated past the cap.
        _n_scroll = sum(1 for h in (history or []) if h.startswith("scroll:"))
        if _n_scroll < 3 and _commit_n < 1:
            return {"kind": "step", "tag": "scroll:%d" % _n_scroll, "step": RecoveryStep(
                kind=STAGED_WRITE, name="scroll",
                action={"op": "scroll", "direction": "down"},
                affordance_target={"role": "page", "label": "scroll", "ref": None},
                manifest={"side_effect_scope": "navigation", "rollback_available": True,
                          "autosave_possible": False, "server_persisted": False})}

        # 5.7) POST-SUBMIT DOCUMENTATION (completion conjunction): the submit subgoal is realized (by the ROOT
        # agent OR this recovery) but is_realized is still False -> a subgoal such as EHR documentation is
        # unmet. Gated on _submit_realized OR recovery-committed, NOT _commit_n>=1 alone -- otherwise the
        # very common "root submitted, only documentation missing" case (recovery _commit_n==0) is unreachable.
        if _commit_n >= 1 or _submit_realized:
            for c in controls or []:
                if (str(c.get("role") or "") in ("button", "link", "a", "menuitem", "tab")
                        and _DOC_NAV_RE.search(str(c.get("label") or ""))):
                    dtag = "docnav:%s" % c.get("ref")
                    if dtag not in (history or []):
                        return {"kind": "step", "tag": dtag, "step": RecoveryStep(
                            kind=STAGED_WRITE, name="navigate",
                            action={"op": "click"},
                            affordance_target={"role": c.get("role"), "label": c.get("label"), "ref": c.get("ref")},
                            manifest={"side_effect_scope": "navigation", "rollback_available": True,
                                      "autosave_possible": False, "server_persisted": False})}

        # 6) nothing actionable left. After a commit with unmet subgoals we still yield to verification (the
        # substrate re-reads state); a residual unmet conjunction is surfaced by the caller, not silently "done".
        if _commit_n >= 1:
            return {"kind": "done", "reason": "committed_awaiting_verify"}
        return {"kind": "block", "reason": "no_actionable_control"}

    _RESOLVER_BY_PATH = [
        (re.compile(r"downloaded", re.I), "download"),
        (re.compile(r"viewed|clicked|opened|\bread\b", re.I), "acquire"),
        (re.compile(r"attachmentnames|faxrecipient|recipient", re.I), "attach"),
        (re.compile(r"submissions|submitted|priorauth", re.I), "commit"),
        (re.compile(r"note|documented|rationale|communication", re.I), "document"),
        (re.compile(r"clearedreferrals|cleared", re.I), "clear"),
    ]

    @staticmethod
    def _resolver_for(path):
        p = str(path or "").replace("_", "")
        for rx, rt in GenericGuiCompletionWorkflow._RESOLVER_BY_PATH:
            if rx.search(p):
                return rt
        return "fill"

    @staticmethod
    def _hints_for(path):
        """Content keywords from a subgoal path leaf (viewedRemittanceImage -> ['remittance'];
        clicked_diagnoses_tab -> ['diagnoses']) so the planner can locate the control that resolves it."""
        leaf = str(path or "").split(".")[-1]
        words = re.sub(r"([a-z])([A-Z])", r"\1 \2", leaf).replace("_", " ").lower().split()
        stop = {"viewed", "clicked", "downloaded", "opened", "read", "tab", "image", "go", "to", "the", "a"}
        hints = [w for w in words if w not in stop and len(w) > 2]
        return hints or [w for w in words if len(w) > 2]

    def _verify_effect(self, goal, state_view):
        """The STRUCTURED completion gap (not a bool) -- {satisfied:[path], unmet:[{path, resolver_type,
        target_hints}]} over the verify conjunction. Lets the planner act on the FIRST actionable unmet node
        instead of blindly filling / navigating. Oracle-blind (paths are public state, no gold values)."""
        raw = getattr(goal, "raw", None) or {}
        specs = raw.get("verify_all") or ([raw.get("verify_spec")] if raw.get("verify_spec") else [])
        satisfied, unmet = [], []
        for sp in specs:
            if not isinstance(sp, dict):
                continue
            path = sp.get("path")
            if self._check_one(sp, state_view):
                satisfied.append(path)
            else:
                unmet.append({"path": path, "resolver_type": self._resolver_for(path),
                              "target_hints": self._hints_for(path)})
        return {"satisfied": satisfied, "unmet": unmet}

    @staticmethod
    def _stage_next(stage_map, observation, controls, history):
        """GENERIC stage-aware navigator (engine). Consumes a PUBLIC stage_map (adapter DATA: ordered stages,
        each {name, markers, advance:{keywords}}); identifies the CURRENT stage by which stage's markers are
        present in the live observation, and returns the transition CONTROL to advance to the next stage.
        Benchmark-neutral: no HAB names here -- the map is data. None if no stage matches / no transition
        control found (caller falls back to keyword nav)."""
        if not stage_map:
            return None
        obs = _norm(observation or "")
        cur = -1
        # A route/URL match is AUTHORITATIVE -- it pins the current stage and defeats persistent-sidebar
        # marker pollution (a sidebar 'Submit Authorizations' link must not read as the auth_dashboard stage).
        for i, st in enumerate(stage_map):
            r = st.get("route")
            if r and _norm(r) in obs:
                cur = i
                break
        else:
            # no route match -> most-advanced stage whose markers are present. all_markers requires the WHOLE
            # conjunction (a distinctive fingerprint), not any single generic keyword.
            for i, st in enumerate(stage_map):
                need = st.get("all_markers")
                if need:
                    if all(_norm(m) in obs for m in need if m):
                        cur = i
                elif any(_norm(m) in obs for m in (st.get("markers") or []) if m):
                    cur = i
        if cur < 0:
            return None
        kws = [_norm(k) for k in ((stage_map[cur].get("advance") or {}).get("keywords") or []) if k]
        if not kws:
            return None
        for c in controls or []:
            if str(c.get("role") or "") not in ("button", "link", "a", "tab", "menuitem"):
                continue
            label = c.get("label") or ""
            if not label or _NAV_STOP_RE.search(label):
                continue
            _is_nav_submit = bool(_NAV_SUBMIT_RE.search(label))
            if (c.get("commit") or _COMMIT_RE.search(label)) and not _is_nav_submit:  # SAFETY: never click a commit
                continue                                     # (submit/send/transmit/file). Clicking it as a plain
                                                             # navigation would bypass the operational-commitment
                                                             # check + irreversible-commit authorization + the
                                                             # decision-readiness gate. Terminal controls are the
                                                             # commit phase's job ONLY -- regardless of the map.
            nl = _norm(label)
            if nl and any(kw in nl for kw in kws) and ("stagenav:%s" % c.get("ref")) not in history:
                return c
        return None

    def _nav_toward(self, goal, controls, history, observation=""):
        """Navigation: PREFER the stage-aware navigator when the adapter supplied a PUBLIC stage_map (drives
        the multi-step flow in order); else fall back to greedy keyword-match on the goal's PUBLIC target
        keywords. Never clicks a stop/cancel control; never repeats."""
        raw = getattr(goal, "raw", None) or {}
        _sc = self._stage_next(raw.get("stage_map"), observation, controls, history)
        if _sc is not None:
            return {"kind": "step", "tag": "stagenav:%s" % _sc.get("ref"), "step": RecoveryStep(
                kind=STAGED_WRITE, name="navigate", action={"op": "click"},
                affordance_target={"role": _sc.get("role"), "label": _sc.get("label"), "ref": _sc.get("ref")},
                manifest={"side_effect_scope": "navigation", "rollback_available": True,
                          "autosave_possible": False, "server_persisted": False})}
        kws = [_norm(k) for k in (raw.get("target_keywords") or []) if k]
        if not kws:
            return None
        best = None
        best_score = 0
        for c in controls or []:
            if str(c.get("role") or "") not in ("button", "link", "a", "tab", "menuitem"):
                continue
            label = c.get("label") or ""
            if not label or _NAV_STOP_RE.search(label):
                continue
            nl = _norm(label)
            if not nl:                                       # an empty normalized label (glyph-only chrome
                continue                                     # button, e.g. window-minimize) must NOT match
                                                             # every keyword via 'nl in kw'
            score = sum(1 for kw in kws if kw and kw in nl)
            tag = "nav:%s" % c.get("ref")
            if score > best_score and tag not in history:
                best, best_score = c, score
        if best is None or best_score == 0:
            return None
        return {"kind": "step", "tag": "nav:%s" % best.get("ref"), "step": RecoveryStep(
            kind=STAGED_WRITE, name="navigate",
            action={"op": "click"},
            affordance_target={"role": best.get("role"), "label": best.get("label"), "ref": best.get("ref")},
            manifest={"side_effect_scope": "navigation", "rollback_available": True,
                      "autosave_possible": False, "server_persisted": False})}

    def payload_covered(self, goal, state_view):
        """Oracle-blind PAYLOAD-coverage signal (NOT a gate): are ALL the values the agent COMMITTED
        (committed_fields) reflected in the landed submission state? True -> the effect carried the agent's
        FULL commitment (no truncation) => VERIFIED_PAYLOAD; False -> a committed value is missing =>
        VERIFIED_EFFECT only; None -> nothing committed to measure. NEVER asserts task correctness: whether
        the agent's commitment was itself complete/correct against the hidden required payload is the native
        scorer's domain (VERIFIED_TASK), never claimed by the harness."""
        committed = getattr(goal, "committed_fields", None) or {}
        vals = [str(v).strip() for v in committed.values()
                if v not in (None, "") and not isinstance(v, (dict, list))]
        if not vals:
            return None
        blob = " ".join(str(x) for x in _flatten(state_view or {}).values()
                        if x not in (None, "") and not isinstance(x, (dict, list))).lower()
        return all(v.lower() in blob for v in vals)

    def verify_effect(self, goal, state_view):
        r = self.is_realized(goal, state_view)
        if r is None:
            return None
        if not (state_view or {}):
            return None
        return bool(r)
