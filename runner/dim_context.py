#!/usr/bin/env python3
"""Context dimension — benchmark-AGNOSTIC CONTEXT MANAGEMENT scorer.

Measures whether the agent *managed its context* (acquired / had sufficient / bound to the right
subject / pulled relevant evidence) — NOT whether the final answer is correct (that is Verification /
Outcome). This module supersedes the MedCTA image-answer `context_grounding` / mm_judge "Context" path
in scoring.py: that path judged the ANSWER against the IMAGE (answer correctness); Context here judges
only the EVIDENCE the agent gathered.

Hard 诚信门 invariants (enforced by construction):
  * Consumes ONLY substrate structures: SemanticTrace (map_trace), EvidenceView (evidence_view),
    DimensionPolicy (dimension_policy), CapabilityManifest. No benchmark name, tool literal
    (OCR/fhir_/click/RegionAttribute), image, FHIR resource, or DOM appears in the scoring logic.
  * NEVER reads the final answer or the gold. `relevance` judges the OBSERVATIONS (evidence payloads)
    against the TASK INSTRUCTION only; the `final`/terminal SemanticEvent payload is explicitly dropped.
  * Applicable-only aggregation: a sub-metric with no opportunity returns status=not_applicable and is
    EXCLUDED from the mean (never a vacuous 1.0). Every sub-metric carries a `reportable` flag.

SHARED CONTRACT (v2, typed context units + evidence provenance) — the plugins now emit:
  * required_context_units as a list of TYPED entries {id, type} (a bare string still parses but
    degrades). The semantic TYPE vocabulary is benchmark-declared (patient_identity /
    current_medication_list / allergy_status / case_identity / form_state / submission_requirements /
    target_image_evidence / region_specific_image_evidence / ...).
  * each EvidenceUnit gains a context_type (the semantic TYPE it carries; matched ONE-TO-ONE against a
    required unit's type), plus information-SOURCE provenance:
        source_channel     — the SOURCE family (radiology_image / fhir_patient_record / gui_portal /
                             external_web); the unit of INDEPENDENCE for cross-source corroboration,
        source_instance_id — the specific instance within that channel (image id / Patient/<id> / url);
                             this is also the SUBJECT identity that `binding` converges on,
        extractor          — the tool/model that read it (OCR / fhir_read / ...).
  Cross-source corroboration counts INDEPENDENT (source_channel, source_instance_id) pairs — NOT distinct
  payload hashes (two OCR reads of the SAME image are ONE source, not two).

Sub-metrics (applicable-only):
  acquisition  — were the dimension_policy.required_context_units obtained, matched ONE-TO-ONE by
                 semantic TYPE: a required unit of type T is satisfied ONLY by an EvidenceUnit whose
                 context_type==T (one evidence unit never fills two required units; unrelated
                 acquisitions cannot auto-fill distinct units). Legacy bare-string units degrade to a
                 distinct-evidence-kind cap. CONTRACT-A: only units delivered_to_agent AND
                 usable_for_context (semantic_status=='success' AND a non-empty progress_token) count — a
                 delivered-but-PARTIAL empty result (empty FHIR Bundle / blank OCR) does NOT satisfy a unit.
  sufficiency  — REUSES the acquisition TYPE-match result (matched required units / required units), so
                 acquisition and sufficiency cannot disagree; gated on required milestones too. Computed
                 from the PRE-BOUNDARY trace prefix only.
  binding      — CONTRACT-C: TWO sub-signals. subject_consistency = evidence converges on ONE dominant
                 SUBJECT identity (source_instance_id / subject:<Type>/<id> token — the EXPECTED subject,
                 NOT a resource's own id, never a broad numeric regex over a year/exam/dose). This sub-metric
                 carries the consistency score.
  expected_subject_match — CONTRACT-C: the SEPARATE signal that the converged subject == dimension_policy.
                 expected_subject (a {type,id} derived from the task). 'Consistently reading the WRONG
                 patient' -> binding consistency high yet expected_subject_match 0 (and that 0 enters the
                 Context mean). not_applicable when the policy declares no expected_subject.
  relevance    — delivered+usable evidence pertains to the task (gateway judge over OBSERVATIONS only;
                 strict parse: RELEVANT->1, IRRELEVANT->0, anything else -> status=error, never fail-open).

Information-leak fix / CONTRACT-D: there is ONE canonical CONTEXT-BOUNDARY predicate — a context boundary
event = terminal in (final, escalate) OR event_role=='commit'. EVERY pre-terminal/pre-commit cutoff here
(acquire prefix, evidence prefix, milestone prefix) uses that SAME predicate, so a post-submit/post-commit
CONFIRMATION can NOT back-fill Context. Context also does NOT require a boundary-only milestone (e.g. a
submit's form_submitted) — that floor belongs to Execution/Lifecycle; such milestones are stripped from the
Context milestone requirement (derived from event_role/terminal, never a literal milestone name).

tier = experimental.
"""
import os
import re

try:
    import substrate as _sub
except Exception:                       # pragma: no cover - allow `from runner import`
    from runner import substrate as _sub  # type: ignore

try:
    from lifecycle_exec import _sm, _aggregate          # reuse the canonical sub-metric helpers
except Exception:                                       # pragma: no cover
    def _sm(score, status="valid", opportunities=None, **kw):
        d = {"score": score, "status": status}
        if opportunities is not None:
            d["opportunities"] = opportunities
        d.update(kw)
        return d

    def _aggregate(subs):
        valid = {k: v for k, v in subs.items()
                 if v.get("status") == "valid" and isinstance(v.get("score"), (int, float))}
        score = round(sum(v["score"] for v in valid.values()) / len(valid), 3) if valid else None
        vals = [v["score"] for v in valid.values()]
        return {"score": score, "submetrics": subs, "applicable_submetrics": sorted(valid),
                "n_applicable": len(valid), "zero_variance": (len(set(vals)) == 1) if vals else None}

CONTEXT_VERSION = "context-1.3-experimental"

# ---------------------------------------------------------------------------- semantic typing
# A required_context_unit is declared as a TYPED entry {"id": ..., "type": "patient_identity"|
# "allergy_status"|"target_image_evidence"|...} OR (legacy) a bare string. The acquisition sub-metric
# matches each required unit to evidence ONE-TO-ONE by semantic TYPE so three IRRELEVANT acquisitions can
# NOT fill three distinct required units. When the policy carries only bare strings we degrade gracefully
# (see _acquisition) but still cap obtained by the number of DISTINCT evidence kinds and never
# double-count one evidence unit across many required units.
#
# An EvidenceUnit's semantic TYPE is read FIRST from the plugin-tagged `context_type` (the contract field);
# when absent (legacy traces / default extractor) it is derived from the SEMANTIC progress_token family
# the substrate emits (content-/type-keyed, never a tool name) and aliased to a type word. Token families:
#   evidence:<source>:<hash8>      -> kind "evidence:<source>"     (OCR text, search snippet, image desc)
#   region:<hash8>:resolved        -> kind "image_region"
#   resource:<Type>/<id>:created   -> kind "resource:<Type>"        (carries a typed subject id)
#   state:read=<Type>/<id>         -> kind "resource:<Type>"        (carries a typed subject id)
#   subject:<Type>/<id>            -> kind "subject:<Type>"         (the EXPECTED subject identity)
#   state:search=<hash8>           -> kind "search"
#   state:page=<hash8>             -> kind "page_state"
#   state:submitted=<hash8>        -> kind "submission"
#   state:<k>=<...>                -> kind "state:<k>"
_TYPED_RES_RE = re.compile(r"^(?:resource:|state:read=)([A-Za-z][A-Za-z0-9_]*)/(.+?)(?::created)?$")
# a SUBJECT token names the expected subject/case identity directly (subject:<Type>/<id>), distinct from a
# resource's OWN id — this is the binding target the contract requires.
_SUBJECT_RE = re.compile(r"^subject:([A-Za-z][A-Za-z0-9_]*)/(.+)$")


def evidence_semantic_kind(progress_token):
    """Benchmark-agnostic SEMANTIC kind of an acquire/commit evidence unit, read from its progress
    token's structural family. None when the token carries no semantic content. Never a tool name."""
    t = str(progress_token or "")
    if not t:
        return None
    m = _SUBJECT_RE.match(t)
    if m:
        return "subject:%s" % m.group(1)
    m = _TYPED_RES_RE.match(t)
    if m:
        return "resource:%s" % m.group(1)          # typed subject (FHIR resourceType / record kind)
    if t.startswith("evidence:"):
        parts = t.split(":")
        return "evidence:%s" % (parts[1] if len(parts) > 1 else "generic")
    if t.startswith("region:"):
        return "image_region"
    if t.startswith("state:search="):
        return "search"
    if t.startswith("state:page="):
        return "page_state"
    if t.startswith("state:submitted="):
        return "submission"
    if t.startswith("state:"):
        k = t[len("state:"):].split("=", 1)[0]
        return "state:%s" % k if k else "state"
    return None


def _typed_subject_id(progress_token):
    """A subject/case identifier carried by a TYPED token — the only place an identifier is GUARANTEED to
    name a subject/case rather than being an arbitrary number (year/dose/exam #). Order of preference:
    a SUBJECT token (subject:<Type>/<id>, the expected subject identity) first, else a resource token
    (resource:<Type>/<id> / state:read=<Type>/<id>). Returns (kind, id) or None. No broad regex."""
    t = str(progress_token or "")
    m = _SUBJECT_RE.match(t)
    if m:
        rtype, rid = m.group(1), m.group(2).strip()
        return ("subject:%s" % rtype, rid) if rid else None
    m = _TYPED_RES_RE.match(t)
    if not m:
        return None
    rtype, rid = m.group(1), m.group(2).strip()
    if not rid:
        return None
    return ("resource:%s" % rtype, rid)


def _unit_type(unit):
    """The declared SEMANTIC type of a required_context_unit. Supports the contract typed entry
    {"id":..., "type": ...} (and {"unit_type": ...}/{"kind": ...} aliases); a bare string has NO declared
    type (returns None -> graceful-degrade path)."""
    if isinstance(unit, dict):
        return unit.get("type") or unit.get("unit_type") or unit.get("kind")
    return None


def evidence_context_type(unit):
    """The semantic TYPE an EvidenceUnit carries, for ONE-TO-ONE matching against a required unit's type.
    Reads the plugin-tagged `context_type` (the contract field). Returns a lower-cased type word or None
    (None -> fall back to the structural progress_token kind for legacy/default-extractor traces)."""
    if not isinstance(unit, dict):
        return None
    ct = unit.get("context_type")
    if ct:
        return str(ct).lower()
    return None


# ---------------------------------------------------------------- CONTRACT-A: usable_for_context
# An EvidenceUnit is ACQUIRED context only when it was delivered_to_agent AND usable_for_context. Per the
# shared contract, usable_for_context = (semantic_status == 'success' AND a non-empty progress_token); the
# producing SemanticEvent's status (success|partial|failure) is the unit's semantic_status. A
# delivered-but-PARTIAL result (an empty FHIR Bundle, a blank OCR/page) is delivered+partial -> NOT acquired
# context (it still feeds Observability/relevance, but cannot satisfy a required unit). A plugin MAY tag the
# unit with semantic_status / usable_for_context directly; when it has not, we DERIVE semantic_status by
# joining the unit back to its SemanticEvent (via the "<tool>#<idx>" id convention / trace_index) and fall
# back to the unit's own status / a partial-when-no-progress_token default. We never fail-open: an
# ambiguous/unjoinable unit needs a non-empty progress_token to be usable.
def _status_by_trace_idx(sem_trace):
    """Map a raw-trace index -> the producing SemanticEvent's status, so an EvidenceUnit (id "<tool>#<idx>"
    or trace_index) can recover its semantic_status when the plugin did not tag it on the unit."""
    out = {}
    for i, s in enumerate(sem_trace or []):
        raw = s.get("raw") if isinstance(s.get("raw"), dict) else None
        idx = raw.get("_idx") if raw else None
        if idx is None:
            idx = i
        out[idx] = s.get("status")
    return out


def _semantic_status(u, status_by_idx):
    """The producing SemanticEvent's status for an EvidenceUnit. Preference: an explicit unit-level
    semantic_status (the contract field a plugin MAY set) > the joined SemanticEvent status (by trace index)
    > the unit's own 'status'. None when undiscoverable (treated as non-success below)."""
    if isinstance(u, dict):
        ss = u.get("semantic_status")
        if ss:
            return str(ss).lower()
    idx = _unit_trace_idx(u)
    if idx is not None and idx in status_by_idx and status_by_idx[idx]:
        return str(status_by_idx[idx]).lower()
    st = u.get("status") if isinstance(u, dict) else None
    return str(st).lower() if st else None


def _usable_for_context(u, status_by_idx):
    """CONTRACT-A: usable_for_context = (semantic_status == 'success' AND a non-empty progress_token).
    Honors an explicit unit-level usable_for_context flag when a plugin set it; otherwise derives it.

    A non-empty TYPED context_type counts as the required non-empty progress signal too: the plugins emit a
    typed context_type ONLY when real context was obtained and set it to None on errors / blank pages /
    empty bundles (the very 'delivered-but-partial' case this excludes), so (progress_token OR context_type)
    is exactly 'a non-empty semantic token of the result'. An empty/blank delivered result (semantic_status
    partial, OR neither token) is NOT usable context.

    semantic_status defaults to 'success' ONLY when it is genuinely undiscoverable (no joinable
    SemanticEvent, no unit-level status) AND the unit still carries a non-empty token — a hand-built/legacy
    delivered unit with content is usable; it never fails-open a TOKENLESS or explicitly-partial unit."""
    if isinstance(u, dict) and "usable_for_context" in u:
        return bool(u.get("usable_for_context"))
    pt = (u.get("progress_token") if isinstance(u, dict) else None)
    ct = (u.get("context_type") if isinstance(u, dict) else None)
    has_token = bool(str(pt or "").strip()) or bool(str(ct or "").strip())
    if not has_token:
        return False                                   # tokenless/blank -> never usable
    status = _semantic_status(u, status_by_idx)
    if status is None:
        status = "success"                             # undiscoverable status + real content -> success
    return status == "success"


def _delivered(evidence, sem_trace=None):
    """CONTRACT-A: a unit counts as ACQUIRED context only when delivered_to_agent AND usable_for_context.
    When sem_trace is None (legacy callers), fall back to delivered_to_agent only (back-compat; callers
    that enforce CONTRACT-A pass sem_trace so usable_for_context is applied)."""
    units = [u for u in (evidence or []) if u.get("delivered_to_agent")]
    if sem_trace is None:
        return units
    status_by_idx = _status_by_trace_idx(sem_trace)
    return [u for u in units if _usable_for_context(u, status_by_idx)]


# Generic type-aliasing so a structural evidence kind can satisfy a domain-named required type WITHOUT
# any benchmark literal: we only ever match on token-family words, never on a tool/resource name. Used
# ONLY for evidence units that carry no plugin `context_type` (legacy / default extractor); a tagged
# context_type is matched directly type==type.
#
# A typed unit is satisfied ONLY by an evidence kind in its SPECIFIC alias set. The aliases are
# token-family words (resource:<rtype> / evidence:<src> / image_region / search / page_state), NOT a
# benchmark/tool literal. We deliberately do NOT give a medical type the blanket "resource:" alias —
# otherwise a Patient-identity unit would be auto-filled by an unrelated Observation/Encounter resource,
# which is exactly the "3 unrelated acquisitions fill 3 distinct units" bug this module fixes.
_TYPE_ALIASES = {
    "patient_identity": ("resource:patient", "subject:patient", "search"),
    "patient_record": ("resource:patient", "subject:patient", "search"),
    "correct_patient": ("resource:patient", "subject:patient", "search"),
    "current_medication_list": ("resource:medicationrequest", "resource:medicationstatement",
                                "resource:medication"),
    "allergy_status": ("resource:allergyintolerance",),
    "current_medications": ("resource:medicationrequest", "resource:medicationstatement",
                            "resource:medication"),
    "current_medication": ("resource:medicationrequest", "resource:medicationstatement",
                           "resource:medication"),
    "target_image_region": ("image_region",),
    "image_region": ("image_region",),
    "region_specific_image_evidence": ("image_region",),
    "target_image_evidence": ("image_region", "evidence:imagedescription", "evidence:ocr"),
    "text_evidence": ("evidence:ocr",),
    "external_reference": ("evidence:search", "search"),
    "case_identity": ("page_state", "resource:case", "subject:case"),
    "current_form_state": ("page_state",),
    "form_state": ("page_state",),
    "page_state": ("page_state",),
    "submission_requirements": ("page_state", "submission"),
    "correct_case": ("page_state", "resource:case", "subject:case"),
}


def _kind_matches_type(kind, unit_type):
    """Does a structural evidence KIND satisfy a declared required-unit TYPE? Benchmark-agnostic:
    matches ONLY against the SPECIFIC alias set for that type (token-family words, never a tool/resource
    literal). No broad "any resource" fallback — an unrelated resource kind must NOT satisfy a typed unit
    (that is the very bug this module fixes). For a type with no alias entry, we degrade to an exact
    family-prefix equality so it still fails closed against unrelated kinds."""
    if not kind or not unit_type:
        return False
    k = str(kind).lower()
    ut = str(unit_type).lower()
    aliases = _TYPE_ALIASES.get(ut)
    if aliases is not None:
        for fam in aliases:
            if k == fam or (fam.endswith(":") and k.startswith(fam)):
                return True
        return False
    # unknown type: accept only when the type word IS the kind family (e.g. type "search" -> kind
    # "search", type "resource:patient" -> kind "resource:patient"). No loose substring auto-match.
    return k == ut or k.startswith(ut + ":") or ("/" not in k and ut.startswith(k))


def _evidence_offer(unit):
    """The TYPE word an evidence unit offers for one-to-one matching: its tagged context_type if present
    (matched type==type, the contract path), else its structural kind (matched via the alias table).
    Returns (mode, value) where mode is 'context_type' or 'kind'; value None when neither exists."""
    ct = evidence_context_type(unit)
    if ct:
        return ("context_type", ct)
    return ("kind", evidence_semantic_kind(unit.get("progress_token")))


def _offer_matches(offer, unit_type):
    """Does an evidence offer (mode, value) satisfy a required unit TYPE? A tagged context_type matches
    ONE-TO-ONE by type equality (contract: context_type==required type); a structural kind matches through
    the benchmark-agnostic alias table."""
    mode, val = offer
    if val is None or not unit_type:
        return False
    if mode == "context_type":
        return str(val).lower() == str(unit_type).lower()
    return _kind_matches_type(val, unit_type)


# ---------------------------------------------------------------- CONTRACT-D: context boundary
# CONTRACT-D: there is ONE canonical "context boundary" predicate, and EVERY pre-terminal/pre-commit cutoff
# in this module (acquire prefix, evidence prefix, milestone prefix) uses it -- so a post-submit/post-commit
# confirmation can NEVER back-fill Context. A context boundary event = a TERMINAL in (final, escalate) OR an
# event whose role is 'commit' (the submit/commit that closes the task). Context is everything the agent had
# BEFORE that boundary.
def _is_context_boundary(s):
    """The single context-boundary predicate (CONTRACT-D): terminal in (final, escalate) OR
    event_role == 'commit'."""
    return s.get("terminal") in ("final", "escalate") or s.get("event_role") == "commit"


def _boundary_milestones_from_plugin(policy):
    """The AUTHORITATIVE set of boundary (commit/terminal) milestones from the benchmark plugin's
    tool_semantics: a milestone whose ONLY producing tools are commit-role (the submit/commit) is a boundary
    milestone, INDEPENDENT of whether the agent actually submitted (so form_submitted is dropped even when
    the agent never reached the submit). Resolved via policy.governance_policy_id (the benchmark name the
    policy already carries) -> the plugin -> its tool_semantics role/success_milestones; this reads only the
    structural ROLE, never a tool literal in the scoring logic. Empty when no plugin resolves (legacy /
    test policies) -- the caller then falls back to the trace-derived classification."""
    bench = (policy or {}).get("governance_policy_id")
    if not bench:
        return set()
    try:
        plugin = _sub.get_plugin(bench)
    except Exception:
        plugin = None
    tsem = (plugin or {}).get("tool_semantics") or {}
    if not tsem:
        return set()
    commit_ms, noncommit_ms = set(), set()
    for _tool, meta in tsem.items():
        if not isinstance(meta, dict):
            continue
        ms = set(meta.get("success_milestones") or [])
        if not ms:
            continue
        if meta.get("role") == "commit":
            commit_ms.update(ms)
        else:
            noncommit_ms.update(ms)
    return commit_ms - noncommit_ms     # produced ONLY by a commit tool


def _boundary_milestones(sem_trace):
    """Trace-derived fallback: milestones produced ONLY by a context-boundary event (commit/terminal) in the
    OBSERVED trace. Used when no plugin resolves from the policy. Benchmark-agnostic: derived from
    event_role/terminal, never a literal milestone name. A milestone also earned by a NON-boundary
    (acquire/act/verify) event is kept as a legitimate Context milestone."""
    boundary_ms, pre_ms = set(), set()
    for s in sem_trace or []:
        added = s.get("milestones_added") or []
        if not added:
            continue
        if _is_context_boundary(s):
            boundary_ms.update(added)
        else:
            pre_ms.update(added)
    return boundary_ms - pre_ms


def _context_required_milestones(policy, sem_trace):
    """The required milestones Context is allowed to gate on: the policy's required_milestones MINUS any
    boundary (commit/submit) milestone (CONTRACT-D drops the commit/submit milestone requirement --
    form_submitted and the like belong to Execution/Lifecycle, NOT Context). The plugin's tool_semantics is
    the authoritative source (works even when the agent never submitted); the trace-derived set is unioned
    in as a fallback for legacy/test policies with no resolvable plugin. Returns a set."""
    req_ms = set(policy.get("required_milestones") or [])
    boundary = _boundary_milestones_from_plugin(policy) | _boundary_milestones(sem_trace)
    return req_ms - boundary


# ---------------------------------------------------------------- pre-terminal scoping
def _acquire_prefix(sem_trace):
    """The SemanticEvent prefix BEFORE the first context boundary (CONTRACT-D predicate) — used so
    post-boundary acquisitions/milestones cannot raise Context (information-leak fix)."""
    pre = []
    for s in sem_trace:
        if _is_context_boundary(s):
            break
        pre.append(s)
    return pre


def _terminal_trace_idx(sem_trace):
    """Raw-trace index of the first context-boundary event (CONTRACT-D: final/escalate OR commit), or None.
    Used to cut the parallel EvidenceView so units rendered AT/AFTER the boundary (incl. a post-submit
    confirmation) do not raise Context — the SAME predicate _acquire_prefix uses, so evidence and the
    semantic prefix cut at the identical point."""
    for i, s in enumerate(sem_trace):
        if _is_context_boundary(s):
            raw = s.get("raw") if isinstance(s.get("raw"), dict) else {}
            idx = raw.get("_idx")
            return idx if idx is not None else i
    return None


def _unit_trace_idx(u):
    """Best-effort trace index of an EvidenceUnit from its id "<tool>#<i>" (the substrate/extractor id
    convention) or an explicit trace_index field. None when not parseable."""
    if isinstance(u, dict) and isinstance(u.get("trace_index"), int):
        return u["trace_index"]
    m = re.search(r"#(\d+)$", str((u or {}).get("id") or ""))
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def _pre_terminal_evidence(sem_trace, evidence):
    """Evidence units delivered BEFORE the first terminal. The EvidenceView is parallel to the tool_call
    events; we drop any unit whose embedded trace index is >= the first terminal's index. When neither a
    terminal index nor unit indices are discoverable (older traces) we keep all units — binding/relevance
    still drop the terminal payload explicitly."""
    term_idx = _terminal_trace_idx(sem_trace)
    if term_idx is None:
        return evidence
    out = []
    for u in evidence:
        idx = _unit_trace_idx(u)
        if idx is None or idx < term_idx:
            out.append(u)
    return out


# ---------------------------------------------------------------- acquisition
def _acquisition(sem_trace, evidence, policy):
    """Map required_context_units to evidence by SEMANTIC TYPE, not raw acquire-event count. Returns the
    sub-metric AND (as `matched_units`/`required_total`) the one-to-one TYPE-match result so sufficiency
    REUSES it (acquisition and sufficiency cannot disagree).

    TYPED policy  ({type: ...} entries): a required unit is satisfied ONLY by a delivered/earned evidence
       unit whose context_type (or, legacy, structural kind) matches its declared type. Each evidence
       OFFER fills at most ONE required unit (no double-counting one evidence unit across many units).
    BARE policy   (plain strings): no per-unit type to match against, so we degrade gracefully — but we
       refuse to let unrelated acquisitions auto-satisfy distinct units. obtained is capped by the count
       of DISTINCT evidence SEMANTIC KINDS actually backed by delivered evidence.
    No required units -> not_applicable (no opportunity, never a vacuous 1.0)."""
    req_units = list(policy.get("required_context_units") or [])
    # CONTRACT-D: drop any boundary-only (commit/submit, e.g. form_submitted) milestone -- Context does NOT
    # require the submit milestone (that is Execution/Lifecycle's).
    req_ms = _context_required_milestones(policy, sem_trace)
    n_req = len(req_units)
    if not req_units:
        return _sm(None, "not_applicable", 0, reportable=False,
                   matched_units=0, required_total=0,
                   reason="policy declares no required_context_units")

    pre = _acquire_prefix(sem_trace)                                  # pre-terminal prefix only
    # CONTRACT-A: only delivered AND usable_for_context (success + progress_token) units count toward
    # acquisition; a delivered-but-PARTIAL empty result does NOT satisfy a required unit.
    deliv = _delivered(_pre_terminal_evidence(sem_trace, evidence), sem_trace)

    typed = [u for u in req_units if _unit_type(u)]
    reached = _sub.milestones_reached(pre)
    ms_hit = req_ms & reached

    if typed and len(typed) == n_req:
        # fully-typed policy: ONE-TO-ONE match. Build each delivered evidence unit's offer (tagged
        # context_type preferred, else structural kind). Each OFFER fills at most ONE required unit.
        # Fall back to earned acquire/commit trace tokens when the evidence view carries no offer.
        offers = [o for o in (_evidence_offer(u) for u in deliv) if o[1] is not None]
        if not offers:
            offers = [("kind", evidence_semantic_kind(s.get("progress_token")))
                      for s in pre if s.get("event_role") in ("acquire", "commit")
                      and s.get("status") == "success" and s.get("progress_token")]
            offers = [o for o in offers if o[1] is not None]

        used = [False] * len(offers)
        cand = {}
        for u in req_units:
            ut = _unit_type(u)
            cand[id(u)] = (ut, [i for i, o in enumerate(offers) if _offer_matches(o, ut)])
        # process MOST CONSTRAINED units first (fewest candidate offers) so a scarce specific offer is not
        # consumed by a looser unit -- deterministic, order-independent.
        order = sorted(req_units, key=lambda u: len(cand[id(u)][1]))
        matched = 0
        matched_pairs = []

        def _generic(i):
            mode, val = offers[i]
            return mode == "kind" and str(val).lower() in ("search", "page_state", "submission")

        for u in order:
            ut, idxs = cand[id(u)]
            avail = [i for i in idxs if not used[i]]
            if not avail:
                continue
            avail.sort(key=lambda i: (_generic(i), str(offers[i][1])))   # prefer SPECIFIC offers
            hit = avail[0]
            used[hit] = True
            matched += 1
            matched_pairs.append({"unit": ut, "evidence_kind": offers[hit][1], "via": offers[hit][0]})
        score = round(min(1.0, matched / max(1, n_req)), 3)
        offered = sorted({str(o[1]) for o in offers})
        return _sm(score, "valid", n_req, reportable=True,
                   matching="typed", required_units=[_unit_type(u) for u in req_units],
                   distinct_evidence_kinds=offered, matched_units=matched, required_total=n_req,
                   matched_pairs=matched_pairs,
                   required_milestones=sorted(req_ms), milestones_reached=sorted(ms_hit))

    # ----- graceful degrade (bare-string units, or mixed): cap by DISTINCT evidence kinds -----
    deliv_kinds = {evidence_semantic_kind(u.get("progress_token")) for u in deliv}
    deliv_kinds.discard(None)
    earned_kinds = {evidence_semantic_kind(s.get("progress_token")) for s in pre
                    if s.get("event_role") in ("acquire", "commit") and s.get("status") == "success"
                    and s.get("progress_token")}
    earned_kinds.discard(None)
    kinds = deliv_kinds or earned_kinds

    obtained = min(len(kinds), n_req)
    note = "bare_units_distinct_kind_cap"
    if req_ms:
        ms_cover = min(len(ms_hit), n_req)
        if ms_cover > obtained:
            obtained = ms_cover
            note = "bare_units_milestone_cover"
    score = round(min(1.0, obtained / max(1, n_req)), 3)
    return _sm(score, "valid", n_req, reportable=True,
               matching="degraded", required_units=[(_unit_type(u) or u) for u in req_units],
               distinct_evidence_kinds=sorted(kinds), obtained_context_kinds=obtained, note=note,
               matched_units=obtained, required_total=n_req,
               required_milestones=sorted(req_ms), milestones_reached=sorted(ms_hit))


# ---------------------------------------------------------------- sufficiency
def _sufficiency(sem_trace, policy, acquisition):
    """Did the agent have ENOUGH context to proceed: a boolean floor that REUSES the acquisition TYPE-match
    result (NOT a raw acquire-event count) so acquisition and sufficiency cannot disagree — every required
    context unit was TYPE-matched AND every required milestone reached, all from the PRE-TERMINAL prefix.
    Distinct from acquisition only in being a 0/1 floor vs a graded ratio."""
    req_units = list(policy.get("required_context_units") or [])
    # CONTRACT-D: gate ONLY on non-boundary required milestones (the submit/commit milestone is dropped --
    # Context must NOT require form_submitted; that floor lives in Execution/Lifecycle).
    req_ms = _context_required_milestones(policy, sem_trace)
    if not req_units and not req_ms:
        return _sm(None, "not_applicable", 0, reportable=False, reason="no required units/milestones to gate on")

    matched = acquisition.get("matched_units", 0)
    required_total = acquisition.get("required_total", len(req_units))
    units_ok = (not req_units) or (required_total > 0 and matched >= required_total)
    pre = _acquire_prefix(sem_trace)                       # milestones from the pre-terminal prefix only
    ms_ok = req_ms.issubset(_sub.milestones_reached(pre))
    score = 1.0 if (units_ok and ms_ok) else 0.0
    return _sm(score, "valid", 1, reportable=True,
               method="reuses_acquisition_type_match",
               required_units=len(req_units), units_type_matched=matched,
               required_milestones=sorted(req_ms), milestones_satisfied=ms_ok)


# ---------------------------------------------------------------- binding
def _subject_from_unit(u):
    """The SUBJECT identity an EvidenceUnit binds to — the EXPECTED subject, NOT a resource's own id.
    Preference order (most subject-specific first):
      1. an explicit SUBJECT token the plugin attaches (subject_token 'subject:<Type>/<id>', or a
         subject_id/subject field) — the contract's designated binding target, so ten Observation reads of
         ONE patient share ONE subject and do NOT scatter binding;
      2. a subject:<Type>/<id> family carried on the progress_token;
      3. ONLY as a last resort, the provenance pair (source_channel, source_instance_id) — used when no
         coarser subject is declared. Because source_instance_id may be a per-resource id, this is the
         fallback, never the primary, so a multi-resource read of one subject is not mis-scattered.
    Returns (kind, id) or None. Never a bare numeric/free-text guess."""
    # 1. explicit subject token / field
    for key in ("subject_token", "subject_id", "subject"):
        val = u.get(key)
        if not val:
            continue
        if isinstance(val, str):
            t = _typed_subject_id(val)
            if t:
                return t
            return ("subject", val)
    # 2. subject token on the progress_token
    t = _typed_subject_id(u.get("progress_token"))
    if t and str(t[0]).startswith("subject:"):
        return t
    # 3. provenance pair fallback
    ch = u.get("source_channel")
    inst = u.get("source_instance_id")
    if ch and inst:
        return ("channel:%s" % str(ch).lower(), str(inst))
    return t            # last: a resource:<Type>/<id> token from the progress_token (legacy)


def _expected_subject(policy):
    """The expected subject the task is ABOUT (CONTRACT-C: dimension_policy.expected_subject =
    {type, id}, derived from the task, e.g. context.patient_ref). Returns (type, id) lower/stripped, or None
    when the policy declares no expected subject (-> expected_subject_match is not_applicable). Accepts a few
    shapes: {'type','id'} | {'resourceType','id'} | a bare 'Type/id' string."""
    es = (policy or {}).get("expected_subject")
    if not es:
        return None
    if isinstance(es, str):
        t = _typed_subject_id(es) or _typed_subject_id("subject:%s" % es)
        if t:
            return (str(t[0]).split(":", 1)[-1].lower(), str(t[1]).strip())
        if "/" in es:
            ty, _, i = es.partition("/")
            return (ty.strip().lower(), i.strip())
        return None
    if isinstance(es, dict):
        ty = es.get("type") or es.get("resourceType") or es.get("kind")
        i = es.get("id") or es.get("reference") or es.get("ref")
        if i and isinstance(i, str) and "/" in i and not ty:
            ty, _, i = i.partition("/")
        if i:
            return (str(ty or "").strip().lower(), str(i).strip())
    return None


def _norm_subject_id(kind, rid):
    """Normalize a bound (kind, id) to the comparable (type_word, id) the expected_subject is expressed in:
    strip the 'subject:'/'resource:'/'channel:' family prefix off the kind, and strip a 'Type/' prefix off
    the id so 'Patient/MRN1' and ('subject:Patient','MRN1') compare equal."""
    k = str(kind or "")
    for pre in ("subject:", "resource:", "channel:"):
        if k.startswith(pre):
            k = k[len(pre):]
            break
    k = k.lower()
    i = str(rid or "").strip()
    if "/" in i:
        ty, _, tail = i.partition("/")
        # keep the id tail; the type moves into k if k was generic
        if not k or k in ("channel", ""):
            k = ty.strip().lower()
        i = tail.strip()
    return (k, i)


def _collect_subjects(sem_trace, evidence):
    """The typed subject identities (kind, id) backing the gathered evidence — delivered + usable_for_context
    units first (CONTRACT-A), then earned acquire/commit trace tokens. Shared by both binding sub-signals."""
    # binding is subject-CONVERGENCE over delivered evidence; CONTRACT-A's usable filter governs ACQUISITION
    # (which unit satisfies a required context unit), not which delivered reads a subject converges over, so
    # binding uses delivered-only (pre-boundary) -- a real read carries a subject regardless of token shape.
    deliv = _delivered(_pre_terminal_evidence(sem_trace, evidence))
    pre = _acquire_prefix(sem_trace)
    typed = []
    for u in deliv:
        s = _subject_from_unit(u)
        if s:
            typed.append(s)
    if not typed:
        for s in pre:
            if s.get("event_role") in ("acquire", "commit") and s.get("status") == "success":
                t = _typed_subject_id(s.get("progress_token"))
                if t:
                    typed.append(t)
    return typed


def _binding(sem_trace, evidence, policy=None):
    """CONTRACT-C: binding reports TWO sub-signals (both attached to this sub-metric; `context()` also
    surfaces them as two scored submetrics):
      * subject_consistency      — does the evidence converge on ONE dominant subject within each subject
                                   KIND (the legacy binding signal; this sub-metric's `score`);
      * expected_subject_match   — is that dominant subject == dimension_policy.expected_subject? (0/1, or
                                   not_applicable when the policy declares no expected subject). A
                                   CONSISTENTLY-WRONG patient -> subject_consistency high but
                                   expected_subject_match 0.
    Binding uses ONLY the contract provenance pair (source_channel, source_instance_id) OR a typed
    SUBJECT/resource token — NOT a broad numeric/ID regex over free-text (year / dose / exam number) and NOT
    a resource's OWN id (ten Observation reads of one patient share ONE subject). Applicable only when typed
    subject identities exist; otherwise not_applicable (never a vacuous score, never a bare-number guess)."""
    from collections import Counter, defaultdict
    typed = _collect_subjects(sem_trace, evidence)
    if not typed:
        return _sm(None, "not_applicable", 0, reportable=False,
                   reason="no typed subject identity (source_instance_id / subject:<Type>/<id>) in evidence")

    by_kind = defaultdict(Counter)
    for kind, rid in typed:
        by_kind[kind][rid] += 1
    kind_scores = {}
    details = {}
    dominant = {}
    for kind, ctr in by_kind.items():
        total = sum(ctr.values())
        dom_id, dom_n = ctr.most_common(1)[0]
        kind_scores[kind] = round(dom_n / max(1, total), 3)
        dominant[kind] = dom_id
        details[kind] = {"dominant_id": dom_id, "dominant_n": dom_n, "total": total,
                         "distinct_ids": len(ctr)}
    score = round(sum(kind_scores.values()) / max(1, len(kind_scores)), 3)

    # CONTRACT-C: expected_subject_match (vs policy.expected_subject). not_applicable when no expected
    # subject is declared. A match requires SOME bound dominant subject to equal the expected (type, id).
    exp = _expected_subject(policy)
    exp_match = {"status": "not_applicable", "score": None}
    if exp:
        exp_norm = (str(exp[0]).lower(), str(exp[1]).strip())
        matched_kind = None
        for kind, dom_id in dominant.items():
            if _norm_subject_id(kind, dom_id) == exp_norm or \
               (exp_norm[0] in ("", None) and _norm_subject_id(kind, dom_id)[1] == exp_norm[1]):
                matched_kind = kind
                break
        exp_match = {"status": "valid", "score": 1.0 if matched_kind else 0.0,
                     "expected_subject": {"type": exp[0], "id": exp[1]},
                     "matched_kind": matched_kind,
                     "bound_dominant": {k: v for k, v in dominant.items()}}
    return _sm(score, "valid", len(typed), reportable=True,
               method="typed_subject_identity", per_kind_focus=kind_scores,
               typed_subjects=len(typed), subject_kinds=sorted(by_kind), details=details,
               subject_consistency=score, expected_subject_match=exp_match)


# ---------------------------------------------------------------- cross-source corroboration
def _corroboration(sem_trace, evidence):
    """How many INDEPENDENT information sources backed the gathered evidence: counts distinct
    (source_channel, source_instance_id) pairs — NOT distinct payload hashes, so two OCR reads of the SAME
    image are ONE source, not two. Reported as an info signal (not folded into the mean). not_applicable
    when no provenance-tagged evidence (legacy traces)."""
    deliv = _delivered(_pre_terminal_evidence(sem_trace, evidence))   # delivered (provenance-counted) only
    pairs = set()
    for u in deliv:
        ch = u.get("source_channel")
        inst = u.get("source_instance_id")
        if ch and inst:
            pairs.add((str(ch).lower(), str(inst)))
    if not pairs:
        return _sm(None, "not_applicable", 0, reportable=False,
                   reason="no (source_channel, source_instance_id) provenance on evidence units")
    channels = sorted({c for c, _i in pairs})
    n = len(pairs)
    score = 1.0 if n >= 2 else 0.0          # >=2 independent sources -> corroborated; single -> 0.0
    return _sm(score, "valid", n, reportable=True,
               method="independent_source_instances", independent_sources=n, channels=channels)


# ---------------------------------------------------------------- relevance (judge over OBSERVATIONS)
def _parse_relevance(content, judge_model, n_units):
    """STRICT relevance parse (no fail-open). First non-empty line, upper-cased, MUST be exactly RELEVANT
    or IRRELEVANT. Anything else (UNKNOWN / empty / a hedged sentence / leading prose) -> status='error',
    score None — never silently credited as relevant."""
    head = ""
    for line in str(content or "").splitlines():
        if line.strip():
            head = line.strip().upper()
            break
    if head == "RELEVANT":
        return _sm(1.0, "valid", 1, reportable=True, judge_model=judge_model,
                   judge_tier="gateway_observation_relevance", judge_backend=judge_model,
                   n_evidence_units=n_units, verdict="RELEVANT", reason=(content or "")[:200])
    if head == "IRRELEVANT":
        return _sm(0.0, "valid", 1, reportable=True, judge_model=judge_model,
                   judge_tier="gateway_observation_relevance", judge_backend=judge_model,
                   n_evidence_units=n_units, verdict="IRRELEVANT", reason=(content or "")[:200])
    return _sm(None, "error", 1, reportable=False,
               judge_model=judge_model, judge_tier="gateway_observation_relevance",
               reason="non-strict relevance verdict (head=%r); not fail-open to relevant" % head[:40])


def _relevance(sem_trace, evidence, task_instruction, judge_model, char_budget=7000, per_unit=700):
    """Gateway judge: do the delivered OBSERVATIONS pertain to the task? Reads evidence payloads +
    the task INSTRUCTION only. The terminal/final SemanticEvent is excluded so the answer never leaks;
    gold is never passed. STRICT parse (no fail-open). No judge backend / instruction / evidence ->
    not_applicable; a judge call that errors or returns a non-strict verdict -> status='error'."""
    deliv = _delivered(_pre_terminal_evidence(sem_trace, evidence))   # delivered observations only
    instr = str(task_instruction or "").strip()
    if not deliv or not instr:
        return _sm(None, "not_applicable", 0, reportable=False, reason="no delivered evidence or no task instruction")
    if not judge_model or os.environ.get("MH_CONTEXT_JUDGE", "1") == "0":
        return _sm(None, "not_applicable", 0, reportable=False, reason="relevance judge backend unavailable/disabled")

    parts, used = [], 0
    for u in deliv:
        seg = "- %s" % str(u.get("payload"))[:per_unit]
        if used + len(seg) > char_budget and parts:
            break
        parts.append(seg)
        used += len(seg)
    obs = "\n".join(parts)
    try:
        import gateway
        sysp = ("You judge whether the EVIDENCE an agent gathered is RELEVANT to the stated task. "
                "Judge only the evidence/observations against the task description — do NOT judge whether "
                "any conclusion is correct, and there is no answer key. Reply with exactly RELEVANT or "
                "IRRELEVANT on the first line, then a one-line reason.")
        usr = "TASK:\n%s\n\nEVIDENCE GATHERED:\n%s" % (instr[:1500], obs[:char_budget])
        r = gateway.chat([{"role": "system", "content": sysp}, {"role": "user", "content": usr}],
                         model=judge_model, max_tokens=200, judge=True)
        if not r.get("ok"):
            return _sm(None, "error", 1, reportable=False,
                       reason="relevance judge error: %s" % r.get("error_type"))
        return _parse_relevance(r.get("content"), judge_model, len(parts))
    except Exception as ex:
        return _sm(None, "error", 1, reportable=False, reason="relevance judge exception: %s" % ex)


# ---------------------------------------------------------------- entry point
def context(sem_trace, evidence, dimension_policy, task_instruction=None, judge_model=None):
    """Context-management score from substrate structures only.

    sem_trace        : list[SemanticEvent]  (substrate.map_trace(trace, plugin))
    evidence         : list[EvidenceUnit]   (substrate.evidence_view(trace, plugin))
    dimension_policy : dict                 (substrate.dimension_policy(task, plugin))
    task_instruction : str | None  — the task GOAL/instruction text (NOT the gold, NOT the answer).
                       Used by the relevance judge only; safe because it is the problem statement.
    judge_model      : str | None  — gateway model for the relevance judge; None -> relevance skipped.
    """
    policy = dimension_policy or {}
    if judge_model is None:
        judge_model = os.environ.get("MH_JUDGE_MODEL")
    # tag each SemanticEvent's raw with its trace position so pre-terminal scoping can locate the terminal.
    for i, s in enumerate(sem_trace):
        raw = s.get("raw")
        if isinstance(raw, dict) and "_idx" not in raw:
            raw["_idx"] = i
    acq = _acquisition(sem_trace, evidence, policy)
    bind = _binding(sem_trace, evidence, policy)
    # CONTRACT-C: surface binding's TWO sub-signals as distinct scored submetrics. `binding` keeps the
    # subject-CONSISTENCY score (convergence on one subject); `expected_subject_match` is the SEPARATE
    # signal that the converged subject == policy.expected_subject -- so 'consistently reading the WRONG
    # patient' shows binding (consistency) high yet expected_subject_match 0 (and lowers the Context mean).
    esm = bind.get("expected_subject_match") if isinstance(bind, dict) else None
    if isinstance(esm, dict) and esm.get("status") == "valid":
        esm_sub = _sm(esm.get("score"), "valid", 1, reportable=True,
                      method="expected_subject_match", **{k: v for k, v in esm.items()
                                                          if k not in ("status", "score")})
    else:
        esm_sub = _sm(None, "not_applicable", 0, reportable=False,
                      reason="dimension_policy declares no expected_subject")
    subs = {
        "acquisition": acq,
        "sufficiency": _sufficiency(sem_trace, policy, acq),
        "binding": bind,
        "expected_subject_match": esm_sub,
        "relevance": _relevance(sem_trace, evidence, task_instruction, judge_model),
    }
    corr = _corroboration(sem_trace, evidence)            # info signal; reported, NOT folded into the mean
    out = _aggregate(subs)
    out["dimension"] = "Context"
    out["tier"] = "experimental"
    out["evaluator_version"] = CONTEXT_VERSION
    out["measures"] = "context_management"               # explicitly NOT answer correctness
    out["reads_final_or_gold"] = False
    out["reportable"] = bool(out.get("n_applicable"))
    out["governance_policy_id"] = policy.get("governance_policy_id")
    out["corroboration"] = corr                           # cross-source independence (not in the mean)
    return out


# ---------------------------------------------------------------- self-verification harness
def _selfcheck():
    import json
    import sys
    base = os.path.expanduser("~/Medical_harness")
    sys.path.insert(0, os.path.join(base, "runner"))

    def _load(bundle, sb):
        with open(os.path.join(base, bundle, "trajectory.jsonl")) as f:
            trace = [json.loads(l) for l in f if l.strip()]
        tj = os.path.join(base, bundle, "task.json")
        task = json.load(open(tj)) if os.path.exists(tj) else {"source_benchmark": sb}
        task.setdefault("source_benchmark", sb)
        return trace, task

    cases = [("results_mctaGov/gpt5/MCTA-0", "MedCTA"),
             ("results_pb_chk3/gpt5/PB-aberrant_drug_screen", "PhysicianBench"),
             ("results_hab10/gpt5/HAB-denial-easy-1", "HealthAdminBench")]
    for bundle, sb in cases:
        trace, task = _load(bundle, sb)
        plugin = _sub.get_plugin(sb)
        sem = _sub.map_trace(trace, plugin)
        ev = _sub.evidence_view(trace, plugin)
        pol = _sub.dimension_policy(task, plugin)
        instr = (task.get("context") or {}).get("text") or task.get("goal")
        res = context(sem, ev, pol, task_instruction=instr, judge_model=None)
        print("\n==== %s [%s] ====" % (bundle, sb))
        print(" score       :", res["score"], "| applicable:", res["applicable_submetrics"],
              "| reportable:", res["reportable"], "| tier:", res["tier"])
        for k, v in res["submetrics"].items():
            print("   %-12s %-15s score=%s  %s" % (
                k, v.get("status"), v.get("score"),
                {kk: vv for kk, vv in v.items() if kk not in ("score", "status")}))
        c = res["corroboration"]
        print("   corroboration  :", c.get("status"), c.get("score"),
              c.get("channels") or c.get("reason"))

    # ---- synthetic invariant 1: 3 UNRELATED same-kind acquisitions must NOT fill 3 required units ----
    print("\n==== SYNTHETIC 1: 3 unrelated same-kind acquisitions vs 3 required units ====")
    sem3 = [_sub.semantic_event("acquire", status="success", capability_id="t",
                                progress_token="evidence:search:%s" % h)
            for h in ("aaaaaaaa", "bbbbbbbb", "cccccccc")]            # 3 distinct tokens, SAME kind
    ev3 = [{"id": "e#%d" % i, "delivered_to_agent": True, "delivery_fidelity": 1.0,
            "error_visible": False, "payload": "snippet %d" % i,
            "progress_token": s["progress_token"]} for i, s in enumerate(sem3)]
    pol3 = {"required_context_units": ["correct_patient", "current_medications", "allergy_status"],
            "required_milestones": []}
    acq = _acquisition(sem3, ev3, pol3)
    print("   acquisition:", acq.get("score"), "distinct_kinds=", acq.get("distinct_evidence_kinds"),
          "obtained=", acq.get("obtained_context_kinds"))
    assert acq["score"] <= round(1 / 3, 3) + 1e-9, ("3 unrelated same-kind acq must NOT fill 3 units", acq)
    print("   PASS: 3 unrelated same-kind acquisitions do NOT fill 3 distinct units (score=%s)" % acq["score"])

    # ---- synthetic invariant 2: typed policy via tagged context_type maps one-to-one by TYPE ----
    print("\n==== SYNTHETIC 2: typed policy, context_type one-to-one ====")
    pol_t = {"required_context_units": [{"id": "p", "type": "patient_identity"},
                                        {"id": "a", "type": "allergy_status"},
                                        {"id": "m", "type": "current_medication_list"}]}
    ev_ct = [
        {"id": "u#0", "delivered_to_agent": True, "delivery_fidelity": 1.0, "error_visible": False,
         "payload": "p", "context_type": "patient_identity"},
        {"id": "u#1", "delivered_to_agent": True, "delivery_fidelity": 1.0, "error_visible": False,
         "payload": "p", "context_type": "allergy_status"},
        {"id": "u#2", "delivered_to_agent": True, "delivery_fidelity": 1.0, "error_visible": False,
         "payload": "p", "context_type": "current_medication_list"}]
    acqt = _acquisition([], ev_ct, pol_t)
    print("   typed(context_type) acquisition:", acqt.get("score"), acqt.get("matched_pairs"))
    assert acqt["matching"] == "typed" and acqt["score"] == 1.0, acqt
    assert all(p["via"] == "context_type" for p in acqt["matched_pairs"]), acqt

    # one evidence unit cannot fill TWO required units (3 units, only 1 patient_identity evidence)
    ev_one = [{"id": "u#0", "delivered_to_agent": True, "delivery_fidelity": 1.0, "error_visible": False,
               "payload": "p", "context_type": "patient_identity"}]
    acq_one = _acquisition([], ev_one, pol_t)
    assert acq_one["score"] == round(1 / 3, 3), ("one evidence unit fills exactly one unit", acq_one)

    # legacy untagged evidence still maps via structural kind (back-compat)
    sem_t = [
        _sub.semantic_event("acquire", status="success", progress_token="state:read=Patient/42"),
        _sub.semantic_event("acquire", status="success", progress_token="state:read=AllergyIntolerance/7"),
        _sub.semantic_event("acquire", status="success", progress_token="state:read=MedicationRequest/9")]
    ev_t = [{"id": "u#%d" % i, "delivered_to_agent": True, "delivery_fidelity": 1.0, "error_visible": False,
             "payload": "p", "progress_token": s["progress_token"]} for i, s in enumerate(sem_t)]
    acq_legacy = _acquisition(sem_t, ev_t, pol_t)
    assert acq_legacy["score"] == 1.0 and acq_legacy["matching"] == "typed", acq_legacy

    # an unrelated SPECIFIC resource kind does NOT satisfy the typed units
    sem_x = [_sub.semantic_event("acquire", status="success", progress_token="state:read=Encounter/%d" % i)
             for i in (1, 2, 3)]
    ev_x = [{"id": "x#%d" % i, "delivered_to_agent": True, "delivery_fidelity": 1.0, "error_visible": False,
             "payload": "p", "progress_token": s["progress_token"]} for i, s in enumerate(sem_x)]
    acqx = _acquisition(sem_x, ev_x, pol_t)
    assert acqx["score"] == 0.0, ("3 unrelated Encounter resources must NOT fill 3 typed units", acqx)
    print("   PASS: context_type one-to-one; one unit can't double-fill; unrelated kinds 0.0")

    # ---- synthetic invariant 3: acquisition == sufficiency (cannot disagree) ----
    print("\n==== SYNTHETIC 3: acquisition and sufficiency reuse the same TYPE match ====")
    polms = {"required_context_units": [{"id": "p", "type": "patient_identity"},
                                        {"id": "a", "type": "allergy_status"}],
             "required_milestones": []}
    suff_ok = _sufficiency([], polms, _acquisition([], ev_ct, polms))
    assert suff_ok["score"] == 1.0 and suff_ok["units_type_matched"] == 2, suff_ok
    acq_partial = _acquisition([], ev_one, polms)
    suff_partial = _sufficiency([], polms, acq_partial)
    assert acq_partial["score"] < 1.0 and suff_partial["score"] == 0.0, (acq_partial, suff_partial)
    print("   PASS: full coverage -> both pass; partial -> acq<1 and sufficiency floor 0")

    # ---- synthetic invariant 4: binding to SUBJECT (multi-resource of one patient does not tank) ----
    print("\n==== SYNTHETIC 4: binding to SUBJECT, not the resource's own id ====")
    ev_subj = [{"id": "b#%d" % i, "delivered_to_agent": True, "delivery_fidelity": 1.0,
                "error_visible": False, "payload": "taken in 2024, dose 500 mg",
                "source_channel": "fhir_patient_record", "source_instance_id": "Patient/MRN1",
                "extractor": "fhir_read"} for i in range(10)]
    b = _binding([], ev_subj)
    print("   binding (10 Observations, one patient):", b.get("score"), b.get("per_kind_focus"))
    assert b["status"] == "valid" and b["score"] == 1.0, ("one subject -> 1.0, must not tank", b)
    ev_scatter = ev_subj + [{"id": "b#10", "delivered_to_agent": True, "delivery_fidelity": 1.0,
                             "error_visible": False, "payload": "x",
                             "source_channel": "fhir_patient_record",
                             "source_instance_id": "Patient/MRN2", "extractor": "fhir_read"}]
    bs = _binding([], ev_scatter)
    assert bs["score"] < 1.0, ("two distinct patient subjects -> binding < 1.0", bs)
    # the contract interlock: an explicit subject_token converges binding even when source_instance_id
    # scatters per-resource (10 Observation/<id> reads of ONE patient -> binding 1.0, not 1/10)
    ev_scatter_inst = [{"id": "o#%d" % i, "delivered_to_agent": True, "delivery_fidelity": 1.0,
                        "error_visible": False, "payload": "obs",
                        "source_channel": "fhir_patient_record",
                        "source_instance_id": "Observation/%d" % (190330 + i),   # per-resource id scatter
                        "subject_token": "subject:Patient/MRN1"} for i in range(10)]
    bsi = _binding([], ev_scatter_inst)
    print("   binding (10 scattering Observation ids, one subject_token):", bsi.get("score"),
          bsi.get("per_kind_focus"))
    assert bsi["score"] == 1.0, ("subject_token must converge over per-resource source_instance_id", bsi)
    sem_subj = [_sub.semantic_event("acquire", status="success", progress_token="subject:Patient/Z"),
                _sub.semantic_event("acquire", status="success", progress_token="subject:Patient/Z")]
    ev_tok = [{"id": "t#%d" % i, "delivered_to_agent": True, "delivery_fidelity": 1.0, "error_visible": False,
               "payload": "p", "progress_token": s["progress_token"]} for i, s in enumerate(sem_subj)]
    bt = _binding(sem_subj, ev_tok)
    assert bt["status"] == "valid" and bt["score"] == 1.0, bt
    ev_none = [{"id": "n#0", "delivered_to_agent": True, "delivery_fidelity": 1.0, "error_visible": False,
                "payload": "year 2024 dose 500 mg exam 12345", "progress_token": "evidence:ocr:deadbeef"}]
    bn = _binding([], ev_none)
    assert bn["status"] == "not_applicable", bn
    print("   PASS: binds on SUBJECT identity; 10 obs of one patient = 1.0; bare numbers never bind")

    # ---- synthetic invariant 5: cross-source corroboration counts source INSTANCES, not payloads ----
    print("\n==== SYNTHETIC 5: corroboration counts independent (channel, instance) pairs ====")
    ev_same = [{"id": "s#%d" % i, "delivered_to_agent": True, "delivery_fidelity": 1.0,
                "error_visible": False, "payload": "read %d (different bytes)" % i,
                "source_channel": "radiology_image", "source_instance_id": "img/1",
                "extractor": "OCR"} for i in range(2)]
    csame = _corroboration([], ev_same)
    print("   2 OCR reads of same image:", csame.get("score"), "n=", csame.get("independent_sources"))
    assert csame["independent_sources"] == 1 and csame["score"] == 0.0, csame
    ev_two = ev_same + [{"id": "s#9", "delivered_to_agent": True, "delivery_fidelity": 1.0,
                         "error_visible": False, "payload": "snippet", "source_channel": "external_web",
                         "source_instance_id": "http://x", "extractor": "GoogleSearch"}]
    ctwo = _corroboration([], ev_two)
    assert ctwo["independent_sources"] == 2 and ctwo["score"] == 1.0, ctwo
    print("   PASS: two reads of one image = 1 source; image+web = 2 independent sources")

    # ---- synthetic invariant 6: relevance STRICT parse (no fail-open on UNKNOWN) ----
    print("\n==== SYNTHETIC 6: relevance strict parse ====")
    assert _parse_relevance("RELEVANT\nbecause...", "m", 1)["score"] == 1.0
    assert _parse_relevance("IRRELEVANT\nbecause...", "m", 1)["score"] == 0.0
    for bad in ("UNKNOWN", "", "Maybe relevant", "I think it is RELEVANT", "RELEVANT-ish"):
        r = _parse_relevance(bad, "m", 1)
        assert r["status"] == "error" and r["score"] is None, ("UNKNOWN must be error, not 1", bad, r)
    print("   PASS: only exact RELEVANT/IRRELEVANT scored; UNKNOWN/empty/hedged -> error (not 1)")

    # ---- synthetic invariant 7: post-terminal evidence does NOT raise Context ----
    print("\n==== SYNTHETIC 7: post-terminal evidence does not raise Context ====")
    pol7 = {"required_context_units": [{"id": "p", "type": "patient_identity"},
                                       {"id": "a", "type": "allergy_status"}],
            "required_milestones": []}
    sem7 = [
        _sub.semantic_event("acquire", status="success", progress_token="state:read=Patient/1"),
        _sub.semantic_event("final", terminal="final"),
        _sub.semantic_event("acquire", status="success", progress_token="state:read=AllergyIntolerance/2")]
    sem7[0]["raw"] = {"_idx": 0}
    sem7[1]["raw"] = {"_idx": 1}
    sem7[2]["raw"] = {"_idx": 2}
    ev7 = [{"id": "fhir#0", "delivered_to_agent": True, "delivery_fidelity": 1.0, "error_visible": False,
            "payload": "p", "progress_token": "state:read=Patient/1"},
           {"id": "fhir#2", "delivered_to_agent": True, "delivery_fidelity": 1.0, "error_visible": False,
            "payload": "p", "progress_token": "state:read=AllergyIntolerance/2"}]
    acq7 = _acquisition(sem7, ev7, pol7)
    print("   acquisition with a post-terminal allergy read:", acq7.get("score"), acq7.get("matched_pairs"))
    assert acq7["matched_units"] == 1, ("post-terminal evidence must NOT count toward acquisition", acq7)
    assert acq7["score"] == round(1 / 2, 3), acq7
    suff7 = _sufficiency(sem7, pol7, acq7)
    assert suff7["score"] == 0.0, ("post-terminal completion must not satisfy sufficiency", suff7)
    bnd7 = _binding(sem7, ev7)
    assert bnd7["details"]["resource:Patient"]["total"] == 1, ("post-terminal allergy not bound", bnd7)
    assert "resource:AllergyIntolerance" not in (bnd7.get("per_kind_focus") or {}), bnd7
    print("   PASS: evidence after the final answer does not raise acquisition/sufficiency/binding")

    # ---- synthetic invariant 8: CONTRACT-A usable_for_context (empty/partial result not acquired) ----
    print("\n==== SYNTHETIC 8: CONTRACT-A empty/partial result is delivered but NOT acquired ====")
    pol8 = {"required_context_units": [{"id": "p", "type": "patient_identity"},
                                       {"id": "a", "type": "allergy_status"}]}
    sem8 = [_sub.semantic_event("acquire", status="success", progress_token="state:read=Patient/1"),
            _sub.semantic_event("acquire", status="partial", progress_token=None)]   # empty bundle -> partial
    sem8[0]["raw"] = {"_idx": 0}; sem8[1]["raw"] = {"_idx": 1}
    ev8 = [{"id": "fhir#0", "delivered_to_agent": True, "delivery_fidelity": 1.0, "error_visible": False,
            "payload": "p", "context_type": "patient_identity", "progress_token": "state:read=Patient/1"},
           {"id": "fhir#1", "delivered_to_agent": True, "delivery_fidelity": 1.0, "error_visible": False,
            "payload": "", "context_type": "allergy_status", "progress_token": None}]   # delivered+partial
    acq8 = _acquisition(sem8, ev8, pol8)
    print("   acquisition (1 good + 1 empty/partial):", acq8.get("score"), acq8.get("matched_pairs"))
    assert acq8["matched_units"] == 1 and acq8["score"] == round(1 / 2, 3), \
        ("a delivered-but-partial empty result must NOT satisfy a required unit", acq8)
    # an explicit unit-level semantic_status='partial' is also excluded even with a token present
    ev8b = ev8[:1] + [dict(ev8[1], progress_token="state:read=AllergyIntolerance/2",
                           semantic_status="partial")]
    acq8b = _acquisition(sem8, ev8b, pol8)
    assert acq8b["matched_units"] == 1, ("semantic_status=partial unit not usable", acq8b)
    # explicit usable_for_context=False overrides even a success-looking unit
    ev8c = ev8[:1] + [dict(ev8[1], progress_token="state:read=AllergyIntolerance/2",
                           usable_for_context=False)]
    assert _acquisition(sem8, ev8c, pol8)["matched_units"] == 1, "usable_for_context=False excludes unit"
    print("   PASS: empty/partial (no token / status partial / usable_for_context False) not acquired")

    # ---- synthetic invariant 9: CONTRACT-D post-submit confirmation does not back-fill Context ----
    print("\n==== SYNTHETIC 9: CONTRACT-D post-commit/submit confirmation does not back-fill Context ====")
    pol9 = {"required_context_units": [{"id": "c", "type": "case_identity"},
                                       {"id": "s", "type": "submission_requirements"}],
            "required_milestones": ["form_submitted"]}            # a COMMIT/submit milestone
    sem9 = [
        _sub.semantic_event("acquire", status="success", capability_id="navigate",
                            progress_token="state:page=case1", milestones_added=["target_page_reached"]),
        _sub.semantic_event("commit", status="success", capability_id="submit",
                            progress_token="state:submitted=abcd1234", milestones_added=["form_submitted"]),
        # post-submit CONFIRMATION page (a submission_confirmation, a DIFFERENT type) appears AFTER commit
        _sub.semantic_event("verify", status="success", capability_id="snapshot",
                            progress_token="state:page=confirm")]
    for i, s in enumerate(sem9):
        s["raw"] = {"_idx": i}
    ev9 = [
        {"id": "navigate#0", "delivered_to_agent": True, "delivery_fidelity": 1.0, "error_visible": False,
         "payload": "case 1 page", "context_type": "case_identity",
         "source_channel": "gui_portal", "source_instance_id": "case1", "extractor": "gui",
         "progress_token": "state:page=case1"},
        # the submission_requirements unit only appears on the POST-submit confirmation page -> must NOT
        # satisfy the pre-submit submission_requirements unit (post-commit, idx 2 >= boundary idx 1)
        {"id": "snapshot#2", "delivered_to_agent": True, "delivery_fidelity": 1.0, "error_visible": False,
         "payload": "submission confirmed", "context_type": "submission_requirements",
         "source_channel": "gui_portal", "source_instance_id": "confirm", "extractor": "gui",
         "progress_token": "state:page=confirm"}]
    acq9 = _acquisition(sem9, ev9, pol9)
    print("   acquisition (post-submit confirmation evidence):", acq9.get("score"), acq9.get("matched_pairs"),
          "| required_milestones gated:", acq9.get("required_milestones"))
    assert acq9["matched_units"] == 1, ("post-commit confirmation must NOT satisfy a Context unit", acq9)
    # CONTRACT-D: Context must NOT require the form_submitted (commit) milestone -> it is dropped
    assert "form_submitted" not in (acq9.get("required_milestones") or []), \
        ("Context must NOT require the form_submitted commit milestone", acq9)
    suff9 = _sufficiency(sem9, pol9, acq9)
    assert "form_submitted" not in (suff9.get("required_milestones") or []), suff9
    # a pre-submit case_identity alone, with NO form_submitted requirement, is still partial (missing the
    # submission_requirements unit), so the post-submit confirmation genuinely cannot back-fill it
    assert suff9["score"] == 0.0, ("missing pre-submit unit -> sufficiency 0 (not back-filled)", suff9)
    print("   PASS: post-submit confirmation does not satisfy submission_requirements; form_submitted dropped")

    # ---- synthetic invariant 10: CONTRACT-C consistently-WRONG subject -> expected_subject_match 0 ----
    print("\n==== SYNTHETIC 10: CONTRACT-C expected_subject_match (consistently wrong patient) ====")
    pol10 = {"required_context_units": [{"id": "p", "type": "patient_identity"}],
             "expected_subject": {"type": "Patient", "id": "RIGHT"}}
    # 6 reads, ALL of the WRONG patient -> highly CONSISTENT but the WRONG subject
    ev_wrong = [{"id": "fhir#%d" % i, "delivered_to_agent": True, "delivery_fidelity": 1.0,
                 "error_visible": False, "payload": "p", "context_type": "patient_identity",
                 "source_channel": "fhir_patient_record", "source_instance_id": "Patient/WRONG",
                 "subject_token": "subject:Patient/WRONG", "progress_token": "state:read=Patient/WRONG"}
                for i in range(6)]
    b_wrong = _binding([], ev_wrong, pol10)
    print("   binding consistency:", b_wrong.get("subject_consistency"),
          "| expected_subject_match:", b_wrong["expected_subject_match"].get("score"))
    assert b_wrong["subject_consistency"] == 1.0, ("wrong-but-consistent -> consistency 1.0", b_wrong)
    assert b_wrong["expected_subject_match"]["score"] == 0.0, \
        ("consistently reading the WRONG patient must fail expected_subject_match", b_wrong)
    # the RIGHT patient -> match 1.0
    ev_right = [dict(u, source_instance_id="Patient/RIGHT", subject_token="subject:Patient/RIGHT",
                     progress_token="state:read=Patient/RIGHT") for u in ev_wrong]
    b_right = _binding([], ev_right, pol10)
    assert b_right["expected_subject_match"]["score"] == 1.0, ("right subject -> match 1.0", b_right)
    # no expected_subject in policy -> expected_subject_match not_applicable (never vacuous)
    b_noexp = _binding([], ev_right, {"required_context_units": [{"id": "p", "type": "patient_identity"}]})
    assert b_noexp["expected_subject_match"]["status"] == "not_applicable", b_noexp
    # the full context() surfaces expected_subject_match as a SEPARATE scored submetric that lowers the mean
    out10 = context([], ev_wrong, pol10)
    esm10 = out10["submetrics"]["expected_subject_match"]
    assert esm10["status"] == "valid" and esm10["score"] == 0.0, esm10
    assert "expected_subject_match" in out10["applicable_submetrics"], out10
    print("   PASS: consistent-but-wrong subject -> consistency 1.0, expected_subject_match 0.0 (in the mean)")

    print("\nALL SYNTHETIC INVARIANTS PASSED")


if __name__ == "__main__":
    _selfcheck()
