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
                 distinct-evidence-kind cap.
  sufficiency  — REUSES the acquisition TYPE-match result (matched required units / required units), so
                 acquisition and sufficiency cannot disagree; gated on required milestones too. Computed
                 from the PRE-TERMINAL trace prefix only.
  relevance    — delivered evidence pertains to the task (gateway judge over OBSERVATIONS only; strict
                 parse: RELEVANT->1, IRRELEVANT->0, anything else -> status=error, never fail-open).
  binding      — evidence bound to a single consistent SUBJECT identity (the source_instance_id /
                 subject:<Type>/<id> token — the EXPECTED subject, NOT a resource's own id), never a
                 broad numeric regex that could mistake a year / exam number / dose for a subject id.

Information-leak fix: delivered evidence AND reached milestones are computed from the trace PREFIX up to
the first terminal (final/escalate/commit) only — evidence/milestones appearing AFTER the final answer
must NOT raise Context.

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

CONTEXT_VERSION = "context-1.2-experimental"

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


def _delivered(evidence):
    return [u for u in (evidence or []) if u.get("delivered_to_agent")]


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


# ---------------------------------------------------------------- pre-terminal scoping
def _acquire_prefix(sem_trace):
    """The SemanticEvent prefix BEFORE the first terminal (final/escalate/commit) — used so post-terminal
    acquisitions/milestones cannot raise Context (information-leak fix)."""
    pre = []
    for s in sem_trace:
        if s.get("terminal") in ("final", "escalate") or s.get("event_role") == "commit":
            break
        pre.append(s)
    return pre


def _terminal_trace_idx(sem_trace):
    """Raw-trace index of the first terminal (final/escalate) event, or None. Used to cut the parallel
    EvidenceView so units rendered after the final answer do not raise Context."""
    for s in sem_trace:
        if s.get("terminal") in ("final", "escalate"):
            raw = s.get("raw") or {}
            return raw.get("_idx")
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
    req_ms = set(policy.get("required_milestones") or [])
    n_req = len(req_units)
    if not req_units:
        return _sm(None, "not_applicable", 0, reportable=False,
                   matched_units=0, required_total=0,
                   reason="policy declares no required_context_units")

    pre = _acquire_prefix(sem_trace)                                  # pre-terminal prefix only
    deliv = _delivered(_pre_terminal_evidence(sem_trace, evidence))

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
    req_ms = set(policy.get("required_milestones") or [])
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


def _binding(sem_trace, evidence):
    """Subject-consistency check binding each evidence unit to its EXPECTED SUBJECT identity, using ONLY
    the contract provenance pair (source_channel, source_instance_id) OR a typed SUBJECT/resource token —
    NOT a broad numeric/ID regex over free-text payloads (which could bind on a year / dose / exam number)
    and NOT a resource's OWN id (ten Observation reads of one patient share ONE subject, so they do not
    scatter binding). Binding asks whether the subjects converge on ONE dominant subject within each
    subject KIND. Applicable only when typed subject identities exist; otherwise not_applicable (never a
    vacuous score, never a guess from a bare number)."""
    from collections import Counter, defaultdict
    deliv = _delivered(_pre_terminal_evidence(sem_trace, evidence))   # pre-terminal only
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
    if not typed:
        return _sm(None, "not_applicable", 0, reportable=False,
                   reason="no typed subject identity (source_instance_id / subject:<Type>/<id>) in evidence")

    by_kind = defaultdict(Counter)
    for kind, rid in typed:
        by_kind[kind][rid] += 1
    kind_scores = {}
    details = {}
    for kind, ctr in by_kind.items():
        total = sum(ctr.values())
        dom_id, dom_n = ctr.most_common(1)[0]
        kind_scores[kind] = round(dom_n / max(1, total), 3)
        details[kind] = {"dominant_id": dom_id, "dominant_n": dom_n, "total": total,
                         "distinct_ids": len(ctr)}
    score = round(sum(kind_scores.values()) / max(1, len(kind_scores)), 3)
    return _sm(score, "valid", len(typed), reportable=True,
               method="typed_subject_identity", per_kind_focus=kind_scores,
               typed_subjects=len(typed), subject_kinds=sorted(by_kind), details=details)


# ---------------------------------------------------------------- cross-source corroboration
def _corroboration(sem_trace, evidence):
    """How many INDEPENDENT information sources backed the gathered evidence: counts distinct
    (source_channel, source_instance_id) pairs — NOT distinct payload hashes, so two OCR reads of the SAME
    image are ONE source, not two. Reported as an info signal (not folded into the mean). not_applicable
    when no provenance-tagged evidence (legacy traces)."""
    deliv = _delivered(_pre_terminal_evidence(sem_trace, evidence))
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
    deliv = _delivered(_pre_terminal_evidence(sem_trace, evidence))
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
    subs = {
        "acquisition": acq,
        "sufficiency": _sufficiency(sem_trace, policy, acq),
        "binding": _binding(sem_trace, evidence),
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

    print("\nALL SYNTHETIC INVARIANTS PASSED")


if __name__ == "__main__":
    _selfcheck()
