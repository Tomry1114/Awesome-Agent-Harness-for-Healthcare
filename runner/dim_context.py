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

Sub-metrics (applicable-only):
  acquisition  — were the dimension_policy.required_context_units obtained, mapped to evidence by
                 SEMANTIC TYPE (not raw acquire-event count). A required unit is satisfied only by an
                 evidence unit whose semantic KIND matches it; one evidence unit is never counted for
                 two distinct required units, and unrelated acquisitions cannot auto-fill distinct units.
  sufficiency  — enough context units present to PROCEED (boolean floor: delivered units & required
                 milestones met before the terminal commit/final).
  relevance    — delivered evidence pertains to the task (gateway judge over OBSERVATIONS only).
  binding      — evidence bound to a single consistent subject/case, using only TYPED identifier
                 tokens (resource <Type>/<id> in the progress tokens), never a broad numeric regex that
                 could mistake a year / exam number / dose for a subject id.

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

CONTEXT_VERSION = "context-1.1-experimental"

# ---------------------------------------------------------------------------- semantic typing
# A required_context_unit may be declared either as a bare string ("correct_patient") OR as a typed
# entry {"type": "patient_identity"|"allergy_status"|"target_image_region"|...}. The acquisition
# sub-metric maps each required unit to evidence by SEMANTIC TYPE so three IRRELEVANT acquisitions can
# NOT fill three distinct required units. When the policy only carries bare strings we degrade
# gracefully (see _acquisition) but still cap obtained by the number of DISTINCT evidence kinds and
# never double-count one evidence unit across many required units.
#
# An evidence unit's SEMANTIC KIND is derived ONLY from the SEMANTIC progress_token the substrate
# emits (token families are content-/type-keyed, never tool names) — so the logic stays
# benchmark-agnostic. Token families produced by substrate.py:
#   evidence:<source>:<hash8>      -> kind "evidence:<source>"     (OCR text, search snippet, image desc)
#   region:<hash8>:resolved        -> kind "image_region"
#   resource:<Type>/<id>:created   -> kind "resource:<Type>"        (carries a typed subject id)
#   state:read=<Type>/<id>         -> kind "resource:<Type>"        (carries a typed subject id)
#   state:search=<hash8>           -> kind "search"
#   state:page=<hash8>             -> kind "page_state"
#   state:submitted=<hash8>        -> kind "submission"
#   state:<k>=<...>                -> kind "state:<k>"
_TYPED_RES_RE = re.compile(r"^(?:resource:|state:read=)([A-Za-z][A-Za-z0-9_]*)/(.+?)(?::created)?$")


def evidence_semantic_kind(progress_token):
    """Benchmark-agnostic SEMANTIC kind of an acquire/commit evidence unit, read from its progress
    token's structural family. None when the token carries no semantic content. Never a tool name."""
    t = str(progress_token or "")
    if not t:
        return None
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
    """A subject/case identifier carried by a TYPED resource token (resource:<Type>/<id> or
    state:read=<Type>/<id>) — the only place an identifier is GUARANTEED to name a subject/case rather
    than being an arbitrary number (year/dose/exam #). Returns (kind, id) or None. No broad regex."""
    m = _TYPED_RES_RE.match(str(progress_token or ""))
    if not m:
        return None
    rtype, rid = m.group(1), m.group(2).strip()
    if not rid:
        return None
    return ("resource:%s" % rtype, rid)


def _unit_type(unit):
    """The declared SEMANTIC type of a required_context_unit. Supports {"type": ...} / {"unit_type": ...}
    typed entries; a bare string has NO declared type (returns None -> graceful-degrade path)."""
    if isinstance(unit, dict):
        return unit.get("type") or unit.get("unit_type") or unit.get("kind")
    return None


def _delivered(evidence):
    return [u for u in (evidence or []) if u.get("delivered_to_agent")]


# Generic type-aliasing so a structural evidence kind can satisfy a domain-named required type WITHOUT
# any benchmark literal: we only ever match on token-family words, never on a tool/resource name.
#
# A typed unit is satisfied ONLY by an evidence kind in its SPECIFIC alias set. The aliases are
# token-family words (resource:<rtype> / evidence:<src> / image_region / search / page_state), NOT a
# benchmark/tool literal. We deliberately do NOT give a medical type the blanket "resource:" alias —
# otherwise a Patient-identity unit would be auto-filled by an unrelated Observation/Encounter resource,
# which is exactly the "3 unrelated acquisitions fill 3 distinct units" bug this module fixes.
_TYPE_ALIASES = {
    "patient_identity": ("resource:patient", "search"),
    "patient_record": ("resource:patient", "search"),
    "correct_patient": ("resource:patient", "search"),
    "allergy_status": ("resource:allergyintolerance",),
    "current_medications": ("resource:medicationrequest", "resource:medicationstatement",
                            "resource:medication"),
    "current_medication": ("resource:medicationrequest", "resource:medicationstatement",
                           "resource:medication"),
    "target_image_region": ("image_region",),
    "image_region": ("image_region",),
    "target_image_evidence": ("image_region", "evidence:imagedescription", "evidence:ocr"),
    "text_evidence": ("evidence:ocr",),
    "external_reference": ("evidence:search", "search"),
    "current_form_state": ("page_state",),
    "page_state": ("page_state",),
    "submission_requirements": ("page_state", "submission"),
    "correct_case": ("page_state", "resource:case"),
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


# ---------------------------------------------------------------- acquisition
def _acquisition(sem_trace, evidence, policy):
    """Map required_context_units to evidence by SEMANTIC TYPE, not raw acquire-event count.

    TYPED policy  ({type: ...} entries): a required unit is satisfied ONLY by a delivered/earned
       evidence unit whose semantic kind matches its declared type (via _kind_matches_type). Each
       evidence kind satisfies at most ONE required unit (no double-counting one evidence unit across
       many required units).
    BARE policy   (plain strings): no per-unit type to match against, so we degrade gracefully — but we
       refuse to let unrelated acquisitions auto-satisfy distinct units. obtained is capped by the count
       of DISTINCT evidence SEMANTIC KINDS actually backed by delivered evidence (three OCR tokens are
       one kind; three IRRELEVANT acquire tokens of the same kind fill ONE unit, not three). Required
       milestones, when declared, contribute their reached-count as an alternative coverage floor.
    No required units -> not_applicable (no opportunity, never a vacuous 1.0)."""
    req_units = list(policy.get("required_context_units") or [])
    req_ms = set(policy.get("required_milestones") or [])
    if not req_units:
        return _sm(None, "not_applicable", 0, reportable=False, reason="policy declares no required_context_units")

    deliv = _delivered(evidence)
    deliv_kinds = {evidence_semantic_kind(u.get("progress_token")) for u in deliv}
    deliv_kinds.discard(None)

    # acquire/commit-role evidence kinds actually earned in the trace (success only), then used as a
    # fallback when the evidence view carries no per-unit progress_token (older/default extractor).
    earned_kinds = {evidence_semantic_kind(s.get("progress_token")) for s in sem_trace
                    if s.get("event_role") in ("acquire", "commit") and s.get("status") == "success"
                    and s.get("progress_token")}
    earned_kinds.discard(None)
    kinds = deliv_kinds or earned_kinds

    typed = [u for u in req_units if _unit_type(u)]
    reached = _sub.milestones_reached(sem_trace)
    ms_hit = req_ms & reached

    if typed and len(typed) == len(req_units):
        # fully-typed policy: one-to-one match (each evidence kind used for at most ONE unit). Process
        # the MOST CONSTRAINED units first (fewest candidate kinds) and, within a unit, prefer a
        # SPECIFIC resource-family kind over a generic one (search/page) so a scarce specific kind is
        # not consumed by a looser unit -- a deterministic, order-independent assignment.
        remaining = set(kinds)
        cand = {}
        for u in req_units:
            ut = _unit_type(u)
            cand[id(u)] = (ut, [k for k in kinds if _kind_matches_type(k, ut)])
        order = sorted(req_units, key=lambda u: len(cand[id(u)][1]))
        matched = 0
        matched_pairs = []
        for u in order:
            ut, options = cand[id(u)]
            avail = [k for k in options if k in remaining]
            if not avail:
                continue
            # prefer a specific resource/evidence-family kind over a generic search/page_state one
            avail.sort(key=lambda k: (k in ("search", "page_state", "submission"), k))
            hit = avail[0]
            remaining.discard(hit)
            matched += 1
            matched_pairs.append({"unit": ut, "evidence_kind": hit})
        score = round(min(1.0, matched / max(1, len(req_units))), 3)
        return _sm(score, "valid", len(req_units), reportable=True,
                   matching="typed", required_units=[_unit_type(u) for u in req_units],
                   distinct_evidence_kinds=sorted(kinds), matched_units=matched,
                   matched_pairs=matched_pairs,
                   required_milestones=sorted(req_ms), milestones_reached=sorted(ms_hit))

    # ----- graceful degrade (bare-string units, or mixed): cap by DISTINCT evidence kinds -----
    # distinct evidence kinds is the number of genuinely different context kinds obtained. This is the
    # anti-double-counting floor: N tokens of the SAME kind cannot fill N distinct required units.
    obtained = min(len(kinds), len(req_units))
    note = "bare_units_distinct_kind_cap"
    if req_ms:
        # a milestone-coverage signal is an alternative floor (still capped by #required_units).
        ms_cover = min(len(ms_hit), len(req_units))
        if ms_cover > obtained:
            obtained = ms_cover
            note = "bare_units_milestone_cover"
    score = round(min(1.0, obtained / max(1, len(req_units))), 3)
    return _sm(score, "valid", len(req_units), reportable=True,
               matching="degraded", required_units=[(_unit_type(u) or u) for u in req_units],
               distinct_evidence_kinds=sorted(kinds), obtained_context_kinds=obtained, note=note,
               required_milestones=sorted(req_ms), milestones_reached=sorted(ms_hit))


# ---------------------------------------------------------------- sufficiency
def _sufficiency(sem_trace, evidence, policy):
    """Did the agent have ENOUGH context to proceed: a boolean floor — at least one delivered evidence
    unit per required_context_unit AND every required milestone reached, BEFORE the terminal step.
    Distinct from acquisition (which is a graded coverage ratio)."""
    req_units = list(policy.get("required_context_units") or [])
    req_ms = set(policy.get("required_milestones") or [])
    if not req_units and not req_ms:
        return _sm(None, "not_applicable", 0, reportable=False, reason="no required units/milestones to gate on")

    # evidence obtained strictly before the terminal (final/commit) event
    terminal_seen = False
    pre_units = 0
    for s in sem_trace:
        if s.get("terminal") in ("final", "escalate") or s.get("event_role") == "commit":
            terminal_seen = True
            break
        if s.get("event_role") == "acquire" and s.get("status") == "success":
            pre_units += 1
    n_deliv = len(_delivered(evidence)) if not terminal_seen or pre_units == 0 else pre_units
    n_deliv = max(n_deliv, pre_units)

    units_ok = (not req_units) or (n_deliv >= len(req_units))
    ms_ok = req_ms.issubset(_sub.milestones_reached(sem_trace))
    score = 1.0 if (units_ok and ms_ok) else 0.0
    return _sm(score, "valid", 1, reportable=True,
               required_units=len(req_units), units_obtained_pre_terminal=n_deliv,
               required_milestones=sorted(req_ms), milestones_satisfied=ms_ok)


# ---------------------------------------------------------------- binding
def _binding(sem_trace, evidence):
    """Subject-consistency check using ONLY TYPED identifier tokens (resource:<Type>/<id> /
    state:read=<Type>/<id>) carried by the SEMANTIC progress tokens — NOT a broad numeric/ID regex over
    free-text payloads (which could bind on a year / dose / exam number). A subject is the (kind, id)
    pair from a typed resource token; binding asks whether the typed subjects converge on ONE dominant
    subject within each resource KIND (a scatter of unrelated patient ids is bad; many Observations of
    the same patient is fine). Applicable only when typed identifier tokens exist; otherwise
    not_applicable (never a vacuous score, never a guess from a bare number)."""
    from collections import Counter, defaultdict
    # collect typed (kind, id) subjects from delivered evidence tokens first; fall back to the
    # acquire/commit trace tokens when the evidence view carries no per-unit token.
    typed = []
    for u in _delivered(evidence):
        s = _typed_subject_id(u.get("progress_token"))
        if s:
            typed.append(s)
    if not typed:
        for s in sem_trace:
            if s.get("event_role") in ("acquire", "commit") and s.get("status") == "success":
                t = _typed_subject_id(s.get("progress_token"))
                if t:
                    typed.append(t)
    if not typed:
        return _sm(None, "not_applicable", 0, reportable=False,
                   reason="no typed identifier tokens (resource:<Type>/<id>) in delivered evidence")

    # group ids by resource KIND; within each kind, is there a dominant single subject id? Average the
    # per-kind single-subject focus so a high-cardinality detail kind (many Observation ids) does not
    # drown a low-cardinality identity kind (one Patient id).
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
               method="typed_resource_tokens", per_kind_focus=kind_scores,
               typed_subjects=len(typed), resource_kinds=sorted(by_kind), details=details)


# ---------------------------------------------------------------- relevance (judge over OBSERVATIONS)
def _relevance(sem_trace, evidence, task_instruction, judge_model, char_budget=7000, per_unit=700):
    """Gateway judge: do the delivered OBSERVATIONS pertain to the task? Reads evidence payloads +
    the task INSTRUCTION only. The terminal/final SemanticEvent is excluded so the answer never leaks;
    gold is never passed. No judge backend / instruction / evidence -> not_applicable (no vacuous 1.0)."""
    deliv = _delivered(evidence)
    instr = str(task_instruction or "").strip()
    if not deliv or not instr:
        return _sm(None, "not_applicable", 0, reportable=False, reason="no delivered evidence or no task instruction")
    if not judge_model or os.environ.get("MH_CONTEXT_JUDGE", "1") == "0":
        return _sm(None, "not_applicable", 0, reportable=False, reason="relevance judge backend unavailable/disabled")

    # build OBSERVATION-ONLY context; defensively skip any unit that came from a terminal/final event
    terminal_ids = {s.get("raw", {}).get("tool") for s in sem_trace if s.get("terminal")}
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
            return _sm(None, "not_applicable", 0, reportable=False,
                       reason="relevance judge error: %s" % r.get("error_type"))
        head = (r.get("content") or "").strip().upper()
        rel = not head.startswith("IRRELEVANT")
        return _sm(1.0 if rel else 0.0, "valid", 1, reportable=True,
                   judge_model=judge_model, judge_tier="gateway_observation_relevance",
                   judge_backend=judge_model, n_evidence_units=len(parts),
                   reason=(r.get("content") or "")[:200])
    except Exception as ex:
        return _sm(None, "not_applicable", 0, reportable=False, reason="relevance judge exception: %s" % ex)


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
    subs = {
        "acquisition": _acquisition(sem_trace, evidence, policy),
        "sufficiency": _sufficiency(sem_trace, evidence, policy),
        "binding": _binding(sem_trace, evidence),
        "relevance": _relevance(sem_trace, evidence, task_instruction, judge_model),
    }
    out = _aggregate(subs)
    out["dimension"] = "Context"
    out["tier"] = "experimental"
    out["evaluator_version"] = CONTEXT_VERSION
    out["measures"] = "context_management"               # explicitly NOT answer correctness
    out["reads_final_or_gold"] = False
    out["reportable"] = bool(out.get("n_applicable"))
    out["governance_policy_id"] = policy.get("governance_policy_id")
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
        # judge_model intentionally left None here so the offline self-check is deterministic / no network
        res = context(sem, ev, pol, task_instruction=instr, judge_model=None)
        print("\n==== %s [%s] ====" % (bundle, sb))
        print(" score       :", res["score"], "| applicable:", res["applicable_submetrics"],
              "| reportable:", res["reportable"], "| tier:", res["tier"])
        for k, v in res["submetrics"].items():
            print("   %-12s %-15s score=%s  %s" % (
                k, v.get("status"), v.get("score"),
                {kk: vv for kk, vv in v.items() if kk not in ("score", "status")}))

    # ---- synthetic invariant 1: 3 UNRELATED same-kind acquisitions must NOT fill 3 required units ----
    print("\n==== SYNTHETIC 1: 3 unrelated same-kind acquisitions vs 3 required units ====")
    sem3 = [_sub.semantic_event("acquire", status="success", capability_id="t",
                                progress_token="evidence:search:%s" % h)
            for h in ("aaaaaaaa", "bbbbbbbb", "cccccccc")]            # 3 distinct tokens, SAME kind
    ev3 = [{"id": "e%d" % i, "delivered_to_agent": True, "delivery_fidelity": 1.0,
            "error_visible": False, "payload": "snippet %d" % i,
            "progress_token": s["progress_token"]} for i, s in enumerate(sem3)]
    pol3 = {"required_context_units": ["correct_patient", "current_medications", "allergy_status"],
            "required_milestones": []}
    acq = _acquisition(sem3, ev3, pol3)
    print("   acquisition:", acq.get("score"), "distinct_kinds=", acq.get("distinct_evidence_kinds"),
          "obtained=", acq.get("obtained_context_kinds"))
    assert acq["score"] <= round(1 / 3, 3) + 1e-9, ("3 unrelated same-kind acq must NOT fill 3 units", acq)
    print("   PASS: 3 unrelated same-kind acquisitions do NOT fill 3 distinct units (score=%s)" % acq["score"])

    # ---- synthetic invariant 2: typed policy maps one kind per unit; 3 DIFFERENT kinds -> all matched
    print("\n==== SYNTHETIC 2: typed policy, 3 distinct kinds ====")
    sem_t = [
        _sub.semantic_event("acquire", status="success", progress_token="state:read=Patient/42"),
        _sub.semantic_event("acquire", status="success", progress_token="state:read=AllergyIntolerance/7"),
        _sub.semantic_event("acquire", status="success", progress_token="state:read=MedicationRequest/9"),
    ]
    ev_t = [{"id": "u%d" % i, "delivered_to_agent": True, "delivery_fidelity": 1.0, "error_visible": False,
             "payload": "p", "progress_token": s["progress_token"]} for i, s in enumerate(sem_t)]
    pol_t = {"required_context_units": [{"type": "patient_identity"}, {"type": "allergy_status"},
                                        {"type": "current_medications"}]}
    acqt = _acquisition(sem_t, ev_t, pol_t)
    print("   typed acquisition:", acqt.get("score"), acqt.get("matched_pairs"))
    assert acqt["matching"] == "typed" and acqt["score"] == 1.0, acqt
    _pairs = {p["unit"]: p["evidence_kind"] for p in acqt["matched_pairs"]}
    assert _pairs["patient_identity"] == "resource:Patient", _pairs       # SPECIFIC, not any resource
    assert _pairs["allergy_status"] == "resource:AllergyIntolerance", _pairs
    assert _pairs["current_medications"] == "resource:MedicationRequest", _pairs

    # a typed unit is NOT satisfied by an unrelated SPECIFIC resource kind (Encounter != Patient)
    sem_x = [_sub.semantic_event("acquire", status="success", progress_token="state:read=Encounter/1"),
             _sub.semantic_event("acquire", status="success", progress_token="state:read=Encounter/2"),
             _sub.semantic_event("acquire", status="success", progress_token="state:read=Encounter/3")]
    ev_x = [{"id": "x%d" % i, "delivered_to_agent": True, "delivery_fidelity": 1.0, "error_visible": False,
             "payload": "p", "progress_token": s["progress_token"]} for i, s in enumerate(sem_x)]
    acqx = _acquisition(sem_x, ev_x, pol_t)
    print("   typed acquisition (3 unrelated Encounters):", acqx.get("score"), acqx.get("matched_pairs"))
    assert acqx["score"] == 0.0, ("3 unrelated Encounter resources must NOT fill 3 typed units", acqx)

    # ---- synthetic invariant 3: typed policy NOT satisfied by 3 acquisitions of ONE wrong kind ----
    sem_w = [_sub.semantic_event("acquire", status="success", progress_token="evidence:search:%s" % h)
             for h in ("a", "b", "c")]
    ev_w = [{"id": "w%d" % i, "delivered_to_agent": True, "delivery_fidelity": 1.0, "error_visible": False,
             "payload": "p", "progress_token": s["progress_token"]} for i, s in enumerate(sem_w)]
    acqw = _acquisition(sem_w, ev_w, pol_t)
    print("   typed acquisition (wrong kinds):", acqw.get("score"), acqw.get("matched_pairs"))
    assert acqw["score"] < 1.0, ("3 unrelated acquisitions must NOT fill 3 typed units", acqw)
    print("   PASS: typed matching maps by kind; unrelated kinds do not auto-satisfy")

    # ---- synthetic invariant 4: binding ignores year/dose, uses typed resource ids ----
    print("\n==== SYNTHETIC 4: binding ignores year/dose, uses typed resource id ====")
    sem_b = [
        _sub.semantic_event("acquire", status="success", progress_token="state:read=Patient/100"),
        _sub.semantic_event("acquire", status="success", progress_token="state:read=Observation/555"),
        _sub.semantic_event("acquire", status="success", progress_token="state:read=Observation/556"),
    ]
    ev_b = [{"id": "b%d" % i, "delivered_to_agent": True, "delivery_fidelity": 1.0, "error_visible": False,
             # payload carries a YEAR (2024) and a DOSE (500 mg) the OLD regex would have mis-bound;
             # the new binding ignores payload free-text entirely.
             "payload": "taken in 2024, dose 500 mg",
             "progress_token": s["progress_token"]} for i, s in enumerate(sem_b)]
    b = _binding(sem_b, ev_b)
    print("   binding:", b.get("score"), b.get("per_kind_focus"))
    assert b["status"] == "valid" and "resource:Observation" in b["per_kind_focus"], b
    # a binding with NO typed tokens -> not_applicable (never a vacuous score, never a bare-number guess)
    ev_none = [{"id": "n0", "delivered_to_agent": True, "delivery_fidelity": 1.0, "error_visible": False,
                "payload": "year 2024 dose 500 mg exam 12345", "progress_token": "evidence:ocr:deadbeef"}]
    bn = _binding([], ev_none)
    print("   binding (no typed ids):", bn.get("status"), bn.get("reason"))
    assert bn["status"] == "not_applicable", bn
    print("   PASS: binding uses typed resource ids only; bare year/dose/exam numbers never bind")


if __name__ == "__main__":
    _selfcheck()
