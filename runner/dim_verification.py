#!/usr/bin/env python3
"""Dimension: VERIFICATION (correctness/epistemic-hygiene layer).

Did the agent's FINAL CLAIMS rest on evidence it actually obtained, did it corroborate them across
INDEPENDENT sources, did it acknowledge contradictions, did it calibrate its uncertainty to the strength
of the evidence, and did it actually carry out the verification steps it had the opportunity to run?

This evaluator is benchmark-AGNOSTIC: it consumes ONLY substrate structures
  - EvidenceView units  (substrate.evidence_view  -> [{id, delivered_to_agent, delivery_fidelity,
                          error_visible, acknowledged?, payload}])
  - SemanticTrace       (substrate.map_trace      -> [SemanticEvent], for verify-role actions + final)
  - DimensionPolicy     (substrate.dimension_policy, optional: thin-evidence threshold etc.)
No benchmark name, tool literal, image, DOM, or FHIR resource appears in the scoring logic.
Benchmark specifics (which tool delivers which evidence, what the claim text is) arrive pre-mapped
through the plugin via the substrate; this module never inspects them.

Supersedes the MedCTA-bound evidence_verification judge wiring: an LLM judge MAY still be used, but it
receives the EVIDENCE VIEW + the claim strings ONLY -- never the benchmark id, gold answer, or task
solution -- so corroboration is judged against what was delivered, not against hidden truth.

ITEM #4b RESTRUCTURE -- five GENUINELY DISTINCT behaviors (no two are an algebraic transform of the same
vector). Each carries status applicable|not_applicable + opportunities; the dimension averages ONLY
applicable ones, so a run with no opportunity for a sub-metric does NOT get a vacuous 1.0:

  evidence_support             : RATE of substantive claims backed by >=1 DELIVERED evidence unit.
                                 Vector = per-claim {supported_by>=1}. (basic grounding)
  cross_source_check           : RATE of corroboration-REQUIRED claims backed by >=2 INDEPENDENT evidence
                                 SOURCES. INDEPENDENCE = distinct (source_channel, source_instance_id) pairs
                                 (substrate v2 provenance contract): two OCR reads of the SAME image are
                                 ONE source, not two -- independence is the SOURCE instance, not the payload
                                 hash and not the extractor. POLICY-GATED: read verification_policy.
                                 cross_source_required_for (claim-type cues) / cross_source_required (global
                                 flag); a claim NOT flagged is EXCLUDED (not_applicable), never auto-failed.
                                 Default (no policy declared): do not force 2 sources -- legacy behavior
                                 treats every content claim as an opportunity for back-compat. A claim
                                 backed by exactly one source scores 0 here even though evidence_support
                                 scores it 1 -> a strictly DIFFERENT per-claim vector {n_indep_sources>=2},
                                 NOT a transform of the evidence_support vector.
  conflict_handling            : contradictions among evidence units (declared conflicts or error-visible
                                 deliveries) are acknowledged/resolved. Vector = per-conflict {acked}.
  uncertainty_calibration      : TWO-SIDED hedging calibration -- hedging PRESENT when the CLAIM-SUPPORTING
                                 evidence is thin AND hedging ABSENT (committed) when it is strong. STRENGTH
                                 IS MEASURED OVER THE UNITS THAT ACTUALLY SUPPORT THE CLAIMS, not the whole
                                 delivered view: '3 high-fidelity but IRRELEVANT units + a confident
                                 unsupported conclusion' has 0 supporting sources -> THIN -> a committed
                                 conclusion is mis-calibrated -> 0 (and evidence_support is also 0). A
                                 correct call in EITHER regime scores 1; mis-hedging (overconfident-when-thin
                                 OR wishy-washy-when-strong) scores 0. Vector = {correct_calibration} over
                                 the single applicable regime. It can PENALIZE excessive hedging on strong
                                 SUPPORTING evidence, so it is not a monotone slice of any support vector.
  verification_action_completion: of the VERIFY-role actions the agent had the chance to complete, the
                                 fraction that actually CONFIRMED (status==success / produced a real verify
                                 result) rather than firing-and-not-confirming (partial/failure). Vector =
                                 per-verify-event {confirmed}. Entirely independent of the claim/evidence
                                 vectors above (it scores the agent's self-checking ACTIONS, not its claims).

Tier: experimental (heuristic corroboration + optional LLM aux signal; not yet human-audited).
"""
import os
import re
import hashlib

TIER = "experimental"
DIMENSION = "Verification"
_SUBMETRICS = ("evidence_support", "cross_source_check", "conflict_handling",
               "uncertainty_calibration", "verification_action_completion")

# Lexical cues used ONLY on the agent's own final-claim/verify text (never on benchmark data) to detect
# epistemic hedging and explicit contradiction-acknowledgement. These are language cues, not tool/bench
# literals, so they keep the module benchmark-agnostic.
_HEDGE = ("uncertain", "unclear", "insufficient", "cannot confirm", "cannot determine", "inconclusive",
          "not enough", "limited evidence", "unable to", "would need", "further", "ambiguous",
          "low confidence", "tentative", "possible", "likely", "may ", "might ", "suggest",
          "no evidence", "not visualized", "not assessed", "indeterminate", "equivocal")
_CONFLICT = ("however", "but ", "contradict", "conflict", "discrepan", "inconsisten", "whereas",
             "on the other hand", "differs", "disagree", "mismatch", "reconcile", "despite")
_STOP = set("the a an of and or to in on at for with is are was were be been being this that these those "
            "it its as by from into per via no not all any can may might will would should could there "
            "here then than thus so if then which who whom whose what when where why how he she they we "
            "you i me my our your their his her them us also more most some such only just very".split())


# --------------------------------------------------------------------------- helpers
def _txt(x):
    return "" if x is None else str(x)


# V11: medical claims hinge on NUMERIC ATOMS -- a measurement ('12 mm'), a percentage ('EF 35%'), a lab
# value ('Hb 8', 'pH 7.2'), a short test abbreviation ('EF', 'Hb', 'pH', 'CRP'). The old word tokenizer
# (leading letter + >=3 chars) silently DROPPED every one of these: pure numbers, '%', units written
# separately, and 2-char abbreviations. Dropping them is catastrophic for corroboration -- 'lesion 12 mm'
# and 'lesion 8 mm' tokenize identically once the numbers vanish, so a contradictory measurement reads as
# support. We therefore tokenize THREE alphabets and keep them all as evidence tokens:
#   - numeric atoms: a number optionally glued/spaced to a unit or '%' ('12mm', '12 mm', '35%', '7.2'),
#     normalized to a single 'num:<value><unit>' token so '12 mm' and '12mm' match but '8 mm' does not.
#   - short medical abbreviations: 2-char alnum tokens that contain a letter (EF, Hb, pH, T3) -- the old
#     >=3 rule excluded these; here a 2-char token is kept only if it is NOT a pure number and NOT a stopword.
#   - ordinary content words: a leading letter + >=2 trailing chars (unchanged), minus stopwords.
_UNIT_RE = (r"(?:%|mm|cm|mg|mcg|ug|kg|ml|dl|l|g|mmhg|bpm|bps|mmol|mol|meq|iu|"
            r"units?|mm3|ml/min|/min|/hr|/dl|/l|cc|sec|hrs?|days?|wks?|mos?|yrs?|"
            r"celsius|fahrenheit|deg|cmh2o)")
_NUM_ATOM_RE = re.compile(r"(?<![a-z0-9])(\d+(?:\.\d+)?)\s*(" + _UNIT_RE + r")?(?![a-z0-9])", re.I)
_WORD_RE = re.compile(r"[a-z][a-z0-9\-]{2,}")          # ordinary content word (>=3 chars, leading letter)
# a WHOLE-TOKEN 2-char abbreviation (EF, Hb, pH, T3): delimited by non-alphanumerics on both sides so it is
# a standalone abbreviation, never a 2-char SUBSTRING of a longer word ('le' in 'lesion').
_ABBR_RE = re.compile(r"(?<![a-z0-9])[a-z0-9]{2}(?![a-z0-9])")
# BARE unit words carry NO independent content (they are already encoded inside the 'num:<v><unit>' atom),
# so a stray 'mm'/'dl'/'mcg' must NOT become a matchable token -- otherwise '8 mm' and '12 mm' would
# spuriously corroborate via the shared bare 'mm'. Dropped from BOTH the word and abbreviation alphabets.
_UNIT_WORDS = {"mm", "cm", "mg", "mcg", "ug", "kg", "ml", "dl", "mmhg", "bpm", "bps", "mmol", "mol",
               "meq", "iu", "mm3", "cc", "sec", "deg", "cmh2o", "hr", "hrs", "wk", "wks", "mo", "mos",
               "yr", "yrs", "day", "days", "min", "celsius", "fahrenheit"}


def _claim_tokens(claim):
    """Content tokens of a claim string for the deterministic corroboration fallback. V11: NUMERIC medical
    atoms (numbers, units, '%', short lab abbreviations) are KEPT as evidence tokens rather than dropped, so
    a contradictory measurement (12 mm vs 8 mm) does not collapse into a match. Returns a list of tokens
    drawn from three alphabets (numeric atoms, 2-char abbreviations, content words)."""
    low = _txt(claim).lower()
    toks = []
    # 1) numeric atoms: number (+ optional unit/%) -> normalized 'num:<value><unit>' token.
    for m in _NUM_ATOM_RE.finditer(low):
        val, unit = m.group(1), (m.group(2) or "")
        # a bare number with no unit is still a distinguishing atom (Hb 8, pH 7.2 -> 'num:8', 'num:7.2').
        toks.append("num:%s%s" % (val, unit))
    # 2) ordinary content words (>=3 chars), stopwords + bare unit words removed.
    toks += [t for t in _WORD_RE.findall(low) if t not in _STOP and t not in _UNIT_WORDS]
    # 3) standalone 2-char abbreviations (EF, Hb, pH, T3): a whole delimited token containing a letter and
    #    not a stopword/unit word. (Pure 2-digit numbers were already captured as numeric atoms in step 1.)
    for m in _ABBR_RE.finditer(low):
        a = m.group(0)
        if a.isdigit() or a in _STOP or a in _UNIT_WORDS:
            continue
        if any(ch.isalpha() for ch in a):
            toks.append(a)
    return toks


# V7: a single final-answer string usually bundles SEVERAL assertions ('The mass is benign. It measures
# 8 mm.'). Calibration must be judged PER statement, so a hedge on one statement does not launder an
# over-confident UNSUPPORTED statement elsewhere. Split on sentence terminators AND newline/semicolon/
# bullet boundaries; drop empty fragments.
_SENT_SPLIT_RE = re.compile(r"(?:[.!?]+\s+|[;\n\r]+|\s+[-*•]\s+)")


def _statements(claim_strings):
    """Split a list of final-claim strings into per-statement units (sentence/clause granularity) for V7
    per-claim calibration. Returns a list of non-empty statement strings."""
    out = []
    for c in claim_strings or []:
        for part in _SENT_SPLIT_RE.split(_txt(c)):
            s = part.strip(" \t\r\n-*•")
            if s:
                out.append(s)
    return out


def _delivered(evidence_items):
    return [u for u in (evidence_items or []) if u.get("delivered_to_agent")]


def _source_identity(unit):
    """A DELIVERED unit's INDEPENDENT-source identity for cross-source corroboration.

    CONTRACT (substrate v2): independence is the (source_channel, source_instance_id) pair -- the
    information SOURCE family ('radiology_image' / 'fhir_patient_record' / 'gui_portal' / 'external_web')
    paired with the specific instance within it (an image id / Patient/<id> / url). TWO READS OF THE SAME
    IMAGE/SOURCE ARE ONE SOURCE: a second OCR pass over the same image carries the same
    (source_channel, source_instance_id) even though its payload bytes differ, so it collapses to one
    independent source. The 'extractor' (OCR / fhir_read / ...) is deliberately NOT part of the identity:
    re-reading one source with a different tool is still the SAME source.

    LEGACY FALLBACK: when a unit predates the provenance contract (no source_channel), fall back to the
    tool-origin of its id (everything before the trailing '#<idx>') joined with a payload content hash, so
    older bundles still get a sensible (if coarser) independence signal."""
    chan = _txt(unit.get("source_channel")).strip().lower()
    if chan:
        inst = _txt(unit.get("source_instance_id")).strip().lower()
        if not inst:
            # channel present but no instance id: fall back to a payload hash WITHIN that channel so two
            # materially different payloads on the same channel still count as 2, but byte-identical ones
            # collapse -- never let a missing instance id silently merge everything on a channel into one.
            payload = _txt(unit.get("payload")).strip().lower()
            inst = hashlib.sha1(payload.encode("utf-8", "replace")).hexdigest()[:8] if payload else "_"
        return "%s::%s" % (chan, inst)
    # --- legacy fallback (no provenance contract on this unit) ---
    uid = _txt(unit.get("id"))
    origin = uid.rsplit("#", 1)[0] if "#" in uid else uid
    payload = _txt(unit.get("payload")).strip().lower()
    h = hashlib.sha1(payload.encode("utf-8", "replace")).hexdigest()[:8] if payload else "empty"
    return "%s::%s" % (origin or "src", h)


def _unit_supports_claim(claim_toks, unit):
    """Does this single delivered unit's payload share enough content tokens with the claim to count as
    supporting it? Generic token overlap -- no benchmark vocabulary baked in."""
    ptoks = set(_claim_tokens(unit.get("payload")))
    if not ptoks or not claim_toks:
        return False
    overlap = claim_toks & ptoks
    return len(overlap) >= max(2, int(round(0.4 * len(claim_toks))))


def _cross_source_policy(policy):
    """Read the verification policy that governs WHICH claims must be corroborated across independent
    sources. Returns (has_policy, required_for) where:
      has_policy   : a verification_policy was declared (so cross-source is policy-GATED -- claims not
                     flagged are EXCLUDED/not_applicable, never auto-failed),
      required_for : list of claim-type cue strings; a claim REQUIRES corroboration iff its text contains
                     one of these cues (case-insensitive) OR the policy globally forces it.
    CONTRACT: 'Default when no policy: do not force 2 sources on every claim.' With NO verification_policy
    we return has_policy=False and the caller keeps the legacy 'every content claim is an opportunity'
    behavior (back-compat). With a policy present, only flagged claims are opportunities."""
    vp = (policy or {}).get("verification_policy")
    if not isinstance(vp, dict):
        return False, []
    req = vp.get("cross_source_required_for")
    return True, (req if isinstance(req, list) else [])


def _claim_requires_cross_source(claim, required_for, global_flag):
    """Per-claim cross-source requirement under a declared policy. Each policy entry is either a STRUCTURED
    {type, patterns:[natural-language keyword/phrase, ...]} (preferred) or a bare string (legacy substring).
    A claim REQUIRES corroboration iff: a global flag forces it; OR the claim's structured claim_type matches
    an entry.type; OR a natural-language pattern appears in the claim text. NOTE: matching the snake_case
    TYPE LABEL itself against natural text is NOT done (a label like 'high_risk_recommendation' never appears
    in a real answer) -- entries must supply patterns, else the type only matches a structured claim_type."""
    if global_flag:
        return True
    if not required_for:
        return False
    ctype = claim.get("claim_type") if isinstance(claim, dict) else None
    low = _txt(claim).lower()
    for entry in required_for:
        if isinstance(entry, dict):
            if ctype and entry.get("type") == ctype:
                return True
            for pat in (entry.get("patterns") or []):
                if str(pat).strip().lower() and str(pat).strip().lower() in low:
                    return True
        elif str(entry).strip().lower() and str(entry).strip().lower() in low:   # legacy bare-string cue
            return True
    return False


# --------------------------------------------------------------------------- optional LLM aux judge
def _llm_corroboration(claims, delivered_units, judge_fn):
    """Optional auxiliary signal for evidence_support. judge_fn(system, user)->str. It is handed ONLY the
    evidence-view payloads + the claim strings; NEVER the benchmark id, task gold, or hidden reference.
    Returns a dict {claim_index: bool} of support verdicts, or None on any failure (caller falls back to
    the deterministic signal). The FULL EvidenceView is sent (never a truncated first-N) -- only each
    individual payload is length-clipped so the prompt stays bounded."""
    if not judge_fn or not claims:
        return None
    ev_lines = "\n".join("E%d: %s" % (i, _txt(u.get("payload"))[:400]) for i, u in enumerate(delivered_units))
    cl_lines = "\n".join("C%d: %s" % (i, _txt(c)[:300]) for i, c in enumerate(claims))
    system = ("You audit whether each CLAIM is supported by the supplied EVIDENCE UNITS. You are given "
              "ONLY the evidence the agent obtained and the agent's claims -- you have NO access to any "
              "gold answer or external truth. A claim is supported=1 only if at least one evidence unit "
              "supports it; otherwise 0. Reply with ONLY JSON {\"verdicts\":{\"0\":0|1,...}}.")
    user = "EVIDENCE UNITS:\n%s\n\nCLAIMS:\n%s" % (ev_lines or "(none)", cl_lines)
    try:
        raw = judge_fn(system, user) or ""
        s, e = raw.find("{"), raw.rfind("}")
        if s == -1 or e == -1:
            return None
        import json as _json
        obj = _json.loads(raw[s:e + 1])
        verds = obj.get("verdicts") or obj
        return {int(k): bool(int(v)) for k, v in verds.items()}
    except Exception:
        return None


# --------------------------------------------------------------------------- V12: judge gating
def _resolve_judge_fn(judge_fn, judge_model):
    """V12 -- distinguish an EXPLICITLY DISABLED judge from an UNSPECIFIED one, mirroring dim_context's
    judge_model contract but for this module's (system,user)->str judge_fn shape:

      judge_fn given (not None)   : use it verbatim (direct injection -- tests / callers wire their own).
      judge_model is False        : EXPLICITLY DISABLED. Offline. NO gateway call, NO MH_JUDGE_MODEL read.
                                    Returns None -> deterministic-only scoring. (The offline cross-check in
                                    aggregate_report passes False to stay offline.)
      judge_model is None         : UNSPECIFIED. MAY read MH_JUDGE_MODEL from the environment; if a model is
                                    configured there, build a gateway-backed judge_fn, else stay offline.
      judge_model is a str        : build a gateway-backed judge_fn for that model.

    Building the gateway judge is lazy + fail-soft: any import/availability problem falls back to None
    (deterministic scoring), never raising into the scorer."""
    if judge_fn is not None:
        return judge_fn
    if judge_model is False:                       # EXPLICIT disable -- stay strictly offline (no env read)
        return None
    model = judge_model
    if model is None:                              # UNSPECIFIED -- env MAY supply a model
        model = os.environ.get("MH_JUDGE_MODEL")
    if not model:
        return None
    try:
        import gateway
    except Exception:
        return None

    def _gw(system, user):
        r = gateway.chat([{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                         model=model, max_tokens=300, judge=True)
        if not isinstance(r, dict) or not r.get("ok"):
            return None
        return r.get("content")
    return _gw


# --------------------------------------------------------------------------- entry point
def verification(evidence_items, verification_actions=None, final_claims=None, conflicts=None,
                 policy=None, judge_fn=None, judge_model=None):
    """Generic Verification scorer.

    Args (all substrate-derived; no benchmark specifics):
      evidence_items      : list[EvidenceUnit] from substrate.evidence_view (the FULL view -- this scorer
                            never truncates the unit list; cross-source counting walks every delivered unit)
      verification_actions: list[SemanticEvent] with event_role == 'verify' (re-checks the agent ran);
                            drives verification_action_completion. Optional -- not_applicable when none.
      final_claims        : list[str] -- substantive assertions the agent committed to (final answer /
                            verify-step conclusions). Pre-extracted from the SemanticTrace.
      conflicts           : optional list[{units:[id,id], acknowledged:bool}] of detected evidence
                            contradictions. If None, conflict_handling is derived from acknowledgement
                            cues in the claim/verify text and is marked not_applicable when no conflict
                            signal exists.
      policy              : DimensionPolicy dict; reads optional 'thin_evidence_min_units' /
                            'thin_evidence_min_fidelity' thresholds (defaults applied).
      judge_fn            : optional gateway callable (system,user)->str for the auxiliary support
                            signal. Receives EVIDENCE VIEW + claims only. Direct injection -- when given it
                            is used verbatim and judge_model is ignored.
      judge_model         : V12 judge gating when judge_fn is NOT supplied. False = EXPLICITLY DISABLED
                            (offline; no gateway call, no MH_JUDGE_MODEL read). None = unspecified (MAY read
                            MH_JUDGE_MODEL). A str = build a gateway judge for that model. The offline
                            cross-check must pass judge_model=False to stay offline.

    Returns {dimension, tier, score, reportable, coverage, submetrics:{name:{score,status,opportunities,
             basis}}, applicable_submetrics}."""
    policy = policy or {}
    judge_fn = _resolve_judge_fn(judge_fn, judge_model)   # V12: False disables (offline); None may read env
    final_claims = _statements([c for c in (final_claims or []) if _txt(c).strip()])  # P0-3: per-statement for ALL sub-metrics
    verification_actions = verification_actions or []
    delivered = _delivered(evidence_items)          # FULL delivered view -- never a first-N slice
    n_units = len(evidence_items or [])
    n_deliv = len(delivered)
    errors_visible = sum(1 for u in (evidence_items or []) if u.get("error_visible"))
    avg_fid = (sum(float(u.get("delivery_fidelity") or 0.0) for u in delivered) / n_deliv) if n_deliv else 0.0

    subs = {}

    # ===== per-claim INDEPENDENT-SOURCE support counts (one walk over the FULL delivered view) =====
    # For each substantive claim we record n_independent_sources = number of DISTINCT delivered sources
    # (distinct _source_identity) whose payload supports it. evidence_support thresholds this at >=1;
    # cross_source_check thresholds it at >=2. The two sub-metrics therefore read TWO DIFFERENT boolean
    # vectors off the same underlying count, and neither vector is an algebraic transform of the other
    # (a claim with exactly one source flips one bit but not the other).
    scorable = [(i, c) for i, c in enumerate(final_claims) if _claim_tokens(c)]
    # optional LLM aux only refines the >=1 (support) verdict; cross-source always uses the deterministic
    # independent-source count so the two signals can never collapse to the same vector.
    llm_supp = _llm_corroboration([c for _, c in scorable], delivered, judge_fn) if scorable else None
    has_csp, cs_cues = _cross_source_policy(policy)
    cs_global = bool(((policy or {}).get("verification_policy") or {}).get("cross_source_required"))
    claim_supp1 = {}        # claim_index -> supported by >=1 source (bool)
    claim_indep = {}        # claim_index -> # of INDEPENDENT delivered sources supporting it (int)
    claim_supp_units = {}   # claim_index -> list of DELIVERED units that actually SUPPORT this claim
    claim_cs_required = {}  # claim_index -> does policy REQUIRE cross-source corroboration for this claim?
    for j, (i, c) in enumerate(scorable):
        ctoks = set(_claim_tokens(c))
        src_ids, sup_units = set(), []
        for u in delivered:
            if _unit_supports_claim(ctoks, u):
                src_ids.add(_source_identity(u))
                sup_units.append(u)
        claim_indep[i] = len(src_ids)
        claim_supp_units[i] = sup_units
        claim_cs_required[i] = has_csp and _claim_requires_cross_source(c, cs_cues, cs_global)
        if llm_supp is not None and j in llm_supp:
            claim_supp1[i] = bool(llm_supp[j])
        else:
            claim_supp1[i] = len(src_ids) >= 1

    # ----- evidence_support : claims backed by >=1 delivered evidence unit (RATE) -----
    if claim_supp1:
        n_sup = sum(1 for v in claim_supp1.values() if v)
        subs["evidence_support"] = {
            "score": round(n_sup / len(claim_supp1), 4), "status": "applicable",
            "opportunities": len(claim_supp1),
            "basis": "%d/%d substantive claims backed by >=1 delivered evidence unit"
                     % (n_sup, len(claim_supp1))}
    else:
        subs["evidence_support"] = {"score": None, "status": "not_applicable", "opportunities": 0,
                                    "basis": "no substantive (content-bearing) final claim to check"}

    # ----- cross_source_check : claims that REQUIRE corroboration backed by >=2 INDEPENDENT sources -----
    # Distinct vector from evidence_support: a claim resting on a single source is the very failure this
    # sub-metric exists to catch, so when it IS in scope it counts as opportunity AND scores 0.
    # POLICY GATING (contract): only claims the policy flags as needing corroboration are opportunities;
    # an unflagged claim is EXCLUDED (not_applicable), never auto-failed. With NO verification_policy we
    # keep the legacy 'every content claim is an opportunity' behavior (claim_cs_required all True).
    cs_claims = {i: n for i, n in claim_indep.items() if claim_cs_required.get(i)}
    if cs_claims:
        n_cross = sum(1 for n in cs_claims.values() if n >= 2)
        n_excluded = len(claim_indep) - len(cs_claims)
        basis = ("%d/%d corroboration-REQUIRED claims backed by >=2 INDEPENDENT sources"
                 % (n_cross, len(cs_claims)))
        if has_csp and n_excluded:
            basis += " (%d claim(s) not flagged for cross-source -> excluded)" % n_excluded
        subs["cross_source_check"] = {
            "score": round(n_cross / len(cs_claims), 4), "status": "applicable",
            "opportunities": len(cs_claims), "basis": basis}
    elif has_csp and claim_indep:
        # a policy IS declared but no claim is flagged for corroboration -> nothing to test, do NOT penalize
        subs["cross_source_check"] = {"score": None, "status": "not_applicable", "opportunities": 0,
                                      "basis": "verification_policy declared but no claim requires "
                                               "cross-source corroboration -> excluded (not penalized)"}
    else:
        subs["cross_source_check"] = {"score": None, "status": "not_applicable", "opportunities": 0,
                                      "basis": "no substantive claim to cross-corroborate"}

    # ----- conflict_handling : contradictions acknowledged/resolved -----
    claim_text = " ".join(_txt(c) for c in final_claims).lower()
    for s in verification_actions:
        claim_text += " " + _txt((s.get("raw") or {}).get("thought")).lower()
    ack_cue = any(k in claim_text for k in _CONFLICT)
    if conflicts is not None and len(conflicts) > 0:
        acked = sum(1 for c in conflicts if c.get("acknowledged") or ack_cue)
        subs["conflict_handling"] = {"score": round(acked / len(conflicts), 4), "status": "applicable",
                                     "opportunities": len(conflicts),
                                     "basis": "%d/%d evidence conflicts acknowledged/resolved" % (acked, len(conflicts))}
    elif errors_visible > 0:
        # an error-visible delivery is a (delivery-level) contradiction the agent must reckon with
        subs["conflict_handling"] = {"score": 1.0 if ack_cue else 0.0, "status": "applicable",
                                     "opportunities": errors_visible,
                                     "basis": "%d error-visible delivery(ies); acknowledgement cue %s"
                                              % (errors_visible, "present" if ack_cue else "absent")}
    else:
        subs["conflict_handling"] = {"score": None, "status": "not_applicable", "opportunities": 0,
                                     "basis": "no detected evidence conflict or error-visible delivery"}

    # ----- uncertainty_calibration : PER-CLAIM (V7) TWO-SIDED hedge-vs-evidence-strength match -----
    # V7 RESTRUCTURE: calibration is computed PER STATEMENT, not over one joined answer blob. The final
    # answer is split into statements (sentence/clause granularity); each statement gets its OWN supporting
    # sources, its OWN hedging detection, and its OWN calibration verdict; the sub-metric AVERAGES the
    # applicable per-statement verdicts. Consequence: a hedge on statement A can NO LONGER excuse a confident
    # UNSUPPORTED statement B -- B is its own opportunity and fails on its own.
    #
    # Per statement: strength is measured over the UNITS THAT SUPPORT THAT STATEMENT (not the whole delivered
    # view), de-duplicated by source identity (so N renders of one source do not inflate 'strong'). A
    # statement with 0 supporting sources is THIN; a confident (un-hedged) THIN statement is mis-calibrated
    # -> 0. A statement that is neither clearly thin nor clearly strong is not_applicable (excluded from the
    # average, no vacuous 1.0). Correct = hedged-when-thin OR committed-when-strong; over-hedging on a strong
    # statement is penalized, so this is NOT a monotone function of any support vector.
    min_units = int(policy.get("thin_evidence_min_units", 2))
    min_fid = float(policy.get("thin_evidence_min_fidelity", 0.75))
    statements = final_claims  # already per-statement (P0-3)
    per_stmt = []          # list of (statement, regime, hedged, correct) for applicable statements
    for st in statements:
        st_toks = set(_claim_tokens(st))
        if not st_toks:
            continue
        # supporting sources for THIS statement, de-duplicated by independent-source identity.
        st_sup = {}
        for u in delivered:
            if _unit_supports_claim(st_toks, u):
                st_sup.setdefault(_source_identity(u), u)
        su = list(st_sup.values())
        n_su = len(su)
        s_err = sum(1 for u in su if u.get("error_visible"))
        s_fid = (sum(float(u.get("delivery_fidelity") or 0.0) for u in su) / n_su) if n_su else 0.0
        err_dom = (n_su > 0 and s_err > 0 and s_err >= 0.5 * n_su)
        thin = (n_su < min_units) or (n_su > 0 and s_fid < min_fid) or err_dom
        strong = (n_su >= max(min_units + 1, 3)) and (s_fid >= 0.9) and (s_err == 0)
        hedged = any(k in st.lower() for k in _HEDGE)
        if thin:
            per_stmt.append((st, "thin", hedged, hedged))            # correct iff hedged
        elif strong:
            per_stmt.append((st, "strong", hedged, not hedged))      # correct iff committed
        # else: neither clearly thin nor strong -> excluded for this statement
    if per_stmt:
        n_correct = sum(1 for _, _, _, ok in per_stmt if ok)
        n_thin = sum(1 for _, r, _, _ in per_stmt if r == "thin")
        n_strong = len(per_stmt) - n_thin
        subs["uncertainty_calibration"] = {
            "score": round(n_correct / len(per_stmt), 4), "status": "applicable",
            "opportunities": len(per_stmt),
            "basis": "%d/%d statements correctly calibrated (per-claim; %d thin->want-hedge, "
                     "%d strong->want-commit)" % (n_correct, len(per_stmt), n_thin, n_strong)}
    else:
        subs["uncertainty_calibration"] = {
            "score": None, "status": "not_applicable", "opportunities": 0,
            "basis": ("no scorable claim" if not claim_supp1 else
                      "no statement is clearly thin or clearly strong (calibration indeterminate)")}

    # ----- verification_action_completion : did the agent finish the verify steps it ran? -----
    # Opportunity = each verify-role action. Completion = it CONFIRMED (status success) rather than firing
    # without a confirming result (partial/failure). Independent of claim/evidence vectors above.
    if verification_actions:
        confirmed = sum(1 for s in verification_actions if str(s.get("status")) == "success")
        subs["verification_action_completion"] = {
            "score": round(confirmed / len(verification_actions), 4), "status": "applicable",
            "opportunities": len(verification_actions),
            "basis": "%d/%d verify-role actions actually confirmed (status=success)"
                     % (confirmed, len(verification_actions))}
    else:
        subs["verification_action_completion"] = {
            "score": None, "status": "not_applicable", "opportunities": 0,
            "basis": "agent ran no verify-role action (no self-check opportunity exercised)"}

    # ----- applicable-only aggregation -----
    # informational stat: distinct independent sources that support ANY scorable claim (de-duped union).
    _sup_src = set()
    for i in claim_supp_units:
        for u in claim_supp_units[i]:
            _sup_src.add(_source_identity(u))
    n_sup_src = len(_sup_src)

    applicable = [k for k in _SUBMETRICS if subs[k]["status"] == "applicable"]
    score = round(sum(subs[k]["score"] for k in applicable) / len(applicable), 4) if applicable else None
    return {
        "dimension": DIMENSION, "tier": TIER, "score": score,
        "reportable": bool(applicable),
        "coverage": round(len(applicable) / len(_SUBMETRICS), 4),
        "applicable_submetrics": applicable,
        "submetrics": subs,
        "stats": {"evidence_units": n_units, "delivered": n_deliv, "avg_fidelity": round(avg_fid, 4),
                  "errors_visible": errors_visible, "claims": len(final_claims),
                  "supporting_sources": n_sup_src, "cross_source_policy": has_csp,
                  "verify_actions": len(verification_actions), "judge_used": llm_supp is not None},
    }


# --------------------------------------------------------------------------- substrate bundle adapter
def extract_claims(sem_trace):
    """Generic claim extraction from a SemanticTrace: the final-answer content + any verify-step
    conclusions. Reads only neutral event fields (thought / canonical_action.content / content). No
    benchmark literal."""
    claims = []
    for s in sem_trace or []:
        raw = s.get("raw") or {}
        if s.get("event_role") == "final" or s.get("terminal") == "final":
            ca = raw.get("canonical_action") or {}
            txt = ca.get("content") or raw.get("content") or raw.get("answer") or raw.get("thought") or ""
            if _txt(txt).strip():
                claims.append(_txt(txt).strip())
        elif s.get("event_role") == "verify":
            txt = raw.get("thought") or ""
            if _txt(txt).strip():
                claims.append(_txt(txt).strip())
    return claims


def score_bundle(trace, plugin, task=None, judge_fn=None, judge_model=False):
    """Convenience: drive `verification` from a raw canonical trace via the substrate only. Used by the
    self-verification harness. Imports substrate lazily so the module stays import-clean standalone.

    V12: judge_model defaults to False (EXPLICITLY OFFLINE) so the self-verification harness and the
    aggregate_report offline cross-check make NO gateway call. Pass judge_model=None to opt into the
    MH_JUDGE_MODEL env, or a model string / judge_fn to wire a real judge."""
    import substrate as S
    sem = S.map_trace(trace, plugin)
    evidence_items = S.evidence_view(trace, plugin)
    verification_actions = [s for s in sem if s.get("event_role") == "verify"]
    final_claims = extract_claims(sem)
    policy = S.dimension_policy(task or {"source_benchmark": (plugin or {}).get("benchmark")}, plugin)
    return verification(evidence_items, verification_actions, final_claims, conflicts=None,
                        policy=policy, judge_fn=judge_fn, judge_model=judge_model)


# --------------------------------------------------------------------------- self-verification
if __name__ == "__main__":
    import sys, json, glob
    sys.path.insert(0, "runner")
    sys.path.insert(0, ".")
    import substrate as S

    def _load_trace(bdir):
        tp = os.path.join(bdir, "trajectory.jsonl")
        return [json.loads(l) for l in open(tp) if l.strip()]

    def _pick_plugin(trace):
        tools = {e.get("tool") for e in trace if e.get("event_type") == "tool_call"}
        best, bestn = None, -1
        for name in S.list_plugins():
            p = S.get_plugin(name)
            sem = set((p.get("tool_semantics") or {}).keys())
            n = len(tools & sem)
            if n > bestn:
                best, bestn = p, n
        return best

    def run(bdir, label):
        trace = _load_trace(bdir)
        plugin = _pick_plugin(trace)
        out = score_bundle(trace, plugin, task={"source_benchmark": plugin["benchmark"]})
        print("\n=== %s  (%s) plugin=%s ===" % (label, bdir, plugin["benchmark"]))
        print("score=%s reportable=%s coverage=%s applicable=%s"
              % (out["score"], out["reportable"], out["coverage"], out["applicable_submetrics"]))
        print("stats:", out["stats"])
        for k in _SUBMETRICS:
            sm = out["submetrics"][k]
            print("  %-30s score=%-6s status=%-15s opp=%s  | %s"
                  % (k, sm["score"], sm["status"], sm["opportunities"], sm["basis"]))
        return out

    mcta = "results_mctaGov/gpt5/MCTA-0"
    pbs = sorted(glob.glob("results_pb_chk3/gpt5/PB-*"))
    pb = next((p for p in pbs if os.path.isdir(p)), None)
    habs = sorted(glob.glob("results_hab10/gpt5/HAB-*"))
    hab = next((h for h in habs if os.path.isdir(h)), None)

    outs = []
    outs.append(("MedCTA", run(mcta, "MedCTA")))
    if pb:
        outs.append(("PhysicianBench", run(pb, "PhysicianBench")))
    if hab:
        outs.append(("HealthAdminBench", run(hab, "HealthAdminBench")))

    # ---------------------------------------------------------------- SYNTHETIC independence proof
    # Construct evidence/claim sets that make EACH pair of sub-metrics DIVERGE, proving no sub-metric is
    # an algebraic transform of another.
    print("\n--- synthetic independence cases (each row shows the 5 sub-metric scores) ---")

    def _ev(uid, payload, delivered=True, fid=1.0, err=False):
        return {"id": uid, "payload": payload, "delivered_to_agent": delivered,
                "delivery_fidelity": fid, "error_visible": err}

    def _scores(ev, claims, vacts=None, conflicts=None, pol=None):
        o = verification(ev, vacts or [], claims, conflicts=conflicts, policy=pol or {})
        return {k: o["submetrics"][k]["score"] for k in _SUBMETRICS}

    ca = _scores([_ev("OCR#0", "liver lesion hypodense segment seven")],
                 ["liver lesion hypodense segment seven"])
    print("  A one-source claim       :", ca)
    cb = _scores([_ev("OCR#0", "liver lesion hypodense segment seven"),
                  _ev("ImageDescription#1", "hypodense liver lesion in segment seven confirmed")],
                 ["liver lesion hypodense segment seven"])
    print("  B two-source claim       :", cb)
    cc = _scores([_ev("OCR#0", "same exact payload text alpha beta gamma"),
                  _ev("OCR#1", "same exact payload text alpha beta gamma")],
                 ["same exact payload text alpha beta gamma"])
    print("  C dup-payload one source :", cc)
    cd = _scores([_ev("OCR#0", "finding alpha beta"), _ev("ImageDescription#1", "finding alpha beta"),
                  _ev("GoogleSearch#2", "finding alpha beta corroborated")],
                 ["finding alpha beta but this is uncertain and inconclusive"])
    print("  D over-hedge on strong   :", cd)
    ce = _scores([_ev("OCR#0", "single thin finding delta")],
                 ["single thin finding delta definitely present"])
    print("  E commit on thin         :", ce)
    vacts = [S.semantic_event("verify", status="success"), S.semantic_event("verify", status="partial")]
    cf = _scores([_ev("OCR#0", "x y z finding")], ["x y z finding"], vacts=vacts)
    print("  F half-confirmed verify  :", cf)
    cg_ack = _scores([_ev("OCR#0", "a b c")], ["a b c however the sources conflict and I reconcile them"],
                     conflicts=[{"units": ["OCR#0", "OCR#1"], "acknowledged": False}])
    cg_no = _scores([_ev("OCR#0", "a b c")], ["a b c stated plainly"],
                    conflicts=[{"units": ["OCR#0", "OCR#1"], "acknowledged": False}])
    print("  G conflict ack vs not    : ack=", cg_ack["conflict_handling"], " no=", cg_no["conflict_handling"])

    # ============================================================ CONTRACT-SPECIFIC self-checks =====
    print("\n--- contract self-checks (provenance independence / policy gating / supported-strength) ---")

    def _evp(channel, instance, payload, extractor="x", delivered=True, fid=1.0, err=False):
        """Provenance-tagged EvidenceUnit (substrate v2 contract fields)."""
        return {"id": "%s#%s" % (extractor, instance), "payload": payload, "delivered_to_agent": delivered,
                "delivery_fidelity": fid, "error_visible": err, "source_channel": channel,
                "source_instance_id": instance, "extractor": extractor}

    def _full(ev, claims, vacts=None, conflicts=None, pol=None):
        return verification(ev, vacts or [], claims, conflicts=conflicts, policy=pol or {})

    ok = True

    # (1) TWO READS OF THE SAME IMAGE = ONE SOURCE (different extractor, same source_instance_id).
    same_img = _full([_evp("radiology_image", "img-7", "liver lesion segment seven hypodense", extractor="OCR"),
                      _evp("radiology_image", "img-7", "hypodense liver lesion segment seven", extractor="ImageDescription")],
                     ["liver lesion segment seven hypodense"])
    n1 = same_img["submetrics"]["cross_source_check"]["score"]
    print("  same-image two reads -> cross_source=%s (want 0.0, ONE source)" % n1)
    ok &= (n1 == 0.0)

    # (2) TWO DIFFERENT SOURCES (distinct source_instance_id) = real corroboration.
    two_src = _full([_evp("radiology_image", "img-7", "liver lesion segment seven hypodense", extractor="OCR"),
                     _evp("fhir_patient_record", "Patient/42", "liver lesion segment seven hypodense documented", extractor="fhir_read")],
                    ["liver lesion segment seven hypodense"])
    n2 = two_src["submetrics"]["cross_source_check"]["score"]
    print("  two distinct sources  -> cross_source=%s (want 1.0)" % n2)
    ok &= (n2 == 1.0)

    # (3) POLICY GATING: with a verification_policy, an UNFLAGGED claim is EXCLUDED (not auto-failed).
    pol_flagged = {"verification_policy": {"cross_source_required_for": ["diagnosis"]}}
    unflagged = _full([_evp("radiology_image", "img-7", "incidental note alpha beta", extractor="OCR")],
                      ["incidental note alpha beta"], pol=pol_flagged)
    st_unf = unflagged["submetrics"]["cross_source_check"]["status"]
    print("  policy + unflagged claim -> cross_source status=%s (want not_applicable)" % st_unf)
    ok &= (st_unf == "not_applicable")

    # (3b) the SAME policy, a FLAGGED claim (text contains 'diagnosis') on one source -> applicable AND 0.
    flagged = _full([_evp("radiology_image", "img-7", "diagnosis pneumonia consolidation", extractor="OCR")],
                    ["diagnosis pneumonia consolidation present"], pol=pol_flagged)
    sm_fl = flagged["submetrics"]["cross_source_check"]
    print("  policy + flagged 1-source -> cross_source status=%s score=%s (want applicable,0.0)"
          % (sm_fl["status"], sm_fl["score"]))
    ok &= (sm_fl["status"] == "applicable" and sm_fl["score"] == 0.0)

    # (3c) NO policy -> legacy behavior: every content claim is an opportunity (back-compat, not excluded).
    nopol = _full([_evp("radiology_image", "img-7", "finding alpha beta", extractor="OCR")],
                  ["finding alpha beta"])
    print("  no policy 1-source    -> cross_source status=%s score=%s (want applicable,0.0)"
          % (nopol["submetrics"]["cross_source_check"]["status"],
             nopol["submetrics"]["cross_source_check"]["score"]))
    ok &= (nopol["submetrics"]["cross_source_check"]["status"] == "applicable"
           and nopol["submetrics"]["cross_source_check"]["score"] == 0.0)

    # (4) THE BUG: 3 high-fidelity but IRRELEVANT units + a confident UNSUPPORTED conclusion.
    #     -> evidence_support 0 AND uncertainty_calibration 0 (NOT 1). This is the core contract fix.
    irrelevant = _full([_evp("radiology_image", "img-1", "kidney cyst left upper pole benign", extractor="OCR"),
                        _evp("fhir_patient_record", "Patient/9", "blood pressure 120 over 80 stable", extractor="fhir_read"),
                        _evp("external_web", "http://x", "guideline on hydration intake daily", extractor="web")],
                       ["the brain tumor is definitely glioblastoma grade four"])
    es = irrelevant["submetrics"]["evidence_support"]["score"]
    uc = irrelevant["submetrics"]["uncertainty_calibration"]
    print("  3 irrelevant + confident-unsupported -> evidence_support=%s uncertainty_calibration=%s/%s"
          % (es, uc["score"], uc["status"]))
    ok &= (es == 0.0)
    ok &= (uc["status"] == "applicable" and uc["score"] == 0.0)   # thin (0 supporting) + committed -> 0, NOT 1

    # (V7) PER-CLAIM calibration: a hedge on statement A must NOT excuse a confident UNSUPPORTED statement
    #      B. Answer = 'Hb is 8 g/dl, this is uncertain. The tumor is definitely glioblastoma.' Evidence
    #      supports only the Hb statement (thin+hedged=correct); the tumor statement is thin+committed=wrong.
    #      -> per-claim avg = 1/2 = 0.5 (a single-blob hedge would have scored the whole thing 1.0).
    v7 = _full([_evp("fhir_patient_record", "Patient/3", "hemoglobin hb 8 g/dl recorded", extractor="fhir_read")],
               ["Hb is 8 g/dl but this is uncertain. The tumor is definitely glioblastoma grade four."])
    uc7 = v7["submetrics"]["uncertainty_calibration"]
    print("  V7 per-claim calib (1 hedged-thin-OK + 1 confident-unsupported) -> score=%s opp=%s status=%s (want 0.5,2)"
          % (uc7["score"], uc7["opportunities"], uc7["status"]))
    ok &= (uc7["status"] == "applicable" and uc7["score"] == 0.5 and uc7["opportunities"] == 2)

    # (V11) NUMERIC ATOMS: '12 mm' must NOT corroborate '8 mm'. A claim 'lesion measures 12 mm' is supported
    #       by an '12 mm' payload but NOT by an '8 mm' payload -- the number is a real evidence token now.
    t_match = _claim_tokens("lesion measures 12 mm")
    print("  V11 tokens('lesion measures 12 mm') -> %s" % t_match)
    ok &= any(tok.startswith("num:12") for tok in t_match)          # numeric atom captured (not dropped)
    ok &= ("ef" in _claim_tokens("EF 35%"))                          # 2-char abbreviation captured
    ok &= any(tok.startswith("num:35") for tok in _claim_tokens("EF 35%"))   # percentage captured
    ok &= ("ph" in _claim_tokens("pH 7.2") and any(t.startswith("num:7.2") for t in _claim_tokens("pH 7.2")))
    # contradictory measurement does NOT support: payload '8 mm' must not back a '12 mm' claim.
    v11 = verification([{"id": "OCR#0", "payload": "the lesion is 8 mm in the right lobe",
                         "delivered_to_agent": True, "delivery_fidelity": 1.0, "error_visible": False}],
                       [], ["the lesion is 12 mm in size"], judge_model=False)
    es11 = v11["submetrics"]["evidence_support"]["score"]
    print("  V11 '8 mm' payload vs '12 mm' claim -> evidence_support=%s (want 0.0, numbers disambiguate)" % es11)
    ok &= (es11 == 0.0)
    # ...and the SAME number DOES support (sanity: not over-restrictive).
    v11b = verification([{"id": "OCR#0", "payload": "the lesion is 12 mm in the right lobe",
                          "delivered_to_agent": True, "delivery_fidelity": 1.0, "error_visible": False}],
                        [], ["the lesion is 12 mm in size"], judge_model=False)
    print("  V11 '12 mm' payload vs '12 mm' claim -> evidence_support=%s (want 1.0)"
          % v11b["submetrics"]["evidence_support"]["score"])
    ok &= (v11b["submetrics"]["evidence_support"]["score"] == 1.0)

    # (V12) JUDGE GATING: judge_model=False -> OFFLINE (no gateway), judge_used False. An injected judge_fn
    #       is used verbatim. A bogus env model with no gateway falls back to deterministic (no raise).
    off = verification([{"id": "OCR#0", "payload": "finding alpha beta", "delivered_to_agent": True,
                         "delivery_fidelity": 1.0, "error_visible": False}],
                       [], ["finding alpha beta"], judge_model=False)
    print("  V12 judge_model=False -> judge_used=%s (want False, offline)" % off["stats"]["judge_used"])
    ok &= (off["stats"]["judge_used"] is False)
    # injected judge_fn that flips every claim to UNsupported is honored (judge_used True, support 0).
    _stub = lambda s, u: '{"verdicts":{"0":0}}'
    inj = verification([{"id": "OCR#0", "payload": "finding alpha beta", "delivered_to_agent": True,
                         "delivery_fidelity": 1.0, "error_visible": False}],
                       [], ["finding alpha beta"], judge_fn=_stub)
    print("  V12 injected judge_fn (forces unsupported) -> judge_used=%s evidence_support=%s (want True,0.0)"
          % (inj["stats"]["judge_used"], inj["submetrics"]["evidence_support"]["score"]))
    ok &= (inj["stats"]["judge_used"] is True and inj["submetrics"]["evidence_support"]["score"] == 0.0)

    print("\nCONTRACT SELF-CHECKS:", "OK" if ok else "FAIL")
    if not ok:
        raise SystemExit("contract self-checks FAILED")

    # ---- algebraic-independence assertion: across the crafted cases, NO sub-metric is a function of
    # another (find a counterexample row where equal-b forces unequal-a). ----
    rows = [ca, cb, cc, cd, ce, cf, cg_ack, cg_no]
    indep_ok = True
    for a in range(len(_SUBMETRICS)):
        for b in range(len(_SUBMETRICS)):
            if a == b:
                continue
            ka, kb = _SUBMETRICS[a], _SUBMETRICS[b]
            seen, functional = {}, True
            for r in rows:
                va, vb = r[ka], r[kb]
                if vb in seen and seen[vb] != va:
                    functional = False
                    break
                seen[vb] = va
            if functional and len(seen) > 1:
                print("  !! POSSIBLE FUNCTIONAL DEPENDENCE: %s = f(%s)" % (ka, kb))
                indep_ok = False
    print("\nINDEPENDENCE (synthetic, no sub-metric is a function of another):", "OK" if indep_ok else "FAIL")
    print("IMPORT OK; module self-contained.")
