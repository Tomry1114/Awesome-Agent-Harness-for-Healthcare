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


def _claim_tokens(claim):
    """Content tokens of a claim string (lowercased, stopwords + short tokens dropped). Used for the
    deterministic corroboration fallback when no LLM judge is wired."""
    toks = re.findall(r"[a-z][a-z0-9\-]{2,}", _txt(claim).lower())
    return [t for t in toks if t not in _STOP]


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
    cues = [str(c).strip().lower() for c in req if str(c).strip()] if isinstance(req, list) else []
    return True, cues


def _claim_requires_cross_source(claim, cues, global_flag):
    """Per-claim cross-source requirement under a declared policy: required iff a global flag forces it OR
    the claim text matches one of the policy's claim-type cues. A claim that matches nothing is NOT
    flagged -> excluded from cross_source (not_applicable), never penalized."""
    if global_flag:
        return True
    if not cues:
        return False
    low = _txt(claim).lower()
    return any(cue in low for cue in cues)


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


# --------------------------------------------------------------------------- entry point
def verification(evidence_items, verification_actions=None, final_claims=None, conflicts=None,
                 policy=None, judge_fn=None):
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
                            signal. Receives EVIDENCE VIEW + claims only.

    Returns {dimension, tier, score, reportable, coverage, submetrics:{name:{score,status,opportunities,
             basis}}, applicable_submetrics}."""
    policy = policy or {}
    final_claims = [c for c in (final_claims or []) if _txt(c).strip()]
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
        claim_cs_required[i] = (not has_csp) or _claim_requires_cross_source(c, cs_cues, cs_global)
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

    # ----- uncertainty_calibration : TWO-SIDED hedge-vs-evidence-strength match -----
    # BUGFIX (contract): 'strong evidence' must be evidence that SUPPORTS THE CLAIM, not just any delivered
    # evidence on the trace. Strength is therefore computed over the UNION of units that actually support
    # the scorable claims (claim_supp_units), NOT over the whole delivered view. Consequence: '3
    # high-fidelity but IRRELEVANT units + a confident UNSUPPORTED conclusion' has ZERO supporting units ->
    # THIN (not strong) -> a confident (un-hedged) conclusion is mis-calibrated -> score 0 (NOT 1). It also
    # keeps evidence_support honest: such a run gets evidence_support 0 AND uncertainty_calibration 0.
    # Correct calibration = hedged when thin OR committed (un-hedged) when strong. This can PENALIZE
    # over-hedging on strong supporting evidence, so it is NOT a monotone function of any support vector.
    min_units = int(policy.get("thin_evidence_min_units", 2))
    min_fid = float(policy.get("thin_evidence_min_fidelity", 0.75))
    # de-duplicate supporting units by identity (one source counts once toward strength, mirroring
    # cross-source independence) so N renders of the SAME source do not inflate 'strong'.
    sup_by_src = {}
    for i in claim_supp_units:
        for u in claim_supp_units[i]:
            sup_by_src.setdefault(_source_identity(u), u)
    sup_units = list(sup_by_src.values())
    n_sup_units = len(sup_units)
    sup_err = sum(1 for u in sup_units if u.get("error_visible"))
    sup_fid = (sum(float(u.get("delivery_fidelity") or 0.0) for u in sup_units) / n_sup_units) if n_sup_units else 0.0
    err_dominant = (n_sup_units > 0 and sup_err > 0 and sup_err >= 0.5 * n_sup_units)
    # thin = the CLAIM-SUPPORTING evidence is sparse/low-fidelity (n_sup_units==0 -> always thin).
    thin = (n_sup_units < min_units) or (n_sup_units > 0 and sup_fid < min_fid) or err_dominant
    # strong = the CLAIM is backed by ample, high-fidelity, error-free SUPPORTING sources.
    strong = (n_sup_units >= max(min_units + 1, 3)) and (sup_fid >= 0.9) and (sup_err == 0)
    hedged = any(k in claim_text for k in _HEDGE)
    if claim_supp1 and thin:
        subs["uncertainty_calibration"] = {
            "score": 1.0 if hedged else 0.0, "status": "applicable", "opportunities": 1,
            "basis": "THIN supporting evidence (supporting_sources=%d, avg_fidelity=%.2f, errors_visible=%d): "
                     "hedging %s (want present)"
                     % (n_sup_units, sup_fid, sup_err, "present" if hedged else "absent")}
    elif claim_supp1 and strong:
        subs["uncertainty_calibration"] = {
            "score": 1.0 if not hedged else 0.0, "status": "applicable", "opportunities": 1,
            "basis": "STRONG supporting evidence (supporting_sources=%d, avg_fidelity=%.2f, errors_visible=%d): "
                     "hedging %s (want absent)"
                     % (n_sup_units, sup_fid, sup_err, "present" if hedged else "absent")}
    else:
        subs["uncertainty_calibration"] = {
            "score": None, "status": "not_applicable", "opportunities": 0,
            "basis": ("no scorable claim" if not claim_supp1 else
                      "supporting evidence neither clearly thin nor clearly strong (supporting_sources=%d, "
                      "avg_fidelity=%.2f, errors_visible=%d)" % (n_sup_units, sup_fid, sup_err))}

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
                  "supporting_sources": n_sup_units, "cross_source_policy": has_csp,
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


def score_bundle(trace, plugin, task=None, judge_fn=None):
    """Convenience: drive `verification` from a raw canonical trace via the substrate only. Used by the
    self-verification harness. Imports substrate lazily so the module stays import-clean standalone."""
    import substrate as S
    sem = S.map_trace(trace, plugin)
    evidence_items = S.evidence_view(trace, plugin)
    verification_actions = [s for s in sem if s.get("event_role") == "verify"]
    final_claims = extract_claims(sem)
    policy = S.dimension_policy(task or {"source_benchmark": (plugin or {}).get("benchmark")}, plugin)
    return verification(evidence_items, verification_actions, final_claims, conflicts=None,
                        policy=policy, judge_fn=judge_fn)


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
