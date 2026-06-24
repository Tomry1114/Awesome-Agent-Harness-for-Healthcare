#!/usr/bin/env python3
"""Dimension: VERIFICATION (correctness/epistemic-hygiene layer).

Did the agent's FINAL CLAIMS rest on evidence it actually obtained, did it acknowledge
contradictions, and did it avoid asserting things no delivered evidence unit supports?

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

Sub-metrics (each carries status applicable|not_applicable + opportunities; the dimension averages ONLY
applicable ones, so a run with no opportunity for a sub-metric does NOT get a vacuous 1.0):
  cross_check              : each substantive claim corroborated by >=1 delivered evidence unit
  conflict_handling        : contradictions among evidence units are acknowledged/resolved
  insufficiency_disclosure : when evidence is thin (few delivered units / low fidelity / errors visible),
                             the agent flagged the uncertainty instead of over-asserting
  no_unsupported_claim     : penalize claims backed by NO delivered evidence unit (hallucination guard)

Tier: experimental (heuristic corroboration + optional LLM aux signal; not yet human-audited).
"""
import os
import re

TIER = "experimental"
DIMENSION = "Verification"
_SUBMETRICS = ("cross_check", "conflict_handling", "insufficiency_disclosure", "no_unsupported_claim")

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


def _evidence_text(units):
    return " \n ".join(_txt(u.get("payload")) for u in units).lower()


def _corroborated_det(claim, delivered_units):
    """Deterministic corroboration: does at least one delivered evidence unit's payload share enough
    content tokens with the claim? Generic token overlap -- no benchmark vocabulary baked in."""
    ctoks = set(_claim_tokens(claim))
    if not ctoks:
        return None  # claim has no scorable content (e.g. an empty/action-only final answer)
    for u in delivered_units:
        ptoks = set(_claim_tokens(u.get("payload")))
        if not ptoks:
            continue
        overlap = ctoks & ptoks
        # corroborated if the claim's content is largely present in this unit, or a strong absolute hit
        if len(overlap) >= max(2, int(round(0.4 * len(ctoks)))):
            return True
    return False


# --------------------------------------------------------------------------- optional LLM aux judge
def _llm_corroboration(claims, delivered_units, judge_fn):
    """Optional auxiliary signal. judge_fn(system, user)->str. It is handed ONLY the evidence-view
    payloads + the claim strings; NEVER the benchmark id, task gold, or hidden reference. Returns a
    dict {claim_index: bool} of corroboration verdicts, or None on any failure (caller falls back to
    the deterministic signal)."""
    if not judge_fn or not claims:
        return None
    ev_lines = "\n".join("E%d: %s" % (i, _txt(u.get("payload"))[:400]) for i, u in enumerate(delivered_units))
    cl_lines = "\n".join("C%d: %s" % (i, _txt(c)[:300]) for i, c in enumerate(claims))
    system = ("You audit whether each CLAIM is corroborated by the supplied EVIDENCE UNITS. You are given "
              "ONLY the evidence the agent obtained and the agent's claims -- you have NO access to any "
              "gold answer or external truth. A claim is corroborated=1 only if at least one evidence unit "
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
      evidence_items      : list[EvidenceUnit] from substrate.evidence_view
      verification_actions: list[SemanticEvent] with event_role == 'verify' (re-checks the agent ran);
                            optional -- contributes evidence-of-self-checking but never required.
      final_claims        : list[str] -- substantive assertions the agent committed to (final answer /
                            verify-step conclusions). Pre-extracted from the SemanticTrace.
      conflicts           : optional list[{units:[id,id], acknowledged:bool}] of detected evidence
                            contradictions. If None, conflict_handling is derived from acknowledgement
                            cues in the claim/verify text and is marked not_applicable when no conflict
                            signal exists.
      policy              : DimensionPolicy dict; reads optional 'thin_evidence_min_units' /
                            'thin_evidence_min_fidelity' thresholds (defaults applied).
      judge_fn            : optional gateway callable (system,user)->str for the auxiliary corroboration
                            signal. Receives EVIDENCE VIEW + claims only.

    Returns {dimension, tier, score, reportable, coverage, submetrics:{name:{score,status,opportunities,
             basis}}, applicable_submetrics}."""
    policy = policy or {}
    final_claims = [c for c in (final_claims or []) if _txt(c).strip()]
    verification_actions = verification_actions or []
    delivered = _delivered(evidence_items)
    n_units = len(evidence_items or [])
    n_deliv = len(delivered)
    errors_visible = sum(1 for u in (evidence_items or []) if u.get("error_visible"))
    avg_fid = (sum(float(u.get("delivery_fidelity") or 0.0) for u in delivered) / n_deliv) if n_deliv else 0.0

    subs = {}

    # ----- per-claim corroboration verdicts (LLM aux if wired, else deterministic token overlap) -----
    scorable = [(i, c) for i, c in enumerate(final_claims) if _claim_tokens(c)]
    verds = _llm_corroboration([c for _, c in scorable], delivered, judge_fn) if scorable else None
    claim_corrob = {}  # claim_index -> bool|None
    for j, (i, c) in enumerate(scorable):
        if verds is not None and j in verds:
            claim_corrob[i] = verds[j]
        else:
            claim_corrob[i] = _corroborated_det(c, delivered)

    # ----- cross_check : substantive claims corroborated by >=1 delivered evidence unit -----
    cc_vals = [v for v in claim_corrob.values() if v is not None]
    if cc_vals:
        subs["cross_check"] = {"score": round(sum(1 for v in cc_vals if v) / len(cc_vals), 4),
                               "status": "applicable", "opportunities": len(cc_vals),
                               "basis": "%d/%d substantive claims corroborated by a delivered evidence unit"
                                        % (sum(1 for v in cc_vals if v), len(cc_vals))}
    else:
        subs["cross_check"] = {"score": None, "status": "not_applicable", "opportunities": 0,
                               "basis": "no substantive (content-bearing) final claim to corroborate"}

    # ----- no_unsupported_claim : penalize claims with NO backing delivered unit -----
    if cc_vals:
        unsupported = sum(1 for v in cc_vals if v is False)
        subs["no_unsupported_claim"] = {
            "score": round(1.0 - unsupported / len(cc_vals), 4), "status": "applicable",
            "opportunities": len(cc_vals),
            "basis": "%d/%d claims lack any delivered evidence support" % (unsupported, len(cc_vals))}
    else:
        subs["no_unsupported_claim"] = {"score": None, "status": "not_applicable", "opportunities": 0,
                                        "basis": "no substantive claim to check for support"}

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

    # ----- insufficiency_disclosure : flags thin evidence instead of over-asserting -----
    min_units = int(policy.get("thin_evidence_min_units", 2))
    min_fid = float(policy.get("thin_evidence_min_fidelity", 0.75))
    # Thin == genuinely sparse/low-fidelity evidence. A few visible errors alone do NOT make a 19-unit
    # run "thin" (those belong to conflict_handling); errors contribute only when they dominate delivery.
    err_dominant = (n_units > 0 and errors_visible > 0 and errors_visible >= 0.5 * n_units)
    thin = (n_deliv < min_units) or (n_deliv > 0 and avg_fid < min_fid) or err_dominant
    if thin and cc_vals:
        hedged = any(k in claim_text for k in _HEDGE)
        subs["insufficiency_disclosure"] = {
            "score": 1.0 if hedged else 0.0, "status": "applicable", "opportunities": 1,
            "basis": "thin evidence (delivered=%d, avg_fidelity=%.2f, errors_visible=%d); disclosure %s"
                     % (n_deliv, avg_fid, errors_visible, "present" if hedged else "absent")}
    else:
        subs["insufficiency_disclosure"] = {
            "score": None, "status": "not_applicable", "opportunities": 0,
            "basis": ("evidence not thin (delivered=%d, avg_fidelity=%.2f, errors_visible=%d)"
                      % (n_deliv, avg_fid, errors_visible)) if not thin else "no scorable claim under thin evidence"}

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
                  "judge_used": verds is not None},
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

    def _bench_of(bdir):
        # neutral: read provenance/agent_model is irrelevant; infer plugin from the registered set by
        # matching the tools present -- but for the harness we just read the result.json source if any.
        rp = os.path.join(bdir, "result.json")
        if os.path.exists(rp):
            r = json.load(open(rp))
            for cand in S.list_plugins():
                # match by tool_semantics overlap with the trace's tools
                pass
        return None

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
            print("  %-26s score=%-6s status=%-15s opp=%s  | %s"
                  % (k, sm["score"], sm["status"], sm["opportunities"], sm["basis"]))
        return out

    mcta = "results_mctaGov/gpt5/MCTA-0"
    pbs = sorted(glob.glob("results_pb_chk3/gpt5/PB-*"))
    pb = pbs[0] if pbs else None
    habs = sorted(glob.glob("results_hab10/gpt5/HAB-*"))
    hab = habs[0] if habs else None

    run(mcta, "MedCTA")
    if pb:
        run(pb, "PhysicianBench")
    if hab:
        run(hab, "HealthAdminBench")
    print("\nIMPORT OK; module self-contained.")
