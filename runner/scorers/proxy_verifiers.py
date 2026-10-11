#!/usr/bin/env python3
"""Trajectory-derived PROXY dimension signals (score_eligible=False, soft, do NOT enter primary
dimension_scores or success). Fills dimension cells a benchmark does not formally test, from the
tool-call trajectory already logged. Heuristics, labeled as such; never counted toward pass/fail.
The aim is that every dataset reports all 7 dimensions (strict where authored, proxy for the rest).

Modality-agnostic: works for FHIR action tasks (PB), GUI tasks (HAB), and QA tool-use (MedCTA);
final_answer events are read as the agent's answer, so QA trajectories also yield signals.

Event schema (trajectory.jsonl): {event_type: tool_call|final_answer|..., tool, args, status,
result|observation, step, thought}.
"""
import json
import re



def _is_mutation(t):
    t = t or ""
    return any(x in t for x in ("create", "write_file", "submit", "update", "_post", "send"))


def _observation(e):
    return str(e.get("result") or e.get("observation") or "").strip()


def _canon_modalities(e):
    """Read the CANONICAL observation layer (canonical_observation is consumed here, not only
    written). Returns the non-empty modality dict of e["canonical_observation"], or None if absent."""
    co = e.get("canonical_observation")
    if isinstance(co, dict) and isinstance(co.get("modalities"), dict):
        return {k: v for k, v in co["modalities"].items() if v}
    return None


def _has_observation(e):
    cm = _canon_modalities(e)
    if cm is not None:        # canonical layer present -> audit-grade signal
        return bool(cm)
    return bool(_observation(e))   # fallback for bundles without a canonical layer


def _errored(e):
    # OUTCOME-based, NOT content-based. A tool's error lives in its STRUCTURED verdict (status /
    # canonical_result / an explicit error key / a FHIR OperationOutcome envelope / a real traceback),
    # never in bare words inside its payload. Free-text substring matching on "error"/"not found" would
    # misfire on clinical documents (a note reading "date not found" would turn a genuine success into a
    # false SILENT failure), polluting delivery / error_transparency and Care Action Fulfillment attribution.
    # An EMPTY search result (zero rows) is NOT an error: the call succeeded and the agent saw "no results".
    s = str(e.get("status", "")).lower()
    if s and s not in ("ok", "success", "done"):
        return True
    if e.get("error") or e.get("error_type"):
        return True
    cr = e.get("canonical_result")
    if isinstance(cr, dict):
        cs = str(cr.get("status", "")).lower()
        ck = str(cr.get("kind", "")).lower()
        if cs in ("error", "failure", "failed") or ck in ("error", "failure"):
            return True
    res = e.get("result")
    if isinstance(res, dict) and isinstance(res.get("output"), dict):
        res = res.get("output")
    if isinstance(res, dict) and (res.get("error") or res.get("errors")):
        return True
    r = _observation(e).lower()
    if '"resourcetype": "operationoutcome"' in r or "traceback (most recent call last)" in r:
        return True
    return False
def proxy_dimensions(events):
    """Return {dim: {score in [0,1], basis}} for dims derivable from one task's trajectory."""
    calls = [e for e in events if e.get("event_type") == "tool_call"]
    has_final = any(e.get("event_type") == "final_answer" for e in events)
    n = len(calls)
    out = {}
    if n == 0:
        return out

    # --- Clinical Resource Competence: tool-use quality = low error + low redundancy ---
    err = sum(_errored(e) for e in calls) / n
    seen, dup = set(), 0
    for e in calls:
        k = (e.get("tool"), json.dumps(e.get("args"), sort_keys=True, ensure_ascii=False))
        if k in seen:
            dup += 1
        else:
            seen.add(k)
    redun = dup / n
    # tool_execution_hygiene: NOT the Clinical Resource Competence dimension (that is tool_use_quality, a strict LLM judge).
    # This only measures whether calls ran smoothly / non-redundantly (execution != selection-correctness).
    out["tool_execution_hygiene"] = {"score": round(max(0.0, 1.0 - 0.5 * err - 0.5 * redun), 3),
                                     "basis": "n=%d err=%.2f redundant=%.2f" % (n, err, redun)}

    # --- Clinical Auditability: fraction of tool calls that produced a recorded observation (audit trail) ---
    # The execution system must DELIVER task evidence/failures to the deciding agent. A 3-layer pipeline
    # is reported EXPLICITLY so "delivered" is never conflated with "used":
    #   availability (env/tools produced a valid result) -> exposure (harness put it in the trace/context)
    #   -> uptake (agent actually referenced it). Plus error_transparency: are failures surfaced?
    exposed = sum(1 for e in calls if _has_observation(e))
    exposure = round(exposed / n, 3)                                    # harness-side delivery
    valid = sum(1 for e in calls if _has_observation(e) and not _errored(e))
    availability = round(valid / n, 3)                                 # env/tools produced usable evidence
    _errs = [e for e in calls if _errored(e)]
    err_transp = round(sum(1 for e in _errs if e.get("error_type") or str(e.get("status", "")).lower() == "error") / len(_errs), 3) if _errs else None
    # Uptake via EVIDENCE-UNIT matching, NOT raw long-word overlap (which would reward copying
    # tool prose and penalize correct paraphrase). Extract salient, matchable units — measurements/
    # numbers+units and key radiology finding terms — and check coverage in the final answer. Still a
    # COARSE deterministic signal (true semantic uptake needs a judge): reported with its method + a
    # reliability flag, and kept at low weight so it does not dominate the composite.
    _fa = " ".join(str(e.get("thought", "")) for e in events if e.get("event_type") == "final_answer").lower()
    _units = set()
    for e in calls:
        for v in (_canon_modalities(e) or {}).values():
            t = str(v).lower()
            _units |= set(re.findall(r"\d+(?:\.\d+)?\s?(?:cm|mm|hu|%|ml|mg|mmhg)?", t))
            _units |= set(re.findall(r"\b(?:left|right|bilateral|anterior|posterior|superior|inferior|hypodense|hyperdense|hypoattenuating|enhancing|mass|lesion|nodule|cyst|stenosis|occlusion|thrombus|calcification|hemorrhage|dilation|dilated)\b", t))
    _units = {u.strip() for u in _units if u.strip() and not u.strip().isdigit() or (u.strip().isdigit() and len(u.strip()) >= 2)}
    uptake = round(sum(1 for u in _units if u in _fa) / len(_units), 3) if (_units and _fa) else None
    # composite reflects the FULL pipeline: delivery (exposure) + failure transparency + agent uptake.
    # uptake is the discriminating layer (delivery alone saturates at 1.0). uptake None -> fall back to exposure.
    # Average ONLY applicable layers. error_transparency is N/A (not 1.0) when there was no
    # error opportunity -> a clean run does not collect free transparency credit. exposure always
    # applies; uptake applies if computable. (This is delivery+uptake = evidence-pipeline effectiveness,
    # NOT pure agent observability -- labelled as such in the basis.)
    _layers = [("exposure", exposure, 0.5)]
    if err_transp is not None: _layers.append(("error_transparency", err_transp, 0.2))
    if uptake is not None: _layers.append(("uptake", uptake, 0.3))
    _tw = sum(w for _, _, w in _layers)
    _score = round(sum(v * w for _, v, w in _layers) / _tw, 3) if _tw else None
    out["Clinical Auditability"] = {"score": _score, "evidence_availability": availability,
                            "evidence_exposure": exposure, "evidence_uptake": uptake,
                            "error_transparency": err_transp,
                            "measures": "evidence_pipeline_effectiveness(delivery+uptake), not pure agent observability",
                            "uptake_method": "evidence_unit_matching(measurements+finding_terms)",
                            "uptake_reliability": "coarse_deterministic_proxy_judge_needed_for_semantic",
                            "applicable_layers": [n for n, _, _ in _layers],
                            "basis": "exposure=%d/%d avail=%d/%d err_transp=%s uptake=%s" % (exposed, n, valid, n, err_transp, uptake)}
    out["trace_observation_coverage"] = {"score": exposure,           # harness-side mirror for agent-vs-harness comparison (-> integrity)
                                         "basis": "%d/%d tool results delivered into canonical trace" % (exposed, n)}

    # Care Action Fulfillment and Clinical Workflow Continuity are not computed here. They are scored by the substrate-based dimension
    # evaluators runner/scorers/dim_execution.execution + runner/scorers/dim_lifecycle.lifecycle (wired in
    # aggregate_report._experimental_evaluators), which fill the Care Action Fulfillment/Clinical Workflow Continuity dimension cells.
    # The Clinical Auditability block above serves only as (a) the trace_observation_coverage harness-side
    # mirror and (b) the conformance contract (test_observability_consumes_canonical); the REPORT's
    # Clinical Auditability dimension cell is filled by runner/scorers/dim_observability.observability over the
    # substrate EvidenceView (single source of truth for the report).
    return out


def average_proxy(per_task):
    """per_task: list of {dim: {score, basis}} -> {dim: {mean, n}}."""
    acc = {}
    for d in per_task:
        for dim, v in d.items():
            acc.setdefault(dim, []).append(v["score"])
    return {dim: {"mean": round(sum(xs) / len(xs), 3), "n": len(xs)} for dim, xs in acc.items()}
