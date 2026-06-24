#!/usr/bin/env python3
"""Trajectory-derived PROXY dimension signals (score_eligible=False, soft, do NOT enter primary
dimension_scores or success). Fills dimension cells a benchmark does not formally test, from the
tool-call trajectory already logged. Honest heuristics — labeled as such; never counted toward
pass/fail. GOAL: every dataset has all 7 dims (strict where authored + proxy for the rest).

Modality-agnostic: works for FHIR action tasks (PB), GUI tasks (HAB), and QA tool-use (MedCTA) —
treats a final_answer as a 'goal' action so QA trajectories also yield Execution/Lifecycle.

Event schema (trajectory.jsonl): {event_type: tool_call|final_answer|..., tool, args, status,
result|observation, step, thought}.
"""
import json
import re


def _is_retrieval(t):
    t = t or ""
    return ("search" in t) or ("read" in t) or t.endswith("_get") or ("_get_" in t) \
        or any(x in t for x in ("ocr", "description", "image", "region", "google", "lookup"))


def _is_mutation(t):
    t = t or ""
    return any(x in t for x in ("create", "write_file", "submit", "update", "_post", "send"))


def _observation(e):
    return str(e.get("result") or e.get("observation") or "").strip()


def _canon_modalities(e):
    """Read the CANONICAL observation layer (Codex: canonical_observation must be consumed, not just
    written). Returns the non-empty modality dict of e["canonical_observation"], or None if absent."""
    co = e.get("canonical_observation")
    if isinstance(co, dict) and isinstance(co.get("modalities"), dict):
        return {k: v for k, v in co["modalities"].items() if v}
    return None


def _has_observation(e):
    cm = _canon_modalities(e)
    if cm is not None:        # canonical layer present -> audit-grade signal
        return bool(cm)
    return bool(_observation(e))   # pre-canonical bundle fallback


def _errored(e):
    s = str(e.get("status", "")).lower()
    r = _observation(e).lower()
    if s and s not in ("ok", "success", "done"):
        return True
    # #11: tighten bare-substring "error" (misfires on legit text mentioning "error").
    # Treat as error ONLY on an explicit structured indicator: a JSON "error" key, a leading
    # [error... marker, or an "error:"/"error " token at the start of a line — NOT any word "error".
    if '"error"' in r:
        return True
    if re.search(r'(^|[\n>\]\}])\s*\[?error[:\s\]"]', r):
        return True
    return ("not found" in r) or ('"total": 0' in r) or ("traceback" in r)


def proxy_dimensions(events):
    """Return {dim: {score in [0,1], basis}} for dims derivable from one task's trajectory."""
    calls = [e for e in events if e.get("event_type") == "tool_call"]
    has_final = any(e.get("event_type") == "final_answer" for e in events)
    n = len(calls)
    out = {}
    if n == 0:
        return out

    # --- Tooling: tool-use quality = low error + low redundancy ---
    err = sum(_errored(e) for e in calls) / n
    seen, dup = set(), 0
    for e in calls:
        k = (e.get("tool"), json.dumps(e.get("args"), sort_keys=True, ensure_ascii=False))
        if k in seen:
            dup += 1
        else:
            seen.add(k)
    redun = dup / n
    # tool_execution_hygiene: NOT the Tooling dimension (that is tool_use_quality, a strict LLM judge).
    # This only measures whether calls ran smoothly / non-redundantly (execution != selection-correctness).
    out["tool_execution_hygiene"] = {"score": round(max(0.0, 1.0 - 0.5 * err - 0.5 * redun), 3),
                                     "basis": "n=%d err=%.2f redundant=%.2f" % (n, err, redun)}

    # --- Observability: fraction of tool calls that produced a recorded observation (audit trail) ---
    # Observability (refined): the execution system must DELIVER task evidence/failures to the deciding
    # agent. 3-layer pipeline reported EXPLICITLY so "delivered" is never conflated with "used":
    #   availability (env/tools produced a valid result) -> exposure (harness put it in the trace/context)
    #   -> uptake (agent actually referenced it). Plus error_transparency (O3): are failures surfaced?
    exposed = sum(1 for e in calls if _has_observation(e))
    exposure = round(exposed / n, 3)                                    # harness-side delivery (was the whole metric)
    valid = sum(1 for e in calls if _has_observation(e) and not _errored(e))
    availability = round(valid / n, 3)                                 # env/tools produced usable evidence
    _errs = [e for e in calls if _errored(e)]
    err_transp = round(sum(1 for e in _errs if e.get("error_type") or str(e.get("status", "")).lower() == "error") / len(_errs), 3) if _errs else None
    _fa = " ".join(str(e.get("thought", "")) for e in events if e.get("event_type") == "final_answer").lower()
    _terms = set()
    for e in calls:
        for v in (_canon_modalities(e) or {}).values():
            _terms.update(w for w in str(v).lower().split() if len(w) > 6)
    uptake = round(sum(1 for w in _terms if w in _fa) / len(_terms), 3) if (_terms and _fa) else None
    # composite reflects the FULL pipeline: delivery (exposure) + failure transparency + agent uptake.
    # uptake is the discriminating layer (delivery alone saturates at 1.0). uptake None -> fall back to exposure.
    _up = uptake if uptake is not None else exposure
    _et = err_transp if err_transp is not None else 1.0
    _score = round(0.5 * exposure + 0.2 * _et + 0.3 * _up, 3)
    out["Observability"] = {"score": _score, "evidence_availability": availability,
                            "evidence_exposure": exposure, "evidence_uptake": uptake,
                            "error_transparency": err_transp,
                            "basis": "exposure=%d/%d avail=%d/%d err_transp=%s uptake=%s" % (exposed, n, valid, n, err_transp, uptake)}
    out["trace_observation_coverage"] = {"score": exposure,           # harness-side mirror for agent-vs-harness comparison (-> integrity)
                                         "basis": "%d/%d tool results delivered into canonical trace" % (exposed, n)}

    # --- Lifecycle: ordering sanity = each goal (mutation OR final answer) preceded by info-gathering ---
    info_seen, goals, ok = False, 0, 0
    for e in events:
        et, t = e.get("event_type"), e.get("tool")
        if et == "tool_call":
            if _is_retrieval(t) or _observation(e):
                info_seen = True
            if _is_mutation(t):
                goals += 1
                ok += 1 if info_seen else 0
        elif et == "final_answer":
            goals += 1
            ok += 1 if info_seen else 0
    if goals:
        out["Lifecycle"] = {"score": round(ok / goals, 3),
                            "basis": "%d/%d goals after info-gathering" % (ok, goals)}

    # --- Execution: operational completion = reached a terminal answer with >=1 successful tool call ---
    ok_calls = sum(1 for e in calls if not _errored(e))
    tool_success_rate = ok_calls / n
    # Codex #3: a trajectory with NO final answer must NOT score 1.0 just because tools succeeded.
    # Split the two concepts and combine; final_reached and tool_success_rate are kept as sub-signals.
    out["Execution"] = {"score": round(0.5 * (1.0 if has_final else 0.0) + 0.5 * tool_success_rate, 3),
                        "final_reached": bool(has_final), "tool_success_rate": round(tool_success_rate, 3),
                        "basis": "0.5*final(%s) + 0.5*tool_success(%d/%d)" % (has_final, ok_calls, n)}
    return out


def average_proxy(per_task):
    """per_task: list of {dim: {score, basis}} -> {dim: {mean, n}}."""
    acc = {}
    for d in per_task:
        for dim, v in d.items():
            acc.setdefault(dim, []).append(v["score"])
    return {dim: {"mean": round(sum(xs) / len(xs), 3), "n": len(xs)} for dim, xs in acc.items()}
