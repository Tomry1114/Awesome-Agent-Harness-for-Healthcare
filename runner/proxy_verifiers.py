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


def _is_retrieval(t):
    t = t or ""
    return ("search" in t) or ("read" in t) or t.endswith("_get") or ("_get_" in t) \
        or any(x in t for x in ("ocr", "description", "image", "region", "google", "lookup"))


def _is_mutation(t):
    t = t or ""
    return any(x in t for x in ("create", "write_file", "submit", "update", "_post", "send"))


def _observation(e):
    return str(e.get("result") or e.get("observation") or "").strip()


def _errored(e):
    s = str(e.get("status", "")).lower()
    r = _observation(e).lower()
    if s and s not in ("ok", "success", "done"):
        return True
    return ("error" in r) or ("not found" in r) or ('"total": 0' in r) or ("traceback" in r)


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
    observed = sum(1 for e in calls if _observation(e))
    out["Observability"] = {"score": round(observed / n, 3),
                            "basis": "%d/%d calls produced an observation" % (observed, n)}

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
    done = has_final and ok_calls > 0
    out["Execution"] = {"score": 1.0 if done else round(ok_calls / n, 3),
                        "basis": "final=%s ok_calls=%d/%d" % (has_final, ok_calls, n)}
    return out


def average_proxy(per_task):
    """per_task: list of {dim: {score, basis}} -> {dim: {mean, n}}."""
    acc = {}
    for d in per_task:
        for dim, v in d.items():
            acc.setdefault(dim, []).append(v["score"])
    return {dim: {"mean": round(sum(xs) / len(xs), 3), "n": len(xs)} for dim, xs in acc.items()}
