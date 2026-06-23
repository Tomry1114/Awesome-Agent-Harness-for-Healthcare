#!/usr/bin/env python3
"""Trajectory-derived PROXY dimension signals (score_eligible=False, soft, do NOT enter primary
dimension_scores or success). Fills dimension cells a benchmark does not formally test, from the
tool-call trajectory already logged. Honest heuristics — labeled as such; never counted toward
pass/fail. See docs/ARCHITECTURE.md (proxy track) and METRICS.md.

Event schema (trajectory.jsonl): {event_type: tool_call|final_answer|..., tool, args, status,
result|observation, step, thought}.
"""
import json


def _is_retrieval(t):
    t = t or ""
    return ("search" in t) or ("read" in t) or t.endswith("_get") or ("_get_" in t)


def _is_mutation(t):
    t = t or ""
    return any(x in t for x in ("create", "write_file", "submit", "update", "_post", "send"))


def _errored(e):
    s = str(e.get("status", "")).lower()
    r = str(e.get("result") or e.get("observation") or "").lower()
    if s and s not in ("ok", "success", "done"):
        return True
    return ("error" in r) or ("not found" in r) or ('"total": 0' in r) or ("traceback" in r)


def proxy_dimensions(events):
    """Return {dim: {score in [0,1], basis}} for dims derivable from one task's trajectory."""
    calls = [e for e in events if e.get("event_type") == "tool_call"]
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
    out["Tooling"] = {"score": round(max(0.0, 1.0 - 0.5 * err - 0.5 * redun), 3),
                      "basis": "n=%d err=%.2f redundant=%.2f" % (n, err, redun)}

    # --- Lifecycle: ordering sanity = mutations preceded by >=1 retrieval ---
    retrieved, muts, ok_order = False, 0, 0
    for e in calls:
        t = e.get("tool")
        if _is_retrieval(t):
            retrieved = True
        if _is_mutation(t):
            muts += 1
            if retrieved:
                ok_order += 1
    if muts:
        out["Lifecycle"] = {"score": round(ok_order / muts, 3),
                            "basis": "%d/%d mutations after retrieval" % (ok_order, muts)}
    # NOTE: deliberately NO Execution proxy — a "produced artifact" heuristic is action-biased
    # (false 0 for QA tasks like MedCTA) and otherwise near-constant. Honest signal only.
    return out


def average_proxy(per_task):
    """per_task: list of {dim: {score, basis}} -> {dim: {mean, n}}."""
    acc = {}
    for d in per_task:
        for dim, v in d.items():
            acc.setdefault(dim, []).append(v["score"])
    return {dim: {"mean": round(sum(xs) / len(xs), 3), "n": len(xs)} for dim, xs in acc.items()}
