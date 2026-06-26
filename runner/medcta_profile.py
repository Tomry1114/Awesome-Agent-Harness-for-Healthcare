"""MedCTA ToolProtocolAgent failure profiler v2 (MCTA-0..9). One process => model loads once.
Deterministic MULTI-LABEL classifier. Separates GENUINE failure tags from formal-metric diagnostics
(cp_tool_selection / cp_arg_accuracy are reference-SEQUENCE match = under-use signal, NOT 'wrong tool').
Heuristic by design. Persists full trajectory + all cp + actual args for offline re-classification."""
import sys, os, json, re
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run

PERCEPTION = {"ImageDescription", "RegionAttributeDescription", "OCR"}
TAGS = ["tool_selection_error", "tool_argument_error", "image_misread", "search_misuse",
        "loop_or_invalid_action", "final_answer_format_error", "outcome_proxy_fail"]

def classify(task, res):
    traj = res.get("_trajectory", [])
    cp = {c["id"]: c["checkpoint_status"] for c in res["checkpoints"]}
    avail = {t.get("name") for t in task.get("available_tools", [])}
    tcs = [(e["tool"], e.get("args") or {}) for e in traj if e.get("event_type") == "tool_call"]
    results = [e.get("result") for e in traj if e.get("event_type") == "tool_call"]
    errs = [e for e in traj if e.get("event_type") == "agent_error"]
    finals = [e.get("thought", "") or "" for e in traj if e.get("event_type") == "final_answer"]
    tags = set()
    perceived = any(t in PERCEPTION for t, _ in tcs)
    last_final = finals[-1] if finals else None

    # final_answer_format_error: leftover protocol in the 'answer', or empty answer after tool calls
    fmt = False
    if last_final is not None:
        if re.search(r'<tool_call|</?answer>|^\s*\{\s*"name"', last_final): fmt = True
        if last_final.strip() == "" and tcs: fmt = True
    if fmt: tags.add("final_answer_format_error")

    # loop_or_invalid_action
    rep = Counter((t, json.dumps(a, sort_keys=True, ensure_ascii=False)) for t, a in tcs)
    rmax = max(rep.values()) if rep else 0
    if any(e.get("error") == "max_steps_exceeded" for e in errs) or rmax >= 3: tags.add("loop_or_invalid_action")
    if any(e.get("error") in ("invalid_action", "bad_action_type") for e in errs): tags.add("loop_or_invalid_action")

    # tool_selection_error: GENUINE only — unknown tool, or answered with zero perception (not a fmt error)
    if any(t not in avail for t, _ in tcs): tags.add("tool_selection_error")
    if finals and not perceived and not fmt: tags.add("tool_selection_error")

    # tool_argument_error: tool returned error dict, OR image tool got a bogus text 'image' arg
    if any(isinstance(r, dict) and "error" in r for r in results): tags.add("tool_argument_error")
    for t, a in tcs:
        if t in PERCEPTION and isinstance(a, dict) and isinstance(a.get("image"), str) and len(a.get("image", "")) > 20:
            tags.add("tool_argument_error")

    # search_misuse
    gs = [r for (t, _), r in zip(tcs, results) if t == "GoogleSearch"]
    if any("no offline result" in str(r) for r in gs) or len(gs) >= 3: tags.add("search_misuse")

    # image_misread: perceived, produced a real answer, outcome proxy wrong, not fmt/loop
    if perceived and last_final and last_final.strip() and cp.get("cp_outcome") == "failed" \
       and "final_answer_format_error" not in tags and "loop_or_invalid_action" not in tags:
        tags.add("image_misread")

    # outcome_proxy_fail
    if cp.get("cp_outcome") == "failed": tags.add("outcome_proxy_fail")

    diag = {"toolacc_formal": cp.get("cp_tool_selection"), "argacc_formal": cp.get("cp_arg_accuracy"),
            "grounding": cp.get("cp_grounding"), "no_fabrication": cp.get("cp_no_fabrication"),
            "underuse_vs_ref": (cp.get("cp_tool_selection") == "failed" and perceived and rmax < 3)}
    facts = {"n_tool_calls": len(tcs), "tools_used": [t for t, _ in tcs][:12], "repeated_max": rmax,
             "n_search": len(gs), "perceived": perceived, "cp": cp, "diag": diag,
             "tool_args": [{"tool": t, "args": a} for t, a in tcs][:12],
             "final_preview": (last_final[:200] if last_final else None)}
    return sorted(tags), facts

def main():
    rows, agg, diag_agg = [], Counter(), Counter()
    for i in range(10):
        tid = "MCTA-%d" % i
        try:
            task = run.load_task("MedCTA", tid)
            res = run.run_task("MedCTA", tid, "qwen", max_steps=10, cleanup=False)
            tags, facts = classify(task, res)
            row = {"task": tid, "success": res["success"], "status": res["evaluation_status"], "tags": tags, **facts}
            if facts["diag"]["underuse_vs_ref"]: diag_agg["underuse_vs_ref"] += 1
            if facts["diag"]["toolacc_formal"] == "failed": diag_agg["toolacc_formal_fail"] += 1
            if facts["diag"]["argacc_formal"] == "failed": diag_agg["argacc_formal_fail"] += 1
        except Exception as e:
            import traceback; traceback.print_exc()
            row = {"task": tid, "error": repr(e), "tags": ["RUN_ERROR"]}
        for t in row["tags"]: agg[t] += 1
        rows.append(row)
        print("[%s] succ=%s steps=%s calls=%s rep=%s tags=%s" % (
            tid, row.get("success"), row.get("n_tool_calls"), row.get("n_tool_calls"), row.get("repeated_max"), row["tags"]))
        sys.stdout.flush()
    out = {"n": len(rows), "aggregate": dict(agg), "diagnostics": dict(diag_agg), "rows": rows}
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "medcta_profile.json"), "w"), ensure_ascii=False, indent=1)
    print("\n=== FAILURE TAGS (tasks hitting each, multi-label) ===")
    for t in TAGS: print("  %-28s %d/%d" % (t, agg.get(t, 0), len(rows)))
    print("=== FORMAL-METRIC DIAGNOSTICS ===")
    for k, v in diag_agg.items(): print("  %-28s %d/%d" % (k, v, len(rows)))
    print("WROTE medcta_profile.json")

if __name__ == "__main__":
    main()
