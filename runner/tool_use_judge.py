#!/usr/bin/env python3
"""tool_use_quality: a STRICT harness-native Tooling metric via LLM judge (no fixed reference
trajectory needed — legitimate alternative tool paths are accepted; the judge scores SEMANTIC
correctness of the path, not step-by-step match to a gold trace).

Distinct from `tool_execution_hygiene` (the 1-0.5err-0.5redun heuristic, proxy): execution success
!= selection correctness. The judge sees COMPLETE evidence: task goal, available tools+docs, every
tool call with arguments AND the observation it returned, and the final answer. It scores five
sub-dimensions 0/1/2 with an explicit rubric, plus an unnecessary-use score reported separately.

  Tooling(tool_use_quality) = (relevance + necessity + argument + sequence + evidence_use) / (5*2)

Writes a strict checkpoint cp_tool_use_quality (dimension=Tooling, score_eligible=True) into each
result.json. Post-hoc over existing bundles; reuses the gateway (gpt-5.5). No agent re-run / GPU.
"""
import json, os, sys, glob, collections
import gateway

_MODEL = os.environ.get("MH_JUDGE_MODEL", "gpt-5.4")
SUBS = ["relevance", "necessity", "argument", "sequence", "evidence_use"]

_SYS = """You are an expert evaluator of a medical AI agent's TOOL USE. You are given a task, the
tools available (with signatures), the agent's FULL trajectory (each tool call with its arguments
AND the observation it returned), and the final answer. Evaluate the SEMANTIC quality of the tool-use
path. Do NOT require a single canonical path — legitimate alternative tool routes are fully acceptable.
IMPORTANT: a tool call returning success only means the API ran; it does NOT mean the right tool/args
were chosen (e.g. querying the WRONG patient is execution-success but selection-wrong).

Score each sub-dimension on this scale:
  2 = good (selection/args/order sound, no critical gap)
  1 = acceptable but minor issue (small gap or minor unnecessary call)
  0 = wrong (wrong tool/args, missing a NECESSARY tool/info source, or tool results not used)

Sub-dimensions:
- relevance: are the called tools relevant to the task?
- necessity: were all NECESSARY tools/info sources used (no critical omission)?
- argument: are arguments appropriate to the patient/resource/goal (right patient, right codes)?
- sequence: is the order sound — especially prerequisite queries BEFORE high-risk actions?
- evidence_use: does the final answer correctly use the information the tools returned?
Also rate `unnecessary` 0/1/2 (2 = no redundant/no-contribution calls; 0 = many).

Reply with ONLY a JSON object, no prose:
{"relevance":0-2,"necessity":0-2,"argument":0-2,"sequence":0-2,"evidence_use":0-2,"unnecessary":0-2,"reason":"<=40 words"}"""


def _gateway(system, user):
    res = gateway.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                       model=_MODEL, max_tokens=1500, judge=True, timeout=200)
    return res["content"] if res["ok"] else ""


def _parse(raw):
    s = raw.find("{"); e = raw.rfind("}")
    if s == -1 or e == -1:
        return None
    try:
        return json.loads(raw[s:e + 1])
    except Exception:
        return None


def _evidence(bdir):
    task = json.load(open(os.path.join(bdir, "task.json")))
    goal = task.get("goal") or task.get("instruction") or ""
    tools = task.get("available_tools") or []
    tool_lines = "\n".join("- %s : %s" % (t.get("name"), t.get("signature", "")) for t in tools)
    steps, answer = [], ""
    tp = os.path.join(bdir, "trajectory.jsonl")
    if os.path.exists(tp):
        for l in open(tp):
            ev = json.loads(l)
            if ev.get("event_type") == "tool_call":
                obs = str(ev.get("result") or ev.get("observation") or "")[:400]
                steps.append("CALL %s args=%s -> %s" % (ev.get("tool"),
                             json.dumps(ev.get("args"), ensure_ascii=False)[:200], obs))
            elif ev.get("event_type") == "final_answer":
                answer = ev.get("thought") or ev.get("answer") or ""
    traj = "\n".join(steps[:40]) if steps else "(no tool calls)"
    return ("TASK:\n%s\n\nAVAILABLE TOOLS:\n%s\n\nTRAJECTORY (call, args, observation):\n%s\n\nFINAL ANSWER:\n%s"
            % (goal[:2500], tool_lines, traj[:6000], answer[:1500]))


def _judge_sampled(system, user, n):
    """#4 judge multi-sampling: average n independent judgments to reduce judge noise (~0.12 measured
    at n=1). MH_JUDGE_SAMPLES controls n. Sub-scores become averaged floats."""
    runs = []
    for _ in range(max(1, n)):
        v = _parse(_gateway(system, user))
        if v:
            runs.append(v)
    if not runs:
        return None
    keys = SUBS + ["unnecessary"]
    avg = {k: round(sum(float(r.get(k, 0)) for r in runs) / len(runs), 3)
           for k in keys if any(k in r for r in runs)}
    avg["_n_samples"] = len(runs)
    avg["reason"] = runs[0].get("reason")
    return avg


def judge_dir(agent_dir):
    rows = []
    n = int(os.environ.get("MH_JUDGE_SAMPLES", "1"))
    for rp in sorted(glob.glob(os.path.join(agent_dir, "*", "result.json"))):
        bdir = os.path.dirname(rp)
        try:
            v = _judge_sampled(_SYS, _evidence(bdir), n)
        except Exception as e:
            v = None
            sys.stderr.write("judge err %s: %r\n" % (bdir, e))
        if not v:
            rows.append((os.path.basename(bdir), None))
            continue
        quality = sum(float(v.get(k, 0)) for k in SUBS) / (len(SUBS) * 2.0)
        r = json.load(open(rp))
        r["checkpoints"] = [c for c in (r.get("checkpoints") or []) if c.get("id") != "cp_tool_use_quality"]
        r["checkpoints"].append({
            "id": "cp_tool_use_quality", "category": "tooling", "type": "llm_judge",
            "dimension": "Tooling", "subdimension": "tool_use_quality",
            "checkpoint_status": "passed" if quality >= 0.5 else "failed",
            "failure_mode": None if quality >= 0.5 else "agent_failure",
            "weight": 1.0, "score": round(quality, 3), "score_eligible": True,
            "evaluator_kind": "tool_use_judge", "judge_backend": _MODEL,
            "subscores": {k: v.get(k) for k in SUBS}, "unnecessary": v.get("unnecessary"),
            "n_samples": v.get("_n_samples"), "detail": {"reason": v.get("reason")}})
        json.dump(r, open(rp, "w"), indent=1, ensure_ascii=False)
        rows.append((os.path.basename(bdir), round(quality, 3)))
    return rows


if __name__ == "__main__":
    rows = judge_dir(sys.argv[1])
    vals = [q for _, q in rows if q is not None]
    for name, q in rows:
        print("  %-34s %s" % (name, q))
    print("tool_use_quality mean: %s over %d" % (round(sum(vals) / len(vals), 3) if vals else None, len(vals)))
