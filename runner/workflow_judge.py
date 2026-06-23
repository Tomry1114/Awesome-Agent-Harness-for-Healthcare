#!/usr/bin/env python3
"""Lifecycle upgrade: strict cp_workflow_quality via LLM judge (clinical/admin workflow ordering &
completeness). Same evidence-complete + explicit-rubric standard as tool_use_judge. Writes a strict
Lifecycle checkpoint. For workflow benchmarks (PB/HAB); QA tasks (MedCTA) keep proxy (no real workflow)."""
import json, os, sys, glob
from tool_use_judge import _gateway, _parse, _evidence, _MODEL

SUBS = ["evidence_before_decision", "logical_progression", "prerequisite_before_action", "completeness"]
_SYS = """You evaluate the WORKFLOW quality of a medical/administrative agent. Given the task, the
agent's full trajectory (calls, args, observations) and final answer, judge whether the agent followed
a sound multi-step workflow. Do NOT require a fixed step order — accept legitimate variants. Score each
0/1/2 (2=sound, 1=minor issue, 0=violated):
- evidence_before_decision: gathered relevant data BEFORE deciding/acting/answering
- logical_progression: steps follow a coherent clinical/admin progression (no out-of-order jumps)
- prerequisite_before_action: high-risk or mutating actions are preceded by their required checks
- completeness: the workflow reached an appropriate end state (not abandoned mid-way)
Reply with ONLY JSON: {"evidence_before_decision":0-2,"logical_progression":0-2,"prerequisite_before_action":0-2,"completeness":0-2,"reason":"<=30 words"}"""


def judge_dir(agent_dir):
    rows = []
    for rp in sorted(glob.glob(os.path.join(agent_dir, "*", "result.json"))):
        bdir = os.path.dirname(rp)
        try:
            v = _parse(_gateway(_SYS, _evidence(bdir)))
        except Exception as e:
            v = None
            sys.stderr.write("err %s: %r\n" % (bdir, e))
        if not v:
            rows.append((os.path.basename(bdir), None))
            continue
        q = sum(float(v.get(k, 0)) for k in SUBS) / (len(SUBS) * 2.0)
        r = json.load(open(rp))
        r["checkpoints"] = [c for c in (r.get("checkpoints") or []) if c.get("id") != "cp_workflow_quality"]
        r["checkpoints"].append({
            "id": "cp_workflow_quality", "category": "lifecycle", "type": "llm_judge",
            "dimension": "Lifecycle", "subdimension": "workflow_quality",
            "checkpoint_status": "passed" if q >= 0.5 else "failed",
            "failure_mode": None if q >= 0.5 else "agent_failure", "weight": 1.0,
            "score": round(q, 3), "score_eligible": True, "evaluator_kind": "workflow_judge",
            "judge_backend": _MODEL, "subscores": {k: v.get(k) for k in SUBS},
            "detail": {"reason": v.get("reason")}})
        json.dump(r, open(rp, "w"), indent=1, ensure_ascii=False)
        rows.append((os.path.basename(bdir), round(q, 3)))
    return rows


if __name__ == "__main__":
    rows = judge_dir(sys.argv[1])
    vals = [q for _, q in rows if q is not None]
    for name, q in rows:
        print("  %-34s %s" % (name, q))
    print("workflow_quality mean: %s over %d" % (round(sum(vals) / len(vals), 3) if vals else None, len(vals)))
