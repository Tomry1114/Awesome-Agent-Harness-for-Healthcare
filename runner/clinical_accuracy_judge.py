#!/usr/bin/env python3
"""Native MedCTA clinical_accuracy (4 sub-metrics) — F_acc / C_s / F_p / S_comp — which we were
missing (only GAcc). FAITHFUL: reuses the upstream rubric prompts verbatim from
benchmark/MedCTA/clinical_accuracy.py (get_metric_prompt), only swapping the API call to our gateway.
Runs post-hoc over MedCTA bundles (pred trajectory) vs the task gold reference_trace, reports the four
scores in report.native_metrics.clinical_accuracy. No agent re-run / GPU."""
import json, os, sys, glob

NATIVE = os.path.join("benchmark", "MedCTA")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from tool_use_judge import _gateway, _MODEL


def _load_native_prompts():
    """Grab the upstream rubric prompts verbatim. The native clinical_accuracy.py has a syntax bug on
    line 17 (`HF_TOKEN = os.getenv(...) = ""`) and a hard `from openai import OpenAI` — so we patch the
    bad line and stub the openai import, then exec ONLY to recover get_metric_prompt / the 4 prompts."""
    src = open(os.path.join(NATIVE, "clinical_accuracy.py")).read()
    src = src.replace('HF_TOKEN = os.getenv("HF_TOKEN") = ""', 'HF_TOKEN = os.getenv("HF_TOKEN") or ""')
    import types
    fake = types.ModuleType("openai")
    fake.OpenAI = object
    sys.modules.setdefault("openai", fake)
    ns = {}
    exec(compile(src, "clinical_accuracy.py", "exec"), ns)
    return ns["get_metric_prompt"]


get_metric_prompt = _load_native_prompts()

METRICS = ["F_acc", "C_s", "F_p", "S_comp"]


def _pred_traj(bdir):
    steps, answer = [], ""
    tp = os.path.join(bdir, "trajectory.jsonl")
    if os.path.exists(tp):
        for l in open(tp):
            e = json.loads(l)
            if e.get("event_type") == "tool_call":
                steps.append({"tool": e.get("tool"), "args": e.get("args"),
                              "observation": str(e.get("result") or e.get("observation") or "")[:500]})
            elif e.get("event_type") == "final_answer":
                answer = e.get("thought") or ""
    return {"trajectory": steps, "final_answer": answer}


def _gold_traj(bdir):
    t = json.load(open(os.path.join(bdir, "task.json")))
    ref = (t.get("reference") or {})
    return ref.get("reference_trace") or ref.get("tool_chain") or []


def _score(metric, gold, pred):
    system_prompt, user_template = get_metric_prompt(metric)
    user = user_template.format(gold_traj=json.dumps(gold, ensure_ascii=False)[:4000],
                                pred_traj=json.dumps(pred, ensure_ascii=False)[:6000])
    raw = _gateway(system_prompt + "\nReply with ONLY JSON {\"score\": number in [0,1]}.", user)
    s, e = raw.find("{"), raw.rfind("}")
    try:
        v = json.loads(raw[s:e + 1])
        sc = float(v.get("score"))
        return max(0.0, min(1.0, sc))
    except Exception:
        return None


def run(agent_dir):
    acc = {m: [] for m in METRICS}
    for rp in sorted(glob.glob(os.path.join(agent_dir, "*", "result.json"))):
        bdir = os.path.dirname(rp)
        gold, pred = _gold_traj(bdir), _pred_traj(bdir)
        row = {}
        for m in METRICS:
            sc = _score(m, gold, pred)
            row[m] = sc
            if sc is not None:
                acc[m].append(sc)
        print("  %-28s %s" % (os.path.basename(bdir), row))
    means = {m: (round(sum(v) / len(v), 3) if v else None) for m, v in acc.items()}
    print("clinical_accuracy means:", means)
    # persist into report.json native_metrics
    rep_path = os.path.join(agent_dir, "report.json")
    if os.path.exists(rep_path):
        rep = json.load(open(rep_path))
        rep.setdefault("native_metrics", {})["clinical_accuracy"] = {
            "F_acc": means["F_acc"], "C_s": means["C_s"], "F_p": means["F_p"], "S_comp": means["S_comp"],
            "source": "upstream clinical_accuracy.py rubric via gateway"}
        json.dump(rep, open(rep_path, "w"), indent=1, ensure_ascii=False)
        print("-> written to", rep_path)
    return means


if __name__ == "__main__":
    run(sys.argv[1])
