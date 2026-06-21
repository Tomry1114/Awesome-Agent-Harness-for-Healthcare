"""v0 metrics report: bundles -> Safety / Efficiency / Integrity-Meta, per benchmark.

Input = run_batch bundles (<out>/<agent>/<tid>/{task.json,trajectory.jsonl,result.json}).
Per SAFETY_SPEC + benchmark_metric/README: report a benchmark x panel matrix; NEVER average a
dimension across benchmarks. Each metric carries its sample size / coverage. Computable-today v0 set.
"""
import json, glob, os, sys
from collections import defaultdict
import risk_annotator as ra, safety_metrics as sm

def load_bundles(root):
    out = defaultdict(list)
    for rj in glob.glob(os.path.join(root, "**", "result.json"), recursive=True):
        bdir = os.path.dirname(rj)
        try:
            res = json.load(open(rj))
            task = json.load(open(os.path.join(bdir, "task.json")))
            traj = [json.loads(l) for l in open(os.path.join(bdir, "trajectory.jsonl"))] \
                if os.path.exists(os.path.join(bdir, "trajectory.jsonl")) else []
        except Exception as e:
            print("skip", bdir, e); continue
        bench = task.get("source_benchmark") or res.get("source_benchmark") or "?"
        out[bench].append({"task": task, "res": res, "traj": traj})
    return out

def _strict(c): return c.get("score_eligible", True) is not False
def _eval(c): return c["checkpoint_status"] in ("passed", "failed")
def _tool_events(traj): return [e for e in traj if e.get("event_type") == "tool_call"]
def _rate(num, den): return (num / den) if den else None

# ---------- Efficiency ----------
def efficiency(items):
    res = [it["res"] for it in items]
    # task_success_rate (strict-based result.success)
    succ = sum(1 for r in res if r.get("success")); tsr = _rate(succ, len(res))
    # subtask_success_rate (strict, evaluated checkpoints)
    scp = [c for r in res for c in r.get("checkpoints", []) if _strict(c) and _eval(c)]
    ssr = _rate(sum(1 for c in scp if c["checkpoint_status"] == "passed"), len(scp))
    # tool_call_success_rate (trajectory status)
    tev = [e for it in items for e in _tool_events(it["traj"])]
    tcs = _rate(sum(1 for e in tev if e.get("status") == "ok"), len(tev))
    # argument_validity: of arg-bearing calls, those NOT rejected for bad args (http_400/422)
    argc = [e for e in tev if e.get("args")]
    bad = sum(1 for e in argc if e.get("error_type") in ("http_400", "http_422"))
    av = _rate(len(argc) - bad, len(argc))
    # redundant_action_rate: consecutive identical (tool,args)
    red = 0
    for it in items:
        ev = _tool_events(it["traj"]); prev = None
        for e in ev:
            key = (e.get("tool"), json.dumps(e.get("args"), sort_keys=True))
            if key == prev: red += 1
            prev = key
    rar = _rate(red, len(tev))
    # workflow_completion_rate (strict workflow_compliance subdim; HAB)
    wcp = [c for r in res for c in r.get("checkpoints", [])
           if _strict(c) and _eval(c) and c.get("subdimension") == "workflow_compliance"]
    wcr = _rate(sum(1 for c in wcp if c["checkpoint_status"] == "passed"), len(wcp)) if wcp else None
    # functional_tool_use (MedCTA: agent used a required core tool)
    import tool_requirements as _TR
    ftu_num = ftu_den = rtc_num = rtc_den = 0
    for it in items:
        used = {e.get("tool") for e in _tool_events(it["traj"])}
        fu = _TR.functional_used(it["task"], used)
        if fu is not None: ftu_den += 1; ftu_num += int(fu)
        rc = _TR.required_complete(it["task"], used)
        if rc is not None: rtc_den += 1; rtc_num += int(rc)
    ftu = _rate(ftu_num, ftu_den) if ftu_den else None
    rtc = _rate(rtc_num, rtc_den) if rtc_den else None
    gsc = [c.get("score") for r in res for c in r.get("checkpoints", []) if isinstance(c.get("score"), (int, float))]
    gacc = (sum(gsc) / len(gsc)) if gsc else None
    return {"task_success_rate": (tsr, "%d/%d" % (succ, len(res))),
            "subtask_success_rate": (ssr, "%d strict cp" % len(scp)),
            "tool_call_success_rate": (tcs, "%d actions" % len(tev)),
            "argument_validity": (av, "%d arg-calls" % len(argc)),
            "redundant_action_rate": (rar, "%d/%d" % (red, len(tev))),
            "workflow_completion_rate": (wcr, "%d cp" % len(wcp)),
            "functional_tool_use": (ftu, "%d/%d tasks" % (ftu_num, ftu_den)),
            "required_tool_completion": (rtc, "%d/%d tasks" % (rtc_num, rtc_den)),
            "gacc_mean": (gacc, "%d cp, 0-1 semantic" % len(gsc))}

# ---------- Safety (action-level, via annotator) ----------
def safety(items):
    anns = [ra.annotate(it["task"], it["traj"], fhir_base=os.environ.get("MH_FHIR_BASE")) for it in items]
    return {m["metric"].split(".")[-1]: m for m in sm.all_safety_metrics(anns)}

# ---------- Meta ----------
def meta(items):
    res = [it["res"] for it in items]
    allcp = [c for r in res for c in r.get("checkpoints", [])]
    cov = [c for c in allcp if _strict(c) and _eval(c)]   # strict & actually executed
    vc = _rate(len(cov), len(allcp))
    # qualification integrity: field present + proxy runs flagged
    viol = 0
    for r in res:
        q = r.get("qualification")
        if q is None: viol += 1; continue
        has_proxy_cp = any(c.get("score_eligible") is False for c in r.get("checkpoints", []))
        if has_proxy_cp and "proxy_scored_checkpoints" not in q: viol += 1
    qi = _rate(len(res) - viol, len(res))
    return {"verifier_coverage": (vc, "%d/%d cp strict-exec" % (len(cov), len(allcp))),
            "qualification_integrity": (qi, "%d/%d runs ok" % (len(res) - viol, len(res)))}

def _fmt(v):
    val, note = v
    return ("%.2f" % val if isinstance(val, float) else ("n/a" if val is None else str(val))) + " (%s)" % note

def main(root):
    data = load_bundles(root)
    if not data:
        print("no bundles under", root); return
    for bench in sorted(data):
        items = data[bench]
        print("\n" + "=" * 70 + "\n%s  (%d tasks)\n" % (bench, len(items)) + "=" * 70)
        eff = efficiency(items); saf = safety(items); mt = meta(items)
        print("-- Efficiency --")
        for k in ("task_success_rate", "subtask_success_rate", "gacc_mean", "functional_tool_use", "required_tool_completion",
                  "tool_call_success_rate", "argument_validity", "workflow_completion_rate", "redundant_action_rate"):
            print("  %-26s %s" % (k, _fmt(eff[k])))
        print("-- Safety (action-level) --")
        for k in ("unsafe_action_rate", "required_check_completion", "patient_scope_correctness"):
            m = saf[k]; v = m.get("value")
            extra = {kk: m[kk] for kk in m if kk not in ("metric", "value")}
            print("  %-26s %s  %s" % (k, "%.2f" % v if isinstance(v, float) else "n/a", json.dumps(extra, ensure_ascii=False)))
        print("-- Integrity/Meta --")
        for k in ("verifier_coverage", "qualification_integrity"):
            print("  %-26s %s" % (k, _fmt(mt[k])))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../results_v0")
