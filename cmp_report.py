"""Formal harness-vs-no-harness comparison — FAIL-CLOSED. Reads PER-TASK result.json, filters to ELIGIBLE
runs (evaluated + schema valid is True + harness active/no-errors/req==eff for harness modes), and declares
a bundle VALID only when: EVERY declared mode is present, all present modes share ONE identical eligible
task set, the git_sha is single across all tasks, and agent/tool/judge identities are single-valued. An
INVALID bundle still prints its numbers but with a loud INVALID banner — never silently comparable. Rates
are macro (eligible mean) AND pooled (sum int numerators / sum denominators). Usage: cmp_report.py [prefix]."""
import json, glob, os, sys

PREFIX = sys.argv[1] if len(sys.argv) > 1 else "res6"
MODES_BY_DS = {"PhysicianBench": ["off", "enforce"],
               "MedCTA": ["off", "observe", "assist", "enforce"],
               "HealthAdminBench": ["off", "enforce"]}
STEM = {"PhysicianBench": "pb", "MedCTA": "mcta", "HealthAdminBench": "hab"}
MODS = ["Execution", "Tooling", "Context", "Lifecycle", "Observability", "Verification", "Governance"]
POOL = {"wrong_scope_action_rate": ("wrong_scope_count", "wrong_scope_opportunities"),
        "missing_prerequisite_rate": ("missing_prerequisite_count", "missing_prerequisite_opportunities"),
        "verified_commit_rate": ("verified_commit_count", "n_commits"),
        "unknown_verification_rate": ("unknown_verification_count", "n_commits"),
        "executed_violation_rate": ("executed_violation_count", "n_proposed_actions"),
        "unsafe_commitment_rate": ("unsafe_commitment_count", "n_commit_proposals"),
        "repair_success_rate": ("repair_success_count", "repair_opportunities")}


def eligible(d, mode):
    if d.get("evaluation_status") not in ("complete", "completed", "evaluated"):
        return False, "not_evaluated"
    sv = d.get("schema_validation")
    if isinstance(sv, dict) and sv.get("valid") is not True:   # fail-closed: present schema must be VALID
        return False, "schema_invalid"
    if mode != "off":
        h = d.get("harness") or {}
        if h.get("status") != "active":
            return False, "harness_%s" % (h.get("status") or "missing")
        if h.get("runtime_errors"):
            return False, "runtime_error"
        if h.get("requested_mode") and h.get("effective_mode") and h["requested_mode"] != h["effective_mode"]:
            return False, "mode_mismatch"
    return True, None


def load(ds, mode):
    base = "%s_%s_%s/gpt5" % (PREFIX, STEM[ds], mode)
    out = {"n": 0, "tids": set(), "excl": {}, "pnum": {}, "pden": {}, "macro": {}, "mcnt": {},
           "cp_pass": 0, "cp_tot": 0, "gacc": [], "pass1": [], "delivered": 0,
           "agents": set(), "tools": set(), "judges": set(), "shas": set()}
    for f in glob.glob(base + "/*/result.json"):
        d = json.load(open(f))
        tid = d.get("task_id") or os.path.basename(os.path.dirname(f))
        ok, why = eligible(d, mode)
        if not ok:
            out["excl"][why] = out["excl"].get(why, 0) + 1
            continue
        out["n"] += 1; out["tids"].add(tid)
        prov = d.get("provenance") or {}
        if prov.get("agent_model"): out["agents"].add(prov["agent_model"])
        if prov.get("tool_backend_model"): out["tools"].add(prov["tool_backend_model"])
        if prov.get("git_sha"): out["shas"].add(prov["git_sha"])
        h = d.get("harness") or {}
        jm = (h.get("audit") or {}).get("runtime_judge_model")
        if jm: out["judges"].add(jm)
        m = h.get("metrics") or {}
        for k in (set(POOL) | {"escalation_rate"}):
            v = m.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out["macro"][k] = out["macro"].get(k, 0) + v; out["mcnt"][k] = out["mcnt"].get(k, 0) + 1
        for k, (nf, df) in POOL.items():
            num, den = m.get(nf), m.get(df)
            if isinstance(num, int) and isinstance(den, int) and den > 0:
                out["pnum"][k] = out["pnum"].get(k, 0) + num; out["pden"][k] = out["pden"].get(k, 0) + den
        out["delivered"] += int(m.get("answer_delivered_count") or m.get("answer_delivered") or 0)
        nm = d.get("native_metrics") or {}
        if isinstance(nm.get("checkpoint_passed"), int): out["cp_pass"] += nm["checkpoint_passed"]
        if isinstance(nm.get("checkpoint_scored"), int): out["cp_tot"] += nm["checkpoint_scored"]
        if isinstance(nm.get("gacc_mean"), (int, float)): out["gacc"].append(nm["gacc_mean"])
        no = d.get("native_outcome") or {}
        if isinstance(no.get("pass_at_1"), (int, float)): out["pass1"].append(no["pass_at_1"])
    return out


def report_native(ds, mode):
    p = "%s_%s_%s/gpt5/report.json" % (PREFIX, STEM[ds], mode)
    if not os.path.exists(p): return None
    r = json.load(open(p)); nm = r.get("native_metrics") or {}; o = (r.get("outcome") or {}).get("score")
    return {"outcome": o, "checkpoint": nm.get("checkpoint_pass_rate"), "gacc": nm.get("gacc_mean"),
            "pass1": nm.get("pass_at_1"), "dims": r.get("harness_dimensions") or {}}


def f(v):
    return "  -  " if v is None else ("%.3f" % v if isinstance(v, float) else str(v))


print("FORMAL COMPARISON  prefix=%s" % PREFIX)
for ds, modes in MODES_BY_DS.items():
    agg = {m: load(ds, m) for m in modes}
    rep = {m: report_native(ds, m) for m in modes}
    present = [m for m in modes if agg[m]["n"]]
    missing_modes = [m for m in modes if not agg[m]["n"]]
    sets = {m: agg[m]["tids"] for m in present}
    same_set = (len({frozenset(s) for s in sets.values()}) <= 1) if sets else False
    shas = set().union(*[agg[m]["shas"] for m in present]) if present else set()
    agents = set().union(*[agg[m]["agents"] for m in present]) if present else set()
    tools = set().union(*[agg[m]["tools"] for m in present]) if present else set()
    judges = set().union(*[agg[m]["judges"] for m in present]) if present else set()
    reasons = []
    if missing_modes: reasons.append("MISSING_MODES:%s" % ",".join(missing_modes))
    if not same_set: reasons.append("UNPAIRED_TASK_SETS")
    if len(shas) > 1: reasons.append("MIXED_SHA:%s" % "|".join(sorted(shas)))
    if len(agents) > 1: reasons.append("MIXED_AGENT")
    if len(tools) > 1: reasons.append("MIXED_TOOL")
    if len(judges) > 1: reasons.append("MIXED_JUDGE")
    valid = not reasons
    print("=" * 96)
    print("%-16s agent=%s tool=%s judge=%s sha=%s"
          % (ds, "|".join(sorted(agents)) or "?", "|".join(sorted(tools)) or "none",
             "|".join(sorted(judges)) or "none", "|".join(sorted(shas)) or "n/a"))
    print("eligible n: " + " ".join("%s=%d" % (m, agg[m]["n"]) for m in modes)
          + "   BUNDLE=%s" % ("VALID" if valid else ("INVALID [" + " ".join(reasons) + "]")))
    for m in modes:
        if agg[m]["excl"]: print("  excluded[%s]: %s" % (m, agg[m]["excl"]))
    if not valid:
        print("  *** INVALID BUNDLE — numbers below are NOT a valid comparison (fail-closed). ***")
    print("  NATIVE (per-task pooled over eligible | report.json fallback):")
    for m in modes:
        a = agg[m]; rj = rep[m] or {}
        cp = (a["cp_pass"] / a["cp_tot"]) if a["cp_tot"] else rj.get("checkpoint")
        gm = (sum(a["gacc"]) / len(a["gacc"])) if a["gacc"] else rj.get("gacc")
        p1 = (sum(a["pass1"]) / len(a["pass1"])) if a["pass1"] else rj.get("pass1")
        print("    %-8s outcome=%s checkpoint=%s gacc=%s pass@1=%s answer_delivered=%d/%d"
              % (m, f(rj.get("outcome")), f(cp), f(gm), f(p1), a["delivered"], a["n"]))
    print("  7 DIMENSIONS (report.json — NOT eligibility-filtered):")
    for dim in MODS:
        print("    %-12s %s" % (dim, " ".join("%s=%s" % (m, f(((rep[m] or {}).get("dims", {}).get(dim) or {}).get("score")))
                                              for m in modes)))
    print("  HARNESS  macro|pooled:")
    for k in list(POOL) + ["escalation_rate"]:
        cells = []
        for m in modes:
            a = agg[m]
            mac = round(a["macro"][k] / a["mcnt"][k], 3) if a["mcnt"].get(k) else None
            poo = round(a["pnum"][k] / a["pden"][k], 3) if a["pden"].get(k) else None
            cells.append("%s=%s|%s" % (m, f(mac), f(poo)))
        print("    %-24s %s" % (k, " ".join(cells)))
print("=" * 96)
print("VALID requires: all declared modes present + one identical eligible task set + single git_sha +")
print("single agent/tool/judge identity. 7 dims are from report.json (not eligibility-filtered) — noted.")
