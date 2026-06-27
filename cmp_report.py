"""Formal harness-vs-no-harness comparison. Reads PER-TASK result.json (never just the aggregate), filters
to ELIGIBLE runs, enforces an IDENTICAL paired task set across modes, reports each rate BOTH macro (eligible
per-task mean) and POOLED (sum integer numerators / sum denominators, never mean-of-rounded-rates), and
prints 0/1 metrics as count AND rate with the denominator. A non-paired / degraded / mixed-SHA bundle is
FLAGGED, not silently averaged. Usage: python cmp_report.py [prefix]   (prefix default res6)."""
import json, glob, os, sys

PREFIX = sys.argv[1] if len(sys.argv) > 1 else "res6"
MODES_BY_DS = {"PhysicianBench": ["off", "enforce"],
               "MedCTA": ["off", "observe", "assist", "enforce"],
               "HealthAdminBench": ["off", "enforce"]}
STEM = {"PhysicianBench": "pb", "MedCTA": "mcta", "HealthAdminBench": "hab"}
MODS = ["Execution", "Tooling", "Context", "Lifecycle", "Observability", "Verification", "Governance"]
# pooled rate -> (integer numerator field, denominator field) in harness.metrics
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
    if isinstance(sv, dict) and sv.get("valid") is False:
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
    """Return per-mode aggregate over ELIGIBLE tasks: task_ids, pooled num/den, macro sums, native, provenance."""
    base = "%s_%s_%s/gpt5" % (PREFIX, STEM[ds], mode)
    out = {"n": 0, "tids": set(), "excl": {}, "pnum": {}, "pden": {}, "macro": {}, "mcnt": {},
           "cp_pass": 0, "cp_tot": 0, "gacc": [], "pass1": [], "delivered": 0, "agent": None, "judge": None,
           "sha": set(), "tool": None}
    for f in glob.glob(base + "/*/result.json"):
        d = json.load(open(f))
        tid = d.get("task_id") or os.path.basename(os.path.dirname(f))
        ok, why = eligible(d, mode)
        if not ok:
            out["excl"][why] = out["excl"].get(why, 0) + 1
            continue
        out["n"] += 1; out["tids"].add(tid)
        prov = d.get("provenance") or {}
        out["agent"] = out["agent"] or prov.get("agent_model") or d.get("agent_model")
        out["tool"] = out["tool"] or prov.get("tool_model") or prov.get("tool_backend_model")
        if prov.get("git_sha"): out["sha"].add(prov.get("git_sha"))
        h = d.get("harness") or {}
        out["judge"] = out["judge"] or (h.get("audit") or {}).get("runtime_judge_model")
        m = h.get("metrics") or {}
        for k in (set(POOL) | {"escalation_rate", "n_interventions", "n_findings", "n_commits"}):
            v = m.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out["macro"][k] = out["macro"].get(k, 0) + v; out["mcnt"][k] = out["mcnt"].get(k, 0) + 1
        for k, (nf, df) in POOL.items():
            num, den = m.get(nf), m.get(df)
            if isinstance(num, int) and isinstance(den, int) and den > 0:
                out["pnum"][k] = out["pnum"].get(k, 0) + num; out["pden"][k] = out["pden"].get(k, 0) + den
        out["delivered"] += int(m.get("answer_delivered_count") or m.get("answer_delivered") or 0)
        # native (per task)
        nm = d.get("native_metrics") or {}
        if isinstance(nm.get("checkpoint_passed"), int): out["cp_pass"] += nm["checkpoint_passed"]
        if isinstance(nm.get("checkpoint_scored"), int): out["cp_tot"] += nm["checkpoint_scored"]
        g = nm.get("gacc_mean")
        if isinstance(g, (int, float)): out["gacc"].append(g)
        no = (d.get("native_outcome") or {})
        p1 = no.get("pass_at_1") if isinstance(no, dict) else None
        if isinstance(p1, (int, float)): out["pass1"].append(p1)
    return out


def report_json(ds, mode):
    p = "%s_%s_%s/gpt5/report.json" % (PREFIX, STEM[ds], mode)
    return json.load(open(p)) if os.path.exists(p) else None


def f(v):
    return "  -  " if v is None else ("%.3f" % v if isinstance(v, float) else str(v))


print("FORMAL COMPARISON  prefix=%s" % PREFIX)
for ds, modes in MODES_BY_DS.items():
    agg = {m: load(ds, m) for m in modes}
    present = [m for m in modes if agg[m]["n"]]
    sets = {m: agg[m]["tids"] for m in present}
    paired = (len({frozenset(s) for s in sets.values()}) <= 1) if sets else False
    shas = set().union(*[agg[m]["sha"] for m in present]) if present else set()
    print("=" * 92)
    a0 = next((agg[m] for m in present), {})
    print("%-16s agent=%s tool=%s judge=%s  sha=%s" % (
        ds, a0.get("agent"), a0.get("tool"), a0.get("judge"),
        ("|".join(sorted(shas)) or "n/a") + ("" if len(shas) <= 1 else "  MIXED-SHA!")))
    print("eligible n: " + " ".join("%s=%d" % (m, agg[m]["n"]) for m in modes)
          + "   PAIRED=%s" % ("yes" if paired else "NO -> INVALID_BUNDLE"))
    for m in modes:
        if agg[m]["excl"]: print("  excluded[%s]: %s" % (m, agg[m]["excl"]))
    if not paired:
        print("  WARNING: modes do not share one identical eligible task set — comparison NOT valid.")
    # native (pooled)
    print("  %-26s %s" % ("NATIVE (pooled over eligible):", ""))
    for m in modes:
        a = agg[m]
        cp = (a["cp_pass"] / a["cp_tot"]) if a["cp_tot"] else None
        gm = (sum(a["gacc"]) / len(a["gacc"])) if a["gacc"] else None
        p1 = (sum(a["pass1"]) / len(a["pass1"])) if a["pass1"] else None
        print("    %-8s checkpoint=%s gacc=%s pass@1=%s answer_delivered=%d/%d"
              % (m, f(cp), f(gm), f(p1), a["delivered"], a["n"]))
    # dimensions (from report.json; flagged when not fully paired/eligible)
    print("  %-26s" % "7 DIMENSIONS (report.json):")
    reps = {m: report_json(ds, m) for m in modes}
    for dim in MODS:
        cells = " ".join("%s=%s" % (m, f(((((reps[m] or {}).get("harness_dimensions") or {}).get(dim) or {}).get("score"))))
                         for m in modes)
        print("    %-12s %s" % (dim, cells))
    # harness metrics: macro | pooled
    print("  %-26s" % "HARNESS metrics  macro | pooled:")
    for k in list(POOL) + ["escalation_rate"]:
        cells = []
        for m in modes:
            a = agg[m]
            mac = (a["macro"][k] / a["mcnt"][k]) if a["mcnt"].get(k) else None
            poo = (a["pnum"][k] / a["pden"][k]) if a["pden"].get(k) else None
            cells.append("%s=%s|%s" % (m, f(round(mac, 3) if mac is not None else None),
                                       f(round(poo, 3) if poo is not None else None)))
        print("    %-24s %s" % (k, " ".join(cells)))
print("=" * 92)
print("Reading: pooled = sum(count)/sum(opportunity) over ELIGIBLE paired tasks. unsafe_commitment_rate is")
print("violations on COMMIT actions only. A NON-paired / MIXED-SHA / degraded bundle is flagged, not averaged.")
