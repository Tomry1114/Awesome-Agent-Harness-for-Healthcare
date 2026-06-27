"""Formal harness comparison — everything from PER-TASK result.json, eligibility-filtered and paired.

A bundle is VALID only when, for a dataset: EVERY declared mode is present; all present modes share ONE
identical eligible task set with NO duplicate task_id; git_sha is single-valued and present on every task;
agent identity is single-valued and present; the tool backend is single-valued where the substrate uses a
model tool; the runtime judge is single-valued where the mode runs a semantic judge (and absent for off);
and each task's harness.requested_mode/effective_mode equal the directory mode. Outcome (per-task `success`),
the 7 dimensions (per-task `dimension_scores`), answer-delivery (trajectory final_answer), and the harness
rates are ALL computed over the SAME eligible paired tasks -- no aggregate-report fallback, no mixed
denominators. A paired off-vs-enforce OUTCOME TRANSITION (preserved / harmed / recovered / unchanged) is
the headline. `--formal`: an INVALID bundle prints diagnostics and exits 2 with NO comparison numbers.
Usage: cmp_report.py [prefix] [--formal]
"""
import json, glob, os, sys, hashlib

ARGS = [a for a in sys.argv[1:]]
FORMAL = "--formal" in ARGS
PREFIX = next((a for a in ARGS if not a.startswith("--")), "res6")
MODES_BY_DS = {"PhysicianBench": ["off", "enforce"],
               "MedCTA": ["off", "observe", "assist", "enforce"],
               "HealthAdminBench": ["off", "enforce"]}
STEM = {"PhysicianBench": "pb", "MedCTA": "mcta", "HealthAdminBench": "hab"}
MODS = ["Execution", "Tooling", "Context", "Lifecycle", "Observability", "Verification", "Governance"]
TOOL_SUBSTRATE = {"MedCTA"}                 # uses a model perception backend -> tool identity REQUIRED
SEMANTIC_JUDGE = {"MedCTA", "HealthAdminBench"}   # semantic grounding judge in assist/enforce
POOL = {"wrong_scope_action_rate": ("wrong_scope_count", "wrong_scope_opportunities"),
        "missing_prerequisite_rate": ("missing_prerequisite_count", "missing_prerequisite_opportunities"),
        "verified_commit_rate": ("verified_commit_count", "n_commits"),
        "unknown_verification_rate": ("unknown_verification_count", "n_commits"),
        "executed_violation_rate": ("executed_violation_count", "n_proposed_actions"),
        "unsafe_commitment_rate": ("unsafe_commitment_count", "n_commit_proposals"),
        "repair_success_rate": ("repair_success_count", "repair_opportunities"),
        "escalation_rate": ("escalation_count", "n_proposed_actions")}


def eligible(d, mode):
    if d.get("evaluation_status") not in ("complete", "completed", "evaluated"):
        return False, "not_evaluated"
    sv = d.get("schema_validation")
    if not isinstance(sv, dict) or sv.get("valid") is not True:   # FAIL-CLOSED: must be PROVEN valid
        return False, "schema_not_proven_valid"
    h = d.get("harness") or {}
    if mode == "off":
        if h and h.get("effective_mode") not in (None, "off"):
            return False, "off_dir_ran_harness"
    else:
        if h.get("status") != "active":
            return False, "harness_%s" % (h.get("status") or "missing")
        if h.get("runtime_errors"):
            return False, "runtime_error"
        if h.get("requested_mode") != mode or h.get("effective_mode") != mode:
            return False, "mode_mismatch(%s/%s!=%s)" % (h.get("requested_mode"), h.get("effective_mode"), mode)
    return True, None


def _delivered(stem_dir, tid):
    tp = os.path.join(stem_dir, tid, "trajectory.jsonl")
    if not os.path.exists(tp):
        return None
    try:
        return 1 if any(json.loads(l).get("event_type") == "final_answer" for l in open(tp)) else 0
    except Exception:
        return None


def load(ds, mode):
    base = "%s_%s_%s/gpt5" % (PREFIX, STEM[ds], mode)
    o = {"n": 0, "tasks": {}, "excl": {}, "dups": set(), "agents": set(), "tools": set(),
         "judges": set(), "shas": set(), "miss_agent": 0, "miss_sha": 0, "miss_tool": 0, "miss_judge": 0}
    for f in glob.glob(base + "/*/result.json"):
        d = json.load(open(f))
        tid = d.get("task_id") or os.path.basename(os.path.dirname(f))
        ok, why = eligible(d, mode)
        if not ok:
            o["excl"][why] = o["excl"].get(why, 0) + 1
            continue
        if tid in o["tasks"]:
            o["dups"].add(tid); continue
        prov = d.get("provenance") or {}
        h = d.get("harness") or {}
        hm = h.get("metrics") or {}
        ds_scores = d.get("dimension_scores") or {}
        dims = {}
        for dim in MODS:
            v = ds_scores.get(dim)
            if isinstance(v, dict): v = v.get("score")
            if isinstance(v, (int, float)): dims[dim] = v
        o["tasks"][tid] = {"success": 1 if d.get("success") else 0, "dims": dims, "hm": hm,
                           "delivered": _delivered(base, tid)}
        o["n"] += 1
        (o["agents"].add(prov["agent_model"]) if prov.get("agent_model") else o.__setitem__("miss_agent", o["miss_agent"] + 1))
        (o["shas"].add(prov["git_sha"]) if prov.get("git_sha") else o.__setitem__("miss_sha", o["miss_sha"] + 1))
        if prov.get("tool_backend_model"): o["tools"].add(prov["tool_backend_model"])
        elif ds in TOOL_SUBSTRATE: o["miss_tool"] += 1
        jm = (h.get("audit") or {}).get("runtime_judge_model")
        if jm: o["judges"].add(jm)
        elif mode in ("assist", "enforce") and ds in SEMANTIC_JUDGE: o["miss_judge"] += 1
    return o


def f(v):
    return "  -  " if v is None else ("%.3f" % v if isinstance(v, float) else str(v))


def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return (sum(xs) / len(xs)) if xs else None


_exit = 0
print("FORMAL COMPARISON  prefix=%s  mode=%s" % (PREFIX, "FORMAL" if FORMAL else "descriptive"))
for ds, modes in MODES_BY_DS.items():
    agg = {m: load(ds, m) for m in modes}
    present = [m for m in modes if agg[m]["n"]]
    sets = {m: set(agg[m]["tasks"]) for m in present}
    same = (len({frozenset(s) for s in sets.values()}) <= 1) if sets else False
    shas = set().union(*[agg[m]["shas"] for m in present]) if present else set()
    agents = set().union(*[agg[m]["agents"] for m in present]) if present else set()
    tools = set().union(*[agg[m]["tools"] for m in present]) if present else set()
    judges = set().union(*[agg[m]["judges"] for m in present]) if present else set()
    r = []
    miss = [m for m in modes if not agg[m]["n"]]
    if miss: r.append("MISSING_MODES:%s" % ",".join(miss))
    if not same: r.append("UNPAIRED_SETS")
    if any(agg[m]["dups"] for m in modes): r.append("DUPLICATE_TASK_ID")
    if len(shas) != 1 or any(agg[m]["miss_sha"] for m in present): r.append("SHA_MISSING_OR_MIXED")
    if len(agents) != 1 or any(agg[m]["miss_agent"] for m in present): r.append("AGENT_MISSING_OR_MIXED")
    if len(tools) > 1 or (ds in TOOL_SUBSTRATE and any(agg[m]["miss_tool"] for m in present)): r.append("TOOL_MISSING_OR_MIXED")
    if len(judges) > 1 or any(agg[m]["miss_judge"] for m in present): r.append("JUDGE_MISSING_OR_MIXED")
    valid = not r
    print("=" * 98)
    print("%-16s agent=%s tool=%s judge=%s sha=%s" % (ds, "|".join(sorted(agents)) or "?",
          "|".join(sorted(tools)) or "none", "|".join(sorted(judges)) or "none", "|".join(sorted(shas)) or "MISSING"))
    print("eligible n: " + " ".join("%s=%d" % (m, agg[m]["n"]) for m in modes)
          + "   BUNDLE=" + ("VALID" if valid else "INVALID [" + " ".join(r) + "]"))
    for m in modes:
        if agg[m]["excl"]: print("  excluded[%s]: %s" % (m, agg[m]["excl"]))
        if agg[m]["dups"]: print("  DUPLICATE task_ids[%s]: %s" % (m, sorted(agg[m]["dups"])))
    if not valid and FORMAL:
        print("  *** INVALID — formal mode: no comparison numbers emitted. ***")
        _exit = 2; continue
    if not valid:
        print("  *** INVALID BUNDLE — DESCRIPTIVE ONLY, not a valid comparison. ***")
    # everything below is over the SAME eligible tasks per mode
    print("  OUTCOME (per-task success rate) | 7 dims (per-task mean) | answer_delivered:")
    print("    %-9s %-8s | %s | deliv" % ("mode", "success", " ".join("%-5s" % d[:5] for d in MODS)))
    for m in modes:
        T = agg[m]["tasks"]
        if not T:
            print("    %-9s   -" % m); continue
        succ = mean([t["success"] for t in T.values()])
        dims = [mean([t["dims"].get(d) for t in T.values()]) for d in MODS]
        dl = sum(1 for t in T.values() if t["delivered"] == 1)
        print("    %-9s %-8s | %s | %d/%d" % (m, f(succ), " ".join(f(round(x, 2) if x is not None else None)[:5].ljust(5) for x in dims), dl, len(T)))
    # paired OUTCOME TRANSITION: off vs enforce on the SAME tasks (the headline harm/recovery analysis)
    if same and agg["off"]["n"] and agg["enforce"]["n"]:
        common = set(agg["off"]["tasks"]) & set(agg["enforce"]["tasks"])
        pres = harmed = recov = unch = 0
        for t in common:
            o0 = agg["off"]["tasks"][t]["success"]; o1 = agg["enforce"]["tasks"][t]["success"]
            if o0 and o1: pres += 1
            elif o0 and not o1: harmed += 1
            elif not o0 and o1: recov += 1
            else: unch += 1
        op = pres / (pres + harmed) if (pres + harmed) else None
        rr = recov / (recov + unch) if (recov + unch) else None
        print("  PAIRED off->enforce OUTCOME TRANSITION (n=%d): preserved=%d harmed=%d recovered=%d unchanged=%d"
              % (len(common), pres, harmed, recov, unch))
        print("    outcome_preservation=%s (preserved/(preserved+harmed))   recovery_rate=%s" % (f(op), f(rr)))
    print("  HARNESS rates  macro|pooled (over eligible):")
    for k in POOL:
        cells = []
        for m in modes:
            T = agg[m]["tasks"]
            mac = mean([t["hm"].get(k) for t in T.values() if isinstance(t["hm"].get(k), (int, float)) and not isinstance(t["hm"].get(k), bool)])
            nf, df = POOL[k]
            num = sum(t["hm"][nf] for t in T.values() if isinstance(t["hm"].get(nf), int))
            den = sum(t["hm"][df] for t in T.values() if isinstance(t["hm"].get(df), int))
            poo = (num / den) if den else None
            cells.append("%s=%s|%s" % (m, f(round(mac, 3) if mac is not None else None), f(round(poo, 3) if poo is not None else None)))
        print("    %-24s %s" % (k, " ".join(cells)))
print("=" * 98)
print("All columns use the SAME eligible paired tasks (outcome=success, dims=per-task dimension_scores).")
print("VALID requires every declared mode present + identical paired set + single present git_sha/agent +")
print("role-appropriate single tool/judge + no duplicate/mode-mismatch. --formal exits 2 on INVALID.")
sys.exit(_exit)
