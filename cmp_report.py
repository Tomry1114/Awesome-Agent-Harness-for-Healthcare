"""Formal harness comparison. OUTCOME = the dataset-NATIVE task outcome via the repo's SINGLE SOURCE OF
TRUTH aggregate_report.native_task_outcome (PB/HAB: Outcome-dimension checkpoints; MedCTA: GAcc>=0.5) --
NEVER result['success'] (that is the strict-checkpoint GATE over all ETCLOVG dims, reported separately as
strict_gate). Governance comes from the canonical post-hoc rescored block (result.rescored.json); the other
6 dims from per-task dimension_scores. Everything is per-task, eligibility-filtered and paired; per-dim
coverage [n/N] is printed so denominators are explicit (PB does not score all 7 dims per task). The headline
is the paired off->enforce NATIVE-outcome transition (preserved/harmed/recovered/unchanged). --formal: an
INVALID bundle prints diagnostics and exits 2 with no numbers. Usage: cmp_report.py [prefix] [--formal]"""
import json, glob, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "runner"))
from aggregate_report import native_task_outcome, _read_dim_block

ARGS = sys.argv[1:]
FORMAL = "--formal" in ARGS
PREFIX = next((a for a in ARGS if not a.startswith("--")), "res6")
MODES_BY_DS = {"PhysicianBench": ["off", "enforce"], "MedCTA": ["off", "observe", "assist", "enforce"],
               "HealthAdminBench": ["off", "enforce"]}
STEM = {"PhysicianBench": "pb", "MedCTA": "mcta", "HealthAdminBench": "hab"}
MODS = ["Execution", "Tooling", "Context", "Lifecycle", "Observability", "Verification", "Governance"]
TOOL_SUBSTRATE = {"MedCTA"}
SEMANTIC_JUDGE = {"MedCTA", "HealthAdminBench"}
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
    if not isinstance(sv, dict) or sv.get("valid") is not True:
        return False, "schema_not_proven_valid"
    h = d.get("harness") or {}
    if mode == "off":
        if h and (h.get("requested_mode") not in (None, "off") or h.get("effective_mode") not in (None, "off")):
            return False, "off_dir_ran_harness"
    else:
        if h.get("status") != "active":
            return False, "harness_%s" % (h.get("status") or "missing")
        if h.get("runtime_errors"):
            return False, "runtime_error"
        if h.get("requested_mode") != mode or h.get("effective_mode") != mode:
            return False, "mode_mismatch"
    return True, None


def delivered(bundle, tid):
    tp = os.path.join(bundle, tid, "trajectory.jsonl")
    if not os.path.exists(tp):
        return None
    try:
        return 1 if any(json.loads(l).get("event_type") == "final_answer" for l in open(tp)) else 0
    except Exception:
        return None


def load(ds, mode):
    base = "%s_%s_%s/gpt5" % (PREFIX, STEM[ds], mode)
    o = {"n": 0, "tasks": {}, "excl": {}, "dups": set(), "agents": set(), "tools": set(), "judges": set(),
         "shas": set(), "miss_agent": 0, "miss_sha": 0, "miss_tool": 0, "miss_judge": 0}
    seen = set()
    for f in sorted(glob.glob(base + "/*/result.json")):
        d = json.load(open(f)); tdir = os.path.dirname(f)
        tid = d.get("task_id") or os.path.basename(tdir)
        if tid in seen:
            o["dups"].add(tid); continue
        seen.add(tid)
        ok, why = eligible(d, mode)
        if not ok:
            o["excl"][why] = o["excl"].get(why, 0) + 1; continue
        dlv = delivered(base, tid)
        if dlv is None:
            o["excl"]["trajectory_missing"] = o["excl"].get("trajectory_missing", 0) + 1; continue
        nat = native_task_outcome(d, ds)               # True / False / None (canonical, NOT success)
        # 6 dims from per-task dimension_scores; Governance from the canonical rescored block.
        raw = d.get("dimension_scores") or {}
        dims = {}
        for dim in MODS:
            if dim == "Governance":
                blk, _err = _read_dim_block(tdir, "Governance")
                v = (blk or {}).get("score") if (blk and blk.get("reportable") is not False) else None
            else:
                v = raw.get(dim)
                if isinstance(v, dict): v = v.get("score")
            dims[dim] = v if isinstance(v, (int, float)) else None
        prov = d.get("provenance") or {}; h = d.get("harness") or {}
        o["tasks"][tid] = {"native": nat, "strict_gate": 1 if d.get("success") else 0,
                           "dims": dims, "hm": h.get("metrics") or {}, "delivered": dlv}
        o["n"] += 1
        o["agents"].add(prov["agent_model"]) if prov.get("agent_model") else o.__setitem__("miss_agent", o["miss_agent"] + 1)
        o["shas"].add(prov["git_sha"]) if prov.get("git_sha") else o.__setitem__("miss_sha", o["miss_sha"] + 1)
        if prov.get("tool_backend_model"): o["tools"].add(prov["tool_backend_model"])
        elif ds in TOOL_SUBSTRATE: o["miss_tool"] += 1
        jm = (h.get("audit") or {}).get("runtime_judge_model")
        if jm: o["judges"].add(jm)
        elif mode in ("assist", "enforce") and ds in SEMANTIC_JUDGE: o["miss_judge"] += 1
    return o


def f(v):
    return "  -  " if v is None else ("%.3f" % v if isinstance(v, float) else str(v))


def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and not isinstance(x, bool)]
    return (sum(xs) / len(xs)) if xs else None


_exit = 0
print("FORMAL COMPARISON  prefix=%s  mode=%s" % (PREFIX, "FORMAL" if FORMAL else "descriptive"))
for ds, modes in MODES_BY_DS.items():
    agg = {m: load(ds, m) for m in modes}
    present = [m for m in modes if agg[m]["n"]]
    sets = {m: set(agg[m]["tasks"]) for m in present}
    same = (len({frozenset(s) for s in sets.values()}) <= 1) if sets else False
    U = lambda key: set().union(*[agg[m][key] for m in present]) if present else set()
    shas, agents, tools, judges = U("shas"), U("agents"), U("tools"), U("judges")
    r = []
    if [m for m in modes if not agg[m]["n"]]: r.append("MISSING_MODES:%s" % ",".join(m for m in modes if not agg[m]["n"]))
    if not same: r.append("UNPAIRED_SETS")
    if any(agg[m]["dups"] for m in modes): r.append("DUPLICATE_TASK_ID")
    if len(shas) != 1 or any(agg[m]["miss_sha"] for m in present): r.append("SHA_MISSING_OR_MIXED")
    if len(agents) != 1 or any(agg[m]["miss_agent"] for m in present): r.append("AGENT_MISSING_OR_MIXED")
    if len(tools) > 1 or (ds in TOOL_SUBSTRATE and any(agg[m]["miss_tool"] for m in present)): r.append("TOOL_MISSING_OR_MIXED")
    if len(judges) > 1 or any(agg[m]["miss_judge"] for m in present): r.append("JUDGE_MISSING_OR_MIXED")
    valid = not r
    print("=" * 100)
    print("%-16s agent=%s tool=%s judge=%s sha=%s" % (ds, "|".join(sorted(agents)) or "?",
          "|".join(sorted(tools)) or "none", "|".join(sorted(judges)) or "none", "|".join(sorted(shas)) or "MISSING"))
    print("eligible n: " + " ".join("%s=%d" % (m, agg[m]["n"]) for m in modes)
          + "   BUNDLE=" + ("VALID" if valid else "INVALID [" + " ".join(r) + "]"))
    for m in modes:
        if agg[m]["excl"]: print("  excluded[%s]: %s" % (m, agg[m]["excl"]))
        if agg[m]["dups"]: print("  DUPLICATE task_ids[%s]: %s" % (m, sorted(agg[m]["dups"])))
    if not valid and FORMAL:
        print("  *** INVALID — formal: no comparison numbers. ***"); _exit = 2; continue
    if not valid:
        print("  *** INVALID BUNDLE — DESCRIPTIVE ONLY. ***")
    print("  NATIVE OUTCOME (canonical, native-resolved subset) | strict_gate(success) | answer_delivered:")
    for m in modes:
        T = agg[m]["tasks"]
        if not T:
            print("    %-9s   -" % m); continue
        nat = [t["native"] for t in T.values() if t["native"] is not None]
        natr = (sum(1 for x in nat if x) / len(nat)) if nat else None
        sg = mean([t["strict_gate"] for t in T.values()])
        dl = sum(1 for t in T.values() if t["delivered"] == 1)
        print("    %-9s native=%s [n=%d/%d] strict_gate=%s deliv=%d/%d"
              % (m, f(natr), len(nat), len(T), f(sg), dl, len(T)))
    print("  7 DIMENSIONS  per-task mean [n scored / N]  (Governance from rescored; N/A if not_rescored):")
    for dim in MODS:
        cells = []
        for m in modes:
            T = agg[m]["tasks"]; vals = [t["dims"].get(dim) for t in T.values() if isinstance(t["dims"].get(dim), (int, float))]
            mv = mean(vals)
            cells.append("%s=%s[%d/%d]" % (m, f(round(mv, 3) if mv is not None else None), len(vals), len(T)))
        print("    %-12s %s" % (dim, " ".join(cells)))
    if same and agg["off"]["n"] and agg["enforce"]["n"]:
        common = set(agg["off"]["tasks"]) & set(agg["enforce"]["tasks"])
        common = [t for t in common if agg["off"]["tasks"][t]["native"] is not None and agg["enforce"]["tasks"][t]["native"] is not None]
        pres = harmed = recov = unch = 0
        for t in common:
            o0 = bool(agg["off"]["tasks"][t]["native"]); o1 = bool(agg["enforce"]["tasks"][t]["native"])
            pres += o0 and o1; harmed += o0 and not o1; recov += (not o0) and o1; unch += (not o0) and not o1
        op = pres / (pres + harmed) if (pres + harmed) else None
        rr = recov / (recov + unch) if (recov + unch) else None
        print("  PAIRED off->enforce NATIVE-OUTCOME TRANSITION (native-resolved n=%d): preserved=%d harmed=%d recovered=%d unchanged=%d"
              % (len(common), pres, harmed, recov, unch))
        print("    outcome_preservation=%s   recovery_rate=%s" % (f(op), f(rr)))
    print("  HARNESS rates  macro|pooled:")
    for k in POOL:
        cells = []
        for m in modes:
            T = agg[m]["tasks"]
            mac = mean([t["hm"].get(k) for t in T.values()])
            nf, df = POOL[k]
            num = sum(t["hm"][nf] for t in T.values() if isinstance(t["hm"].get(nf), int))
            den = sum(t["hm"][df] for t in T.values() if isinstance(t["hm"].get(df), int))
            poo = (num / den) if den else None
            cells.append("%s=%s|%s" % (m, f(round(mac, 3) if mac is not None else None), f(round(poo, 3) if poo is not None else None)))
        print("    %-24s %s" % (k, " ".join(cells)))
print("=" * 100)
print("OUTCOME = aggregate_report.native_task_outcome (NOT result['success']). strict_gate = the all-")
print("checkpoint gate, shown separately. Dims show [n scored/N]; Governance from result.rescored.json.")
sys.exit(_exit)
