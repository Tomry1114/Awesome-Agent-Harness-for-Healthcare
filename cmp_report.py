"""Formal harness comparison. OUTCOME = the dataset-NATIVE task outcome via the repo's SINGLE SOURCE OF
TRUTH aggregate_report.native_task_outcome (PB/HAB: Outcome-dimension checkpoints; MedCTA: GAcc>=0.5) --
NEVER result['success'] (that is the strict-checkpoint GATE over all ETCLOVG dims, reported separately as
strict_gate).

The 7 ETCLOVG dimensions are READ from the canonical aggregate pipeline (strict + proxy + Governance +
admission) restricted to the eligible PAIRED task set per mode, via aggregate_report.harness_seven_for_tasks
-- NOT re-derived from raw result.json['dimension_scores']. Governance is sourced from each task's
result.rescored.json block (reportable/evaluation_error honoured) and is admissible at the dataset level only
when the rescore artifact is CURRENT (governance_consistency).

Admissions (all enforced; --formal => an INVALID bundle prints diagnostics and exits 2 with NO numbers):
  * NATIVE_RESOLVED_SET_MISMATCH    -- the set of native-resolved task ids must be identical across modes.
  * DIMENSION_NOT_SCORED:<dim>      -- every present mode must carry a dataset-level numeric reportable score
                                       for ALL 7 dims (Governance included; not_rescored => not scored).
  * GOVERNANCE_NOT_CURRENT:<status> -- a Governance number is admitted only when the rescore artifact is
                                       current (overall_artifact_status==current, metadata_agrees,
                                       disk_equals_report).
  * <ROLE>_MISSING_OR_MIXED         -- agent_model / tool_backend_model / runtime_judge / native_outcome_judge
                                       / governance_rescore_judge each recorded SEPARATELY and required to be
                                       present (for the applicable role/mode) and single-valued across modes.
  * MISSING_MODES / DUPLICATE_TASK_ID + schema fail-closed, mode==dir consistency, trajectory-missing excl.

The headline is the paired off->enforce NATIVE-outcome 5-state transition (preserved/harmed/recovered/
unchanged + unresolved_under_mode -- never silently dropped). Per-dim coverage [n/N] and pooled harness-rate
coverage [valid_pairs/N] are printed so denominators are explicit. Usage: cmp_report.py [prefix] [--formal]
"""
import json, glob, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "runner"))
from aggregate_report import native_task_outcome, harness_seven_for_tasks, build as _build_report

ARGS = sys.argv[1:]
FORMAL = "--formal" in ARGS
PREFIX = next((a for a in ARGS if not a.startswith("--")), "res6")
_MANIFEST = json.load(open("%s_manifest.json" % PREFIX)) if os.path.exists("%s_manifest.json" % PREFIX) else None
MODES_BY_DS = {"PhysicianBench": ["off", "enforce"], "MedCTA": ["off", "observe", "assist", "enforce"],
               "HealthAdminBench": ["off", "enforce"]}
STEM = {"PhysicianBench": "pb", "MedCTA": "mcta", "HealthAdminBench": "hab"}
MODS = ["Execution", "Tooling", "Context", "Lifecycle", "Observability", "Verification", "Governance"]
TOOL_SUBSTRATE = {"MedCTA"}
SEMANTIC_JUDGE = {"MedCTA", "HealthAdminBench"}   # runtime semantic judge (observe/assist/enforce)
NATIVE_OUTCOME_JUDGE = {"MedCTA"}                  # native outcome via GAcc judge (HAB native = Outcome checkpoints)
POOL = {"wrong_scope_action_rate": ("wrong_scope_count", "wrong_scope_opportunities"),
        "missing_prerequisite_rate": ("missing_prerequisite_count", "missing_prerequisite_opportunities"),
        "verified_commit_rate": ("verified_commit_count", "n_commits"),
        "unknown_verification_rate": ("unknown_verification_count", "n_commits"),
        "executed_violation_rate": ("executed_violation_count", "n_proposed_actions"),
        "repair_success_rate": ("verified_repair_count", "repair_opportunities"),
        "escalation_rate": ("escalation_count", "n_proposed_actions")}


HARD_EXCLUDE = {"api_backend_error", "degraded_tool_health", "mock_env", "replay_tool_backend",
                "uses_hidden_reference", "scorer_validation_only", "non_independent_judge", "outcome_proxy"}


def eligible(d, mode):
    _q = set(d.get("qualification") or []) & HARD_EXCLUDE
    if _q:
        return False, "qualification:%s" % ",".join(sorted(_q))
    if d.get("evaluation_status") not in ("complete", "completed", "evaluated", "partial"):  # partial = some proxy checkpoints; still evaluated + schema-valid
        return False, "not_evaluated"
    sv = d.get("schema_validation")
    if not isinstance(sv, dict) or sv.get("valid") is not True:          # schema fail-closed
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
        if h.get("requested_mode") != mode or h.get("effective_mode") != mode:   # mode == dir consistency
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
    o = {"base": base, "n": 0, "tasks": {}, "excl": {}, "dups": set(),
         "agent": set(), "tool": set(), "rjudge": set(), "njudge": set(),
         "miss_agent": 0, "miss_tool": 0, "miss_rjudge": 0, "miss_njudge": 0,
         "shas": set(), "miss_sha": 0}
    seen = set()
    for fpath in sorted(glob.glob(base + "/*/result.json")):
        d = json.load(open(fpath)); tdir = os.path.dirname(fpath)
        tid = d.get("task_id") or os.path.basename(tdir)
        if tid in seen:
            o["dups"].add(tid); continue
        seen.add(tid)
        ok, why = eligible(d, mode)
        if not ok:
            o["excl"][why] = o["excl"].get(why, 0) + 1; continue
        dlv = delivered(base, tid)
        if dlv is None:                                          # trajectory-missing exclusion
            o["excl"]["trajectory_missing"] = o["excl"].get("trajectory_missing", 0) + 1; continue
        nat = native_task_outcome(d, ds)                         # True / False / None (canonical, NOT success)
        prov = d.get("provenance") or {}; h = d.get("harness") or {}
        o["tasks"][tid] = {"native": nat, "strict_gate": 1 if d.get("success") else 0,
                           "hm": (h.get("metrics") or {}), "delivered": dlv}
        o["n"] += 1
        # ---- JUDGE / MODEL ROLES recorded SEPARATELY (each validated single-valued across modes) ----
        am = prov.get("agent_model")
        o["agent"].add(am) if am else o.__setitem__("miss_agent", o["miss_agent"] + 1)
        sh = prov.get("git_sha")
        o["shas"].add(sh) if sh else o.__setitem__("miss_sha", o["miss_sha"] + 1)
        tb = prov.get("tool_backend_model")
        if tb:
            o["tool"].add(tb)
        elif ds in TOOL_SUBSTRATE:
            o["miss_tool"] += 1
        rj = (h.get("audit") or {}).get("runtime_judge_model")   # harness runtime judge
        if rj:
            o["rjudge"].add(rj)
        elif mode != "off" and ds in SEMANTIC_JUDGE:             # observe/assist/enforce on a judge dataset
            o["miss_rjudge"] += 1
        nj = ((prov.get("judges") or {}).get("outcome") or {}).get("model")   # native (GAcc) outcome judge
        if nj:
            o["njudge"].add(nj)
        elif ds in NATIVE_OUTCOME_JUDGE:
            o["miss_njudge"] += 1
    return o


def f(v):
    return "  -  " if v is None else ("%.3f" % v if isinstance(v, float) else str(v))


def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and not isinstance(x, bool)]
    return (sum(xs) / len(xs)) if xs else None


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


_exit = 0
print("FORMAL COMPARISON  prefix=%s  mode=%s" % (PREFIX, "FORMAL" if FORMAL else "descriptive"))
for ds, modes in MODES_BY_DS.items():
    agg = {m: load(ds, m) for m in modes}
    present = [m for m in modes if agg[m]["n"]]
    # PAIRED: the FULL eligible task set must be identical across modes; every column then uses ONE common
    # paired_ids (not each mode's own subset), else this is not a paired comparison.
    _elig = {m: frozenset(agg[m]["tasks"]) for m in present}
    _set_mismatch = (len(set(_elig.values())) > 1) if _elig else False
    paired_ids = sorted(next(iter(_elig.values()))) if (present and not _set_mismatch) else []
    # canonical whitelisted 7-dim profile + dataset-level governance currentness, per present mode.
    dims, gov = {}, {}
    for m in present:
        wl = paired_ids
        try:
            dims[m] = harness_seven_for_tasks(agg[m]["base"], ds, wl) or {}
        except Exception as e:
            dims[m] = {}; agg[m]["excl"]["harness_seven_error:%s" % type(e).__name__] = 1
        try:
            gov[m] = (_build_report(agg[m]["base"], ds).get("governance_consistency") or {})
        except Exception:
            gov[m] = {}

    def U(key):
        s = set()
        for m in present:
            s |= agg[m][key]
        return s
    agents, tools, rjs, njs = U("agent"), U("tool"), U("rjudge"), U("njudge")
    gjs = set()
    for m in present:
        for x in (gov[m].get("disk_judge_models") or []):
            gjs.add(x)

    r = []
    missing_modes = [m for m in modes if not agg[m]["n"]]
    if missing_modes:
        r.append("MISSING_MODES:%s" % ",".join(missing_modes))
    if any(agg[m]["dups"] for m in modes):
        r.append("DUPLICATE_TASK_ID")
    if _set_mismatch:
        r.append("ELIGIBLE_TASK_SET_MISMATCH")
    if _MANIFEST:                                   # eligible paired set must equal the DECLARED task universe
        _decl = set((_MANIFEST.get("declared") or {}).get(ds) or [])
        if _decl and frozenset(paired_ids) != _decl:
            r.append("DECLARED_TASK_SET_INCOMPLETE:%d/%d" % (len(paired_ids), len(_decl)))
    # ---- NATIVE admission: native-resolved id set identical across the compared (present) modes ----
    resolved = {m: frozenset(t for t, v in agg[m]["tasks"].items() if v["native"] is not None) for m in present}
    if len(present) >= 2 and len(set(resolved.values())) > 1:
        r.append("NATIVE_RESOLVED_SET_MISMATCH")
    for m in present:
        if paired_ids and frozenset(resolved[m]) != frozenset(paired_ids):
            r.append("NATIVE_OUTCOME_UNRESOLVED:%s" % m)
    # ---- ROLE admission: each role present (where applicable) + single-valued across modes ----
    if len(agents) > 1 or any(agg[m]["miss_agent"] for m in present):
        r.append("AGENT_MODEL_MISSING_OR_MIXED")
    shas = U("shas")
    if len(shas) != 1 or any(agg[m]["miss_sha"] for m in present):   # one present code SHA across the whole bundle
        r.append("SHA_MISSING_OR_MIXED")
    if ds in TOOL_SUBSTRATE:
        if len(tools) > 1 or any(agg[m]["miss_tool"] for m in present):
            r.append("TOOL_BACKEND_MISSING_OR_MIXED")
    elif len(tools) > 1:
        r.append("TOOL_BACKEND_MIXED")
    if ds in SEMANTIC_JUDGE:
        if len(rjs) > 1 or any(agg[m]["miss_rjudge"] for m in present):
            r.append("RUNTIME_JUDGE_MISSING_OR_MIXED")
    if ds in NATIVE_OUTCOME_JUDGE:
        if len(njs) != 1 or any(agg[m]["miss_njudge"] for m in present):
            r.append("NATIVE_OUTCOME_JUDGE_MISSING_OR_MIXED")
    elif len(rjs) > 1:
        r.append("RUNTIME_JUDGE_MIXED")
    _gov_scored = any(_num((dims[m].get("Governance") or {}).get("score")) for m in present)
    if (_gov_scored and len(gjs) != 1) or len(gjs) > 1:
        r.append("GOVERNANCE_RESCORE_JUDGE_MISSING_OR_MIXED")
    # ---- DIMENSION COMPLETENESS admission: every present mode scores ALL 7 dims (reportable numeric) ----
    not_scored, not_admitted = set(), set()
    for m in present:
        for dim in MODS:
            blk = dims[m].get(dim) or {}
            if not _num(blk.get("score")):
                not_scored.add(dim)
            elif blk.get("adapter_admission") != "ok":   # must be PROVEN ok (missing/None is not success)
                not_admitted.add(dim)
    for dim in sorted(not_scored):
        r.append("DIMENSION_NOT_SCORED:%s" % dim)
    for dim in sorted(not_admitted):
        r.append("DIMENSION_NOT_ADMITTED:%s" % dim)
    # role INDEPENDENCE: a runtime/governance judge must differ from the agent brain AND the tool backend
    _nm = lambda x: str(x or "").strip().lower().split("/")[-1].split(" (")[0]
    _an = {_nm(x) for x in agents}; _tn = {_nm(x) for x in tools}
    if {_nm(x) for x in rjs} & (_an | _tn):
        r.append("RUNTIME_JUDGE_NOT_INDEPENDENT")
    if {_nm(x) for x in gjs} & (_an | _tn):
        r.append("GOVERNANCE_JUDGE_NOT_INDEPENDENT")
    # ---- GOVERNANCE strictness: a present-and-scored Governance is admitted only if the artifact is current
    for m in present:
        if _num((dims[m].get("Governance") or {}).get("score")):
            g = gov[m]
            if not (g.get("overall_artifact_status") == "current" and g.get("metadata_agrees") is True
                    and g.get("disk_equals_report") is True):
                r.append("GOVERNANCE_NOT_CURRENT:%s@%s" % (g.get("overall_artifact_status"), m))
    valid = not r

    print("=" * 110)
    print("%-16s sha=%s agent=%s tool=%s runtime_judge=%s native_outcome_judge=%s governance_rescore_judge=%s" % (
          ds, "|".join(sorted(shas)) or "MISSING",
        "|".join(sorted(agents)) or "?",
        "|".join(sorted(tools)) or ("none" if ds in TOOL_SUBSTRATE else "n/a"),
        "|".join(sorted(rjs)) or ("none" if ds in SEMANTIC_JUDGE else "n/a"),
        "|".join(sorted(njs)) or ("none" if ds in SEMANTIC_JUDGE else "n/a"),
        "|".join(sorted(gjs)) or "none"))
    print("eligible n: " + " ".join("%s=%d" % (m, agg[m]["n"]) for m in modes)
          + "   BUNDLE=" + ("VALID" if valid else "INVALID [" + " ".join(r) + "]"))
    for m in modes:
        if agg[m]["excl"]:
            print("  excluded[%s]: %s" % (m, agg[m]["excl"]))
        if agg[m]["dups"]:
            print("  DUPLICATE task_ids[%s]: %s" % (m, sorted(agg[m]["dups"])))
    if not valid and FORMAL:
        print("  *** INVALID -- formal: no comparison numbers. ***"); _exit = 2; continue
    if not valid:
        print("  *** INVALID BUNDLE -- DESCRIPTIVE ONLY. ***")

    # ---- NATIVE OUTCOME (canonical, native-resolved subset) | strict_gate | answer_delivered ----
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

    # ---- paired off->enforce 5-state NATIVE transition (never silently drops a task) ----
    if "off" in present and "enforce" in present:
        To, Te = agg["off"]["tasks"], agg["enforce"]["tasks"]
        allt = set(To) | set(Te)
        pres = harmed = recov = unch = unres = unres_both = 0
        for t in sorted(allt):
            o0 = To.get(t, {}).get("native"); o1 = Te.get(t, {}).get("native")
            if o0 is None and o1 is None:
                unres_both += 1; continue                         # unresolved in BOTH (counted, not dropped)
            if o0 is None or o1 is None:
                unres += 1; continue                              # native None in exactly one mode
            o0, o1 = bool(o0), bool(o1)
            pres += o0 and o1; harmed += o0 and not o1
            recov += (not o0) and o1; unch += (not o0) and not o1
        print("  PAIRED off->enforce NATIVE-OUTCOME: preserved(1->1)=%d harmed(1->0)=%d recovered(0->1)=%d "
              "unchanged(0->0)=%d unresolved_one_mode=%d unresolved_both=%d  (total=%d)"
              % (pres, harmed, recov, unch, unres, unres_both, pres + harmed + recov + unch + unres + unres_both))
        op = pres / (pres + harmed) if (pres + harmed) else None
        rr = recov / (recov + unch) if (recov + unch) else None
        print("    outcome_preservation=%s   recovery_rate=%s" % (f(op), f(rr)))

    # ---- 7 DIMENSIONS from the canonical whitelisted pipeline: score [n_with/n_qualified  r=n_reportable] ----
    print("  7 DIMENSIONS  canonical harness_seven (whitelisted paired set)  score [n_scored/N  r=reportable]:")
    for dim in MODS:
        cells = []
        for m in modes:
            if m not in present:
                cells.append("%s=--" % m); continue
            blk = dims[m].get(dim) or {}
            cells.append("%s=%s[%s/%s r=%s]" % (m, f(blk.get("score")), blk.get("n_with_evidence"),
                                                blk.get("n_qualified"), blk.get("n_reportable")))
        print("    %-12s %s" % (dim, " ".join(cells)))

    # ---- HARNESS rates: pooled = sum(num)/sum(den) over PAIRED tasks (both present); coverage=valid_pairs/N ----
    print("  HARNESS rates  pooled [valid_pairs/N]:")
    for k, (nf, df) in POOL.items():
        cells = []
        for m in modes:
            if m not in present:
                cells.append("%s=--" % m); continue
            T = agg[m]["tasks"]
            num = den = vp = 0
            for t in T.values():
                n, dd = t["hm"].get(nf), t["hm"].get(df)
                if isinstance(n, int) and isinstance(dd, int):    # both numerator AND denominator present
                    num += n; den += dd; vp += 1
            poo = (num / den) if den else None
            cells.append("%s=%s [%d/%d]" % (m, f(poo), vp, len(T)))
        print("    %-24s %s" % (k, " ".join(cells)))

print("=" * 110)
print("OUTCOME = aggregate_report.native_task_outcome (NOT result['success']). 7 dims = canonical")
print("harness_seven_for_tasks (strict+proxy+Governance+admission), whitelisted to the paired task set.")
sys.exit(_exit)
