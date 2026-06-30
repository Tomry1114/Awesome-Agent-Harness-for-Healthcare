#!/usr/bin/env python3
"""Intervention funnel + F->T/T->F net-benefit report (roadmap item 8b).

Usage: intervention_funnel.py <off_dir/gpt5> <full_dir/gpt5> [--lambda 2.0]

Funnel (full arm, from the serialized ledger): discovered -> admitted(enforce|advisory) -> delivered ->
attempted -> resolved/regressed/exhausted, plus repair-execution and effect-resolution rates.
Net benefit (full vs off): F->T (rescued) minus lambda * T->F (broken). In medicine breaking a correct
task is worse than missing a rescue, so lambda > 1.
"""
import json, glob, os, sys

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
LAMBDA = 2.0
for a in sys.argv[1:]:
    if a.startswith("--lambda"):
        LAMBDA = float(a.split("=")[-1]) if "=" in a else 2.0
OFF, FULL = ARGS[0], ARGS[1]


def _ledger(d):
    r = json.load(open(d))
    return r, ((r.get("harness") or {}).get("audit") or {}).get("ledger") or {}


def _cp_pass(r):
    cps = r.get("checkpoints", [])
    return sum(1 for c in cps if str(c.get("checkpoint_status")).lower() == "passed"), len(cps)


def _outcome_pass(r):
    """Native task pass proxy. MedCTA: Outcome cp carries detail.gacc_score -> mean GAcc >= 0.5 (matches the
    canonical aggregator's native_task_success; NOT result.success). PB/HAB: all Outcome checkpoints pass.
    Else: r['success']."""
    oc = [c for c in r.get("checkpoints", []) if c.get("dimension") == "Outcome"]
    if oc:
        gaccs = [c["detail"]["gacc_score"] for c in oc
                 if isinstance(c.get("detail"), dict) and isinstance(c["detail"].get("gacc_score"), (int, float))]
        if gaccs:                                  # continuous-grounding Outcome (MedCTA): mean GAcc >= 0.5
            return (sum(gaccs) / len(gaccs)) >= 0.5
        return all(str(c.get("checkpoint_status")).lower() == "passed" for c in oc)   # discrete (PB/HAB)
    return bool(r.get("success"))


def funnel(full_dir):
    disc = enf = adv = deliv = att = res = reg = exh = 0
    for d in glob.glob(full_dir + "/*/result.json"):
        _, led = _ledger(d)
        rl = led.get("repair_lifecycle") or []
        ad = led.get("advisories") or []
        disc += len(rl) + len(ad); enf += len(rl); adv += len(ad)
        for x in rl:
            if x.get("delivery_count", 0) > 0: deliv += 1
            if x.get("repair_attempts", 0) > 0: att += 1
            st = x.get("status")
            res += (st == "resolved"); reg += (st == "regressed"); exh += (st == "exhausted")
    print("=== INTERVENTION FUNNEL (full arm) ===")
    if disc == 0:
        print("  (no serialized repair_lifecycle -- run predates item-8a serialization)")
    print("  discovered=%d  admitted: enforce=%d advisory=%d" % (disc, enf, adv))
    print("  delivered=%d  attempted=%d  resolved=%d  regressed=%d  exhausted=%d" % (deliv, att, res, reg, exh))
    if deliv:
        print("  repair_execution_rate(attempted/delivered)=%.2f  effect_resolution_rate(resolved/delivered)=%.2f"
              % (att / deliv, res / deliv))


def net_benefit(off_dir, full_dir):
    def by_task(dirp):
        out = {}
        for d in glob.glob(dirp + "/*/result.json"):
            r = json.load(open(d))
            out[r.get("task_id")] = {"outcome": _outcome_pass(r), "cp": _cp_pass(r)}
        return out
    O, F = by_task(off_dir), by_task(full_dir)
    keys = sorted(set(O) & set(F))
    f2t = t2f = same = cp_delta = 0
    rows = []
    for k in keys:
        op, fp = O[k]["outcome"], F[k]["outcome"]
        oc, fc = O[k]["cp"][0], F[k]["cp"][0]
        cp_delta += (fc - oc)
        tag = "="
        if not op and fp: f2t += 1; tag = "F->T"
        elif op and not fp: t2f += 1; tag = "T->F"
        else: same += 1
        if tag != "=" or fc != oc:
            rows.append("  %-26s off:%s/%d  full:%s/%d  %s" % (k, op, oc, fp, fc, tag))
    print("\n=== NET BENEFIT (full vs off) ===")
    for r in rows: print(r)
    print("  F->T(rescued)=%d  T->F(broken)=%d  unchanged=%d  checkpoint_delta=%+d" % (f2t, t2f, same, cp_delta))
    print("  NET = F->T - %.1f*T->F = %.1f   (>0 means the harness is worth it; medicine penalizes T->F)"
          % (LAMBDA, f2t - LAMBDA * t2f))


funnel(FULL)
net_benefit(OFF, FULL)
