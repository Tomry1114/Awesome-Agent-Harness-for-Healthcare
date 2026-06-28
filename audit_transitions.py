"""Causal audit of paired-outcome transitions. For every RECOVERED (off-fail -> enforce-pass) and HARMED
(off-pass -> enforce-fail) task, dump both trajectories + the harness interventions in the treated mode and
emit a VERDICT on whether the outcome flip was CAUSED BY a harness intervention or is just agent run-to-run
variance. A flip with ZERO harness decisions in the treated trajectory cannot be harness-caused.
Usage: python audit_transitions.py <prefix> <dataset> [base_mode=off] [treat_mode=enforce]"""
import json, glob, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "runner"))
from aggregate_report import native_task_outcome

PREFIX = sys.argv[1] if len(sys.argv) > 1 else "res6"
DS = sys.argv[2] if len(sys.argv) > 2 else "MedCTA"
BASE = sys.argv[3] if len(sys.argv) > 3 else "off"
TREAT = sys.argv[4] if len(sys.argv) > 4 else "enforce"
STEM = {"PhysicianBench": "pb", "MedCTA": "mcta", "HealthAdminBench": "hab"}[DS]


def load(mode):
    out = {}
    for f in glob.glob("%s_%s_%s/gpt5/*/result.json" % (PREFIX, STEM, mode)):
        out[os.path.basename(os.path.dirname(f))] = json.load(open(f))
    return out


def traj(mode, tid):
    p = "%s_%s_%s/gpt5/%s/trajectory.jsonl" % (PREFIX, STEM, mode, tid)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def final_answer(evs):
    fa = [e for e in evs if e.get("event_type") == "final_answer"]
    return (str(fa[-1].get("thought", ""))[:160], fa[-1].get("verification_flag")) if fa else (None, None)


def harness_decisions(evs):
    return [e for e in evs if e.get("event_type") in ("harness_decision", "harness_escalation")]


def gacc(d):
    return [c.get("score") for c in (d.get("checkpoints") or []) if c.get("evaluator_kind") == "gacc_judge"]


b, t = load(BASE), load(TREAT)
recovered, harmed = [], []
for tid in sorted(set(b) & set(t)):
    o, e = native_task_outcome(b[tid], DS), native_task_outcome(t[tid], DS)
    if (not o) and e:
        recovered.append(tid)
    elif o and (not e):
        harmed.append(tid)

print("CAUSAL AUDIT  %s %s  %s->%s   recovered=%d harmed=%d" % (PREFIX, DS, BASE, TREAT, len(recovered), len(harmed)))
caused = {"recovered": 0, "harmed": 0}
for kind, ids in (("RECOVERED", recovered), ("HARMED", harmed)):
    for tid in ids:
        te, be = traj(TREAT, tid), traj(BASE, tid)
        hd = harness_decisions(te)
        fa_t, flag_t = final_answer(te)
        fa_b, _ = final_answer(be)
        n_int = len([e for e in hd if e.get("event_type") == "harness_decision"])
        # CAUSAL verdict: a flip with NO harness decision in the treated run is NOT harness-caused.
        verdict = "AGENT_VARIANCE (0 harness interventions)" if n_int == 0 else "HARNESS_INTERVENED -> review below"
        if n_int > 0:
            caused["recovered" if kind == "RECOVERED" else "harmed"] += 1
        print("\n############ %s: %s  ->  %s ############" % (kind, tid, verdict))
        print("  GAcc  %s=%s  %s=%s" % (BASE, gacc(b[tid]), TREAT, gacc(t[tid])))
        print("  %s tools: %s" % (BASE, [e.get("tool") for e in be if e.get("event_type") == "tool_call"]))
        print("  %s tools: %s" % (TREAT, [e.get("tool") for e in te if e.get("event_type") == "tool_call"]))
        print("  harness decisions in %s (%d): %s" % (TREAT, len(hd),
              [(e.get("decision"), e.get("rule_id"), e.get("stage")) for e in hd] or "NONE"))
        print("  %s answer: %s" % (BASE, fa_b))
        print("  %s answer: %s  flag=%s" % (TREAT, fa_t, flag_t))
print("\nSUMMARY: of %d recovered, %d had a harness intervention; of %d harmed, %d had a harness intervention."
      % (len(recovered), caused["recovered"], len(harmed), caused["harmed"]))
print("(A flip with 0 interventions is agent run-to-run variance, NOT a harness causal effect. The"
      " intervened cases still need the dumped trajectory read to confirm the intervention CAUSED the flip.)")
