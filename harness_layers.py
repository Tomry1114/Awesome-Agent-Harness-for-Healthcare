"""Harness LAYER-ACTIVITY report (the evolvability instrument). Aggregates every enforce bundle's harness
decisions by LAYER (infrastructure | compensation | amplification) and reports per-mechanism fire-rate, so
the harness self-reports which parts are load-bearing, which compensation has gone vestigial (retire), and
how the provisional amplification layer behaves (trigger / adoption). Usage:
  python harness_layers.py <bundle_glob> [<bundle_glob> ...]      e.g. 'res6_*_enforce'
"""
import json, glob, os, sys, collections
sys.path.insert(0, "runner")
from harness.layers import layer_of, DISPOSITION_LAYER, INFRASTRUCTURE, COMPENSATION, AMPLIFICATION

VESTIGIAL = 0.05   # a compensation mechanism firing on < 5% of tasks is announcing it is now vestigial

def collect(globs):
    fired = collections.Counter()       # mechanism -> tasks where it fired >=1
    layer_fired = collections.Counter()  # layer -> tasks with >=1 decision in that layer
    disp = collections.Counter()
    unclassified = collections.Counter()
    n = 0
    for g in globs:
        for f in glob.glob("%s/gpt5/*/trajectory.jsonl" % g):
            n += 1
            ev = [json.loads(l) for l in open(f)]
            seen_mech, seen_layer = set(), set()
            for e in ev:
                if e.get("event_type") == "harness_decision":
                    lay, mech = layer_of(e.get("reason_code"), e.get("rule_id"))
                    if lay is None:
                        unclassified[mech] += 1; continue
                    seen_mech.add((lay, mech)); seen_layer.add(lay)
                if e.get("event_type") == "final_answer" and e.get("final_disposition") in DISPOSITION_LAYER:
                    disp[e["final_disposition"]] += 1
            for lm in seen_mech: fired[lm] += 1
            for la in seen_layer: layer_fired[la] += 1
    return n, fired, layer_fired, disp, unclassified

def main():
    globs = sys.argv[1:] or ["res6_mcta_enforce", "res6_hab_enforce"]
    n, fired, layer_fired, disp, unclassified = collect(globs)
    if not n:
        print("no enforce bundles matched:", globs); return
    print("=" * 84)
    print("HARNESS LAYER-ACTIVITY REPORT   bundles=%s   tasks=%d" % (",".join(globs), n))
    print("=" * 84)
    for layer in (INFRASTRUCTURE, COMPENSATION, AMPLIFICATION):
        rate = layer_fired.get(layer, 0) / n
        print("\n[%s]  fired on %d/%d tasks (%.0f%%)" % (layer.upper(), layer_fired.get(layer, 0), n, 100 * rate))
        mechs = sorted([(m, c) for (l, m), c in fired.items() if l == layer], key=lambda x: -x[1])
        for m, c in mechs:
            fr = c / n
            tag = ""
            if layer == COMPENSATION and fr < VESTIGIAL:
                tag = "   <-- VESTIGIAL on this model: retire-candidate"
            print("    %-46s %d/%d (%.0f%%)%s" % (m, c, n, 100 * fr, tag))
        if not mechs:
            tag = "   (compensation silent here = the model does not need it)" if layer == COMPENSATION else ""
            print("    (none fired)%s" % tag)
    if disp:
        print("\n[AMPLIFICATION outcome dispositions]")
        for d, c in disp.most_common():
            print("    %-46s %d" % (d, c))
        adopted = disp.get("revised_commit_adopted", 0); kept = disp.get("kept_original", 0)
        trig = adopted + kept + disp.get("kept_original_no_candidate", 0)
        if trig:
            print("    repair trigger=%d  adoption=%d/%d (%.0f%%)" % (trig, adopted, trig, 100 * adopted / trig))
    if unclassified:
        print("\n[UNCLASSIFIED mechanisms -- add to layers.LAYER_OF]:", dict(unclassified))
    print("\nReading: a COMPENSATION mechanism near 0%% is vestigial on this model tier -> ablate it. "
          "INFRASTRUCTURE rates are the durable safety surface. AMPLIFICATION adoption is provisional, "
          "re-measure each model generation.")

main()
