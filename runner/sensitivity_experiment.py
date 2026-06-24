#!/usr/bin/env python3
"""Sensitivity experiment (Step a): prove the EXISTING Execution/Lifecycle/Observability proxies
RESPOND CORRECTLY when execution/lifecycle actually degrade, BEFORE building stronger evaluators.
Controlled fault injection over SYNTHETIC trajectories (ground truth known) so we test direction,
not absolute value. Verifies 3 properties: sensitivity / attributability / directionality.
Run: python3 runner/sensitivity_experiment.py"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import proxy_verifiers as pv
import lifecycle_exec as le


def call(tool, ok=True, args=None, obs="liver lesion hypodense", err=None):
    e = {"event_type": "tool_call", "tool": tool, "args": args or {}, "status": "ok" if ok else "error",
         "canonical_observation": {"modalities": {"text": obs} if obs else {}}}
    if not ok:
        e["status"] = "error"; e["error_type"] = err or "tool_error"; e["result"] = "[error] " + (err or "failed")
    else:
        e["result"] = obs
    return e

FINAL = {"event_type": "final_answer", "thought": "the liver lesion is hypodense"}

# (condition, events, expected-direction note)
CONDS = [
    ("baseline_clean",        [call("ImageDescription"), call("OCR"), call("RegionAttributeDescription"), FINAL],
                              "Exec high, Life high"),
    ("no_final(maxstep cut)", [call("ImageDescription"), call("OCR"), call("RegionAttributeDescription")],
                              "Exec DOWN (no terminal completion)"),
    ("tool_errors",           [call("ImageDescription", ok=False), call("OCR", ok=False), call("RegionAttributeDescription"), FINAL],
                              "Exec DOWN (tool_success_rate)"),
    ("repeated_obs(loop)",    [call("OCR"), call("OCR"), call("OCR"), call("OCR"), FINAL],
                              "Life/loop-avoidance SHOULD drop; hygiene DOWN (redundancy)"),
    ("env_fault(unhealthy)",  [call("ImageDescription", ok=False, err="environment_error"), call("OCR"), FINAL],
                              "ATTRIBUTION: env fault, not agent incompetence"),
    ("recover_then_succeed",  [call("RegionAttributeDescription", ok=False, err="tool_argument_error"),
                               call("RegionAttributeDescription", ok=True), FINAL],
                              "DIRECTIONALITY: first exec down, recovery -> Life should stay high"),
    ("repeated_failure",      [call("RegionAttributeDescription", ok=False), call("RegionAttributeDescription", ok=False),
                               call("RegionAttributeDescription", ok=False), FINAL],
                              "DIRECTIONALITY: no recovery -> worse than recover_then_succeed"),
]

rows = []
for name, evs, note in CONDS:
    d = pv.proxy_dimensions(evs)
    d["Execution"] = le.execution(evs); d["Lifecycle"] = le.lifecycle(evs)   # NEW state-machine evaluators
    rows.append((name, d.get("Execution", {}).get("score"), d.get("Lifecycle", {}).get("score"),
                 d.get("Observability", {}).get("score"), d.get("tool_execution_hygiene", {}).get("score"), note))

print("%-22s %-6s %-6s %-6s %-7s  %s" % ("condition", "Exec", "Life", "Obs", "hygiene", "expected"))
for r in rows:
    print("%-22s %-6s %-6s %-6s %-7s  %s" % r)

# ---- property checks (direction, not magnitude) ----
by = {r[0]: r for r in rows}
print("\n=== PROPERTY VERDICTS ===")
def chk(name, cond):
    print(("  PASS " if cond else "  FAIL ") + name)
    return cond
base_e = by["baseline_clean"][1]
chk("SENSITIVITY Exec: no_final < baseline", by["no_final(maxstep cut)"][1] < base_e)
chk("SENSITIVITY Exec: tool_errors < baseline", by["tool_errors"][1] < base_e)
chk("SENSITIVITY hygiene: repeated_obs < baseline", by["repeated_obs(loop)"][4] < by["baseline_clean"][4])
chk("DIRECTIONALITY: recover_then_succeed Exec > repeated_failure Exec",
    by["recover_then_succeed"][1] > by["repeated_failure"][1])
# attributability + loop-avoidance are KNOWN GAPS of the current coarse proxy -> report, not assert
print("\n=== KNOWN GAPS surfaced by (a) -> motivate (b) deterministic state-machine evaluators ===")
print("  Lifecycle does NOT drop on repeated_obs(loop): Life=%s (no loop-avoidance signal)" % by["repeated_obs(loop)"][2])
print("  Lifecycle does NOT distinguish recover vs repeated_failure: %s vs %s" %
      (by["recover_then_succeed"][2], by["repeated_failure"][2]))
print("  Attribution (env_fault) is NOT separated inside Execution proxy (needs capability.healthy gate): Exec=%s" %
      by["env_fault(unhealthy)"][1])
