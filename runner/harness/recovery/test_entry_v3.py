"""Bounded Clinical Recovery v3 - feature-flag entry point unit tests.

Standalone: prints PASS/FAIL per check; sys.exit(0) iff every check passes, non-zero otherwise.
Run: python3 runner/harness/recovery/test_entry_v3.py

No network / no model calls / no live driver. Asserts run_recovery_v3(...) behaves as the flag entry:
  (a) unknown env                 -> not dispatched, reason 'no_stack_for_env'.
  (b) lifecycle not triggered     -> not dispatched, reason 'lifecycle_not_triggered'.
  (c) fhir + no judge             -> dispatched, one DECLINED_NO_COMMITMENT episode, a trajectory event
                                     appended (oracle-blind: no committed order without the agent's judge).
  (d) gui + explicit commitment   -> dispatched, a real episode runs end-to-end OFFLINE (driver=None) and
                                     terminates in a known kernel state WITHOUT raising.
  (e) caller task is NOT mutated  -> _augment_task copies; the original dict is untouched.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from harness.recovery.entry import run_recovery_v3
from harness.recovery import contracts as C

FAILS = []


def check(name, cond):
    if cond:
        print("PASS %s" % name)
    else:
        print("FAIL %s" % name)
        FAILS.append(name)


def test_a_unknown_env():
    s = run_recovery_v3({"environment": {"type": "nope"}}, "x", "deliverable_confirmed", [])
    check("a1 unknown env not dispatched", s["dispatched"] is False)
    check("a2 unknown env reason", s.get("reason") == "no_stack_for_env")


def test_b_not_triggered():
    traj = []
    s = run_recovery_v3({"environment": {"type": "fhir"}}, "x", "random_event", traj)
    check("b1 not triggered -> not dispatched", s["dispatched"] is False)
    check("b2 not triggered reason", s.get("reason") == "lifecycle_not_triggered")
    check("b3 no trajectory event on non-trigger", traj == [])


def test_c_fhir_declined():
    traj = []
    s = run_recovery_v3({"environment": {"type": "fhir"}, "goal": "order a pelvic ultrasound"},
                        "Plan: order a pelvic ultrasound.", "deliverable_confirmed", traj)
    check("c1 fhir triggered", s["triggered"] is True)
    check("c2 fhir dispatched", s["dispatched"] is True)
    check("c3 fhir env_type", s["env_type"] == "fhir")
    check("c4 one episode (no judge -> declined)", len(s["episodes"]) == 1)
    check("c5 episode is DECLINED_NO_COMMITMENT",
          s["episodes"][0]["episode_state"] == C.DECLINED_NO_COMMITMENT)
    check("c6 trajectory event appended", len(traj) == 1 and traj[0]["event_type"] == "recovery_v3")


def test_d_gui_offline_episode():
    task = {"environment": {"type": "gui"}, "goal": "submit the prior auth",
            "recovery_commitments": [{"goal_type": "submit_prior_auth", "payer": "a",
                                      "committed_fields": {"requestType": "prior_auth"}}]}
    traj = []
    ok = True
    try:
        s = run_recovery_v3(task, "I will submit the prior authorization request.",
                            "before_final", traj, driver=None)
    except Exception as e:  # must not raise, even with no live driver
        ok = False
        print("   (raised) %r" % e)
        s = None
    check("d1 gui entry did not raise", ok)
    if s is not None:
        check("d2 gui dispatched", s["dispatched"] is True)
        check("d3 gui produced an episode", len(s["episodes"]) >= 1)
        check("d4 episode terminated in a known state",
              s["episodes"][0]["episode_state"] in C.TERMINAL_STATES
              or s["episodes"][0]["episode_state"] in (C.DECLINED_NO_COMMITMENT, C.NOT_APPLICABLE))
        check("d5 gui trajectory event appended",
              any(e.get("event_type") == "recovery_v3" for e in traj))


def test_e_task_not_mutated():
    task = {"environment": {"type": "fhir"}, "goal": "order"}
    before = dict(task)
    run_recovery_v3(task, "some deliverable content", "deliverable_confirmed", [])
    check("e1 caller task keys unchanged", set(task.keys()) == set(before.keys()))
    check("e2 no deliverable/answer leaked into caller task",
          "deliverable" not in task and "answer" not in task)


def main():
    test_a_unknown_env()
    test_b_not_triggered()
    test_c_fhir_declined()
    test_d_gui_offline_episode()
    test_e_task_not_mutated()
    if FAILS:
        print("\nFAILED: %s" % FAILS)
        sys.exit(1)
    print("\nALL GREEN")
    sys.exit(0)


if __name__ == "__main__":
    main()
