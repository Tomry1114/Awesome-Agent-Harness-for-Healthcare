"""Loop/agent feedback conformance: the shared contract for harness->agent feedback.

Guards:
  - the text workspace programs (PB, HAB, MedCTA) AND the function-calling workspace programs surface
    state["harness_feedback"] to the model as an explicit [HARNESS ...] line, ADDITIVELY
    (the env observation handling is untouched), including reason + missing_obligations.
  - run.py before_action / before_final REVISE/BLOCK do not OVERWRITE last_obs/last_res;
    they hand the feedback in as a separate additive turn input and count `_repair_turns`.
  - run.py after_action RESERVES room for the [HARNESS] note instead of truncating obs+note to the
    same cap, so the note is never sliced off a large observation.

No model calls; deterministic. Run: python3 runner/tests/test_harness_loop.py — expects all PASS.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HF = {"decision": "REVISE", "reason": "missing labs before order",
      "missing_obligations": ["order_basic_metabolic_panel"], "stage": "before_action"}


def _user_text(agent):
    return "\n".join(m["content"] for m in agent.messages if m.get("role") == "user")


def _workspace_program(env_type, protocol, reply):
    from workspace import H0, Workspace
    ws = Workspace(H0[env_type], protocol)
    return ws.program_class()({"goal": "q", "context": {}, "available_tools": []}, ws, lambda *a, **k: reply)


def test_function_calling_workspace_programs_surface_harness_feedback():
    # MedCTA and HealthAdminBench (PhysicianBench's program: runner/tests/test_fc_revision_scope.py)
    for env_type in ("tool_sandbox", "gui"):
        ag = _workspace_program(env_type, "function_calling", {"content": "done", "tool_calls": []})
        res = ag.act({"harness_feedback": dict(HF), "last_result": None})
        assert res["type"] == "final", res
        ut = _user_text(ag)
        assert "[HARNESS REVISE]" in ut, (env_type, ut)
        assert "order_basic_metabolic_panel" in ut, (env_type, ut)
        assert "ENVIRONMENT observation below is still current" in ut, (env_type, ut)


def test_text_workspace_programs_surface_harness_feedback():
    for env_type in ("fhir", "gui", "tool_sandbox"):
        ag = _workspace_program(env_type, "text", "<answer>done</answer>")
        res = ag.act({"harness_feedback": dict(HF), "last_result": None, "last_observation": None})
        assert res == {"type": "final", "answer": "done"}, (env_type, res)
        ut = _user_text(ag)
        assert "[HARNESS REVISE]" in ut and "missing labs before order" in ut and "order_basic_metabolic_panel" in ut, \
            (env_type, ut)
        assert "ENVIRONMENT observation below is still current" in ut, (env_type, ut)
        ag = _workspace_program(env_type, "text", "<answer>done</answer>")
        ag.act({"last_result": None, "last_observation": None})
        assert "[HARNESS" not in _user_text(ag), env_type


def _run_src():
    # The per-action pipeline (execute/normalize/read-back/event) lives in harness/executor.py, so the runner "source"
    # for these mechanism-presence checks spans run.py and executor.py. `base` is runner/ (this file sits in
    # runner/tests/, see the sys.path line at the top); a wrong base would make every check read nothing.
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = ""
    for rel in ("run.py", os.path.join("harness", "executor.py")):
        with open(os.path.join(base, rel), encoding="utf-8") as f:
            src += f.read() + "\n"
    return src


def test_run_before_action_is_additive_not_overwrite():
    src = _run_src()
    # the destructive overwrite of last_obs at before_action REVISE/BLOCK must be absent.
    assert 'last_res = {"harness_decision": _hb.type, "harness_feedback": _hb.feedback}' not in src
    assert 'last_res = {"harness_decision": _hf.type, "harness_feedback": _hf.feedback}' not in src
    # the feedback goes through the additive turn input + repair counter instead.
    assert "_turn.feedback(_next_feedback(_hb, \"before_action\")" in src   # one turn input, never an overwrite
    assert "_repair_turns += 1" in src
    assert "max_repair_turns" in src
    # the feedback handed to the agent must be the FULL harness feedback (suggested_capabilities etc. NOT
    # hand-picked away) -- regression guard for the actionable-repair signal.
    assert "def _next_feedback(hr, stage):" in src
    assert "fb = dict(hr.feedback or {})" in src


def test_run_state_dict_passes_harness_feedback_and_clears():
    src = _run_src()
    # Assert the CONTRACT, not run.py's line breaks: the turn inputs (the harness_feedback view among them) are
    # handed into the state dict, then cleared once the agent's act returned. Both halves are checked, because
    # dropping the clearing check would let a leak-the-feedback-forever regression pass.
    assert "_state.update(_turn.state(last_res, last_obs))" in src
    assert "action = agent.act(_state)\n        _turn.consumed()" in src
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "turn_inputs.py"),
              encoding="utf-8") as f:
        ti = f.read()
    assert '"harness_feedback": fb' in ti and "self._items = []" in ti


def test_run_before_final_preserves_terminal_answer():
    src = _run_src()
    # on repair-budget exhaustion at before_final, the final answer is delivered WITH a flag.
    assert "\"stage\": \"before_final\"" in src
    assert "\"verification_flag\": \"unverified_grounding\"" in src


def test_run_after_action_reserves_room_for_harness_note():
    src = _run_src()
    # after_action must reserve room for the note instead of re-truncating obs+note to the same cap.
    assert "_htxt = \"\\n[HARNESS] \" + json.dumps(_hpost.feedback, ensure_ascii=False)" in src
    assert "obs[:_max - len(_htxt)] + _htxt" in src
    # a single-slice fold (obs + note truncated together) must be absent.
    assert "(obs + \"\\n[HARNESS] \" + json.dumps(_hpost.feedback, ensure_ascii=False))[:int(os.environ.get(\"MH_OBS_MAX_LEN\", \"10000\"))]" not in src



def test_run_answer_attempt_logged_and_must_resolve_abstains():
    src = _run_src()
    # every submitted final answer is logged as an answer_attempt (not only the one that lands).
    assert '"event_type": "answer_attempt"' in src
    assert "_answer_attempts += 1" in src
    # must-resolve (localized contradiction) is NOT eligible for unverified_grounding flagged delivery:
    # on repair-budget exhaustion the violating answer is WITHHELD -> safe abstention.
    assert "_mr_viol" in src and "_MH.withhold_unresolved_violation(" in src
    # the withhold is one shell helper, shared by this path and an on_answer Accept
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(base, "middleware_host.py"), encoding="utf-8") as f:
        host = f.read()
    assert '"abstained_unresolved_contradiction"' in host
    assert '"final_disposition": "abstained_unresolved_violation"' in host


def test_run_layer2_candidate_selection_flow():
    src = _run_src()
    # Layer-2: a before_final ACQUIRE or a candidate-mode REVISE holds the ORIGINAL and the next final is compared
    # and only conservatively adopted. The strategy is MedCTA's workspace middleware
    # (workspaces/medcta/h0/middleware/answer_candidate.py: the hold in after_final_verdict, the candidate in on_answer,
    # adjudicated by evaluate_candidate into ADOPT_B / KEEP_A); run.py keeps the loop semantics: the candidate repair
    # turn, its cap, kept_original_no_candidate.
    assert "_pending_iv" not in src and "consume_candidate" not in src
    assert '"final_disposition": "kept_original_no_candidate"' in src
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with open(os.path.join(repo, "workspaces", "medcta", "h0", "middleware", "answer_candidate.py"),
              encoding="utf-8") as f:
        surf = f.read()
    assert "def evaluate_candidate(" in surf and "def on_answer(" in surf and "def after_final_verdict(" in surf
    assert "ADOPT_B" in surf and "KEEP_A" in surf



def test_run_active_readback_wired():
    src = _run_src()
    assert "env.reconcile_write(" in src                       # run.py performs the active read-back
    assert '"readback_unconfirmed"' in src                     # an unconfirmed write becomes an error
    assert '"event_type": "reconciliation"' in src             # and is recorded
    assert '"readback" in _estr' in src                        # routed to unknown commit-state, not silent pass



def test_run_reconcile_recovery_not_terminal():
    src = _run_src()
    # an UNKNOWN commit enters restricted read-back recovery (does NOT terminate); writes are blocked until resolved.
    assert 'if _hpost.type == "RECONCILE"' in src and '_reconcile.enter(action, res, "agent"' in src
    assert '"event_type": "reconcile_enter"' in src
    assert '"event_type": "reconcile_block"' in src                # writes blocked during recovery
    assert '"event_type": "reconcile_resolved"' in src             # read-back resolves committed/failed
    assert "reconcile_budget_exhausted" in src                     # only then escalate



def _run():
    fns = [v for kk, v in sorted(globals().items()) if kk.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn(); passed += 1; print("PASS", fn.__name__)
        except AssertionError as e:
            print("FAIL", fn.__name__, "->", e)
        except Exception as e:
            print("ERROR", fn.__name__, "->", repr(e))
    print("\nharness loop/agent conformance: %d/%d passed" % (passed, len(fns)))
    return passed == len(fns)


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
