"""Loop/agent feedback conformance for cluster B-loop-feedback-agent (P0-1, P0-6, P0-11).

Guards the SHARED CONTRACT for harness->agent feedback:
  (P0-1) tool_agent.ToolProtocolAgent.act AND api_agent.ApiToolAgent.act_fc surface
         state["harness_feedback"] to the model as an explicit [HARNESS ...] line, ADDITIVELY
         (the env observation handling is untouched), including reason + missing_obligations.
  (P0-6) run.py before_action / before_final REVISE/BLOCK no longer OVERWRITE last_obs/last_res;
         they set the separate additive `pending_harness_feedback` and count `_repair_turns`.
  (P0-11) run.py after_action RESERVES room for the [HARNESS] note (no longer re-truncates to the
          same cap so the note is never sliced off a large observation).

No model calls; deterministic. Run: python3 runner/test_harness_loop.py — expects all PASS.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HF = {"decision": "REVISE", "reason": "missing labs before order",
      "missing_obligations": ["order_basic_metabolic_panel"], "stage": "before_action"}


def _user_text(agent):
    return "\n".join(m["content"] for m in agent.messages if m.get("role") == "user")


def test_tool_agent_surfaces_harness_feedback():
    import tool_agent
    ag = object.__new__(tool_agent.ToolProtocolAgent)
    ag.messages = []
    ag.et = "tool_sandbox"
    ag._pending = False
    ag._chat = lambda *a, **k: "<answer>done</answer>"
    res = ag.act({"harness_feedback": dict(HF), "last_result": None, "last_observation": None})
    assert res["type"] == "final", res
    ut = _user_text(ag)
    assert "[HARNESS REVISE]" in ut, ut
    assert "missing labs before order" in ut, ut
    assert "order_basic_metabolic_panel" in ut, ut
    assert "ENVIRONMENT observation below is still current" in ut, ut


def test_tool_agent_no_feedback_no_harness_line():
    import tool_agent
    ag = object.__new__(tool_agent.ToolProtocolAgent)
    ag.messages = []
    ag.et = "tool_sandbox"
    ag._pending = False
    ag._chat = lambda *a, **k: "<answer>done</answer>"
    ag.act({"last_result": None, "last_observation": None})
    assert "[HARNESS" not in _user_text(ag)


def test_api_agent_act_fc_surfaces_harness_feedback():
    import api_agent
    ag = object.__new__(api_agent.ApiToolAgent)
    ag.messages = []
    ag._fc_call_id = None
    ag._chat_fc = lambda msgs: {"content": "done", "tool_calls": []}
    res = ag.act_fc({"harness_feedback": dict(HF), "last_result": None})
    assert res["type"] == "final", res
    ut = _user_text(ag)
    assert "[HARNESS REVISE]" in ut, ut
    assert "order_basic_metabolic_panel" in ut, ut
    assert "ENVIRONMENT observation below is still current" in ut, ut


def _run_src():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "run.py"), encoding="utf-8") as f:
        return f.read()


def test_run_before_action_is_additive_not_overwrite():
    src = _run_src()
    # P0-6: the old destructive overwrite of last_obs at before_action REVISE/BLOCK must be GONE.
    assert 'last_res = {"harness_decision": _hb.type, "harness_feedback": _hb.feedback}' not in src
    assert 'last_res = {"harness_decision": _hf.type, "harness_feedback": _hf.feedback}' not in src
    # and replaced by the contract additive key + repair counter.
    assert "pending_harness_feedback = _next_feedback(_hb, \"before_action\")" in src
    assert "_repair_turns += 1" in src
    assert "max_repair_turns" in src
    # the feedback handed to the agent must be the FULL harness feedback (suggested_capabilities etc. NOT
    # hand-picked away) -- regression guard for the actionable-repair signal.
    assert "def _next_feedback(hr, stage):" in src
    assert "fb = dict(hr.feedback or {})" in src


def test_run_state_dict_passes_harness_feedback_and_clears():
    src = _run_src()
    assert "\"harness_feedback\": pending_harness_feedback})" in src
    assert "pending_harness_feedback = None   # consumed this turn" in src


def test_run_before_final_preserves_terminal_answer():
    src = _run_src()
    # CONTRACT(5): on repair-budget exhaustion at before_final, the final answer is delivered WITH a flag.
    assert "\"stage\": \"before_final\"" in src
    assert "\"verification_flag\": \"unverified_grounding\"" in src


def test_run_after_action_reserves_room_for_harness_note():
    src = _run_src()
    # P0-11: must reserve room instead of re-truncating obs+note to the same cap.
    assert "_htxt = \"\\n[HARNESS] \" + json.dumps(_hpost.feedback, ensure_ascii=False)" in src
    assert "obs[:_max - len(_htxt)] + _htxt" in src
    # the old single-slice fold must be gone.
    assert "(obs + \"\\n[HARNESS] \" + json.dumps(_hpost.feedback, ensure_ascii=False))[:int(os.environ.get(\"MH_OBS_MAX_LEN\", \"10000\"))]" not in src



def test_run_answer_attempt_logged_and_must_resolve_abstains():
    src = _run_src()
    # every submitted final answer is logged as an answer_attempt (not only the one that lands).
    assert '"event_type": "answer_attempt"' in src
    assert "_answer_attempts += 1" in src
    # must-resolve (localized contradiction) is NOT eligible for unverified_grounding flagged delivery:
    # on repair-budget exhaustion the violating answer is WITHHELD -> safe abstention.
    assert "_mr_viol" in src
    assert '"abstained_unresolved_contradiction"' in src
    assert '"final_disposition": "abstained_unresolved_violation"' in src


def test_run_layer2_candidate_selection_flow():
    src = _run_src()
    # Layer-2: the answer-layer repair mode is an ablation flag; a candidate-mode REVISE holds the ORIGINAL
    # and the next final is compared and only conservatively adopted.
    assert 'os.environ.get("MH_REPAIR"' in src
    assert "_pending_candidate" in src
    assert "compare_answer_candidates" in src and "adopt_revised" in src
    assert '"revised_commit_adopted"' in src and '"kept_original"' in src
    # soft = naive adopt-if-preferred; select/full = conservative adopt_revised.
    assert 'if _repair_mode == "soft"' in src



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
