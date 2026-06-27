"""REAL control-flow tests for the run.py agent/harness loop (not source greps). Monkeypatches a fake
env + scripted agent and actually RUNS run.run_task, asserting on the produced trajectory. Run:
  python3 runner/test_harness_loop_exec.py  -> all PASS, exit 0."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.pop("MH_GACC", None); os.environ.pop("MH_GACC_MODEL", None)
os.environ.pop("MH_HARNESS_JUDGE_MODEL", None)   # no semantic judge -> claim-support is UNKNOWN (unverifiable)
import run as R


class FakeEnv:
    type = "tool_sandbox"
    full_state = None
    def __init__(self, script_tools): self._tools = script_tools
    def reset(self): pass
    def available_tools(self): return [{"name": "ImageDescription", "signature": "ImageDescription()"},
                                       {"name": "OtherTool", "signature": "OtherTool()"}]
    def call_tool(self, name, args): return {"output": "a 3cm spiculated RUL nodule on the CT", "ok": True}
    def capabilities(self): return {}
    def teardown(self): pass


class ScriptedAgent:
    name = "scripted-fake"
    def __init__(self, actions):
        self._actions = list(actions); self.seen_feedback = []
    def act(self, state):
        self.seen_feedback.append(state.get("harness_feedback"))
        return self._actions.pop(0) if self._actions else {"type": "final", "answer": "done"}


def _run(actions, mode=None, max_steps=2):
    task = {"task_id": "T", "goal": "What is the finding on the CT?", "context": {"text": "chest/abdomen CT"},
            "environment": {"type": "tool_sandbox"}, "available_tools": [{"name": "ImageDescription"}, {"name": "OtherTool"}],
            "checkpoints": []}
    agent = ScriptedAgent(actions)
    _ot = (R.load_task, R.environments.make_env, R.agents.make_agent)
    R.load_task = lambda bench, tid: task
    R.environments.make_env = lambda *a, **k: FakeEnv(None)
    R.agents.make_agent = lambda name, t: agent
    if mode is None: os.environ.pop("MH_HARNESS_MODE", None)
    else: os.environ["MH_HARNESS_MODE"] = mode
    try:
        res = R.run_task("MedCTA", "T", agent_name="scripted-fake", max_steps=max_steps, cleanup=False)
    finally:
        R.load_task, R.environments.make_env, R.agents.make_agent = _ot
    traj = res.get("_trajectory", [])
    return traj, agent


def _types(traj): return [e.get("event_type") for e in traj]
def _final(traj): return [e for e in traj if e.get("event_type") == "final_answer"]
def _tools(traj): return [e for e in traj if e.get("event_type") == "tool_call"]
import harness as _H_mod

def _capture_build_kernel():
    cap = {}
    _orig = _H_mod.build_kernel
    def _wrapped(*a, **k):
        cap.update(k)
        return _orig(*a, **k)
    _H_mod.build_kernel = _wrapped
    return cap, _orig

PASS = 0; FAIL = 0
def check(name, cond):
    global PASS, FAIL
    if cond: PASS += 1; print("PASS", name)
    else: FAIL += 1; print("FAIL", name)


# T1: after spending the full env-action budget, the agent STILL gets a turn to emit final (no answer erased).
tool = {"type": "tool_call", "tool": "ImageDescription", "args": {}}
final = {"type": "final", "answer": "RUL nodule"}
traj, ag = _run([tool, tool, final], mode=None, max_steps=2)
check("T1_final_delivered_after_full_env_budget", len(_final(traj)) == 1)
check("T1_no_max_steps_exceeded", "max_steps_exceeded" not in _types(traj))

# T2: the agent TRIES 4 tools with max_steps=2; only 2 actually execute (repair/refusal turns don't add env actions),
#     and the final is still delivered.
traj2, ag2 = _run([tool, tool, tool, tool, final], mode=None, max_steps=2)
check("T2_env_budget_enforced_only_2_tools_ran", len(_tools(traj2)) == 2)
check("T2_final_still_delivered", len(_final(traj2)) == 1)

# T4 (feedback channel): once the env budget is spent, the runtime injects feedback that the agent ACTUALLY receives.
check("T4_runtime_budget_feedback_reaches_agent",
      any(isinstance(fb, dict) and fb.get("stage") == "runtime_budget" for fb in ag2.seen_feedback))

# T3: enforce + NO judge -> the final answer is UNVERIFIABLE (ESCALATE). Because an answer is EPISTEMIC
#     (no env side effect), it must be DELIVERED WITH A FLAG, never erased.
traj3, ag3 = _run([tool, final], mode="enforce", max_steps=6)
_f3 = _final(traj3)
check("T3_epistemic_escalated_final_is_delivered", len(_f3) == 1)
check("T3_delivered_with_verification_flag", bool(_f3 and _f3[0].get("verification_flag")))
check("T3_not_aborted_to_nothing", "harness_escalation" not in _types(traj3) or len(_f3) == 1)

# T5: run.py resolves the ACTUAL perception/tool backend model and passes it to build_kernel (independence).
_cap, _orig = _capture_build_kernel()
try:
    _run([tool, final], mode="enforce", max_steps=4)
finally:
    _H_mod.build_kernel = _orig
check("T5_runner_passes_tool_model_to_kernel", _cap.get("tool_model") is not None)

# T6: INSUFFICIENT grounding -> at most ONE revise per evidence_version, then deliver-with-flag (no loop/erase).
_insuff = lambda pr: '{"relation": "insufficient", "supported": false, "confidence": 0.9, "reason": "under-covers"}'
_orig_bk6 = _H_mod.build_kernel
def _bk_insuff(*a, **k):
    kk = _orig_bk6(*a, **k)
    if kk is not None:
        kk.ctx.judge_fn = _insuff
    return kk
_H_mod.build_kernel = _bk_insuff
try:
    traj6, ag6 = _run([tool, final, final, final], mode="enforce", max_steps=6)
finally:
    _H_mod.build_kernel = _orig_bk6
_f6 = _final(traj6)
check("T6_insufficient_one_revise_per_ev_then_flagged_delivery",
      len(_f6) == 1 and bool(_f6[0].get("verification_flag")))

print("\nloop exec conformance: %d/%d passed" % (PASS, PASS + FAIL))
sys.exit(0 if FAIL == 0 else 1)
