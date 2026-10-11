# -*- coding: utf-8 -*-
"""Agent-vs-harness checkpoint credit on PhysicianBench.

Real run.run_task through runner/tests/inline_loop_rig.py (model, FHIR network, judge and scoring stood in at their
boundaries; kernel decisions from a scripted workspace middleware). Pinned:

  * PB's forced broad read (the kernel middleware fhir_empty_targeted_search_broad_reacquire: an ACQUIRE at
    after_action on an empty filtered search) is recorded origin harness, requester kernel_acquire; it reaches the
    kernel with that origin; it is not in the agent-credited scoring inputs (logs/agent/trajectory.log,
    agent_tool_calls, agent_trajectory) and is recorded in the harness view
  * the kernel's no_new_progress loop detection does not fire on the harness's read
  * a deliverable that exists only through forced_deliverable_write is harness-completed: the checkpoints whose test
    reads it are credited harness, and the agent-credited outcome is not achieved where the native one is

The tests fail if the forced read is logged as origin "recovery" with no requester, enters the kernel with no origin,
is written to trajectory.log or agent_tool_calls, or if no credit record exists.
"""
import copy
import json
import os

import inline_loop_rig as L

PB = "PhysicianBench"
PB_TASK = "PB-pretransplant_covid_clearance"
DELIVERABLE_TASK = "PB-buprenorphine_mood_support"
DELIVERABLE = "/workspace/output/management_plan.txt"
LABS = "fhir_observation_search_labs"
BROAD_REACQUIRE = "fhir_empty_targeted_search_broad_reacquire"


def _pb(monkeypatch, tmp_path, task=PB_TASK):
    rig = L.Rig(monkeypatch, tmp_path, PB, task, protocol="function_calling")
    rig.mrn = rig.task()["context"]["patient_ref"]
    return rig


def _labs(rig, **extra):
    return L.fc_tool(LABS, dict({"patient": "Patient/" + rig.mrn}, **extra))


def _spy_kernel_origin(monkeypatch):
    """(hook, tool, forced, origin, requester) for every kernel before_action / after_action call (a pass-through)."""
    import harness.kernel as K
    calls = []
    before, after = K.HarnessKernel.before_action, K.HarnessKernel.after_action

    def before_action(self, action, env_state=None, step=0, **kw):
        calls.append(("before_action", action.get("tool"), bool(action.get("_harness_acquire")), kw.get("origin"),
                      kw.get("requester")))
        return before(self, action, env_state, step=step, **kw)

    def after_action(self, action, *a, **kw):
        calls.append(("after_action", action.get("tool"), bool(action.get("_harness_acquire")), kw.get("origin"),
                      kw.get("requester")))
        return after(self, action, *a, **kw)
    monkeypatch.setattr(K.HarnessKernel, "before_action", before_action)
    monkeypatch.setattr(K.HarnessKernel, "after_action", after_action)
    return calls


def _capture_scoring_ctx(rig):
    seen = []
    stub = rig.R.scoring.run_checkpoint

    def run_checkpoint(cp, ctx):
        seen.append(ctx)
        return stub(cp, ctx)
    rig.mp.setattr(rig.R.scoring, "run_checkpoint", run_checkpoint)
    return seen


def test_a_forced_read_is_harness_origin_in_the_kernel_and_not_an_agent_scoring_input(monkeypatch, tmp_path):
    rig = _pb(monkeypatch, tmp_path)
    kernel_calls = _spy_kernel_origin(monkeypatch)
    ctxs = _capture_scoring_ctx(rig)
    broad = {"patient": "Patient/" + rig.mrn, "count": 50}    # the rig's FHIR stand-in answers every search empty
    rig.replies = [_labs(rig, code="94500-6"), L.fc_final("answer 1")]
    rig.run(max_steps=10)

    calls = rig.events("tool_call")
    assert [(e["origin"], e.get("requester"), e.get("requester_stage"), e.get("requester_capability"), e["args"])
            for e in calls] == [("agent", None, None, None, {"patient": "Patient/" + rig.mrn, "code": "94500-6"}),
                                ("harness", "kernel_acquire", "after_action", BROAD_REACQUIRE, broad)]
    # the kernel sees the forced read as the harness's, before and after it runs
    assert [c for c in kernel_calls if c[1] == LABS] == [
        ("before_action", LABS, False, None, None), ("after_action", LABS, False, None, None),
        ("before_action", LABS, True, "harness", "kernel_acquire"),
        ("after_action", LABS, True, "harness", "kernel_acquire")]
    rows = [(p.get("action_origin"), p.get("action_requester")) for p in rig.kernel.ledger.proposed_actions]
    assert rows[:2] == [(None, None), ("harness", "kernel_acquire")]

    # scoring inputs: the agent-credited view has only the agent's call; the harness view has the forced read
    job = tmp_path / "job"
    log = [json.loads(x) for x in open(str(job / "logs" / "agent" / "trajectory.log")).read().splitlines()]
    assert [x["metadata"]["input"] for x in log] == [calls[0]["args"]]
    hlog = [json.loads(x) for x in open(str(job / "logs" / "harness" / "harness_tool_calls.jsonl")).read().splitlines()]
    assert [(x["metadata"]["requester"], x["metadata"]["input"]) for x in hlog] == [("kernel_acquire", broad)]
    assert ctxs, "no checkpoint was scored"
    ctx = ctxs[0]
    assert ctx["agent_tool_calls"] == [(LABS, calls[0]["args"])]
    assert [e["event_type"] for e in ctx["agent_trajectory"]] == ["tool_call", "answer_attempt", "final_answer"]
    assert ctx["final_texts"] == ["answer 1"]

    cr = rig.result["credit"]
    assert "error" not in cr, cr
    assert cr["origin_counts"]["tool_call"] == {"agent": 1, "harness": 1}
    assert cr["origin_counts"]["tool_call_harness_by_requester"] == {"kernel_acquire": 1}
    assert cr["final_answer_author"] == rig.result["final_answer_author"] == "agent"


def test_no_new_progress_is_not_charged_on_the_harness_read(monkeypatch, tmp_path):
    """The agent reads labs (an empty read closes check_labs: progress), then repeats the labs search with a filter (no
    progress, the first repeat); the harness's broad re-read of labs follows (no progress either). Without origin
    tagging the kernel would count the harness's read as the agent's second no-progress repeat, raise no_new_progress
    on it and hand the agent a REVISE repair turn for it. A harness-origin action is not the agent's progress to judge."""
    rig = _pb(monkeypatch, tmp_path)
    rig.tool_result = lambda name, args: {"entries": [], "total": 0}
    rig.replies = [_labs(rig), _labs(rig, code="94500-6"), L.fc_final("answer 1")]
    rig.run(max_steps=10)

    npg = [e for e in rig.events("harness_decision") if e.get("rule_id") == "no_new_progress"]
    assert npg == [], npg
    assert [f for f in rig.kernel.ledger.findings if f.get("reason_code") == "no_new_progress"] == []
    assert [r for r in rig.actuation_records() if r["stage"] == "after_action"] == []     # no repair turn charged
    assert (rig.feedback_at(2) or {}).get("rule_id") != "no_new_progress"
    assert [(e["origin"], e.get("requester")) for e in rig.events("tool_call")] == [
        ("agent", None), ("agent", None), ("harness", "kernel_acquire")]


def _write(content="plan"):
    return L.fc_tool("write_file", {"file_path": DELIVERABLE, "content": content})


def test_a_deliverable_only_the_forced_write_produced_is_harness_completed(monkeypatch, tmp_path):
    rig = _pb(monkeypatch, tmp_path, task=DELIVERABLE_TASK)
    kernel_calls = _spy_kernel_origin(monkeypatch)
    rig.tool_result = lambda name, args: {"entries": [], "total": 0}
    # the agent never writes the deliverable; the deliverable middleware turns its finals back (pre_final_nudge x3),
    # then after_episode forces one more turn, which the agent spends on the write
    rig.replies = [_labs(rig)] + [L.fc_final("answer %d" % i) for i in range(1, 5)] + [_write("the plan")]
    rig.run(max_steps=10)

    forced = [e for e in rig.events("tool_call") if e.get("forced_deliverable")]
    assert [(e["origin"], e["requester"], e["content_author"]) for e in forced] == \
        [("harness", "forced_deliverable_write", "agent")]
    assert ("before_action", "write_file", False, "harness", "forced_deliverable_write") in kernel_calls
    assert os.path.isfile(str(tmp_path / "job" / "workspace" / "output" / "management_plan.txt"))
    log = open(str(tmp_path / "job" / "logs" / "agent" / "trajectory.log")).read().splitlines()
    assert [json.loads(x)["content"] for x in log] == [LABS]              # the forced write is not the agent's call

    cr = rig.result["credit"]
    assert "error" not in cr, cr
    assert cr["harness_written"]["files_harness_completed"] == ["management_plan.txt"]
    assert cr["output_files"]["management_plan.txt"]["agent_writes"] == 0
    task = rig.task()
    reads_plan = [cp["id"] for cp in task["checkpoints"] if cp.get("type") == "native_pytest"
                  and (cr["checkpoints"][cp["id"]].get("test_inputs") or {}).get("reads_output_dir")]
    assert reads_plan, "the task's native tests read the deliverable"
    assert {cr["checkpoints"][c]["credit"] for c in reads_plan} == {"harness"}
    others = [cp["id"] for cp in task["checkpoints"] if cp.get("type") == "native_pytest" and cp["id"] not in reads_plan]
    assert {cr["checkpoints"][c]["credit"] for c in others} <= {"agent"}

    # the aggregation hook: the native outcome rule with every checkpoint passing is achieved; the agent-credited one is
    # not, because the Outcome checkpoints reading the harness-completed deliverable do not count for the agent
    import credit
    res = copy.deepcopy(rig.result)
    for c in res["checkpoints"]:
        c["checkpoint_status"] = c["pass_status"] = "passed"
    assert credit.native_outcome(res, PB) is True
    assert credit.agent_credited_outcome(res, PB) is False
    att = credit.attribution(res, PB)
    assert set(reads_plan) <= set(att["harness"]) and att["harness_completed_files"] == ["management_plan.txt"]


def test_an_agent_written_deliverable_is_agent_credited(monkeypatch, tmp_path):
    rig = _pb(monkeypatch, tmp_path, task=DELIVERABLE_TASK)
    rig.tool_result = lambda name, args: {"entries": [], "total": 0}
    rig.replies = [_labs(rig), _write("the plan"), L.fc_final("answer 1")]
    rig.run(max_steps=10)

    cr = rig.result["credit"]
    assert [e["origin"] for e in rig.events("tool_call")] == ["agent", "agent"]
    assert cr["harness_written"]["files_harness_completed"] == []
    assert {r["credit"] for r in cr["checkpoints"].values() if "test_inputs" in r} == {"agent"}
    import credit
    res = copy.deepcopy(rig.result)
    for c in res["checkpoints"]:
        c["checkpoint_status"] = c["pass_status"] = "passed"
    assert credit.native_outcome(res, PB) is True and credit.agent_credited_outcome(res, PB) is True
