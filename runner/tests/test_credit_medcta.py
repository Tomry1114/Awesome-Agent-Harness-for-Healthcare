# -*- coding: utf-8 -*-
"""Agent-vs-harness checkpoint credit on MedCTA.

The real run loop (run.run_task) on the middleware-host contract rig of runner/tests/test_middleware_host_contract.py:
a fake tool-sandbox environment, a scripted agent, host middleware returned by the test, the kernel's hooks stubbed.
Scoring is real where it needs no model (the Clinical Resource Competence checkpoint); the outcome checkpoint's GAcc judge is a test double
that scores the text it is handed, so the test reads which text native scoring read. Pinned:

  * a candidate-merge final (an on_answer Accept of a text none of the agent's answers is) and the [WITHHELD] stub are
    recorded with their author, are not the agent's final_texts, and the task is not agent-credited
  * a harness read (next_read) is not in agent_tool_calls: the Clinical Resource Competence checkpoint is scored on the agent's calls only

The tests fail if the final has no author, if final_texts carries the merge or the stub, if agent_tool_calls carries
the harness read, or if no credit record exists.
"""
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_middleware_host_contract as C  # noqa: E402
from test_middleware_host_contract import final, kernel_script, R, MH, H  # noqa: E402

MEDCTA = "MedCTA"
GOLD = "RUL nodule"
CHECKPOINTS = [
    {"id": "cp_outcome", "type": "llm_judge", "dimension": "Outcome", "subdimension": "result_verification",
     "check": {"rubric": "Does the final answer semantically match the gold answer?",
               "whitelist_ref": "reference.gold_answer.whitelist"}},
    {"id": "cp_tool_selection", "type": "deterministic", "dimension": "Clinical Resource Competence", "subdimension": "tool_use_quality",
     "check": {"method": "toolset_contains", "expected_ref": "reference.sufficient_tools"}}]


@pytest.fixture
def run_medcta(monkeypatch, tmp_path):
    for k in ("MH_GACC", "MH_GACC_MODEL", "MH_HARNESS_JUDGE_MODEL", "MH_MM_JUDGE", "MH_MM_JUDGE_MODEL",
              "LLM_JUDGE_MODEL"):
        monkeypatch.delenv(k, raising=False)
    # run_task refuses before any agent step without the bench's judge configuration; configure stub judge
    # names the same way the host-contract rig does (the outcome judge is replaced by the GAcc double below).
    for k in ("MH_HARNESS_JUDGE_MODEL", "LLM_JUDGE_MODEL", "MH_JUDGE_MODEL", "MH_GACC_MODEL"):
        monkeypatch.setenv(k, "contract-stub-judge")
    monkeypatch.setattr(H, "_build_judge", lambda: (None, None))   # no harness judge wired: no gateway call

    def run(actions, middlewares, before_final=None, max_steps=4):
        task = {"task_id": "T", "goal": "What is the finding on the CT?", "context": {"text": "chest CT"},
                "environment": {"type": "tool_sandbox"}, "available_tools": C.TOOLS, "checkpoints": CHECKPOINTS,
                "reference": {"gold_answer": {"whitelist": [[GOLD]]}, "required_tool_groups": [["ImageDescription"]]}}
        agent, env, box, ctxs = C.Agent(actions), C.FakeEnv(), {}, []
        monkeypatch.setattr(R, "load_task", lambda bench, tid: task)
        monkeypatch.setattr(R.environments, "make_env", lambda *a, **k: env)
        monkeypatch.setattr(R.agents, "make_agent", lambda name, t, workspace=None: agent)
        monkeypatch.setattr(MH, "load_middleware",
                            lambda env_type, t, instrumentation=None: MH.MiddlewareStack(middlewares, instrumentation))
        build = H.build_kernel

        def build_stubbed(*a, **k):
            kernel = build(*a, **k)
            kernel.before_action = C.scripted([])
            kernel.after_action = C.scripted([])
            kernel.before_final = before_final or C.scripted([])
            kernel.evaluate_final = kernel.before_final      # verify_final: a query of the same scripted verdicts
            kernel.record_final = C.record_stub
            box["kernel"] = kernel
            return kernel
        monkeypatch.setattr(H, "build_kernel", build_stubbed)
        real = R.scoring.run_checkpoint

        def run_checkpoint(cp, ctx):
            ctxs.append(ctx)
            if cp["id"] != "cp_outcome":
                return real(cp, ctx)
            text = " ".join(ctx["final_texts"])          # a GAcc double: 1.0 when the gold phrase is in the text
            sc = 1.0 if GOLD.lower() in text.lower() else 0.0
            return {"id": cp["id"], "dimension": cp["dimension"], "subdimension": cp["subdimension"], "weight": 1.0,
                    "checkpoint_status": "passed" if sc >= 0.5 else "failed", "score": sc, "score_eligible": True,
                    "evaluator_kind": "gacc_judge", "judge_backend": "gacc-double", "detail": {"text": text}}
        monkeypatch.setattr(R.scoring, "run_checkpoint", run_checkpoint)
        res = R.run_task(MEDCTA, "T", agent_name="scripted", max_steps=max_steps, job_dir=str(tmp_path / "job"),
                         cleanup=False)
        return SimpleNamespace(result=res, traj=C.loop_events(res), ctx=ctxs[0] if ctxs else None, env=env)
    return run


def _final_event(r):
    return [e for e in r.traj if e["event_type"] == "final_answer"][-1]


def test_a_candidate_merge_final_is_not_agent_credited(run_medcta):
    merged = "RUL nodule, spiculated, 3 cm (merged claims)"
    acc = MH.Accept(merged, "adopted_evidence_supported_core_change+claimmerge", "REVISE", True, None)
    r = run_medcta([C.TOOL, final("RUL nodule")], [SimpleNamespace(on_answer=lambda ctx, answer: acc)])

    ev = _final_event(r)
    assert (ev["thought"], ev["origin"], ev["requester"], ev["final_answer_author"]) == \
        (merged, "harness", "candidate_merge", "candidate_merge")
    # native scoring reads the agent's own answer (the loop's no-final rule), not the merge
    assert r.ctx["final_texts"] == ["RUL nodule"]
    cr = r.result["credit"]
    assert "error" not in cr, cr
    assert (cr["final_answer_author"], cr["final_answer"]["final_texts_source"]) == \
        ("candidate_merge", "fallback_first_answer")
    assert cr["final_answer"]["delivered_sha"] != cr["final_answer"]["scored_sha"]
    assert cr["checkpoints"]["cp_outcome"]["credit"] == "harness"
    assert (cr["native_outcome"], cr["agent_credited_outcome"]) == (True, False)


def test_an_agent_authored_accept_is_agent_credited(run_medcta):
    acc = MH.Accept("RUL nodule", "kept_original_candidate_not_clearly_better", "REVISE", True, None)
    r = run_medcta([C.TOOL, final("RUL nodule")], [SimpleNamespace(on_answer=lambda ctx, answer: acc)])
    ev = _final_event(r)
    assert (ev["origin"], ev["final_answer_author"], "requester" in ev) == ("agent", "agent", False)
    cr = r.result["credit"]
    assert cr["checkpoints"]["cp_outcome"]["credit"] == "agent"
    assert (cr["native_outcome"], cr["agent_credited_outcome"]) == (True, True)


def test_a_withheld_stub_final_is_not_agent_credited(run_medcta):
    acc = MH.Accept("RUL nodule", "kept_original_core_flip_without_new_evidence", "REVISE", False, None)
    calls = []
    r = run_medcta([C.TOOL, final("RUL nodule")], [SimpleNamespace(on_answer=lambda ctx, answer: acc)],
                   before_final=kernel_script([("RUL nodule", "contradiction")], calls))
    ev = _final_event(r)
    assert ev["thought"] == MH.WITHHELD_THOUGHT
    assert (ev["origin"], ev["requester"], ev["final_answer_author"]) == ("harness", "withhold", "withheld_stub")
    assert r.ctx["final_texts"] == ["RUL nodule"]          # the stub is not the agent's text
    cr = r.result["credit"]
    assert cr["final_answer_author"] == "withheld_stub"
    assert cr["checkpoints"]["cp_outcome"]["credit"] == "harness"
    assert cr["agent_credited_outcome"] is False


def test_tooling_is_scored_on_the_agents_calls_not_the_harness_read(run_medcta):
    """The reference path needs ImageDescription. The agent only calls OtherTool; the harness's turn-start read calls
    ImageDescription. Without origin tagging the harness's call would satisfy the agent's tool selection."""
    mw = SimpleNamespace(next_read=lambda ctx: MH.Read("ImageDescription", {"region": "r"}) if ctx.step == 1 else None)
    r = run_medcta([{"type": "tool_call", "tool": "OtherTool", "args": {}}, final("RUL nodule")], [mw])

    assert [(e["tool"], e["origin"], e.get("requester")) for e in r.traj if e["event_type"] == "tool_call"] == [
        ("OtherTool", "agent", None), ("ImageDescription", "harness", "next_read")]
    assert r.ctx["agent_tool_calls"] == [("OtherTool", {})]
    tooling = [c for c in r.result["checkpoints"] if c["id"] == "cp_tool_selection"][0]
    assert tooling["checkpoint_status"] == "failed"
    assert tooling["detail"]["used"] == ["OtherTool"]
    cr = r.result["credit"]
    assert cr["checkpoints"]["cp_tool_selection"]["input_view"] == "agent_tool_calls"
    assert cr["origin_counts"]["tool_call"] == {"agent": 1, "harness": 1}
