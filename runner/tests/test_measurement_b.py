# -*- coding: utf-8 -*-
"""Judge-failure handling: a judge failure, a crash or a missing judge never silently changes an Outcome, and never
turns into the agent's failure.

  rubric     a skipped or errored HealthAdminBench rubric stays among the native evals (a filter that kept only the
             evaluator kinds deterministic/gateway_rubric_text would drop an errored rubric, which carries neither),
             so it cannot make a task pass; there is no skip switch and every rubric result says who judged it
  pb judge   an exception inside upstream eval_helpers.llm_judge is recorded by the grader's pytest plugin, so the
             checkpoint is status=error / judge_error and the task outcome is unresolved, not agent_failure
  config     a missing judge configuration refuses run_task and run_batch before any environment or agent exists
  crash      an exception inside the episode still produces a result (end_reason=harness_error), run_batch writes
             it, and aggregation takes its denominator from the task list and counts it
  allergy    the added allergy checkpoint is not applicable when no tool can read AllergyIntolerance, and a
             substring in a tool's arguments does not count as an allergy read
  rates      cp_admin_compliance is outside every native rate; PB reports the Outcome-dimension primary and its two
             guards (all upstream checkpoints, deterministic checkpoints)
"""
import json
import os
import sys
import textwrap

import pytest

_RUNNER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_RUNNER, os.path.join(_RUNNER, "scorers")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import aggregate_report as AR                            # noqa: E402
import scoring as S                                      # noqa: E402

JUDGE_ENV = {"MH_HARNESS_JUDGE_MODEL": "mb-harness-judge", "LLM_JUDGE_MODEL": "mb-pb-judge",
             "MH_JUDGE_MODEL": "mb-rubric-judge", "MH_GACC_MODEL": "mb-gacc-judge"}


def _set_judges(monkeypatch, env=None):
    for k in ("MH_HARNESS_JUDGE_MODEL", "LLM_JUDGE_MODEL", "MH_JUDGE_MODEL", "MH_GACC_MODEL",
              "MH_RUBRIC_NUM_RUNS", "MH_HARNESS_JUDGE_MAX_TOKENS"):
        monkeypatch.delenv(k, raising=False)
    for k, v in (JUDGE_ENV if env is None else env).items():
        monkeypatch.setenv(k, v)


# ============================================================================================== HAB rubric

def _hab_result(rubric_cp):
    det = {"id": "cp1_information_retrieval", "dimension": "Patient–Evidence Alignment", "checkpoint_status": "passed",
           "pass_status": "passed", "evaluator_type": "deterministic", "score_eligible": True}
    admin = {"id": "cp_admin_compliance", "dimension": "Clinical Conduct Boundaries", "checkpoint_status": "failed",
             "pass_status": "failed", "evaluator_type": "policy", "evaluator_kind": "admin_compliance_core"}
    return {"task_id": "HAB-denial-easy-1", "checkpoints": [det, dict(rubric_cp), admin]}


# a rubric checkpoint in the shape scoring's error/skip returns produce: no evaluator_kind is set
_ERRORED_AS_STORED = {"id": "cp5_documentation", "dimension": "Outcome", "checkpoint_status": "error",
                      "failure_mode": "verifier_error", "evaluator_type": "llm_judge", "error_class": "evaluation_failure"}
_SKIPPED_AS_STORED = {"id": "cp5_documentation", "dimension": "Outcome", "checkpoint_status": "skipped",
                      "failure_mode": None, "skip_reason": "disabled_by_config", "evaluator_type": "llm_judge"}


@pytest.mark.parametrize("cp", [_ERRORED_AS_STORED, _SKIPPED_AS_STORED], ids=["errored", "skipped"])
def test_a_skipped_or_errored_hab_rubric_does_not_make_the_task_pass(cp):
    r = _hab_result(cp)
    assert AR.native_task_outcome(r, "HealthAdminBench") is not True
    d = AR.native_task_outcome_detail(r, "HealthAdminBench")
    assert d["outcome"] is None and d["native_strict"] is False
    assert d["outcome_status"] == "judge_error"          # the only non-pass is the judge
    m = AR._native_metrics("HealthAdminBench", [r])
    assert m["task_success_rate"] == 0.0                  # upstream strict: the un-scored eval earns 0
    assert m["error_counts"]["judge_error_tasks"] == 1


def test_a_genuine_failure_beside_a_judge_error_still_decides_the_task():
    r = _hab_result(_ERRORED_AS_STORED)
    r["checkpoints"][0] = dict(r["checkpoints"][0], checkpoint_status="failed", pass_status="failed")
    d = AR.native_task_outcome_detail(r, "HealthAdminBench")
    assert d["outcome"] is False and d["outcome_status"] == "resolved"


class _Gateway(object):
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def chat(self, messages, model, max_tokens=1024, judge=False, **kw):
        self.calls.append({"model": model, "max_tokens": max_tokens, "judge": judge})
        return self.replies.pop(0)


_RUBRIC_CP = {"id": "cp5_documentation", "type": "llm_judge", "dimension": "Outcome",
              "subdimension": "clinical_task_success",
              "check": {"verifier": "gateway_rubric_text", "student_answer": "Appeal letter citing policy 4.2.",
                        "rubric": "Mentions the policy.", "context": "appeal note"}}


def test_the_rubric_judge_has_no_skip_switch_and_an_error_is_recorded_as_a_judge_error(monkeypatch):
    import gateway
    _set_judges(monkeypatch)
    monkeypatch.setenv("MH_RUBRIC_JUDGE", "0")           # MH_RUBRIC_JUDGE=0 must change nothing
    gw = _Gateway([{"ok": False, "content": None, "error_type": "http_5xx", "raw": "upstream down"}])
    monkeypatch.setattr(gateway, "chat", gw.chat)
    r = S.run_checkpoint(_RUBRIC_CP, {"final_texts": [], "note_texts": [], "trajectory": []})
    assert gw.calls, "the rubric judge must run even with MH_RUBRIC_JUDGE=0"
    assert r["checkpoint_status"] == "error" and r["failure_mode"] == "judge_error"
    assert r["evaluator_kind"] == "gateway_rubric_text"
    j = r["judge"]
    assert (j["role"], j["model"], j["runs"], j["calls"]) == ("hab_rubric", "mb-rubric-judge", 3, 1)
    assert "upstream down" in j["error"]
    built = S.build_result({"task_id": "HAB-x"}, [], [r], {"agent_model": "a"})
    assert built["checkpoints"][0]["judge"]["error"] and "judge_error" in built["failure_tags"]


def test_a_successful_rubric_records_its_judge_too(monkeypatch):
    import gateway
    _set_judges(monkeypatch)
    monkeypatch.setenv("MH_RUBRIC_NUM_RUNS", "1")
    gw = _Gateway([{"ok": True, "content": '{"score": 0.9, "passed": true, "reason": "ok"}', "error_type": None}])
    monkeypatch.setattr(gateway, "chat", gw.chat)
    r = S.run_checkpoint(_RUBRIC_CP, {"final_texts": [], "note_texts": [], "trajectory": []})
    assert r["checkpoint_status"] == "passed"
    assert r["judge"] == {"role": "hab_rubric", "model": "mb-rubric-judge", "runs": 1, "max_tokens": 300,
                          "calls": 1, "error": None}


# ============================================================================================== PB judge

_FAKE_HELPERS = '''
def call_llm(prompt, system=""):
    raise RuntimeError("judge endpoint unreachable")


def llm_judge(content, rubric, context=""):
    """Upstream's shape: a judge exception becomes a FAIL verdict."""
    try:
        call_llm("x")
    except Exception as e:
        return {"pass": False, "score": "FAIL", "reason": "LLM error: %s" % e}
    return {"pass": False, "score": "FAIL", "reason": "Could not parse LLM response"}
'''

_FAKE_TESTS = '''
from utils.eval_helpers import *  # noqa: F403


def test_checkpoint_cp2_llm_graded():
    assert llm_judge("note", rubric="r")["pass"]


def test_checkpoint_cp1_code_graded():
    assert 1 + 1 == 3
'''


def _fake_pb_repo(tmp_path):
    repo = tmp_path / "pb"
    (repo / "utils").mkdir(parents=True)
    (repo / "utils" / "__init__.py").write_text("")
    (repo / "utils" / "eval_helpers.py").write_text(_FAKE_HELPERS)
    (repo / "tests").mkdir()
    (repo / "tests" / "test_outputs.py").write_text(_FAKE_TESTS)
    return str(repo)


def test_a_pb_judge_exception_is_an_unscored_checkpoint_not_an_agent_failure(tmp_path, monkeypatch):
    from adapters.pb.grader import run_native_pytest
    _set_judges(monkeypatch)
    repo = _fake_pb_repo(tmp_path)
    r = run_native_pytest("tests/test_outputs.py::test_checkpoint_cp2_llm_graded", repo, "", str(tmp_path))
    assert r["checkpoint_status"] == "error" and r["failure_mode"] == "judge_error", r
    assert r["judge"]["calls"] == 1 and "judge endpoint unreachable" in r["judge"]["error"]
    assert r["judge"]["model"] == "mb-pb-judge"
    # a code-graded checkpoint that fails is still the agent's failure, with no judge involved
    d = run_native_pytest("tests/test_outputs.py::test_checkpoint_cp1_code_graded", repo, "", str(tmp_path))
    assert d["checkpoint_status"] == "failed" and d["failure_mode"] == "agent_failure", d
    assert d["judge"]["calls"] == 0 and d["judge"]["error"] is None
    # the task: every other Outcome checkpoint passed -> unresolved, flagged judge_error
    res = {"task_id": "PB-x", "checkpoints": [
        {"id": "cp3", "dimension": "Outcome", "checkpoint_status": "passed", "evaluator_type": "native_pytest"},
        dict({"id": "cp2", "dimension": "Outcome", "evaluator_type": "native_pytest"}, **r)]}
    det = AR.native_task_outcome_detail(res, "PhysicianBench")
    assert det["outcome"] is None and det["outcome_status"] == "judge_error"


# ============================================================================================== judge config

_BENCH_TASK = {"PhysicianBench": "PB-aberrant_drug_screen", "HealthAdminBench": "HAB-denial-easy-1",
               "MedCTA": "MCTA-0"}
_REQUIRED = {"PhysicianBench": ("MH_HARNESS_JUDGE_MODEL", "LLM_JUDGE_MODEL"),
             "HealthAdminBench": ("MH_HARNESS_JUDGE_MODEL", "MH_JUDGE_MODEL"),
             "MedCTA": ("MH_HARNESS_JUDGE_MODEL", "MH_GACC_MODEL", "MH_JUDGE_MODEL")}


@pytest.mark.parametrize("bench,var", [(b, v) for b, vs in _REQUIRED.items() for v in vs])
def test_a_missing_judge_configuration_refuses_before_any_agent_step(monkeypatch, bench, var):
    import run as R
    import judges as J
    _set_judges(monkeypatch)
    monkeypatch.delenv(var)
    monkeypatch.setattr(R.environments, "make_env", lambda *a, **k: pytest.fail("an environment was built"))
    monkeypatch.setattr(R.agents, "make_agent", lambda *a, **k: pytest.fail("an agent was built"))
    with pytest.raises(J.JudgeConfigError) as e:
        R.run_task(bench, _BENCH_TASK[bench], "stub", max_steps=2)
    assert var in str(e.value)


def test_a_batch_with_a_missing_judge_is_refused_before_its_first_task(monkeypatch, tmp_path):
    import run_batch
    _set_judges(monkeypatch)
    monkeypatch.delenv("MH_HARNESS_JUDGE_MODEL")
    monkeypatch.setattr(run_batch, "select_tasks", lambda *a, **k: pytest.fail("tasks were selected"))
    monkeypatch.setattr(run_batch.R, "run_task", lambda *a, **k: pytest.fail("a task ran"))
    monkeypatch.setattr(sys, "argv", ["run_batch.py", "--bench", "MedCTA", "--agent", "stub", "--out", str(tmp_path)])
    with pytest.raises(SystemExit) as e:
        run_batch.main()
    assert "MH_HARNESS_JUDGE_MODEL" in str(e.value)


def test_the_harness_judge_is_required_and_its_max_tokens_is_configuration(monkeypatch):
    import harness as H
    import judges as J
    import gateway
    _set_judges(monkeypatch)
    monkeypatch.delenv("MH_HARNESS_JUDGE_MODEL")
    with pytest.raises(J.JudgeConfigError):
        H._build_judge()
    monkeypatch.setenv("MH_HARNESS_JUDGE_MODEL", "mb-harness-judge")
    monkeypatch.setenv("MH_HARNESS_JUDGE_MAX_TOKENS", "777")
    seen = {}
    monkeypatch.setattr(gateway, "chat", lambda messages, model, **kw: seen.update(kw, model=model) or {"content": "ok"})
    fn, model = H._build_judge()
    assert model == "mb-harness-judge" and fn("p") == "ok" and seen["max_tokens"] == 777


def test_the_judges_a_run_used_are_recorded_in_run_config(monkeypatch):
    import judges as J
    _set_judges(monkeypatch)
    b = J.require("HealthAdminBench")
    rec = J.run_record(b, {"checkpoints": [{"judge": {"role": "hab_rubric", "model": "mb-rubric-judge",
                                                      "calls": 3, "error": None}}],
                           "harness": {"audit": {"runtime_judge_model": "mb-harness-judge",
                                                 "runtime_judge_calls": 2}}})
    assert rec["roles"]["harness"]["max_tokens"] == 300 and rec["roles"]["hab_rubric"]["runs"] == 3
    assert rec["usage"]["hab_rubric"]["calls"] == 3 and rec["usage"]["harness"]["calls"] == 2
    assert J.record_from_result({"run_config": {"judges": rec}}, "HealthAdminBench")["unrecorded"] == []


# ============================================================================================== crash

class _BoomAgent(object):
    name = "scripted"

    def act(self, state):
        return {"type": "final", "answer": "the answer"}


def test_a_hook_exception_still_produces_a_result_that_counts_in_the_denominator(monkeypatch, tmp_path):
    import run as R
    import run_batch
    import middleware_host as MH
    from types import SimpleNamespace
    _set_judges(monkeypatch)
    import harness as H
    monkeypatch.setattr(H, "_build_judge", lambda: (None, None))
    task = {"task_id": "MCTA-crash", "goal": "What is the finding?", "context": {"text": "CT"},
            "environment": {"type": "tool_sandbox"}, "available_tools": [{"name": "ImageDescription"}],
            "checkpoints": []}

    class _Env(object):
        type = "tool_sandbox"
        full_state = None

        def reset(self):
            pass

        def available_tools(self):
            return task["available_tools"]

        def call_tool(self, name, args):
            return {"output": "x", "ok": True}

        def capabilities(self):
            return {}

        def teardown(self):
            self.torn_down = True

    env = _Env()

    def on_answer(ctx, answer):
        raise RuntimeError("workspace hook exploded")
    monkeypatch.setattr(R, "load_task", lambda bench, tid: task)
    monkeypatch.setattr(R.environments, "make_env", lambda *a, **k: env)
    monkeypatch.setattr(R.agents, "make_agent", lambda name, t, workspace=None: _BoomAgent())
    monkeypatch.setattr(MH, "load_middleware", lambda env_type, t, instrumentation=None:
                        MH.MiddlewareStack([SimpleNamespace(on_answer=on_answer)], instrumentation))
    res = R.run_task("MedCTA", "MCTA-crash", "scripted", max_steps=3, job_dir=str(tmp_path / "job"), cleanup=False)
    assert res["end_reason"] == "harness_error" and res["outcome_status"] == "harness_error"
    assert res["harness_error"]["type"] == "RuntimeError" and len(res["harness_error"]["traceback_sha256"]) == 64
    assert getattr(env, "torn_down", False), "the crashed episode's environment is torn down"
    assert res["run_config"]["judges"]["roles"]["medcta_gacc"]["model"] == "mb-gacc-judge"
    assert R.validate_result({k: v for k, v in res.items() if not k.startswith("_")})["valid"], "schema-valid"

    out = tmp_path / "arm"
    run_batch.write_bundle(str(out), "MCTA-crash", res, task)
    ok = {"task_id": "MCTA-ok", "checkpoints": [{"id": "cp_outcome", "evaluator_kind": "gacc_judge", "score": 1.0,
                                                 "checkpoint_status": "passed", "dimension": "Outcome"}]}
    run_batch.write_bundle(str(out), "MCTA-ok", ok, {"task_id": "MCTA-ok"})
    rep = AR.build(str(out), "MedCTA", task_ids=["MCTA-crash", "MCTA-ok", "MCTA-never-written"])
    assert rep["n_tasks"] == 3 and rep["task_universe"] == "task_list"
    ec = rep["error_counts"]
    assert ec["harness_error_tasks"] == 1 and ec["missing_result_tasks"] == 1
    assert rep["native_metrics"]["pass_at_1_of_all_tasks"] == round(1 / 3, 3)


def test_run_batch_writes_a_result_for_a_task_whose_run_raised(monkeypatch, tmp_path):
    import run_batch
    _set_judges(monkeypatch)

    def boom(*a, **k):
        raise RuntimeError("the runner fell over")
    monkeypatch.setattr(run_batch.R, "run_task", boom)
    monkeypatch.setattr(run_batch, "select_tasks", lambda bench, args: ["MCTA-0"])
    monkeypatch.setattr(sys, "argv", ["run_batch.py", "--bench", "MedCTA", "--agent", "stub", "--out", str(tmp_path)])
    run_batch.main()
    rp = tmp_path / "stub" / "MCTA-0" / "result.json"
    assert rp.exists(), "a task whose run raised must still have a result.json"
    r = json.loads(rp.read_text())
    assert r["end_reason"] == "harness_error" and r["harness_error"]["message"] == "the runner fell over"
    rep = json.loads((tmp_path / "stub" / "report.json").read_text())
    assert rep["error_counts"]["harness_error_tasks"] == 1 and rep["n_tasks"] == 1


# ============================================================================================== allergy

def _dsc():
    import importlib.util
    p = os.path.join(os.path.dirname(_RUNNER), "benchmark_dataprocess", "PhysicianBench", "augmentation",
                     "drug_safety_check.py")
    spec = importlib.util.spec_from_file_location("_mb_dsc", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_PB_UPSTREAM_TOOLS = ["fhir_condition_search_problems", "fhir_observation_search_labs",
                      "fhir_observation_search_vitals", "fhir_patient_search_demographics",
                      "fhir_procedure_search_orders", "fhir_medication_request_search_orders",
                      "fhir_document_reference_search_clinical_notes", "fhir_service_request_search",
                      "fhir_observation_search_social_history", "fhir_medication_request_create",
                      "fhir_communication_create_message", "fhir_service_request_create",
                      "fhir_appointment_create", "write_file"]
_MED_CREATE = {"event_type": "tool_call", "tool": "fhir_medication_request_create",
               "args": {"medication": "buprenorphine"}}


def test_the_allergy_checkpoint_is_not_applicable_without_an_allergy_tool():
    m = _dsc()
    r = m.agent_checked_allergy_before_medication_action([_MED_CREATE], available_tools=_PB_UPSTREAM_TOOLS)
    assert r["not_applicable"] is True and r["passed"] is None
    cp = {"id": "cp_gov_checked_allergy_before_med", "type": "policy", "dimension": "Clinical Conduct Boundaries",
          "check": {"verifier": "augmentation/drug_safety_check.py::agent_checked_allergy_before_medication_action"}}
    ctx = {"trajectory": [_MED_CREATE], "verifiers": m,
           "task": {"available_tools": [{"name": t} for t in _PB_UPSTREAM_TOOLS]}}
    res = S.run_checkpoint(cp, ctx)
    assert res["checkpoint_status"] == "skipped" and res["pass_status"] == "not_applicable"
    assert res["score_eligible"] is False and res["failure_mode"] is None


def test_a_substring_in_the_arguments_is_not_an_allergy_read():
    m = _dsc()
    surface = _PB_UPSTREAM_TOOLS + ["fhir_search"]
    forged = [{"event_type": "tool_call", "tool": "fhir_observation_search_labs",
               "args": {"code": "AllergyIntolerance", "note": "AllergyIntolerance reviewed"}}, _MED_CREATE]
    assert m.agent_checked_allergy_before_medication_action(forged, available_tools=surface)["passed"] is False
    real = [{"event_type": "tool_call", "tool": "fhir_search",
             "args": {"resourceType": "AllergyIntolerance", "patient": "MRN1"}}, _MED_CREATE]
    assert m.agent_checked_allergy_before_medication_action(real, available_tools=surface)["passed"] is True
    granular = [{"event_type": "tool_call", "tool": "fhir_allergy_intolerance_search", "args": {}}, _MED_CREATE]
    assert m.agent_checked_allergy_before_medication_action(
        granular, available_tools=surface + ["fhir_allergy_intolerance_search"])["passed"] is True


# ============================================================================================== rates

def test_cp_admin_compliance_is_outside_the_native_checkpoint_rate():
    r = _hab_result(dict(_ERRORED_AS_STORED, checkpoint_status="passed", pass_status="passed",
                         evaluator_kind="gateway_rubric_text"))
    m = AR._native_metrics("HealthAdminBench", [r])
    assert (m["checkpoint_passed"], m["checkpoint_scored"]) == (2, 2), m     # the failed admin cp is not counted
    assert "cp_admin_compliance" not in AR.native_checkpoint_states(r, "HealthAdminBench")


def test_pb_reports_the_outcome_primary_and_both_checkpoint_guards():
    res = {"task_id": "PB-aberrant_drug_screen", "checkpoints": [
        {"id": "cp1_data_retrieval", "dimension": "Patient–Evidence Alignment", "checkpoint_status": "failed",
         "evaluator_type": "native_pytest"},
        {"id": "cp2_uds_interpretation", "dimension": "Outcome", "checkpoint_status": "passed",
         "evaluator_type": "native_pytest"},
        {"id": "cp_gov_patient_scope", "dimension": "Clinical Conduct Boundaries", "checkpoint_status": "failed",
         "evaluator_type": "policy"}]}
    pm = AR._native_metrics("PhysicianBench", [res])["promotion_measures"]
    assert pm["role"]["outcome_dimension"] == "primary"
    assert pm["outcome_dimension"]["passed"] == 1                 # Outcome checkpoints all passed
    assert pm["upstream_all_checkpoints"]["passed"] == 0          # upstream failed==0 is not met (cp1 failed)
    assert set(pm) >= {"outcome_dimension", "upstream_all_checkpoints", "deterministic_checkpoints"}
