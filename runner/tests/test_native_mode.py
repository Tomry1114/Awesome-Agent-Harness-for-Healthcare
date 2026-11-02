"""Native/OFF runs use the same agent and substrate without kernel interventions."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import run as R
import inline_loop_rig as L


def test_native_runs_tool_observation_final_without_kernel(monkeypatch, tmp_path):
    rig = L.Rig(monkeypatch, tmp_path, "MedCTA", "MCTA-0")
    rig.replies = [
        '<tool_call>{"name": "ImageDescription", "arguments": {}}</tool_call>',
        '<answer>Observed image evidence; final answer.</answer>',
    ]
    res = R.run_task(rig.bench, rig.task_id, "gpt5", max_steps=3, job_dir=str(tmp_path / "job"), cleanup=False,
                     execution_mode="native")
    assert res["run_config"]["execution_mode"] == "native"
    assert res["harness"]["status"] == "disabled"
    assert rig.kernels == []
    assert rig.hooks.calls == []
    assert rig.env_calls, res.get("harness_error")
    assert len(rig.model_calls) == 2
    assert "inline-loop observation" in repr(rig.model_calls[1]["messages"])
    assert any(e.get("event_type") == "tool_call" for e in res["_trajectory"])
    assert any(e.get("event_type") == "final_answer" for e in res["_trajectory"])
    assert res["resume_key"]["execution_mode"] == "native"
    assert res.get("schema_validation", {}).get("valid") is True


def test_execution_mode_changes_resume_identity():
    binding = {"harness": {"model": "test"}}
    a = R.resume_key("T", "digest", binding, "stub", 3)
    b = R.resume_key("T", "digest", binding, "stub", 3, execution_mode="native")
    assert a != b


def test_batch_refuses_mixed_mode_bundle(monkeypatch, tmp_path):
    import run_batch as B
    monkeypatch.setattr(B.R, "resolve_workspace", lambda *a: (None, {"digest": "same"}))
    monkeypatch.setattr(B, "stored_result", lambda *a: {"run_config": {"execution_mode": "harness"}})
    with pytest.raises(SystemExit, match="separate --out"):
        B.resume_plan("MedCTA", ["MCTA-0"], {"MCTA-0": {}}, "stub", None, {},
                      lambda tid: 3, str(tmp_path), execution_mode="native")
