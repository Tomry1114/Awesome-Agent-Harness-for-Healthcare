# -*- coding: utf-8 -*-
"""Instrumentation (runner/instrumentation.py): the record of which workspace components a run loaded and which
of their hooks ran. It must see every workspace hook call, never shell code, and never change the run (the trajectory
and result must be identical with the record on and off).
"""
import hashlib
import os
import shutil
import sys

import pytest

_RUNNER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RUNNER not in sys.path:
    sys.path.insert(0, _RUNNER)

import harness as H                                     # noqa: E402
import middleware_host as MH                            # noqa: E402
import workspace as W                                   # noqa: E402
from harness import decision as D                       # noqa: E402
from harness.capabilities.scope_evidence import ScopeEvidenceBinding   # noqa: E402
from instrumentation import Instrumentation, record_components         # noqa: E402


@pytest.fixture(autouse=True)
def _no_harness_judge(monkeypatch):
    """build_kernel binds the harness judge from configuration and refuses without it (runner/judges.py). These
    tests exercise the kernel with no judge wired, so the judge builder is replaced by one that returns none."""
    import harness as _H
    monkeypatch.setattr(_H, "_build_judge", lambda: (None, None))

PB_SEARCH_NONE = {"entries": [], "pages": 0, "total": 0}
WORKSPACE_CAPS = ["goal_alignment", "required_context", "active_evidence", "evidence_coverage",
                  "commitment_execution", "fhir_empty_targeted_search_broad_reacquire"]


def _task():
    return {"task_id": "INSTR", "goal": "Work up suspected osteomyelitis.",
            "context": {"patient_ref": "Patient/1", "text": "case"},
            "available_tools": [{"name": "fhir_condition_search_problems"}, {"name": "write_file"}],
            "environment": {"type": "fhir", "config": {}}, "checkpoints": []}


def test_shell_code_is_never_recorded():
    assert W.component_path(ScopeEvidenceBinding()) is None
    assert Instrumentation().call(ScopeEvidenceBinding(), "before_action", 0, None) is None


def test_every_kernel_middleware_call_is_recorded_and_the_winner_says_what_it_became():
    inst = Instrumentation()
    k = H.build_kernel(_task(), env_type="fhir", workspace=W.H0["fhir"], instrumentation=inst)
    assert [r["name"] for r in inst.calls if r["hook"] == "on_contract"] == WORKSPACE_CAPS
    act = {"type": "tool_call", "tool": "fhir_condition_search_problems", "args": {"patient": "P1", "code": "NARROW"}}
    eff = k.after_action(act, PB_SEARCH_NONE, "h0", "h0", step=7, result_ok=True, raw_observation=PB_SEARCH_NONE,
                         result_status="ok")
    calls = [r for r in inst.calls if r["hook"] == "after_action"]
    assert [r["name"] for r in calls] == WORKSPACE_CAPS, "one record per workspace capability, none for the shell's"
    assert all(r["step"] == 7 and r["path"].startswith("workspaces/pb/h0/middleware/") for r in calls)
    reacq = calls[-1]
    assert reacq["returned"] == D.ACQUIRE and reacq["reason_code"] == "empty_targeted_search"
    assert reacq["won"] is True and reacq["effective"] == eff.type == D.ACQUIRE
    assert all(r["returned"] is None and "won" not in r for r in calls[:-1])


def test_a_hook_that_raises_is_recorded_as_raised(tmp_path):
    path = str(tmp_path / "ws")
    shutil.copytree(W.H0["fhir"], path)
    p = os.path.join(path, "middleware", "commitment_execution.py")
    src = open(p).read()
    head = "    def before_final(self, answer, ctx):\n"
    assert src.count(head) == 1
    open(p, "w").write(src.replace(head, head + "        raise RuntimeError('edited to fail')\n"))
    inst = Instrumentation()
    k = H.build_kernel(_task(), env_type="fhir", workspace=path, instrumentation=inst)
    k.before_final("done", step=3)
    rec = [r for r in inst.calls if r["hook"] == "before_final" and r["name"] == "commitment_execution"][0]
    assert rec["returned"] == D.ESCALATE and "edited to fail" in rec["raised"]
    assert rec["path"] == W.repo_path(p)


class _Env(object):
    def __init__(self, ws):
        self.workspace = str(ws)

    def available_tools(self):
        return ["write_file"]


def test_middleware_calls_are_recorded_with_what_they_returned(tmp_path):
    inst = Instrumentation()
    task = {"goal": "Write your assessment to output/note.md", "context": ""}
    stack = MH.load_middleware(W.H0["fhir"], task, instrumentation=inst)
    ctx = MH.MiddlewareContext(env=_Env(tmp_path), task=task, trajectory=[], step=4, max_steps=100, used_actions=10,
                               judge_factory=lambda: None, errors=[])
    r = stack.before_final(ctx, "done")
    assert r is not None
    assert inst.calls[-1] == {"kind": "middleware_call", "path": "workspaces/pb/h0/middleware/deliverable.py",
                              "name": "DeliverableMiddleware", "hook": "before_final", "step": 4,
                              "returned": "Revision", "stage": "deliverable", "summary": inst.calls[-1]["summary"]}
    assert inst.calls[-1]["summary"]["kind"] == "Revision" and inst.calls[-1]["summary"]["stage"] == "deliverable"
    assert stack.exempt_from_budget(ctx, {"tool": "write_file", "args": {"file_path": "output/note.md"}}) is True
    assert inst.calls[-1]["hook"] == "exempt_from_budget" and inst.calls[-1]["returned"] is True


def test_the_components_a_pb_run_loads_are_listed_once_with_their_digest(monkeypatch):
    monkeypatch.setenv("MH_OPENAI_KEY", "test")
    task = _task()
    agent = W.WorkspaceAgent(task, W.H0["fhir"])
    inst = Instrumentation()
    stack = MH.load_middleware(W.H0["fhir"], task, instrumentation=inst)
    k = H.build_kernel(task, env_type="fhir", workspace=W.H0["fhir"], instrumentation=inst)
    record_components(inst, agent, stack, k, W.H0["fhir"])
    roles = [c["role"] for c in inst.components]
    assert roles.count("manifest") == 1 and roles.count("agent_program") == 1 and roles.count("system_prompt") == 1
    assert roles.count("tool_description") == 2 and roles.count("middleware") == 1
    assert roles.count("kernel_middleware") == 6 and roles.count("policy") == 1
    for c in inst.components:
        assert c["sha256"] == hashlib.sha256(open(os.path.join(W.REPO, c["path"]), "rb").read()).hexdigest()


def test_the_medcta_workspace_lists_its_kernel_middleware_and_its_middleware():
    task = dict(_task(), environment={"type": "tool_sandbox", "config": {}},
                available_tools=[{"name": "ImageDescription"}])
    inst = Instrumentation()
    k = H.build_kernel(task, env_type="tool_sandbox", workspace=W.H0["tool_sandbox"], instrumentation=inst)
    record_components(inst, None, MH.load_middleware(W.H0["tool_sandbox"], task), k, W.H0["tool_sandbox"])
    assert sorted(set(c["role"] for c in inst.components)) == ["kernel_middleware", "manifest", "middleware"]
    assert [c["path"] for c in inst.components if c["role"] == "middleware"] == [
        "workspaces/medcta/h0/middleware/answer_candidate.py"]
    assert all(c["path"].startswith("workspaces/medcta/h0/") for c in inst.components)


def _roles(inst):
    out = {}
    for c in inst.components:
        out.setdefault(c["role"], []).append(c["path"])
    return out


def test_a_hab_run_lists_its_program_prompt_template_fragments_and_tool_descriptions(monkeypatch):
    monkeypatch.setenv("MH_OPENAI_KEY", "test")
    task = {"goal": "g", "context": {"text": "q"}, "available_tools": [{"name": "click"}, {"name": "done"}],
            "environment": {"type": "gui", "config": {"payer_portal": "payer_a", "challenge_type": "denial_triage"}}}
    inst = Instrumentation()
    record_components(inst, W.WorkspaceAgent(task, W.H0["gui"]), None, None, W.H0["gui"])
    roles = _roles(inst)
    assert roles["agent_program"] == ["workspaces/hab/h0/agent.py"]
    assert roles["system_prompt"] == ["workspaces/hab/h0/system_prompt.md"]
    assert roles["prompt_fragment"] == ["workspaces/hab/h0/prompt_fragments/%s.md" % n
                                        for n in ("emr_denials", "payer_a", "shared_execution", "syntax_dom")]
    assert roles["tool_description"] == ["workspaces/hab/h0/tool_descriptions/click.yaml",
                                         "workspaces/hab/h0/tool_descriptions/done.yaml"]
    for c in inst.components:
        assert c["sha256"] == hashlib.sha256(open(os.path.join(W.REPO, c["path"]), "rb").read()).hexdigest()


def test_a_medcta_text_run_lists_the_text_program_and_its_prompt(monkeypatch):
    import models
    monkeypatch.setenv("MH_OPENAI_KEY", "test")
    monkeypatch.setitem(models.table(), "test-text-model", models.TEXT)
    monkeypatch.setenv("MH_API_MODEL", "test-text-model")
    task = {"goal": "g", "context": {"text": "q"}, "environment": {"type": "tool_sandbox"},
            "available_tools": [{"name": "OCR", "signature": "(image)->text"}]}
    inst = Instrumentation()
    record_components(inst, W.WorkspaceAgent(task, W.H0["tool_sandbox"]), None, None, W.H0["tool_sandbox"])
    roles = _roles(inst)
    assert roles["agent_program"] == ["workspaces/medcta/h0/agent_text.py"]
    assert roles["system_prompt"] == ["workspaces/medcta/h0/system_prompt_text.md"]
    assert roles["tool_description"] == ["workspaces/medcta/h0/tool_descriptions/OCR.yaml"]
    assert "prompt_fragment" not in roles
