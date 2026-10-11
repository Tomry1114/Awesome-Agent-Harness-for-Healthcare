# -*- coding: utf-8 -*-
"""The workspace loader and the default (H0) workspaces.

These tests pin the workspace contract: what makes a workspace loadable, that H0's components are the benchmark's own
text, which runs get which agent program, and that a program sees only public task fields."""
import copy
import importlib.util
import json
import os
import shutil
import sys

import pytest
import yaml

_RUNNER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RUNNER not in sys.path:
    sys.path.insert(0, _RUNNER)

import models as M                                      # noqa: E402
import workspace as W                                   # noqa: E402

FC_MODEL, TEXT_MODEL = "test-function-calling-model", "test-text-model"


@pytest.fixture
def model(monkeypatch):
    """model(protocol): make the agent object built next use a stand-in model of that protocol (via its model-table entry)."""
    monkeypatch.setenv("MH_OPENAI_KEY", "test")
    monkeypatch.setitem(M.table(), FC_MODEL, M.FUNCTION_CALLING)
    monkeypatch.setitem(M.table(), TEXT_MODEL, M.TEXT)

    def use(protocol):
        monkeypatch.setenv("MH_API_MODEL", FC_MODEL if protocol == M.FUNCTION_CALLING else TEXT_MODEL)
    return use


def _copy_h0(tmp_path):
    dst = tmp_path / "ws"
    shutil.copytree(W.H0["fhir"], str(dst))
    return str(dst)


def test_h0_loads_and_describes_every_tool_the_pb_env_exposes():
    try:
        from harness import substrate_catalog as SC
        tools = SC.vocabulary("fhir")
    except Exception as exc:
        pytest.skip("PB substrate unavailable: %s" % exc)
    for protocol in W.PROTOCOLS:
        assert W.Workspace(W.H0["fhir"], protocol).check(tools) == [], protocol


def test_h0_tool_descriptions_are_the_upstream_registry_text():
    """H0 carries the benchmark's own description of its tools, verbatim from the upstream PhysicianBench tool registry.
    Another workspace may rewrite these descriptions; H0 may not."""
    try:
        import environments as E
        sys.path.insert(0, E._PB_UPSTREAM)
        from agent.tool_registry import FHIR_TOOL_SCHEMAS, FILE_TOOL_SCHEMAS
    except Exception as exc:
        pytest.skip("upstream registry unavailable: %s" % exc)
    ws = W.Workspace(W.H0["fhir"])
    for sch in list(FHIR_TOOL_SCHEMAS) + list(FILE_TOOL_SCHEMAS):
        d = ws.tool_description(sch["name"])
        assert d["description"] == sch.get("description", "") and d["parameters"] == sch.get("parameters", {}), sch["name"]


def test_the_system_prompt_is_read_without_the_file_newline():
    ws = W.Workspace(W.H0["fhir"])
    assert not ws.system_prompt.endswith("\n")
    assert ws.system_prompt.startswith("You are a clinical AI assistant")


def test_a_missing_tool_description_refuses_the_workspace(tmp_path):
    path = _copy_h0(tmp_path)
    os.remove(os.path.join(path, "tool_descriptions", "write_file.yaml"))
    problems = W.Workspace(path).check(["write_file"])
    assert problems and "write_file" in problems[0]


def test_a_description_that_names_another_tool_is_refused(tmp_path):
    path = _copy_h0(tmp_path)
    p = os.path.join(path, "tool_descriptions", "write_file.yaml")
    doc = yaml.safe_load(open(p))
    doc["name"] = "something_else"
    open(p, "w").write(yaml.safe_dump(doc))
    assert W.Workspace(path).check(["write_file"])


@pytest.mark.parametrize("protocol", W.PROTOCOLS)
def test_a_manifest_without_the_models_agent_program_cannot_run_an_agent(tmp_path, model, protocol):
    path = _copy_h0(tmp_path)
    m = yaml.safe_load(open(os.path.join(path, "manifest.yaml")))
    m.pop("agent_program" if protocol == W.FUNCTION_CALLING else "agent_program_text")
    open(os.path.join(path, "manifest.yaml"), "w").write(yaml.safe_dump(m))
    task = {"goal": "g", "context": {}, "available_tools": [{"name": "write_file"}]}
    model(protocol)
    with pytest.raises(W.WorkspaceError):
        W.WorkspaceAgent(task, path)


def test_an_edited_system_prompt_reaches_the_model(tmp_path, monkeypatch):
    """The point of a workspace: editing one of its files changes what the model is sent, with no code change."""
    path = _copy_h0(tmp_path)
    open(os.path.join(path, "system_prompt.md"), "w").write("EDITED PROMPT\n")
    monkeypatch.setenv("MH_OPENAI_KEY", "test")
    sent = []
    import transport
    monkeypatch.setattr(transport, "chat_fc", lambda messages, tools, *a, **k: sent.append(messages) or {"content": "done"})
    task = {"goal": "g", "context": {"text": "question"},
            "available_tools": [{"name": "write_file", "description": "", "parameters": {}}]}
    agent = W.WorkspaceAgent(task, path)
    assert agent.act({})["type"] == "final"
    assert sent[0][0] == {"role": "system", "content": "EDITED PROMPT"}
    assert sent[0][1] == {"role": "user", "content": "question"}


def test_a_pb_run_of_the_default_model_gets_the_function_calling_workspace_program(monkeypatch):
    import agents
    monkeypatch.delenv("MH_API_MODEL", raising=False)
    monkeypatch.delenv("MH_OPENAI_MODEL", raising=False)
    monkeypatch.setenv("MH_OPENAI_KEY", "test")
    task = {"goal": "g", "context": {}, "environment": {"type": "fhir"},
            "available_tools": [{"name": "write_file"}]}
    a = agents.make_agent("gpt5", task, W.H0["fhir"])
    assert isinstance(a, W.WorkspaceAgent)
    assert a.workspace.path == os.path.abspath(W.H0["fhir"])
    assert (a.model, a.protocol) == ("gpt-5.5", W.FUNCTION_CALLING)


# ------------------------------------------------------------------------------ HealthAdminBench and MedCTA
def _bench_tasks(bench):
    p = os.path.join(W.REPO, "benchmark_dataprocess", bench, "tasks_unified.jsonl")
    if not os.path.isfile(p):
        pytest.skip("%s tasks unavailable" % bench)
    return [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]


def _copy(env_type, tmp_path):
    dst = str(tmp_path / "ws")
    shutil.copytree(W.H0[env_type], dst)
    return dst


@pytest.mark.parametrize("bench,env_type", [("HealthAdminBench", "gui"), ("MedCTA", "tool_sandbox")])
def test_hab_and_medcta_h0_load_and_describe_every_tool_of_every_task(bench, env_type):
    for protocol in W.PROTOCOLS:
        ws = W.Workspace(W.H0[env_type], protocol)
        for t in _bench_tasks(bench):
            assert ws.check([x["name"] for x in t["available_tools"]]) == [], (protocol, t["task_id"])


def test_hab_and_medcta_tool_descriptions_are_the_text_each_task_gives_the_model():
    """H0 carries the benchmark's own description of its tools, exactly as every task's available_tools carries it
    (benchmark_dataprocess/HealthAdminBench/convert.py GUI_TOOLS, benchmark_dataprocess/MedCTA/convert.py ALL_TOOLS)."""
    hab = W.Workspace(W.H0["gui"])
    for t in _bench_tasks("HealthAdminBench"):
        for x in t["available_tools"]:
            assert hab.tool_description(x["name"]) == {"name": x["name"], "description": x["description"],
                                                       "parameters": x["parameters"]}, (t["task_id"], x["name"])
    med = W.Workspace(W.H0["tool_sandbox"])
    for t in _bench_tasks("MedCTA"):
        for x in t["available_tools"]:
            assert med.tool_description(x["name"]) == {"name": x["name"], "signature": x["signature"]}, t["task_id"]


def test_hab_portal_guidance_is_upstreams_assembly_for_every_portal_and_challenge_type():
    p = os.path.join(W.REPO, "benchmark", "HealthAdminBench", "harness", "healthcare_hints.py")
    if not os.path.isfile(p):
        pytest.skip("upstream HealthAdminBench harness unavailable")
    spec = importlib.util.spec_from_file_location("hab_healthcare_hints_under_test", p)
    hints = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hints)
    ws = W.Workspace(W.H0["gui"])
    program = sys.modules[ws.program_class().__module__]
    cases = {(t["environment"]["config"].get("payer_portal"), t["environment"]["config"].get("challenge_type"))
             for t in _bench_tasks("HealthAdminBench")}
    cases |= {(portal, ct) for portal in list(hints._PORTAL_MAP) + ["PAYER_A", "unknown", None]
              for ct in ("denial_triage", "Electronic_Appeals", "workflow", "verify_coverage", None)}
    for portal, ct in sorted(cases, key=str):
        assert program.portal_guidance(ws, portal, ct) == hints.get_hints_for_task(
            portal=portal, task_type=ct, action_space="dom"), (portal, ct)


def test_an_edited_hab_prompt_fragment_reaches_the_model(tmp_path, monkeypatch):
    path = _copy("gui", tmp_path)
    open(os.path.join(path, "prompt_fragments", "payer_b.md"), "w").write("EDITED PAYER B GUIDANCE\n")
    monkeypatch.setenv("MH_OPENAI_KEY", "test")
    sent = []
    import transport
    monkeypatch.setattr(transport, "chat_fc",
                        lambda messages, tools, *a, **k: sent.append(copy.deepcopy(messages)) or {"content": "done"})
    task = {"goal": "g", "context": {"text": "question"}, "available_tools": [{"name": "click"}],
            "environment": {"type": "gui", "config": {"payer_portal": "anthem", "challenge_type": "urgent"}}}
    agent = W.WorkspaceAgent(task, path)
    assert agent.act({}) == {"type": "final", "answer": "done"}
    system = sent[0][0]["content"]
    assert "EDITED PAYER B GUIDANCE" in system and "{portal_guidance}" not in system
    assert system.startswith("You are an autonomous web agent")
    assert sent[0][1] == {"role": "user", "content": "question"}


def test_an_edited_medcta_text_prompt_and_tool_description_reach_the_model(tmp_path, monkeypatch, model):
    path = _copy("tool_sandbox", tmp_path)
    open(os.path.join(path, "system_prompt_text.md"), "w").write("EDITED. Tools:\n{tools}\n")
    open(os.path.join(path, "tool_descriptions", "OCR.yaml"), "w").write("name: OCR\nsignature: (image)->every word\n")
    sent = []
    import transport
    monkeypatch.setattr(transport, "chat_text",
                        lambda messages, *a, **k: sent.append(copy.deepcopy(messages)) or "<answer>done</answer>")
    task = {"goal": "g", "context": {"text": "question"}, "environment": {"type": "tool_sandbox"},
            "available_tools": [{"name": "OCR", "signature": "(image)->text"}]}
    model(W.TEXT)
    agent = W.WorkspaceAgent(task, path)
    assert agent.act({}) == {"type": "final", "answer": "done"}
    assert sent[0][0] == {"role": "system", "content": "EDITED. Tools:\n- OCR : (image)->every word"}


def test_the_medcta_programs_differ_only_by_protocol(model):
    task = {"goal": "g", "context": {"text": "q"}, "environment": {"type": "tool_sandbox"},
            "available_tools": [{"name": "RegionAttributeDescription", "signature": "(image,region)->text"}]}
    model(W.FUNCTION_CALLING)
    fc = W.WorkspaceAgent(task, W.H0["tool_sandbox"])
    model(W.TEXT)
    text = W.WorkspaceAgent(task, W.H0["tool_sandbox"])
    assert fc.program.tools_schema == [{"type": "function", "function": {
        "name": "RegionAttributeDescription", "description": "(image,region)->text",
        "parameters": {"type": "object", "properties": {"region": {"type": "string"}}, "required": []}}}]
    assert fc.program.messages[0]["content"].endswith("give your final answer as plain text when done.")
    assert "- RegionAttributeDescription : (image,region)->text\n" in text.program.messages[0]["content"]
    assert "<tool_call>" in text.program.messages[0]["content"]


def test_the_text_program_reports_a_repaired_markup_variant_through_the_agent(monkeypatch, model):
    import transport
    reply = '<tool_calls>[{"name": "ocr", "arguments": {"image": "x"}}]'
    monkeypatch.setattr(transport, "chat_text", lambda messages, *a, **k: reply)
    task = {"goal": "g", "context": {"text": "q"}, "environment": {"type": "tool_sandbox"},
            "available_tools": [{"name": "OCR", "signature": "(image)->text"}]}
    model(W.TEXT)
    agent = W.WorkspaceAgent(task, W.H0["tool_sandbox"])
    assert agent.act({}) == {"type": "tool_call", "tool": "OCR", "args": {}}
    assert agent._protocol_variant_last == "<tool_calls>"
    agent._protocol_variant_last = None
    assert agent.program.protocol_variant_last is None
    model(W.FUNCTION_CALLING)
    pb = W.WorkspaceAgent({"goal": "g", "context": {}, "available_tools": [{"name": "write_file"}]}, W.H0["fhir"])
    assert pb._protocol_variant_last is None


def test_an_agent_program_is_given_only_public_task_fields(tmp_path, monkeypatch):
    path = _copy("gui", tmp_path)
    open(os.path.join(path, "agent.py"), "w").write(
        "class AgentProgram(object):\n"
        "    def __init__(self, task, workspace, chat):\n"
        "        self.task = task\n"
        "    def act(self, state):\n"
        "        return {'type': 'final', 'answer': ''}\n")
    monkeypatch.setenv("MH_OPENAI_KEY", "test")
    task = {"task_id": "HAB-x", "goal": "g", "context": {"text": "q"}, "available_tools": [{"name": "click"}],
            "environment": {"type": "gui", "config": {"payer_portal": "payer_a", "challenge_type": "denial_triage",
                                                      "possible": False, "native_config": {"denial_id": "DEN-1"},
                                                      "task_type": "denial_triage"}},
            "checkpoints": [{"id": "cp"}], "reference": {"gold": "x"}, "expected_outcome": "x", "policy": {},
            "scoring": {}}
    agent = W.WorkspaceAgent(task, path)
    assert agent.program.task == {"goal": "g", "context": {"text": "q"}, "available_tools": [{"name": "click"}],
                                  "environment": {"type": "gui", "config": {"payer_portal": "payer_a",
                                                                            "challenge_type": "denial_triage"}}}


_TOOLS = {"fhir": [{"name": "write_file"}], "gui": [{"name": "click"}],
          "tool_sandbox": [{"name": "OCR", "signature": "(image)->text"}]}


def _manifest(env_type):
    return yaml.safe_load(open(os.path.join(W.H0[env_type], "manifest.yaml"), encoding="utf-8"))


def test_every_h0_workspace_holds_both_agent_programs():
    """The protocol (function calling or text) is a property of the model, so every benchmark's H0 must be able to run a
    model of either protocol."""
    for env_type in ("fhir", "gui", "tool_sandbox"):
        m = _manifest(env_type)
        for protocol in W.PROTOCOLS:
            ws = W.Workspace(W.H0[env_type], protocol)
            assert ws.check([t["name"] for t in _TOOLS[env_type]]) == [], (env_type, protocol)
            assert callable(getattr(ws.program_class(), "act", None)), (env_type, protocol)
            assert ws.system_prompt.strip(), (env_type, protocol)
        assert m["agent_program"] != m["agent_program_text"] and m["system_prompt"] != m["system_prompt_text"], env_type


@pytest.mark.parametrize("env_type", ["fhir", "gui"])
def test_an_edited_pb_or_hab_text_prompt_reaches_the_model(tmp_path, monkeypatch, model, env_type):
    path = _copy(env_type, tmp_path)
    open(os.path.join(path, "system_prompt_text.md"), "w").write("EDITED TEXT PROMPT. Tools:\n{tools}\n")
    if env_type == "gui":
        open(os.path.join(path, "prompt_fragments", "portal_operating_guidance.md"), "w").write("EDITED GUIDANCE\n")
    sent = []
    import transport
    monkeypatch.setattr(transport, "chat_text",
                        lambda messages, *a, **k: sent.append(copy.deepcopy(messages)) or "<answer>done</answer>")
    task = {"goal": "g", "context": {"text": "question"}, "environment": {"type": env_type, "config": {}},
            "available_tools": _TOOLS[env_type]}
    model(W.TEXT)
    agent = W.WorkspaceAgent(task, path)
    assert agent.act({}) == {"type": "final", "answer": "done"}
    system = sent[0][0]["content"]
    assert system.startswith("EDITED TEXT PROMPT. Tools:\n- %s : " % _TOOLS[env_type][0]["name"]), system
    assert "{tools}" not in system
    if env_type == "gui":
        assert system.endswith("\n\nEDITED GUIDANCE"), system
    assert sent[0][1] == {"role": "user", "content": "question"}


@pytest.mark.parametrize("env_type", ["fhir", "gui", "tool_sandbox"])
@pytest.mark.parametrize("protocol", W.PROTOCOLS)
def test_make_agent_runs_the_program_for_the_models_protocol(model, env_type, protocol):
    import agents
    model(protocol)
    task = {"goal": "g", "context": {}, "environment": {"type": env_type, "config": {}},
            "available_tools": _TOOLS[env_type]}
    a = agents.make_agent("gpt5", task, W.H0[env_type])
    assert isinstance(a, W.WorkspaceAgent)
    assert a.workspace.path == os.path.abspath(W.H0[env_type]) and a.protocol == protocol
    key = "agent_program" if protocol == W.FUNCTION_CALLING else "agent_program_text"
    assert W.component_path(a.program) == os.path.join(os.path.abspath(W.H0[env_type]), _manifest(env_type)[key])


@pytest.mark.parametrize("name", sorted(M.table()))
def test_each_listed_model_selects_its_protocol(monkeypatch, name):
    import agents
    monkeypatch.setenv("MH_OPENAI_KEY", "test")
    monkeypatch.setenv("MH_API_MODEL", name)
    a = agents.make_agent("openai", {"goal": "g", "context": {}, "environment": {"type": "fhir", "config": {}},
                                     "available_tools": _TOOLS["fhir"]}, W.H0["fhir"])
    assert (a.model, a.protocol) == (name, M.table()[name])


def test_an_unknown_model_fails_closed(monkeypatch, tmp_path):
    import agents
    import run_batch
    monkeypatch.setenv("MH_OPENAI_KEY", "test")
    monkeypatch.setenv("MH_API_MODEL", "no-such-model")
    task = {"goal": "g", "context": {}, "environment": {"type": "fhir", "config": {}}, "available_tools": _TOOLS["fhir"]}
    with pytest.raises(M.UnknownModelError):
        agents.make_agent("gpt5", task, W.H0["fhir"])
    with pytest.raises(M.UnknownModelError):
        W.WorkspaceAgent(task, W.H0["fhir"])
    # a batch stops before its first task
    monkeypatch.setattr(run_batch.R, "run_task", lambda *a, **k: pytest.fail("a task ran for an unknown model"))
    monkeypatch.setattr(run_batch, "select_tasks", lambda *a, **k: pytest.fail("tasks selected for an unknown model"))
    monkeypatch.setattr(sys, "argv", ["run_batch.py", "--bench", "MedCTA", "--agent", "gpt5", "--out", str(tmp_path)])
    with pytest.raises(M.UnknownModelError):
        run_batch.main()


def test_an_unknown_protocol_or_a_missing_text_program_refuses_the_workspace(tmp_path, model):
    with pytest.raises(W.WorkspaceError):
        W.Workspace(W.H0["tool_sandbox"], "xml")
    path = _copy("gui", tmp_path)
    m = yaml.safe_load(open(os.path.join(path, "manifest.yaml")))
    m.pop("agent_program_text")
    open(os.path.join(path, "manifest.yaml"), "w").write(yaml.safe_dump(m))
    model(W.TEXT)
    with pytest.raises(W.WorkspaceError):
        W.WorkspaceAgent({"goal": "g", "context": {}, "available_tools": [{"name": "click"}]}, path)
