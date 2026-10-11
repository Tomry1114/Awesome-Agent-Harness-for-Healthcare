# -*- coding: utf-8 -*-
"""A workspace is a run input, with an identity.

    identity     workspace_digest / component_digests cover every file under the workspace except the byte-code
                 cache; a symlink, or a path resolving outside the workspace, is refused
    refusal      a manifest that does not declare the run's benchmark and env_type stops the run (and the batch)
                 before an environment or an agent exists
    run input    run_task(..., workspace=DIR): every reader of workspace content loads the named copy and nothing of
                 H0; an edited file in the copy changes what the scripted run does (PhysicianBench: the system prompt
                 the model is sent; HealthAdminBench and MedCTA: a middleware constant)
    result.json  workspace {path, digest, component_digests, parent_digest}, run_config.workspace_digest, resume_key
    resume       run_batch reuses a stored result only when its resume key (task id, workspace digest, judge
                 configuration, agent model, protocol, max_steps) equals the run's

These pin that a result is always attributable to the exact workspace bytes it ran with, and that a stored result is
never reused for a different configuration. The runs drive the real run.run_task through the scripted rigs
(runner/tests/inline_loop_rig.py for PhysicianBench and MedCTA, runner/tests/inline_hab_stub.py for HealthAdminBench):
no model, judge or network is reached.
"""
import hashlib
import json
import os
import re
import shutil
import sys

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inline_hab_stub as S  # noqa: E402
import inline_loop_rig as L  # noqa: E402
from inline_hab_stub import control, final  # noqa: E402

import judges as J  # noqa: E402
import run as R  # noqa: E402
import workspace as W  # noqa: E402
from harness import obligation_gate as OG  # noqa: E402
import harness.engines.policy as P  # noqa: E402

PB, PB_TASK = "PhysicianBench", "PB-pretransplant_covid_clearance"
HAB_TASK = "HAB-denial-easy-1"
MEDCTA_TASK = "MCTA-0"
H0_BENCH = {"fhir": ("PhysicianBench", "pb"), "gui": ("HealthAdminBench", "hab"), "tool_sandbox": ("MedCTA", "medcta")}
DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


def _copy(src, dst):
    shutil.copytree(src, str(dst), ignore=shutil.ignore_patterns("__pycache__"))
    return str(dst)


def _edit(path, old, new):
    src = open(path, encoding="utf-8").read()
    assert src.count(old) == 1, (path, old)
    open(path, "w", encoding="utf-8").write(src.replace(old, new))


def _set_manifest(ws, **kv):
    mp = os.path.join(ws, "manifest.yaml")
    m = yaml.safe_load(open(mp, encoding="utf-8"))
    for k, v in kv.items():
        if v is None:
            m.pop(k, None)
        else:
            m[k] = v
    open(mp, "w", encoding="utf-8").write(yaml.safe_dump(m, sort_keys=False))


def _write_json(path, obj):
    with open(str(path), "w", encoding="utf-8") as f:
        json.dump(obj, f)


def _inside(path, root):
    real, root = os.path.realpath(path), os.path.realpath(root)
    return real == root or real.startswith(root + os.sep)


# =====================================================================================================================
# identity
# =====================================================================================================================
def test_the_digest_covers_every_file_and_excludes_only_the_bytecode_cache(tmp_path):
    h0 = W.H0["fhir"]
    digest = W.workspace_digest(h0)
    assert DIGEST.match(digest)
    ws = _copy(h0, tmp_path / "ws")
    assert W.workspace_digest(ws) == digest                       # the bytes, not the place, are the identity
    comps = W.component_digests(ws)
    assert {"manifest.yaml", "system_prompt.md", "agent.py", "policies/chart_review.yaml",
            "middleware/deliverable.py"} <= set(comps)
    for rel, h in comps.items():
        assert h == hashlib.sha256(open(os.path.join(ws, rel), "rb").read()).hexdigest(), rel
    h = hashlib.sha256()
    for rel in sorted(comps):
        h.update(rel.encode("utf-8") + b"\0" + comps[rel].encode("ascii") + b"\n")
    assert digest == W.digest_of(comps) == "sha256:" + h.hexdigest()

    os.makedirs(os.path.join(ws, "middleware", "__pycache__"))
    open(os.path.join(ws, "middleware", "__pycache__", "deliverable.cpython-310.pyc"), "wb").write(b"\0cache")
    open(os.path.join(ws, "stray.pyc"), "wb").write(b"\0cache")
    assert W.workspace_digest(ws) == digest                       # the byte-code cache is not part of it

    for i, rel in enumerate(("notes.txt", "middleware/README.md", ".hidden", "policies/extra.yaml")):
        other = _copy(h0, tmp_path / ("add%d" % i))
        open(os.path.join(other, rel), "w").write("x")
        assert W.workspace_digest(other) != digest
        assert set(W.component_digests(other)) - set(comps) == {rel}

    _edit(os.path.join(ws, "system_prompt.md"), "You are", "You are,")
    edited = W.component_digests(ws)
    assert W.workspace_digest(ws) != digest
    assert [rel for rel in comps if comps[rel] != edited[rel]] == ["system_prompt.md"] and set(edited) == set(comps)


def test_a_directory_the_digest_cannot_list_is_refused(tmp_path):
    ws = _copy(W.H0["fhir"], tmp_path / "ws")
    locked = os.path.join(ws, "policies")
    os.chmod(locked, 0)
    try:
        if os.access(locked, os.R_OK):
            pytest.skip("running with permissions that read a mode-0 directory")
        with pytest.raises(W.WorkspaceError, match="cannot list"):
            W.workspace_digest(ws)
    finally:
        os.chmod(locked, 0o755)


def _symlinked_file(tmp_path):
    ws = _copy(W.H0["fhir"], tmp_path / "file_link")
    os.symlink(os.path.join(ws, "system_prompt.md"), os.path.join(ws, "alias.md"))     # even one pointing inside
    return ws


def _symlinked_dir(tmp_path):
    ws = _copy(W.H0["fhir"], tmp_path / "dir_link")
    os.symlink(os.path.join(ws, "policies"), os.path.join(ws, "more_policies"))
    return ws


def _symlinked_cache(tmp_path):
    ws = _copy(W.H0["fhir"], tmp_path / "cache_link")
    os.makedirs(str(tmp_path / "elsewhere"), exist_ok=True)
    os.symlink(str(tmp_path / "elsewhere"), os.path.join(ws, "middleware", "__pycache__"))
    return ws


def _symlinked_root(tmp_path):
    ws = _copy(W.H0["fhir"], tmp_path / "real_root")
    link = str(tmp_path / "root_link")
    os.symlink(ws, link)
    return link


@pytest.mark.parametrize("make", [_symlinked_file, _symlinked_dir, _symlinked_cache, _symlinked_root],
                         ids=["file", "directory", "excluded_name", "workspace_itself"])
def test_a_symlink_is_refused_by_the_digest_and_by_the_run(tmp_path, make):
    ws = make(tmp_path)
    with pytest.raises(W.WorkspaceError):
        W.workspace_digest(ws)
    with pytest.raises(W.WorkspaceError):
        R.resolve_workspace(PB, {"environment": {"type": "fhir"}}, ws)


def test_a_path_resolving_outside_the_workspace_is_refused(tmp_path):
    ws = _copy(W.H0["fhir"], tmp_path / "ws")
    shutil.copy(os.path.join(ws, "agent.py"), str(tmp_path / "outside_agent.py"))
    _set_manifest(ws, agent_program="../outside_agent.py")                  # a manifest path out of the workspace
    with pytest.raises(W.WorkspaceError, match="outside"):
        W.Workspace(ws).program_class()
    assert any("outside" in p for p in W.Workspace(ws).check([]))
    _set_manifest(ws, agent_program="agent.py", policies=["../../elsewhere/chart_review.yaml"])
    with pytest.raises(W.WorkspaceError, match="outside"):
        W.Workspace(ws).policy_files()

    # workspace code loading through a symlink that leaves the workspace: confinement is by realpath
    hab = _copy(W.H0["gui"], tmp_path / "hab")
    elsewhere = str(tmp_path / "hab_elsewhere")
    os.makedirs(elsewhere)
    shutil.copy(os.path.join(hab, "recovery", "contracts.py"), os.path.join(elsewhere, "contracts.py"))
    os.symlink(elsewhere, os.path.join(hab, "linked"))
    by = os.path.join(hab, "middleware", "recovery.py")
    with pytest.raises(W.WorkspaceError):
        W.load_relative(by, "../linked/contracts.py")
    with pytest.raises(W.WorkspaceError):
        W.load_relative(by, "../../hab_elsewhere/contracts.py")


def test_the_parent_digest_is_read_from_candidate_json_next_to_the_workspace(tmp_path):
    ws = _copy(W.H0["fhir"], tmp_path / "version" / "workspace")
    ident = W.identity(ws, PB, "fhir")
    assert ident == {"path": W.repo_path(ws), "digest": W.workspace_digest(ws), "component_digests":
                     W.component_digests(ws), "parent_digest": None}
    parent = W.workspace_digest(W.H0["fhir"])
    _write_json(tmp_path / "version" / "candidate.json", {"parent_digest": parent, "rationale": "x"})
    assert W.identity(ws, PB, "fhir")["parent_digest"] == parent
    _write_json(tmp_path / "version" / "candidate.json", {"parent_digest": "h0"})
    with pytest.raises(W.WorkspaceError, match="parent_digest"):
        W.identity(ws, PB, "fhir")
    open(str(tmp_path / "version" / "candidate.json"), "w").write("{not json")
    with pytest.raises(W.WorkspaceError):
        W.identity(ws, PB, "fhir")


# =====================================================================================================================
# refusal
# =====================================================================================================================
def test_every_h0_manifest_declares_its_benchmark_and_env_type():
    for env_type, (bench, key) in H0_BENCH.items():
        assert W.H0[env_type] == os.path.join(W.REPO, "workspaces", key, "h0")
        m = yaml.safe_load(open(os.path.join(W.H0[env_type], "manifest.yaml"), encoding="utf-8"))
        assert (m["benchmark"], m["env_type"]) == (bench, env_type)
        assert W.identity(W.H0[env_type], bench, env_type)["path"] == "workspaces/%s/h0" % key


@pytest.mark.parametrize("edit", [{"benchmark": "HealthAdminBench"}, {"env_type": "gui"}, {"benchmark": None},
                                  {"env_type": None}],
                         ids=["other_benchmark", "other_env_type", "no_benchmark", "no_env_type"])
def test_a_workspace_that_does_not_declare_the_runs_benchmark_and_env_type_is_refused(monkeypatch, tmp_path, edit):
    ws = _copy(W.H0["fhir"], tmp_path / "ws")
    _set_manifest(ws, **edit)
    for k, v in L.JUDGE_CONFIG_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(R.environments, "make_env", lambda *a, **k: pytest.fail("an environment was built"))
    monkeypatch.setattr(R.agents, "make_agent", lambda *a, **k: pytest.fail("an agent was built"))
    with pytest.raises(W.WorkspaceError, match="REFUSING"):
        R.run_task(PB, PB_TASK, "stub", max_steps=2, job_dir=str(tmp_path / "job"), workspace=ws)


def test_an_api_agent_is_not_built_without_the_runs_workspace(monkeypatch):
    import agents
    monkeypatch.setenv("MH_OPENAI_KEY", "test")
    task = {"goal": "g", "context": {}, "environment": {"type": "fhir"}, "available_tools": [{"name": "write_file"}]}
    with pytest.raises(W.WorkspaceError):
        agents.make_agent("gpt5", task)


# =====================================================================================================================
# run input: every reader loads the named copy
# =====================================================================================================================
def _spy_readers(mp):
    """Record every workspace path the run's readers touch: modules (agent program, middleware, kernel middleware,
    workspace-relative modules), manifests, text files (prompts, fragments), confined manifest paths (tool
    descriptions, policies, completion content), the policy loader's files and the obligation engine's workspace."""
    seen = []
    module, init, read_text, inside = W._module, W.Workspace.__init__, W._read_text, getattr(W, "_inside", None)
    load_content, read_policy = OG.load_completion_obligations, P._read_file

    def _module(path):
        seen.append(("module", path))
        return module(path)

    def _init(self, path, *a, **k):
        init(self, path, *a, **k)
        seen.append(("manifest", self.path))

    def _read(path):
        seen.append(("text", path))
        return read_text(path)

    def _in(path, real_root):
        seen.append(("path", path))
        return inside(path, real_root)

    def _content(workspace):
        seen.append(("completion_obligations", workspace))
        return load_content(workspace)

    def _policy(p):
        if not _inside(p, os.path.join(W.REPO, "runner")):     # the shell's own adapter / substrate / clinical modules
            seen.append(("policy", p))
        return read_policy(p)
    mp.setattr(W, "_module", _module)
    mp.setattr(W.Workspace, "__init__", _init)
    mp.setattr(W, "_read_text", _read)
    if inside is not None:
        mp.setattr(W, "_inside", _in)
    mp.setattr(OG, "load_completion_obligations", _content)
    mp.setattr(P, "_read_file", _policy)
    return seen


def _assert_only(seen, ws, instrumentation, kinds):
    outside = [(k, p) for k, p in seen if not _inside(p, ws)]
    assert outside == [], outside
    assert kinds <= set(k for k, _p in seen), set(k for k, _p in seen)
    comps = [c for c in instrumentation if c.get("kind") == "component_loaded"]
    assert comps and all(_inside(c["path"], ws) for c in comps), comps


def test_pb_every_reader_loads_the_named_copy_and_its_edited_prompt_reaches_the_model(monkeypatch, tmp_path):
    h0_prompt = W.Workspace(W.H0["fhir"]).system_prompt
    parent = W.workspace_digest(W.H0["fhir"])
    out = {}
    for arm in ("copy", "edited"):
        ws = _copy(W.H0["fhir"], tmp_path / arm / "workspace")
        _write_json(tmp_path / arm / "candidate.json", {"parent_digest": parent})
        if arm == "edited":
            _edit(os.path.join(ws, "system_prompt.md"), "You are a clinical AI assistant",
                  "E1 EDITED PROMPT. You are a clinical AI assistant")
        with monkeypatch.context() as mp:
            rig = L.Rig(mp, tmp_path / arm / "rig", PB, PB_TASK, protocol="function_calling")
            seen = _spy_readers(mp)
            rig.replies = [L.fc_final("answer")]
            res = rig.run(max_steps=4, workspace=ws)
            _assert_only(seen, ws, res["_instrumentation"],
                         {"module", "manifest", "text", "path", "policy"})
            loaded = set(os.path.relpath(p, ws) for k, p in seen if k == "module")
            assert {"agent.py", "middleware/deliverable.py", "middleware/goal_alignment.py",
                    "middleware/fhir_empty_targeted_search_broad_reacquire.py"} <= loaded, loaded
            assert any(os.path.relpath(p, ws) == "policies/chart_review.yaml" for k, p in seen if k == "policy")
            out[arm] = {"system": rig.model_calls[0]["messages"][0], "result": res, "ws": ws}
    assert out["copy"]["system"] == {"role": "system", "content": h0_prompt}
    assert out["edited"]["system"] == {"role": "system", "content": "E1 EDITED PROMPT. " + h0_prompt}
    for arm, o in out.items():
        res, ws = o["result"], o["ws"]
        assert res["workspace"] == {"path": W.repo_path(ws), "digest": W.workspace_digest(ws),
                                    "component_digests": W.component_digests(ws), "parent_digest": parent}
        assert res["run_config"]["workspace_digest"] == res["workspace"]["digest"]
        assert res["resume_key"] == {
            "task_id": PB_TASK, "workspace_digest": res["workspace"]["digest"],
            "judge_config_digest": "sha256:" + hashlib.sha256(json.dumps(J.signature(J.require(PB, env=dict(
                L.JUDGE_CONFIG_ENV))), sort_keys=True).encode("utf-8")).hexdigest(),
            "agent_model": L.ENV_SET["MH_API_MODEL"], "protocol": "function_calling", "max_steps": 4}
    assert out["copy"]["result"]["workspace"]["digest"] == parent
    assert out["edited"]["result"]["workspace"]["digest"] != parent
    assert [k for k, v in out["edited"]["result"]["workspace"]["component_digests"].items()
            if out["copy"]["result"]["workspace"]["component_digests"][k] != v] == ["system_prompt.md"]


def test_a_default_run_records_the_h0_workspace(monkeypatch, tmp_path):
    rig = L.Rig(monkeypatch, tmp_path, PB, PB_TASK, protocol="function_calling")
    monkeypatch.setitem(W.H0, "fhir", os.path.join(W.REPO, "workspaces", "pb", "h0"))   # the rig's copy undone
    rig.replies = [L.fc_final("answer")]
    res = rig.run(max_steps=3)
    assert res["workspace"]["path"] == "workspaces/pb/h0" and res["workspace"]["parent_digest"] is None
    assert res["workspace"]["digest"] == W.workspace_digest(W.H0["fhir"]) == res["run_config"]["workspace_digest"]
    assert res["resume_key"]["workspace_digest"] == res["workspace"]["digest"]


def _hab_reentry(h):
    h.portal.controls = [control("disposition-input", role="textbox", label="Disposition", required=True),
                         control("submit-disposition-button", label="Submit Disposition")]
    return [final("a"), final("b"), final("c")], 12


def test_hab_every_reader_loads_the_named_copy_and_its_edited_streak_stop_changes_the_run(monkeypatch, tmp_path):
    out = {}
    for arm in ("copy", "edited"):
        ws = _copy(W.H0["gui"], tmp_path / arm / "workspace")
        if arm == "edited":
            _edit(os.path.join(ws, "middleware", "recovery.py"), "MAX_BLOCKED_RECOVERY = 2 ", "MAX_BLOCKED_RECOVERY = 1 ")
        with monkeypatch.context() as mp:
            h = S.Harness(mp, tmp_path / arm / "run")
            h.use_task(S.load_hab_task(HAB_TASK))
            seen = _spy_readers(mp)
            script, max_steps = _hab_reentry(h)
            res = h.run(script, max_steps=max_steps, workspace=ws)
            assert h.net == [] and h.portal.unscripted == [] and h.runtime_errors() == []
            _assert_only(seen, ws, res["_instrumentation"],
                         {"module", "manifest", "path", "completion_obligations"})
            loaded = set(os.path.relpath(p, ws) for k, p in seen if k == "module")
            assert {"middleware/recovery.py", "recovery/entry.py", "recovery/hab.py", "middleware/obligation_nudge.py",
                    "middleware/goal_alignment.py"} <= loaded, loaded
            content = [p for k, p in seen if k == "completion_obligations"]
            assert content and all(os.path.realpath(p) == os.path.realpath(ws) for p in content)
            out[arm] = {k: [e["step"] for e in h.events(k)] for k in ("recovery_v3", "agent_reentry", "obligation_nudge")}
            assert res["workspace"]["digest"] == W.workspace_digest(ws)
    # the unedited copy runs as H0 does (see runner/tests/test_inline_hab_recovery.py): two blocked invocations re-enter
    # at 0 and 1
    assert (out["copy"]["recovery_v3"], out["copy"]["agent_reentry"], out["copy"]["obligation_nudge"][:1]) == (
        [0, 1], [0, 1], [2])
    # the edited copy stops after ONE blocked invocation
    assert (out["edited"]["recovery_v3"], out["edited"]["agent_reentry"], out["edited"]["obligation_nudge"][:1]) == (
        [0], [0], [1])


def test_medcta_every_reader_loads_the_named_copy_and_its_edited_rederive_text_reaches_the_model(monkeypatch, tmp_path):
    """The kernel verdict that opens MedCTA's answer-candidate strategy comes from the rig's scripted kernel middleware
    (an ACQUIRE at the first final); both arms copy that workspace, and the edited arm changes one constant of H0's
    middleware/answer_candidate.py, the re-derive text the strategy hands the agent."""
    out = {}
    for arm in ("copy", "edited"):
        with monkeypatch.context() as mp:
            rig = L.Rig(mp, tmp_path / arm / "rig", "MedCTA", MEDCTA_TASK)
            ws = _copy(rig.workspace, tmp_path / arm / "workspace")
            if arm == "edited":
                _edit(os.path.join(ws, "middleware", "answer_candidate.py"), "Re-derive your answer to the question",
                      "E1 EDITED: re-derive your answer to the question")
            finals = []

            def before_final(answer, ctx):
                finals.append(answer)
                if len(finals) > 1:
                    return None
                return L.decision("ACQUIRE", rule_id="scripted_acquire", reason="scripted acquire",
                                  extra={"next_action": {"tool": "RegionAttributeDescription",
                                                         "args": {"region": "left lower lobe", "attribute": "density"},
                                                         "read_only": True}})
            rig.hooks.before_final = before_final
            seen = _spy_readers(mp)
            rig.replies = ["<answer>first answer</answer>", "<answer>second answer</answer>",
                           "<answer>third answer</answer>"]
            res = rig.run(max_steps=6, workspace=ws)
            _assert_only(seen, ws, res["_instrumentation"], {"module", "manifest", "text", "path"})
            loaded = set(os.path.relpath(p, ws) for k, p in seen if k == "module")
            assert {"agent_text.py", "middleware/answer_candidate.py", "middleware/answer_repair.py"} <= loaded, loaded
            out[arm] = [m["content"] for call in rig.model_calls for m in call["messages"] if m.get("role") == "user"]
            assert res["workspace"]["digest"] == W.workspace_digest(ws)
    h0_text = [t for t in out["copy"] if "Re-derive your answer to the question" in t]
    edited_text = [t for t in out["edited"] if "E1 EDITED: re-derive your answer to the question" in t]
    assert h0_text and edited_text
    assert not any("E1 EDITED" in t for t in out["copy"])
    assert not any("Re-derive your answer to the question" in t for t in out["edited"])


# =====================================================================================================================
# resume
# =====================================================================================================================
def _batch(mp, out, ws, *extra):
    import run_batch as RB
    mp.setattr(sys, "argv", ["run_batch.py", "--bench", PB, "--agent", "gpt5", "--task-id", PB_TASK, "--limit", "1",
                             "--out", out, "--max-steps", "4", "--workspace", ws] + list(extra))
    RB.main()
    return json.load(open(os.path.join(out, "gpt5", "summary.json")))


def _no_report(mp):
    """Outside what is tested and not zero-cost in a real batch: run_batch's report.json (one experimental dimension
    asks the gateway) and run_task's FHIR stub-resource cleanup (a local HTTP call). Both stand-ins record nothing."""
    import aggregate_report
    mp.setattr(aggregate_report, "build", lambda *a, **k: {"native_metrics": {"stubbed": True}})
    mp.setattr(R, "cleanup_stub", lambda base: {"deleted": 0, "error": None})


def test_run_batch_reuses_a_stored_result_only_under_the_same_resume_key(monkeypatch, tmp_path):
    rig = L.Rig(monkeypatch, tmp_path / "rig", PB, PB_TASK, protocol="function_calling")
    _no_report(monkeypatch)
    ran = []
    real = R.run_task

    def run_task(*a, **k):
        ran.append(k.get("workspace"))
        rig.replies = [L.fc_final("answer")]
        return real(*a, **k)
    monkeypatch.setattr(R, "run_task", run_task)
    out = str(tmp_path / "results")
    a = _copy(W.H0["fhir"], tmp_path / "a" / "workspace")
    b = _copy(W.H0["fhir"], tmp_path / "b" / "workspace")
    _edit(os.path.join(b, "system_prompt.md"), "You are", "You are,")
    _write_json(tmp_path / "b" / "candidate.json", {"parent_digest": W.workspace_digest(a)})
    stored = os.path.join(out, "gpt5", PB_TASK, "result.json")

    def step(ws, *extra, **env):
        n = len(ran)
        with monkeypatch.context() as mp:
            for k, v in env.items():
                mp.setenv(k, v)
            summary = _batch(mp, out, ws, *extra)
        return len(ran) - n, summary

    assert step(a)[0] == 1
    first = json.load(open(stored))
    assert first["workspace"]["digest"] == W.workspace_digest(a) and first["workspace"]["parent_digest"] is None
    assert first["resume_key"]["workspace_digest"] == W.workspace_digest(a)
    n, summary = step(a)
    assert n == 0 and summary["n_reused"] == 1 and summary["tasks"][0]["reused"] is True
    assert json.load(open(stored)) == first                                        # the bundle is left as it was
    assert step(b)[0] == 1                                                         # another workspace digest
    second = json.load(open(stored))
    assert second["workspace"]["digest"] == W.workspace_digest(b) != first["workspace"]["digest"]
    assert second["workspace"]["parent_digest"] == W.workspace_digest(a)
    assert step(b)[0] == 0
    assert step(b, "--max-steps", "5")[0] == 1                                    # another max_steps
    assert step(b, "--max-steps", "5", LLM_JUDGE_MODEL="another-judge")[0] == 1   # another judge configuration
    assert step(b, "--max-steps", "5", LLM_JUDGE_MODEL="another-judge")[0] == 0
    monkeypatch.setitem(__import__("models").table(), "another-brain", "function_calling")
    assert step(b, "--max-steps", "5", LLM_JUDGE_MODEL="another-judge", MH_API_MODEL="another-brain")[0] == 1
    res = json.load(open(stored))
    assert (res["resume_key"]["agent_model"], res["resume_key"]["protocol"], res["resume_key"]["max_steps"]) == (
        "another-brain", "function_calling", 5)
    assert step(b, "--max-steps", "5", LLM_JUDGE_MODEL="another-judge", MH_API_MODEL="another-brain")[0] == 0
    del res["resume_key"]                                                          # a stored result without a resume key
    _write_json(stored, res)
    assert step(b, "--max-steps", "5", LLM_JUDGE_MODEL="another-judge", MH_API_MODEL="another-brain")[0] == 1
    assert ran == [a, b, b, b, b, b]
    assert rig.network == []


def test_run_batch_refuses_a_workspace_of_another_benchmark_before_any_task(monkeypatch, tmp_path):
    L.Rig(monkeypatch, tmp_path / "rig", PB, PB_TASK, protocol="function_calling")
    monkeypatch.setattr(R, "run_task", lambda *a, **k: pytest.fail("a task ran"))
    ws = _copy(W.H0["gui"], tmp_path / "hab")
    out = str(tmp_path / "results")
    with pytest.raises(SystemExit, match="REFUSING"):
        _batch(monkeypatch, out, ws)
    assert not os.path.exists(os.path.join(out, "gpt5", PB_TASK))
