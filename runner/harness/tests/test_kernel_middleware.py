# -*- coding: utf-8 -*-
"""The harness kernel's capabilities: the fixed shell's own, and the workspace's kernel middleware around them.

The editable capabilities (goal_alignment, required_context, active_evidence, evidence_coverage,
commitment_execution, fhir_empty_targeted_search_broad_reacquire) live in each default (H0) workspace, not in
runner/harness/capabilities; the shell's capabilities run first. Order is behaviour -- resolve() breaks ties by it --
so the order is pinned here for all three benchmarks."""
import os
import shutil
import sys

import pytest
import yaml

_RUNNER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _RUNNER not in sys.path:
    sys.path.insert(0, _RUNNER)

import harness as H                                     # noqa: E402
import workspace as W                                   # noqa: E402

SHELL = ["scope_evidence", "obligation_lifecycle", "reference_integrity", "verify_commit"]
WORKSPACE = ["goal_alignment", "required_context", "active_evidence", "evidence_coverage", "commitment_execution",
             "fhir_empty_targeted_search_broad_reacquire"]
# MedCTA's workspace also holds the final audit's repair policy, first, right after verify_commit whose audit it
# reads. PhysicianBench and HealthAdminBench compose no claim-support postcondition, so the audit never runs there.
WORKSPACE_OF = {"fhir": WORKSPACE, "gui": WORKSPACE, "tool_sandbox": ["answer_repair"] + WORKSPACE}


def _file_of(cap):
    return os.path.realpath(sys.modules[type(cap).__module__].__file__)


@pytest.mark.parametrize("env_type", ["fhir", "gui", "tool_sandbox"])
def test_each_h0_runs_the_shell_capabilities_first_then_its_own(env_type):
    caps = H.compose_capabilities(env_type, W.H0[env_type])
    assert [c.name for c in caps] == SHELL + WORKSPACE_OF[env_type]
    ws = os.path.realpath(W.H0[env_type]) + os.sep
    shell = os.path.join(os.path.realpath(_RUNNER), "harness", "capabilities") + os.sep
    for c in caps:
        assert _file_of(c).startswith(shell if c.name in SHELL else ws), (c.name, _file_of(c))


def test_the_editable_capabilities_no_longer_live_in_the_shell():
    for name in WORKSPACE_OF["tool_sandbox"]:
        assert not os.path.exists(os.path.join(_RUNNER, "harness", "capabilities", name + ".py")), name


def test_a_run_without_a_workspace_runs_the_shell_capabilities_alone():
    assert [c.name for c in H.compose_capabilities("perceptual", None)] == SHELL


def _copy(tmp_path, env_type="fhir"):
    dst = str(tmp_path / "ws")
    shutil.copytree(W.H0[env_type], dst)
    return dst


def _edit_order(path, edit):
    mp = os.path.join(path, "manifest.yaml")
    m = yaml.safe_load(open(mp))
    edit(m["kernel_middleware"])
    open(mp, "w").write(yaml.safe_dump(m))


def test_a_workspace_cannot_place_a_shell_capability(tmp_path):
    path = _copy(tmp_path)
    _edit_order(path, lambda km: km.insert(0, "shell:verify_commit"))
    with pytest.raises(W.WorkspaceError):
        H.compose_capabilities("fhir", workspace=path)


def test_a_middleware_named_like_a_shell_capability_is_refused(tmp_path):
    path = _copy(tmp_path)
    p = os.path.join(path, "middleware", "goal_alignment.py")
    src = open(p).read()
    assert src.count('name = "goal_alignment"') == 1
    open(p, "w").write(src.replace('name = "goal_alignment"', 'name = "verify_commit"'))
    with pytest.raises(W.WorkspaceError):
        H.compose_capabilities("fhir", workspace=path)


def test_a_workspace_may_remove_and_move_its_own_middleware(tmp_path):
    path = _copy(tmp_path)

    def edit(km):
        km.remove("middleware/goal_alignment.py:GoalAlignment")
        km.remove("middleware/commitment_execution.py:CommitmentExecution")
        km.insert(0, "middleware/commitment_execution.py:CommitmentExecution")
    _edit_order(path, edit)
    assert [c.name for c in H.compose_capabilities("fhir", workspace=path)] == SHELL + [
        "commitment_execution", "required_context", "active_evidence", "evidence_coverage",
        "fhir_empty_targeted_search_broad_reacquire"]


def test_the_kernel_runs_the_workspace_file(tmp_path):
    """Editing the workspace's file changes what the kernel runs, with no change under runner/."""
    path = _copy(tmp_path)
    p = os.path.join(path, "middleware", "fhir_empty_targeted_search_broad_reacquire.py")
    src = open(p).read()
    assert src.count('name = "fhir_empty_targeted_search_broad_reacquire"') == 1
    open(p, "w").write(src.replace('name = "fhir_empty_targeted_search_broad_reacquire"', 'name = "edited_reacquire"'))
    assert H.compose_capabilities("fhir", workspace=path)[-1].name == "edited_reacquire"
    assert H.compose_capabilities("fhir", W.H0["fhir"])[-1].name == "fhir_empty_targeted_search_broad_reacquire"


def test_a_repeated_capability_name_is_refused(tmp_path):
    path = _copy(tmp_path)
    _edit_order(path, lambda km: km.append("middleware/goal_alignment.py:GoalAlignment"))
    with pytest.raises(W.WorkspaceError):
        H.compose_capabilities("fhir", workspace=path)


def test_a_file_is_loaded_once_per_process_like_an_import():
    ws = W.Workspace(W.H0["fhir"])
    assert ws.load("middleware/goal_alignment.py", "GoalAlignment") is \
        W.Workspace(W.H0["fhir"]).load("middleware/goal_alignment.py", "GoalAlignment")
