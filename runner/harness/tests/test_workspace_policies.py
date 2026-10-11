# -*- coding: utf-8 -*-
"""Clinical policy modules split by rule semantics: medication_safety is the shell's (runner/policies/clinical),
chart_review is PhysicianBench default (H0) workspace content (workspaces/pb/h0/policies), composed after the shell's.
These tests pin the composition rules."""
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
from harness.compiler import build_contract             # noqa: E402
from harness.semantics import canonicalize              # noqa: E402

TASK = {"task_id": "t", "goal": "order", "context": {"patient_ref": "Patient/1"}, "environment": {"type": "fhir"}}


def test_pb_composes_the_shell_module_then_the_workspace_module_then_the_substrate():
    pol = H.load_policy(env_type="fhir", workspace=W.H0["fhir"])
    assert pol["_clinical_modules"] == ["medication_safety", "chart_review"] and not pol["_errors"]
    rules = [cp.get("requires_rule") for cp in pol["commit_points"]]
    assert rules[:2] == ["medication_requires_safety_review", "deliverable_requires_chart_review"], rules


def test_chart_review_lives_in_the_workspace_and_medication_safety_in_the_shell():
    assert not os.path.exists(os.path.join(_RUNNER, "policies", "clinical", "chart_review.yaml"))
    assert os.path.exists(os.path.join(_RUNNER, "policies", "clinical", "medication_safety.yaml"))
    for adapter in ("hapi_fhir", "clinic_emr"):
        doc = yaml.safe_load(open(os.path.join(_RUNNER, "adapters", adapter + ".yaml")))
        assert doc["clinical_modules"] == ["medication_safety"], adapter


@pytest.mark.parametrize("env_type,modules", [("gui", []), ("tool_sandbox", ["evidence_grounding"])])
def test_benchmarks_without_workspace_policies_compose_as_before(env_type, modules):
    pol = H.load_policy(env_type=env_type, workspace=W.H0[env_type])
    assert pol["_clinical_modules"] == modules and not pol["_errors"]


def _copy(tmp_path):
    dst = str(tmp_path / "ws")
    shutil.copytree(W.H0["fhir"], dst)
    return dst


def _add_policy(path, rel, doc):
    open(os.path.join(path, rel), "w").write(yaml.safe_dump(doc))
    mp = os.path.join(path, "manifest.yaml")
    m = yaml.safe_load(open(mp))
    m["policies"].append(rel)
    open(mp, "w").write(yaml.safe_dump(m))


def test_a_workspace_policy_cannot_reuse_a_shell_module_name(tmp_path):
    path = _copy(tmp_path)
    _add_policy(path, "policies/shadow.yaml", {"name": "medication_safety", "commit_points": []})
    errors = H.load_policy(env_type="fhir", workspace=path)["_errors"]
    assert any(e.startswith("workspace_policy_reuses_module_name:medication_safety") for e in errors), errors


def test_a_missing_workspace_policy_is_a_composition_error(tmp_path):
    path = _copy(tmp_path)
    m = yaml.safe_load(open(os.path.join(path, "manifest.yaml")))
    m["policies"].append("policies/nope.yaml")
    open(os.path.join(path, "manifest.yaml"), "w").write(yaml.safe_dump(m))
    assert any(e.startswith("workspace_policy_not_found:") for e in H.load_policy(env_type="fhir", workspace=path)["_errors"])
    assert W.Workspace(path).check([])


def test_a_workspace_policy_can_add_to_a_shell_rule_but_not_weaken_it(tmp_path):
    path = _copy(tmp_path)
    _add_policy(path, "policies/lenient.yaml", {"name": "lenient", "commit_points": [
        {"match": {"semantic_type": "create", "resource": "MedicationRequest"}, "risk": "R0", "requires": []}]})
    pol = H.load_policy(env_type="fhir", workspace=path)
    assert not pol["_errors"], pol["_errors"]
    contract = build_contract(TASK, env_type="fhir", policy=pol)
    create = canonicalize({"type": "tool", "tool": "fhir_medication_request_create", "args": {"patient": "Patient/1"}},
                          pol["manifest"])
    cp = contract.commit_point_for(create)
    assert cp["risk"] == "R2" and "medication_safety_review" in cp["requires"], cp


def test_an_edited_workspace_policy_is_what_composes(tmp_path):
    path = _copy(tmp_path)
    p = os.path.join(path, "policies", "chart_review.yaml")
    src = open(p).read()
    assert src.count("requires_rule: deliverable_requires_chart_review") == 1
    open(p, "w").write(src.replace("requires_rule: deliverable_requires_chart_review", "requires_rule: edited_rule"))
    rules = [cp.get("requires_rule") for cp in H.load_policy(env_type="fhir", workspace=path)["commit_points"]]
    assert "edited_rule" in rules and "deliverable_requires_chart_review" not in rules
