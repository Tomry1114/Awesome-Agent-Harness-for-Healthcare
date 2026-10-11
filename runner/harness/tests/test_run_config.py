# -*- coding: utf-8 -*-
"""A run must be able to say what it was.

The judge models, the agent model and the harness mode decide what a run IS. Unrecorded, two
runs made under different conditions are indistinguishable afterwards and would pool.
"""
import os
import sys

import pytest

_RUNNER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _RUNNER not in sys.path:
    sys.path.insert(0, _RUNNER)

from harness import run_config as RC       # noqa: E402

_VARS = ("MH_HARNESS_JUDGE_MODEL", "LLM_JUDGE_MODEL", "MH_PROCESS_JUDGE_MODEL", "MH_API_MODEL",
         "MH_TARGET_MODEL")


@pytest.fixture
def clean(monkeypatch):
    for v in _VARS:
        monkeypatch.delenv(v, raising=False)
    return monkeypatch


def test_the_conditions_are_all_present(clean):
    c = RC.snapshot()
    for k in ("harness_judge_model", "harness_judge_bound", "benchmark_judge_model", "process_judge_model",
              "agent_model", "harness_mode", "pid"):
        assert k in c, k


def test_an_unbound_harness_judge_is_stated_not_merely_absent(clean):
    """The condition that silently turns every semantic check off. It has to be legible without knowing the
    field name."""
    c = RC.snapshot()
    assert c["harness_judge_bound"] is False, "a boolean, not a missing string"
    assert "no harness judge bound" in RC.render(c)

    clean.setenv("MH_HARNESS_JUDGE_MODEL", "gpt-5.4")
    c2 = RC.snapshot()
    assert c2["harness_judge_bound"] is True and c2["harness_judge_model"] == "gpt-5.4"
    assert "no harness judge bound" not in RC.render(c2)


def test_values_come_from_the_process_environment(clean):
    clean.setenv("MH_TARGET_MODEL", "gpt-5.5")
    c = RC.snapshot()
    assert c["harness_mode"] == "enforce"
    assert c["agent_model"] == "gpt-5.5"


# ---------------------------------------------------------------------------- it reaches the artifacts
def test_every_result_json_carries_it():
    src = open(os.path.join(_RUNNER, "harness", "kernel.py"), encoding="utf-8").read()
    assert '_audit["run_config"] = _rc.snapshot()' in src
