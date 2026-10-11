"""The PB content judge (upstream eval_helpers) follows the run's judge route: LLM_JUDGE_API_KEY / LLM_JUDGE_BASE_URL
come from MH_JUDGE_KEY / MH_JUDGE_BASE, so a judge key is never sent to another provider's base (which answers 401)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import run


def test_judge_route_reaches_the_upstream_judge_vars(monkeypatch):
    for k in ("LLM_JUDGE_API_KEY", "LLM_JUDGE_BASE_URL", "OPENAI_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("LLM_JUDGE_MODEL", "gpt-5.6-sol")
    monkeypatch.setenv("OPENAI_API_KEY", "agent-key")
    monkeypatch.setenv("MH_JUDGE_KEY", "judge-key")
    monkeypatch.setenv("MH_JUDGE_BASE", "https://judge.example")
    run.pb_judge_env()
    assert os.environ["LLM_JUDGE_API_KEY"] == "judge-key"
    assert os.environ["LLM_JUDGE_BASE_URL"] == "https://judge.example/v1"


def test_without_a_judge_route_nothing_is_forced(monkeypatch):
    for k in ("LLM_JUDGE_API_KEY", "LLM_JUDGE_BASE_URL", "MH_JUDGE_KEY", "MH_JUDGE_BASE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("LLM_JUDGE_MODEL", "gpt-5.6-sol")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://agent.example/v1")
    run.pb_judge_env()
    assert "LLM_JUDGE_API_KEY" not in os.environ and "LLM_JUDGE_BASE_URL" not in os.environ
