"""The PB judge guard waits out a rate-limit refusal of upstream call_llm (e.g. a relay's 429 concurrency refusal)
and records only a final failure, so a transient refusal does not turn the checkpoint into a judge_error; other errors
fail at once."""
import importlib
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
GUARD = os.path.join(HERE, "..", "adapters", "pb", "pytest_judge_guard")


def _load(monkeypatch, tmp_path, behaviour):
    utils = types.ModuleType("utils")
    eh = types.ModuleType("utils.eval_helpers")
    calls = []

    def call_llm(prompt, system=""):
        calls.append(1)
        b = behaviour[min(len(calls), len(behaviour)) - 1]
        if isinstance(b, Exception):
            raise b
        return b
    eh.call_llm = call_llm
    eh.llm_judge = lambda content, rubric, context="": {"pass": True}
    utils.eval_helpers = eh
    monkeypatch.setitem(sys.modules, "utils", utils)
    monkeypatch.setitem(sys.modules, "utils.eval_helpers", eh)
    log = tmp_path / "judge.jsonl"
    monkeypatch.setenv("MH_PB_JUDGE_LOG", str(log))
    monkeypatch.syspath_prepend(GUARD)
    sys.modules.pop("mh_pb_judge_guard", None)
    g = importlib.import_module("mh_pb_judge_guard")
    monkeypatch.setattr(g.time, "sleep", lambda s: None)
    return eh, calls, log


class RateLimitError(Exception):
    pass


def _rows(log):
    return [json.loads(l) for l in open(str(log))] if os.path.exists(str(log)) else []


def test_rate_limit_is_waited_out(monkeypatch, tmp_path):
    eh, calls, log = _load(monkeypatch, tmp_path, [RateLimitError("Error code: 429 - Concurrency limit exceeded"), "PASS"])
    assert eh.call_llm("p") == "PASS" and len(calls) == 2
    assert [r["ok"] for r in _rows(log)] == [True]


def test_other_errors_fail_at_once(monkeypatch, tmp_path):
    import pytest
    eh, calls, log = _load(monkeypatch, tmp_path, [ValueError("bad request"), "PASS"])
    with pytest.raises(ValueError):
        eh.call_llm("p")
    assert len(calls) == 1 and [r["ok"] for r in _rows(log)] == [False]


def test_an_unparseable_reply_is_waited_out(monkeypatch, tmp_path):
    """An SDK error from a reply that came back as a string -- "'str' object has no attribute 'choices'" -- is waited
    out like a rate limit, not turned into a judge_error at once."""
    eh, calls, log = _load(monkeypatch, tmp_path, [AttributeError("'str' object has no attribute 'choices'"), "PASS"])
    assert eh.call_llm("p") == "PASS" and len(calls) == 2


def test_a_524_on_chat_moves_the_judge_to_responses(monkeypatch, tmp_path):
    eh, calls, log = _load(monkeypatch, tmp_path, [RuntimeError("Error code: 524 - bad_response_status_code")])
    import responses_api as RA
    monkeypatch.setattr(RA, "_FALLBACK", set())
    monkeypatch.delenv("MH_RESPONSES_MODELS", raising=False)
    monkeypatch.setenv("LLM_JUDGE_MODEL", "gpt-5.6-sol")
    monkeypatch.setattr(RA, "post", lambda *a, **k: {"choices": [{"message": {"content": "PASS"}}]})
    assert eh.call_llm("p") == "PASS" and len(calls) == 1 and RA.wants_responses("gpt-5.6-sol")
