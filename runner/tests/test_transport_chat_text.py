# -*- coding: utf-8 -*-
"""transport.chat_text, the shell's text-protocol chat: the arguments of its gateway call (model, max_tokens, timeout,
reasoning_effort) with and without MH_OPENAI_MAX_TOKENS / MH_OPENAI_TIMEOUT set, and the text protocol's sentinel
final answer on a failed call."""
import os
import sys

_RUNNER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RUNNER not in sys.path:
    sys.path.insert(0, _RUNNER)

import gateway                                          # noqa: E402
import transport                                        # noqa: E402


def _record(monkeypatch, result):
    calls = []
    monkeypatch.setattr(gateway, "chat", lambda messages, **kw: calls.append((messages, kw)) or dict(result))
    return calls


def test_the_gateway_call(monkeypatch):
    monkeypatch.delenv("MH_OPENAI_MAX_TOKENS", raising=False)
    monkeypatch.delenv("MH_OPENAI_TIMEOUT", raising=False)
    calls = _record(monkeypatch, {"ok": True, "content": "<answer>x</answer>"})
    msgs = [{"role": "user", "content": "q"}]
    assert transport.chat_text(msgs, "m", "high", 1500) == "<answer>x</answer>"
    assert calls == [(msgs, {"model": "m", "max_tokens": 16000, "judge": False, "timeout": 300,
                             "extra": {"reasoning_effort": "high"}})]
    monkeypatch.setenv("MH_OPENAI_MAX_TOKENS", "100")
    monkeypatch.setenv("MH_OPENAI_TIMEOUT", "7")
    transport.chat_text(msgs, "m", "", 1500)
    assert calls[-1][1] == {"model": "m", "max_tokens": 1500, "judge": False, "timeout": 7, "extra": None}


def test_a_failed_call_is_the_sentinel_answer(monkeypatch):
    _record(monkeypatch, {"ok": False, "error_type": "billing", "raw": "quota"})
    assert transport.chat_text([], "m", "high") == "<answer>API_BRAIN_ERROR: BILLING/QUOTA quota</answer>"
    _record(monkeypatch, {"ok": False, "error_type": "http", "raw": "503"})
    assert transport.chat_text([], "m", "high") == "<answer>API_BRAIN_ERROR: 503</answer>"
