"""A relay's daily-quota 429 (USAGE_LIMIT_EXCEEDED / DAILY_LIMIT_EXCEEDED) refuses the key at once (no backed-off
retries), its concurrency 429 is retried, and an agent whose last key is refused leaves the stop file
(MH_QUOTA_STOP_FILE) its job's watchdog cancels on."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import transport as T  # noqa: E402
from test_agent_key_failover import _Agent, _http, _run_chat_fc  # noqa: E402

DAILY = ('{"code":"USAGE_LIMIT_EXCEEDED","message":"error: code=429 reason=\\"DAILY_LIMIT_EXCEEDED\\" '
         'message=\\"daily usage limit exceeded\\" metadata=map[]"}')
CONC = ('{"error":{"code":"gateway_concurrency_limit","message":"Concurrency limit exceeded for user, please retry later",'
        '"type":"rate_limit_error"}}')


def test_relay_daily_limit_is_a_refusal(monkeypatch):
    msg, calls = _run_chat_fc(monkeypatch, lambda: _http(429, DAILY))
    assert msg.get("key_refused") is True and len(calls) == 1


def test_relay_concurrency_is_retried(monkeypatch):
    msg, calls = _run_chat_fc(monkeypatch, lambda: _http(429, CONC))
    assert not msg.get("key_refused") and len(calls) == 3


def test_last_key_refused_writes_stop_file(tmp_path, monkeypatch):
    stop = tmp_path / "quota_stop"
    monkeypatch.setenv("MH_QUOTA_STOP_FILE", str(stop))
    monkeypatch.setattr(T, "chat_fc", lambda *a: {"content": "API_BRAIN_ERROR", "key_refused": True})
    assert _Agent(["k1", "k2"], "k1")._chat([], []) == {"content": "API_BRAIN_ERROR"}
    assert "all agent keys refused" in stop.read_text()


def test_switch_to_live_key_writes_no_stop_file(tmp_path, monkeypatch):
    stop = tmp_path / "quota_stop"
    monkeypatch.setenv("MH_QUOTA_STOP_FILE", str(stop))
    monkeypatch.setattr(T, "chat_fc", lambda m, t, mo, b, key, r: ({"content": "API_BRAIN_ERROR", "key_refused": True}
                                                                    if key == "dead" else {"content": "ok"}))
    assert _Agent(["dead", "live"], "dead")._chat([], []) == {"content": "ok"}
    assert not stop.exists()
