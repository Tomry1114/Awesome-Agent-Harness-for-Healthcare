"""A gateway no attempt can reach (TLS failure such as a certificate for another host name, refused connection) stops
the job like a spent key; one that answers with errors does not."""
import os
import ssl
import sys
import urllib.error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import transport as T  # noqa: E402
from test_agent_key_failover import _Agent, _http, _run_chat_fc  # noqa: E402


def test_tls_failure_on_every_attempt_is_unreachable(monkeypatch):
    msg, calls = _run_chat_fc(monkeypatch, lambda: urllib.error.URLError(ssl.SSLCertVerificationError(1, "mismatch")))
    assert msg.get("unreachable") is True and len(calls) == 3


def test_an_answering_gateway_is_not_unreachable(monkeypatch):
    msg, calls = _run_chat_fc(monkeypatch, lambda: _http(503, "overloaded"))
    assert not msg.get("unreachable")


def test_unreachable_writes_the_stop_file(tmp_path, monkeypatch):
    stop = tmp_path / "stop"
    monkeypatch.setenv("MH_QUOTA_STOP_FILE", str(stop))
    monkeypatch.setattr(T, "chat_fc", lambda *a: {"content": "API_BRAIN_ERROR", "unreachable": True})
    assert _Agent(["k"], "k")._chat([], []) == {"content": "API_BRAIN_ERROR"}
    assert "agent gateway unreachable" in stop.read_text()


def test_answered_failure_writes_no_stop_file(tmp_path, monkeypatch):
    stop = tmp_path / "stop"
    monkeypatch.setenv("MH_QUOTA_STOP_FILE", str(stop))
    monkeypatch.setattr(T, "chat_fc", lambda *a: {"content": "API_BRAIN_ERROR", "unreachable": False})
    _Agent(["k"], "k")._chat([], [])
    assert not stop.exists()
