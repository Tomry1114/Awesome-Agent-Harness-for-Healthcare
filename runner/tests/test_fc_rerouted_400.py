"""Some relays answer with a 400 "short-input distillation or heartbeat probing" that is a routing outcome (the
identical request can pass on a later try), so chat_fc retries it; any other 400 still gives up after one request."""
import io, json, urllib.error

import pytest

from runner import transport

BLOCK = b'{"error":{"message":"Upstream rejected illegal short-input distillation or heartbeat probing.","type":"upstream_error"}}'


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _rig(monkeypatch, bodies):
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(req)
        b = bodies[min(len(calls), len(bodies)) - 1]
        if b is None:
            return _Resp(json.dumps({"choices": [{"message": {"role": "assistant", "content": "done"}}]}).encode())
        raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", {}, io.BytesIO(b))
    monkeypatch.setattr(transport.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(transport.time, "sleep", lambda s: None)
    monkeypatch.setenv("MH_FC_RETRIES", "4")
    return calls


def _call():
    return transport.chat_fc([{"role": "user", "content": "q"}], [], "gpt-5.6-sol", "http://relay", "k", "high")


def test_distillation_block_is_retried_until_it_passes(monkeypatch):
    calls = _rig(monkeypatch, [BLOCK, BLOCK, None])
    assert _call().get("content") == "done"
    assert len(calls) == 3


def test_other_400_still_gives_up_after_one_request(monkeypatch):
    calls = _rig(monkeypatch, [b'{"error":{"message":"invalid tools schema"}}', None])
    assert _call() == {"content": "API_BRAIN_ERROR"}
    assert len(calls) == 1


def test_persistent_block_ends_in_brain_error_after_the_retry_budget(monkeypatch):
    calls = _rig(monkeypatch, [BLOCK])
    assert _call() == {"content": "API_BRAIN_ERROR"}
    assert len(calls) == 4
