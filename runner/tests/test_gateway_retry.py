"""gateway.chat retries what can pass on a later try with some relays: a 429 that is not billing, the rerouted
400 "short-input distillation", and -- with MH_GATEWAY_RETRY_BUDGET -- a read timeout that used the whole `timeout`.
Billing 429s and other 4xx still return at once; without the budget a timeout still ends the call."""
import io
import json
import os
import socket
import sys
import urllib.error

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import gateway as G

CONC = b'{"error":{"code":"gateway_concurrency_limit","message":"Concurrency limit exceeded for user, please retry later","type":"rate_limit_error"}}'
QUOTA = '{"code":"API_KEY_QUOTA_EXHAUSTED","message":"API key 额度已用完"}'.encode()
DISTILL = b'{"error":{"message":"Upstream rejected illegal short-input distillation or heartbeat probing."}}'


class _Resp(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _rig(monkeypatch, script, budget=None):
    """script: list of ("ok",) | (code, body) | ("timeout",); the fake clock advances by the socket timeout on a
    timeout and by each backoff sleep."""
    clock = [1000.0]
    calls = []
    monkeypatch.setattr(G.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(G.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s))

    def fake_urlopen(req, timeout=None):
        step = script[min(len(calls), len(script) - 1)]
        calls.append(timeout)
        if step[0] == "ok":
            return _Resp(json.dumps({"choices": [{"message": {"content": "fine"}}]}).encode())
        if step[0] == "timeout":
            clock[0] += timeout
            raise socket.timeout("The read operation timed out")
        raise urllib.error.HTTPError(req.full_url, step[0], "x", {}, io.BytesIO(step[1]))
    monkeypatch.setattr(G.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("MH_JUDGE_KEY", "k")
    if budget is None:
        monkeypatch.delenv("MH_GATEWAY_RETRY_BUDGET", raising=False)
    else:
        monkeypatch.setenv("MH_GATEWAY_RETRY_BUDGET", str(budget))
    return calls


def _chat():
    return G.chat([{"role": "user", "content": "q"}], model="m", judge=True, timeout=120, retries=4,
                  base="http://judge")


def test_concurrency_429_is_retried(monkeypatch):
    calls = _rig(monkeypatch, [(429, CONC), (429, CONC), ("ok",)])
    assert _chat()["ok"] and len(calls) == 3


def test_persistent_429_is_waited_out_to_the_deadline_then_ends_as_http_4xx(monkeypatch):
    """A throttled call is not one of the `retries` attempts: it waits until the deadline."""
    calls = _rig(monkeypatch, [(429, CONC)])
    r = _chat()
    assert not r["ok"] and r["error_type"] == "http_4xx" and len(calls) > 4


RATE503 = b'{"error":{"code":"model:rate_limited","message":"You are sending requests too frequently in your current group. Please try again later.","type":"new_api_error"}}'


def test_rate_limit_503_is_waited_out_past_the_retries(monkeypatch):
    """A relay's rate-limit 503 ("sending requests too frequently") is waited out within the retry budget instead of
    counting against `retries`, so a judge call does not give up after a few quick tries."""
    calls = _rig(monkeypatch, [(503, RATE503)] * 6 + [("ok",)], budget=900)
    assert _chat()["ok"] and len(calls) == 7


def test_other_503_still_counts_against_retries(monkeypatch):
    calls = _rig(monkeypatch, [(503, b"upstream down")], budget=900)
    r = _chat()
    assert r["error_type"] == "http_5xx" and len(calls) == 4


def test_billing_429_is_not_retried(monkeypatch):
    calls = _rig(monkeypatch, [(429, QUOTA), ("ok",)])
    assert _chat()["error_type"] == "billing" and len(calls) == 1


def test_rerouted_400_is_retried_other_400_is_not(monkeypatch):
    calls = _rig(monkeypatch, [(400, DISTILL), ("ok",)])
    assert _chat()["ok"] and len(calls) == 2
    calls = _rig(monkeypatch, [(400, b'{"error":"bad schema"}'), ("ok",)])
    assert _chat()["error_type"] == "auth" and len(calls) == 1


def test_timeout_without_budget_still_ends_the_call(monkeypatch):
    calls = _rig(monkeypatch, [("timeout",), ("ok",)])
    r = _chat()
    assert r["error_type"] == "timeout" and len(calls) == 1


def test_timeout_with_budget_is_retried(monkeypatch):
    calls = _rig(monkeypatch, [("timeout",), ("ok",)], budget=360)
    assert _chat()["ok"] and len(calls) == 2 and calls[0] == 120


def test_an_upstream_403_is_retried_a_billing_403_is_not(monkeypatch):
    up = b'{"error":{"message":"BPS upstream response failed","type":"upstream_error"}}'
    calls = _rig(monkeypatch, [(403, up), ("ok",)], budget=900)
    assert _chat()["ok"] and len(calls) == 2
    bill = '{"error":{"message":"\u7528\u6237\u989d\u5ea6\u4e0d\u8db3","type":"new_api_error"}}'.encode()
    calls = _rig(monkeypatch, [(403, bill), ("ok",)], budget=900)
    assert _chat()["error_type"] == "billing" and len(calls) == 1


DAILY = (b'{"code":"USAGE_LIMIT_EXCEEDED","message":"error: code=429 reason=\\"DAILY_LIMIT_EXCEEDED\\" '
         b'message=\\"daily usage limit exceeded\\""}')


def test_a_spent_daily_allowance_is_billing_not_throttling(monkeypatch):
    """A spent daily usage allowance (DAILY_LIMIT_EXCEEDED in a 429 body) is billing: it returns at once instead of being
    waited out to the deadline as throttling on every call."""
    calls = _rig(monkeypatch, [(429, DAILY), ("ok",)], budget=900)
    assert _chat()["error_type"] == "billing" and len(calls) == 1
