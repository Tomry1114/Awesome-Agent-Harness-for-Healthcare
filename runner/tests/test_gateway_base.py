"""A caller with an endpoint of its own reaches it: an explicit `base` passed to gateway.chat (e.g. the MedCTA
perception backend's MH_VLM_API_BASE) is the endpoint called, instead of MH_OPENAI_BASE; without a base the shared
endpoint is used."""
import io
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import gateway as G


class _Resp(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_an_explicit_base_is_the_endpoint_called(monkeypatch):
    seen = []

    def fake(req, timeout=None):
        seen.append(req.full_url)
        return _Resp(json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode())
    monkeypatch.setenv("MH_OPENAI_BASE", "https://shared.example")
    monkeypatch.setattr(G.urllib.request, "urlopen", fake)
    G.chat([{"role": "user", "content": "x"}], model="m", key="k", base="https://own.example/", retries=1)
    G.chat([{"role": "user", "content": "x"}], model="m", key="k", retries=1)
    assert seen[0].startswith("https://own.example/v1/chat/completions")
    assert seen[1].startswith("https://shared.example/v1/chat/completions"), "no base: unchanged"
