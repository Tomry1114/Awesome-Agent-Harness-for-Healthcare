"""The agent brain can have an endpoint of its own while everything else stays on MH_OPENAI_BASE."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import workspace as W


def test_default_is_the_shared_endpoint(monkeypatch):
    for k in ("MH_AGENT_BASE", "MH_AGENT_KEY_FILE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("MH_OPENAI_BASE", "https://gw.example/")
    monkeypatch.setenv("MH_OPENAI_KEY", "shared")
    assert W.agent_route() == ("https://gw.example", "shared")


def test_agent_endpoint_of_its_own(monkeypatch, tmp_path):
    kf = tmp_path / "k"
    kf.write_text("relay\n")
    monkeypatch.setenv("MH_OPENAI_BASE", "https://gw.example")
    monkeypatch.setenv("MH_OPENAI_KEY", "shared")
    monkeypatch.setenv("MH_AGENT_BASE", "https://relay.example/v1")
    monkeypatch.setenv("MH_AGENT_KEY_FILE", str(kf))
    assert W.agent_route() == ("https://relay.example", "relay")
