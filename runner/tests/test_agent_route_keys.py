"""A key file listing several keys: each agent draws one of them, so concurrent agents spread over the keys; a
single-key file always yields that key."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import workspace as W  # noqa: E402


def test_several_keys_each_drawn(tmp_path, monkeypatch):
    kf = tmp_path / "keys"
    kf.write_text("k1\n\nk2\n")
    monkeypatch.setenv("MH_AGENT_BASE", "https://relay.example/v1")
    monkeypatch.setenv("MH_AGENT_KEY_FILE", str(kf))
    drawn = {W.agent_route()[1] for _ in range(200)}
    assert drawn == {"k1", "k2"}
    assert W.agent_route()[0] == "https://relay.example"


def test_one_key_as_before(tmp_path, monkeypatch):
    kf = tmp_path / "key"
    kf.write_text("only\n")
    monkeypatch.setenv("MH_AGENT_BASE", "https://relay.example")
    monkeypatch.setenv("MH_AGENT_KEY_FILE", str(kf))
    assert W.agent_route() == ("https://relay.example", "only")
