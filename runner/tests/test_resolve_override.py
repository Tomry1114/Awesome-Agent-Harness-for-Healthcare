"""MH_RESOLVE pins listed hosts to given addresses and leaves every other lookup alone."""
import os
import socket
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import resolve_override as RO  # noqa: E402


def test_pinned_host_goes_to_the_given_address(monkeypatch):
    seen = []
    monkeypatch.setattr(RO, "_ORIG", lambda host, *a, **k: seen.append(host) or [("af", 1, 6, "", (host, a[0]))])
    monkeypatch.setattr(socket, "getaddrinfo", socket.getaddrinfo)
    assert RO.install("relay.example:203.0.113.10,203.0.113.11") == {"relay.example": ["203.0.113.10", "203.0.113.11"]}
    out = socket.getaddrinfo("Relay.Example", 443)
    assert [o[4][0] for o in out] == ["203.0.113.10", "203.0.113.11"]
    socket.getaddrinfo("api.other.example", 443)
    assert seen[-1] == "api.other.example"


def test_unset_installs_nothing(monkeypatch):
    before = socket.getaddrinfo
    monkeypatch.delenv("MH_RESOLVE", raising=False)
    assert RO.install() == {} and socket.getaddrinfo is before
