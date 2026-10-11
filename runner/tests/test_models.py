# -*- coding: utf-8 -*-
"""runner/models.yaml, the agent model table: what it lists, that a model it does not list fails closed unless its
protocol is declared beside its id, and that a declaration never overrides the table."""
import os
import sys

import pytest
import yaml

_RUNNER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RUNNER not in sys.path:
    sys.path.insert(0, _RUNNER)

import models as M                                      # noqa: E402


def test_every_listed_model_maps_to_a_protocol():
    raw = yaml.safe_load(open(M.TABLE_PATH, encoding="utf-8"))["models"]
    assert M.table() == {str(k): v for k, v in raw.items()}
    assert set(M.table().values()) == set(M.PROTOCOLS)
    for model, protocol in M.table().items():
        assert M.protocol(model) == protocol


def test_an_unknown_model_is_an_error_not_a_default(monkeypatch):
    monkeypatch.delenv(M.DECLARED_VAR, raising=False)          # unlisted AND undeclared
    for model in ("no-such-model", "", None, "GPT-5.5", "gpt-5.5 "):
        with pytest.raises(M.UnknownModelError):
            M.protocol(model)


def test_an_unlisted_model_runs_when_its_protocol_is_declared(monkeypatch):
    """Naming a model means naming the protocol it speaks: a model the table does not list runs with the protocol
    declared beside its id, and does not run without one."""
    monkeypatch.delenv(M.DECLARED_VAR, raising=False)
    with pytest.raises(M.UnknownModelError):
        M.protocol("not-in-the-table")
    monkeypatch.setenv(M.DECLARED_VAR, M.FUNCTION_CALLING)
    assert M.protocol("not-in-the-table") == M.FUNCTION_CALLING
    assert M.protocol_for("not-in-the-table", M.TEXT) == M.TEXT


def test_a_declaration_never_overrides_the_table(monkeypatch):
    """For a model the table lists, the table is the record -- a declaration that contradicts it is an error, not an
    override (the protocol follows the model)."""
    listed, proto = sorted(M.table().items())[0]
    other = M.TEXT if proto == M.FUNCTION_CALLING else M.FUNCTION_CALLING
    assert M.protocol_for(listed, proto) == proto
    with pytest.raises(M.UnknownModelError):
        M.protocol_for(listed, other)
    monkeypatch.setenv(M.DECLARED_VAR, other)
    with pytest.raises(M.UnknownModelError):
        M.protocol(listed)


def test_a_mistyped_declaration_is_an_error_not_a_default(monkeypatch):
    monkeypatch.setenv(M.DECLARED_VAR, "functioncalling")
    with pytest.raises(M.UnknownModelError):
        M.declared_protocol()
    monkeypatch.setenv(M.DECLARED_VAR, "  ")
    assert M.declared_protocol() is None


@pytest.mark.parametrize("text", ["models:\n  m: xml\n", "models: {}\n", "other: 1\n", ""])
def test_a_table_that_is_not_model_to_protocol_is_refused(monkeypatch, tmp_path, text):
    p = tmp_path / "models.yaml"
    p.write_text(text)
    monkeypatch.setattr(M, "TABLE_PATH", str(p))
    monkeypatch.setattr(M, "_table", None)
    with pytest.raises(M.UnknownModelError):
        M.table()


def test_the_agent_model_is_the_wire_model_string(monkeypatch):
    monkeypatch.delenv("MH_API_MODEL", raising=False)
    monkeypatch.delenv("MH_OPENAI_MODEL", raising=False)
    assert M.agent_model() == "gpt-5.5"
    monkeypatch.setenv("MH_OPENAI_MODEL", "kimi-k2.7-code")
    assert M.agent_model() == "kimi-k2.7-code"
    monkeypatch.setenv("MH_API_MODEL", "deepseek-v4-flash")
    assert M.agent_model() == "deepseek-v4-flash"
