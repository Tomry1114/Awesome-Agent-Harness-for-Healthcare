# -*- coding: utf-8 -*-
"""Model configuration: the protocol each agent model is run with.

Part of the fixed shell. Whether a model is driven with native function calling (transport.chat_fc) or with the
<tool_call>/<answer> text protocol (transport.chat_text) is a property of the model: some models support function calling
and some only the text protocol. So it is configuration like the model id, kept in runner/models.yaml, and not a run
switch. Every default (H0) workspace holds one agent program per protocol (manifest agent_program /
agent_program_text); the shell's agent object runs the one for its model's protocol (workspace.WorkspaceAgent, built by
agents.make_agent).

A model the table lists runs with the protocol the table gives it, always. A model it does not list runs only when
its protocol is DECLARED beside its model id (MH_MODEL_PROTOCOL, the way MH_API_MODEL names the model): naming a model
means naming the protocol it speaks, so this is model configuration like the model id itself, not a run switch. A
model that is neither listed nor declared does not run: protocol() raises UnknownModelError, which make_agent hits
before the first turn and run_batch.py before the first task. A declaration that contradicts the table is an error,
not an override -- for a listed model the table is the record (the protocol follows the model).
"""
import os

import yaml

FUNCTION_CALLING = "function_calling"
TEXT = "text"
PROTOCOLS = (FUNCTION_CALLING, TEXT)
TABLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models.yaml")

_table = None


class UnknownModelError(RuntimeError):
    """An agent model runner/models.yaml does not list, or a table that cannot be read as model -> protocol."""


def agent_model():
    """The agent model a run puts on the wire: MH_API_MODEL (MH_OPENAI_MODEL kept for back-compat), else gpt-5.5."""
    return os.environ.get("MH_API_MODEL") or os.environ.get("MH_OPENAI_MODEL", "gpt-5.5")


def table():
    """model id -> protocol, as runner/models.yaml lists it. Read once per process."""
    global _table
    if _table is None:
        with open(TABLE_PATH, encoding="utf-8") as f:
            models = (yaml.safe_load(f) or {}).get("models")
        if not isinstance(models, dict) or not models:
            raise UnknownModelError("%s lists no models" % TABLE_PATH)
        bad = sorted("%s: %r" % (m, p) for m, p in models.items() if p not in PROTOCOLS)
        if bad:
            raise UnknownModelError("%s: not a protocol (%s): %s" % (TABLE_PATH, " or ".join(PROTOCOLS), "; ".join(bad)))
        _table = {str(m): p for m, p in models.items()}
    return _table


DECLARED_VAR = "MH_MODEL_PROTOCOL"


def declared_protocol():
    """The protocol declared for this run's agent model (MH_MODEL_PROTOCOL), or None. A value that is not a protocol
    is an error: a mistyped declaration must not fall back to a default."""
    d = (os.environ.get(DECLARED_VAR) or "").strip()
    if not d:
        return None
    if d not in PROTOCOLS:
        raise UnknownModelError("%s=%r is not a protocol (%s)" % (DECLARED_VAR, d, " or ".join(PROTOCOLS)))
    return d


def protocol_for(model, declared=None):
    """The protocol `model` runs with. runner/models.yaml decides for a model it lists, and a `declared` protocol that
    contradicts the table is an error. For a model the table does not list, `declared` is the protocol -- and without
    one the model does not run."""
    p = table().get(model)
    if p is not None:
        if declared and declared != p:
            raise UnknownModelError("%s is listed in %s with protocol %r, not %r: the protocol follows the model"
                                    % (model, TABLE_PATH, p, declared))
        return p
    if declared:
        return declared
    raise UnknownModelError("agent model %r is not in %s and no protocol was declared for it; give its protocol with "
                            "%s=%s, or add it to the table" % (model, TABLE_PATH, DECLARED_VAR,
                                                               "|".join(PROTOCOLS)))


def protocol(model):
    """The protocol `model` runs with, reading the declaration this run was launched with."""
    return protocol_for(model, declared_protocol())
