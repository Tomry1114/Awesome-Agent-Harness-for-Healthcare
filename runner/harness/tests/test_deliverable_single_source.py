# -*- coding: utf-8 -*-
"""The deliverable has ONE definition: the adapter that performs the write declares it, and the runtime reads
that declaration.

A length heuristic (any create/update/submit carrying a 200+ character string is the deliverable) would take a
long ServiceRequest note_text for the "artifact", while an offline replay counts only the file write: two
definitions under one name. These tests pin the single declared definition.
"""
import inspect
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_RUNNER = os.path.dirname(os.path.dirname(_HERE))
_REPO = os.path.dirname(_RUNNER)
for _p in (_RUNNER, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import canonical_schema as canon                       # noqa: E402
import harness as H                                    # noqa: E402
import workspace as W                                  # noqa: E402
from harness import substrate_catalog as SC            # noqa: E402
from harness.unknown import UNKNOWN                    # noqa: E402


@pytest.fixture(autouse=True)
def _no_harness_judge(monkeypatch):
    """build_kernel binds the harness judge from configuration and refuses without it (runner/judges.py). These
    tests exercise the kernel with no judge wired, so the judge builder is replaced by one that returns none."""
    import harness as _H
    monkeypatch.setattr(_H, "_build_judge", lambda: (None, None))

NOTE = "Rule out autoimmune hepatitis given persistently elevated transaminases in pregnancy; " * 3
assert len(NOTE) >= 200


def _kernel(env_type="fhir"):
    try:
        tools = SC.vocabulary(env_type)
    except SC.SubstrateUnavailable as exc:
        pytest.skip("substrate unavailable: %s" % exc)
    task = {"task_id": "DELIV", "goal": "Work up elevated liver enzymes.",
            "context": {"patient_ref": "Patient/1", "text": "case"},
            "available_tools": [{"name": t} for t in tools],
            "environment": {"type": env_type, "config": {}}, "checkpoints": []}
    return H.build_kernel(task, env_type=env_type, workspace=W.H0[env_type])


def _after(k, step, tool, args, ok=True):
    """One executed action, driven the way run.py drives it: before_action, then after_action. Calling only
    after_action would miss logic that runs in before_action."""
    res = {"output": "ok", "id": "X/%d" % step} if ok else {"error": "failed"}
    k.before_action({"type": "tool_call", "tool": tool, "args": args}, step=step)
    k.after_action({"type": "tool_call", "tool": tool, "args": args}, res, "s", "s", step=step,
                   canonical_observation=canon.canonical_observation(res, "fhir"),
                   result_ok=ok, raw_observation=res, result_status=("ok" if ok else "failed"))


# ------------------------------------------------------------------------------------ long notes are not deliverables
def test_a_long_order_note_is_not_the_deliverable():
    k = _kernel()
    _after(k, 31, "fhir_service_request_create", {"patient_reference": "Patient/1", "note_text": NOTE})
    assert k.ctx.last_write is None, "a ServiceRequest note was taken for the deliverable again"


def test_the_declared_write_is_the_deliverable_whatever_its_length():
    """A length floor would make a short deliverable ABSENT. The tool is what makes it the deliverable. The path
    is the parameter the tool declares, file_path (reading `path` instead gives None for native writes)."""
    k = _kernel()
    _after(k, 1, "write_file", {"file_path": "plan.txt", "content": "short"})
    art = k.ctx.last_write
    assert isinstance(art, dict) and art["content"] == "short" and art["path"] == "plan.txt"
    assert art["content_from_arg"] == "content" and art["path_from_arg"] == "file_path"
    _after(k, 2, "fhir_service_request_create", {"note_text": NOTE})
    assert k.ctx.last_write["content"] == "short", "a later non-deliverable write replaced the deliverable"


def test_a_write_that_did_not_execute_produced_no_deliverable():
    k = _kernel()
    k.before_action({"type": "tool_call", "tool": "write_file", "args": {"path": "p.txt", "content": "x" * 300}},
                    step=1)
    assert k.ctx.last_write is None, "a proposed write is recorded before anything ran"
    _after(k, 1, "write_file", {"path": "p.txt", "content": "x" * 300}, ok=False)
    assert k.ctx.last_write is None, "a failed write produced no deliverable"


def test_a_substrate_that_declares_no_deliverable_answers_unknown_not_absent():
    """MedCTA's adapter writes nothing. "The deliverable is missing" is not a fact there."""
    assert SC.deliverable_writes("tool_sandbox") is None
    k = _kernel("tool_sandbox")
    assert k.ctx.last_write is UNKNOWN


def test_an_unknown_substrate_is_not_a_substrate_without_a_deliverable():
    with pytest.raises(SC.SubstrateUnavailable):
        SC.deliverable_writes("no_such_env")


# ------------------------------------------------------------------------------------ one source, two readers
def test_the_kernel_reads_the_declaration_instead_of_carrying_its_own_list():
    from harness import kernel as K
    src = inspect.getsource(K.HarnessKernel._note_deliverable)
    assert "write_file" not in src, "the kernel names the tool instead of reading the declaration"
    assert "_deliverables" in src
    assert not hasattr(K.HarnessKernel, "_note_write"), "the length heuristic is gone, not bypassed"
