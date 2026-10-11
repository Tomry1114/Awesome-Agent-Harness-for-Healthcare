"""Decision-Boundary at the GUI fill point: a SEMANTIC slot binds ONLY from a semantic source via a reliable
match; never from system_metadata, never from a loose substring guess (no fabrication). Unbindable semantic
slot -> None -> caller BLOCKs. Operational slot may use metadata + loose match."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from workspace import H0, Workspace
from harness.bindings import (AGENT_COMMITMENT, AUTHORITATIVE_STATE, SYSTEM_METADATA)
GenericGuiCompletionWorkflow = Workspace(H0["gui"]).module("recovery/gui_completion.py").GenericGuiCompletionWorkflow

WF = GenericGuiCompletionWorkflow()
def sc(**kw):  # build class-keyed source dict
    return {AGENT_COMMITMENT: kw.get("commit", {}), AUTHORITATIVE_STATE: kw.get("auth", {}),
            "bound_evidence": {}, SYSTEM_METADATA: kw.get("meta", {})}
MEMBER = {"ref": "member-id-input", "name": "memberId", "label": "Member ID", "role": "input"}

def test_semantic_never_binds_from_system_metadata():
    assert WF._bind_value(MEMBER, sc(meta={"memberid": "AET123", "member-id-input": "AET123"})) is None

def test_semantic_binds_from_agent_commitment_by_ref():
    assert WF._bind_value(MEMBER, sc(commit={"member-id-input": "AET123"})) == "AET123"

def test_semantic_binds_from_authoritative_copyable_fact():
    assert WF._bind_value(MEMBER, sc(auth={"memberid": "AET999"})) == "AET999"

def test_semantic_no_fabrication_from_loose_substring():
    # a decision slot with only an unrelated 'type' key present -> must NOT be fuzzy-grabbed
    rt = {"ref": "request-type-select", "name": "requestType", "label": "Request Type", "role": "select"}
    assert WF._bind_value(rt, sc(auth={"someUnrelatedType": "outpatient"})) is None

def test_operational_slot_may_use_metadata():
    ts = {"ref": "ts", "name": "timestamp", "label": "Timestamp", "role": "input"}
    assert WF._bind_value(ts, sc(meta={"timestamp": "2026-07-04"})) == "2026-07-04"
