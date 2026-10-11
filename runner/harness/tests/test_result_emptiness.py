# -*- coding: utf-8 -*-
"""One fact, one definition: is an action result empty?

Two independent readings of emptiness can disagree on every PhysicianBench action:

  base capability  fhir_empty_targeted_search_broad_reacquire reads the RAW result -- entries, total,
                   result_status == "empty_result".
  a second reader  that reads canonical_observation(res)["modalities"] depends on canonical_observation keeping
                   the body. A PB FHIR result is {"entries": [...], "pages": ..., "total": ...} and has neither
                   res["observation"] nor res["output"]; if only those two channels were copied, modalities would
                   be {} and every PB action -- including a search that returned a full Patient -- would read
                   as empty.

Fixtures must use shapes the producer actually produces. A fixture like {"total": 3, "output": "three
conditions"} gives canonical_observation an `output` channel the real FHIR environment never sends, so
assertions about emptiness pass on a payload that cannot occur. The fixtures and the E2E below use the shape
read back off a real trajectory.

Run:  python -m pytest runner/harness/tests/test_result_emptiness.py -q
"""
import json
import os
import sys

import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
for _p in (_REPO, os.path.join(_REPO, "runner")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import canonical_schema as canon                                  # noqa: E402
from harness.result_facts import result_is_empty                  # noqa: E402
from workspace import H0, Workspace                              # noqa: E402


@pytest.fixture(autouse=True)
def _no_harness_judge(monkeypatch):
    """build_kernel binds the harness judge from configuration and refuses without it (runner/judges.py). These
    tests exercise the kernel with no judge wired, so the judge builder is replaced by one that returns none."""
    import harness as _H
    monkeypatch.setattr(_H, "_build_judge", lambda: (None, None))

_Reacquire = Workspace(H0["fhir"]).load("middleware/fhir_empty_targeted_search_broad_reacquire.py", "FhirEmptyTargetedSearchBroadReacquire")


# The exact shapes a PhysicianBench FHIR tool returns, read back off a real trajectory of
# PB-opioid_use_disorder -- not invented, and carrying no `output` key.
PB_PATIENT_FULL = {"entries": [{"resourceType": "Patient", "id": "MRN7567496105",
                                "meta": {"versionId": "1"}}], "pages": 1, "total": 1}
PB_SEARCH_NONE = {"entries": [], "pages": 0, "total": 0}


# ------------------------------------------------------------------ non-empty and empty PB results

def test_a_pb_patient_search_that_returned_a_patient_is_not_empty():
    assert result_is_empty(PB_PATIENT_FULL) is False


def test_a_pb_search_that_returned_nothing_is_empty():
    assert result_is_empty(PB_SEARCH_NONE) is True


# ------------------------------------------------------------------ the same answer through every shape

@pytest.mark.parametrize("wrap", [
    lambda r: r,
    lambda r: canon.canonical_observation(r, "fhir"),
    lambda r: {"result_ok": True, "result_status": "ok",
               "observation": canon.canonical_observation(r, "fhir")},
], ids=["raw_result", "canonical_observation", "view_envelope"])
def test_one_answer_whichever_shape_the_fact_is_asked_about(wrap):
    assert result_is_empty(wrap(PB_PATIENT_FULL)) is False
    assert result_is_empty(wrap(PB_SEARCH_NONE)) is True


def test_the_canonicaliser_no_longer_drops_a_payload_it_has_no_named_channel_for():
    """A body with neither `observation` nor `output` must not be discarded; otherwise the judge's excerpt is
    empty too -- not only its emptiness flag."""
    mods = canon.canonical_observation(PB_PATIENT_FULL, "fhir")["modalities"]
    assert mods, "a FHIR result must survive canonicalisation"
    assert "MRN7567496105" in json.dumps(mods), mods


# ------------------------------------------------------------------ every consumer reads the one definition

def test_the_base_capability_reads_the_one_definition():
    """THE invariant. The capability's emptiness reading must agree with the shared fact on every shape."""
    cap = _Reacquire()
    for payload in (PB_PATIENT_FULL, PB_SEARCH_NONE,
                    {"result_status": "empty_result"}, {"output": "three conditions"}, {"output": ""}):
        fact = result_is_empty(payload)
        if fact is True:
            assert cap._empty_result(payload) is True, payload
        elif fact is False:
            assert cap._empty_result(payload) is False, payload


def test_the_capability_keeps_its_own_policy_without_redefining_the_fact():
    """A malformed-filter ERROR is not an empty result -- result_status already says "failed" -- but it
    is still not evidence of absence, so this capability acts on it. Separating the two keeps that
    behaviour without letting `result_empty` claim an error was an empty result."""
    err = {"error": "search() got an unexpected keyword argument 'date'"}
    assert result_is_empty(err) is False, "an error is a failure, not an empty result"
    assert _Reacquire()._empty_result(err) is True, "the capability's own policy must be unchanged"


# ------------------------------------------------------------------ unknown is not True

def test_an_unrecognised_shape_reports_unknown_instead_of_empty():
    """THE mechanism. A reader must not turn "I do not know this shape" into "it was empty".
    A future environment with a new payload shape must be visible, not silently reported as empty."""
    assert result_is_empty({"some_new_env_field": 1}) is None
    assert result_is_empty(None) is None


def test_an_artifact_counts_as_content():
    assert result_is_empty({"written": "/workspace/output/assessment.txt"}) is False


def test_an_explicit_declaration_wins_over_the_body():
    assert result_is_empty({"result_status": "empty_result", "entries": [{"x": 1}]}) is True


# ------------------------------------------------------------------ through the real kernel

def test_e2e_broad_search_has_results_and_the_narrow_one_does_not(monkeypatch):
    """End to end through the real kernel, with the payload shape the real FHIR tools return (NO `output`
    key -- a fixture with that key exercises a shape the environment never sends). The narrow search that
    came back empty is the one the base capability acts on; the broad one with results is not."""
    from harness import build_kernel                                       # noqa: E402
    p = os.path.join(_REPO, "benchmark_dataprocess", "PhysicianBench", "tasks_unified.jsonl")
    if not os.path.exists(p):
        pytest.skip("no PB task file")
    tools = sorted({(x.get("name") if isinstance(x, dict) else x)
                    for line in open(p, encoding="utf-8")
                    for x in (json.loads(line).get("available_tools") or []) if x})
    task = {"task_id": "EMPTY-E2E", "goal": "Work up suspected osteomyelitis.",
            "context": {"patient_ref": "Patient/1", "text": "case"},
            "available_tools": [{"name": t} for t in tools],
            "environment": {"type": "fhir", "config": {}}, "checkpoints": []}
    k = build_kernel(task, env_type="fhir", workspace=H0["fhir"])

    def _reacquire_findings():
        return [f for f in k.ledger.findings if f.get("capability") == _Reacquire.name]

    runs = [({"patient": "P1", "code": "BROAD_HAS_RESULTS"}, PB_PATIENT_FULL),   # has results
            ({"patient": "P1", "code": "NARROW_A"}, PB_SEARCH_NONE)]            # none
    seen = []
    for i, (args, res) in enumerate(runs):
        act = {"type": "tool_call", "tool": "fhir_condition_search_problems", "args": args}
        k.after_action(act, res, "h0", "h0", step=i,
                       canonical_observation=canon.canonical_observation(res, "fhir"),
                       result_ok=True, raw_observation=res, result_status="ok")
        seen.append(len(_reacquire_findings()))

    assert seen == [0, 1], (seen, k.ledger.findings)
    assert _reacquire_findings()[0]["reason_code"] == "empty_targeted_search"


def test_a_fhir_create_returns_the_resource_and_that_is_content():
    """`create` answers with the resource it made -- {"resourceType": "ServiceRequest", "id": ...} with
    no entries/total/output (e.g. ServiceRequest, MedicationRequest). Such a result is content."""
    made = {"resourceType": "ServiceRequest", "id": "212152", "status": "active", "intent": "order"}
    assert result_is_empty(made) is False


def test_a_search_result_is_not_mistaken_for_a_returned_resource():
    """resourceType is checked AFTER entries/total, so a Bundle-shaped empty search stays empty."""
    assert result_is_empty({"resourceType": "Bundle", "id": "b1", "entries": [], "total": 0}) is True


def test_every_real_result_shape_in_the_last_round_is_classified():
    """The mechanism check on real data: no shape the environment actually produces may land on unknown.
    Replays every tool result from a set of stored PhysicianBench trajectories (skipped when they are not on
    this machine); every result must classify as empty or non-empty."""
    import glob
    pats = sorted(glob.glob(os.path.join(_REPO, "examples", "pb", "*", "trajectory.jsonl")))
    if not pats:
        pytest.skip("no PhysicianBench trajectory under examples/pb")
    seen, unknown = 0, []
    for p in pats:
        for line in open(p, encoding="utf-8"):
            ev = json.loads(line)
            if ev.get("event_type") != "tool_call":
                continue
            seen += 1
            if result_is_empty(ev.get("result")) is None:
                unknown.append((ev.get("tool"),
                                tuple(sorted(ev["result"].keys()))[:6]
                                if isinstance(ev.get("result"), dict) else type(ev.get("result")).__name__))
    assert seen > 0, seen
    assert not unknown, "unclassified real shapes: %s" % sorted(set(unknown))[:5]


def test_a_serialised_result_still_states_its_own_emptiness():
    """The capability's reader also accepts a serialised (JSON string) result; the shared definition keeps that,
    so moving to it does not silently change the base harness's behaviour."""
    assert result_is_empty(json.dumps({"entries": [], "total": 0})) is True
    assert result_is_empty(json.dumps({"entries": [{"a": 1}], "total": 1})) is False
    assert result_is_empty("empty_result") is True
