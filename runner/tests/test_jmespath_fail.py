# -*- coding: utf-8 -*-
"""A JMESPath checkpoint whose query fails on the state's shape fails, as upstream HAB's evaluator does; a query that
does not parse stays a verifier error. Example of a shape failure: a join() over a null list, as in checkpoint 7 of
HAB-denial-medium-8/9."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scorers"))

import scoring as SC                                              # noqa: E402


def _cp(query, **kw):
    chk = dict({"method": "jmespath", "query": query}, **kw)
    return {"id": "cp7_document_handling", "type": "deterministic", "dimension": "Outcome", "check": chk,
            "verifier": "deterministic"}


def _score(cp, state):
    return SC._ev_deterministic(cp, {"full_state": state}, {"id": cp["id"]})


def test_a_type_error_on_the_state_fails_the_checkpoint():
    r = _score(_cp("join(', ', full_state.agentActions.uploadedDocs)", contains="denial"),
               {"full_state": {"agentActions": {}}})
    assert r["checkpoint_status"] == "failed" and r["failure_mode"] == "agent_failure"
    assert "JMESPathTypeError" in r["detail"]["query_error"]


def test_a_query_that_does_not_parse_stays_a_verifier_error():
    r = _score(_cp("join(', ', full_state.[", contains="x"), {"full_state": {}})
    assert r["checkpoint_status"] == "error" and r["failure_mode"] == "verifier_error"


def test_a_good_query_is_scored_as_before():
    r = _score(_cp("full_state.agentActions.done", expected=True), {"full_state": {"agentActions": {"done": True}}})
    assert r["checkpoint_status"] == "passed"
