#!/usr/bin/env python3
"""Deterministic shared scorer HELPERS (NOT an LLM judge).

Care Action Fulfillment and Clinical Workflow Continuity are scored by runner/scorers/dim_execution.execution and
runner/scorers/dim_lifecycle.lifecycle (the substrate-SemanticTrace path). This module holds three
deterministic helpers imported elsewhere:
  * error_attribution(ev) -> agent/environment/external_service/harness/unknown  (runner/substrate.py)
  * _sm(...)              -> one {score,status,opportunities} sub-metric cell     (dim_context, dim_tooling)
  * _aggregate(subs)      -> average ONLY applicable sub-metrics, never a vacuous 1.0  (dim_context, dim_tooling)
"""
import proxy_verifiers as _pv

# ENVIRONMENT / BACKEND faults. `vlm_api_error` / "does not represent a valid image" cover the MedCTA
# case where ImageDescription returns "[vlm_api_error] http_400: The image data you provided does not
# represent a valid image" -- the perception BACKEND rejecting the image the ADAPTER supplied (args can be
# empty), i.e. an environment/adapter defect, NOT an agent mistake. Without these markers such failures
# fall into "unknown" and failure diagnosability cannot separate them from agent errors.
_ENV = ("503", "502", "504", "timeout", "timed out", "connection", "unreachable", "refused",
        "crash", "econnreset", "service unavailable", "gateway",
        "vlm_api_error", "does not represent a valid image")
_HARNESS = ("adapter", "serialize", "serialization", "missing_synthetic_context", "not wired", "_as_entries")
# CALLER-FAULT signals. These are GENERIC (a Python TypeError from a bad call, an HTTP 4xx, a tool-level
# validation rejection of the caller request) -- not benchmark vocabulary -- so they belong here rather than
# in a plugin. _ENV and _HARNESS are matched FIRST, so a gateway/5xx/adapter failure still attributes away
# from the agent. "unknown" is EXCLUDED from the tool_invocation_success denominator, so an unclassified
# agent failure would silently vanish and the score could only go UP. Typical agent mistakes matched here:
# "got an unexpected keyword argument" (called a tool with a parameter that does not exist),
# "Appointment rejected: ... no based_on ServiceRequest ... Referrals must first ..." (violated a stated
# precondition), "400 Client Error" (malformed request body).
_AGENT = ("tool_argument", "tool_selection", "missing key", "unknown tool", "invalid argument",
          "bad_action", "invalid_action", "truncated_tool_call", "missing required key",
          "unexpected keyword argument", "client error", "rejected")


def error_attribution(ev):
    """Who is responsible for this failure? agent / environment / external_service / harness / unknown.
    Returns None if the event is not an error."""
    if not (ev.get("event_type") in ("tool_call", "agent_error") and (_pv._errored(ev) or ev.get("event_type") == "agent_error")):
        return None
    fm = ev.get("failure_mode"); et = str(ev.get("error_type", "")).lower()
    txt = (str(ev.get("result", "")) + " " + et + " " + str(ev.get("error", ""))).lower()
    if fm == "environment_error" or any(m in txt for m in _ENV):
        return "external_service" if any(m in txt for m in ("503", "502", "504", "gateway", "service unavailable")) else "environment"
    if fm == "verifier_error" or any(m in txt for m in _HARNESS):
        return "harness"
    if ev.get("event_type") == "agent_error" or et.startswith("http_4") or any(m in txt for m in _AGENT):
        return "agent"
    return "unknown"


# _sm / _aggregate are defined in dimension_core; these re-exports keep the import path
# `from lifecycle_exec import _sm, _aggregate` working.
try:                                                   # flat sys.path (runner/scorers on the path)
    from dimension_core import submetric as _sm, aggregate_applicable as _aggregate
except ImportError:                                    # pragma: no cover - packaged import
    from .dimension_core import submetric as _sm, aggregate_applicable as _aggregate
