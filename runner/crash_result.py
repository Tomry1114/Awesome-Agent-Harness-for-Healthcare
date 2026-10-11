# -*- coding: utf-8 -*-
"""The result a task gets when the run itself fails -- an exception out of the loop, a workspace hook, the
kernel, or the runner around it.

Every task gets a result.json, crashed or not: aggregation globs result.json, so a task with no result would
vanish from the denominator, and a harness that crashed on hard tasks would report a higher mean than one
that finished them.

A crash is never the agent's failure and never a pass. The result says so explicitly:

    end_reason      "harness_error"
    outcome_status  "harness_error"   (native outcome is None: unresolved, with the reason)
    checkpoints     []                (nothing was scored -- scoring a half-run episode would invent a verdict)
    harness_error   exception type, message, the traceback's frames and a sha256 digest of them, and which part
                    of the system the innermost repository frame belongs to
"""
import hashlib
import os
import traceback

MODULES = ["Care Action Fulfillment", "Clinical Resource Competence", "Patient–Evidence Alignment", "Clinical Workflow Continuity", "Clinical Auditability", "Closed-Loop Verification", "Clinical Conduct Boundaries"]
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _origin(path):
    p = os.path.abspath(path or "")
    rel = os.path.relpath(p, _REPO) if p.startswith(_REPO + os.sep) else None
    if rel is None:
        return "external"
    if rel.startswith("workspaces" + os.sep):
        return "workspace"
    if rel.startswith(os.path.join("runner", "harness") + os.sep) or rel == os.path.join("runner", "middleware_host.py"):
        return "harness"
    if rel.startswith("runner" + os.sep):
        return "runner"
    return "repository"


def digest(exc, tb=None):
    """-> {type, message, frames, traceback_sha256, origin}. The digest covers the frame sequence (file, line,
    function) and the exception type, not the message, so the same failure at the same place hashes the same
    even when the message carries a volatile value."""
    tb = tb if tb is not None else getattr(exc, "__traceback__", None)
    frames = traceback.extract_tb(tb) if tb is not None else []
    rows = ["%s:%d in %s" % (os.path.relpath(f.filename, _REPO) if os.path.abspath(f.filename).startswith(_REPO)
                             else f.filename, f.lineno or 0, f.name) for f in frames]
    h = hashlib.sha256(("%s\n%s" % (type(exc).__name__, "\n".join(rows))).encode("utf-8")).hexdigest()
    inner = next((f for f in reversed(frames) if _origin(f.filename) != "external"), None)
    return {"type": type(exc).__name__, "message": str(exc)[:500], "frames": rows[-12:],
            "traceback_sha256": h, "origin": _origin(inner.filename) if inner is not None else "external"}


def crash_result(bench, task_id, exc, *, agent_name=None, max_steps=None, trajectory=None, git_sha=None):
    """A schema-valid result for a task whose run raised `exc`. Private keys (leading underscore) follow the
    run_task convention and are stripped before result.json is written."""
    traj = list(trajectory or [])
    d = digest(exc)
    return {
        "task_id": task_id, "success": False, "evaluation_status": "error",
        "end_reason": "harness_error", "outcome_status": "harness_error",
        "harness_error": d,
        "checkpoints": [], "dimension_scores": {m: None for m in MODULES},
        "strict_checkpoint_dimensions": {m: None for m in MODULES},
        "failure_tags": [], "tool_calls": sum(1 for e in traj if isinstance(e, dict)
                                               and e.get("event_type") == "tool_call"),
        "budget": {"max_steps": max_steps},
        "harness": {"mode": "enforce", "requested_mode": "enforce", "effective_mode": "crashed",
                    "status": "failed", "runtime_errors": ["harness_error: %s: %s" % (d["type"], d["message"][:200])]},
        "provenance": {"agent_model": str(agent_name or "unknown"), "task_id": task_id, "source_benchmark": bench,
                       "api_model_string": os.environ.get("MH_API_MODEL") or os.environ.get("MH_OPENAI_MODEL"),
                       "git_sha": git_sha},
        "qualification": ["harness_error"],
        "_trajectory": traj, "_instrumentation": [], "_checkpoints_full": [],
    }
