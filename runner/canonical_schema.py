#!/usr/bin/env python3
"""Canonical schema normalization layer (Canonical Contract §1-4). Unifies at the TRACE level: the
agent emits its protocol, envs execute native actions, and this layer normalizes both into typed
CanonicalAction / CanonicalObservation / CanonicalResult / CanonicalError so one audit applies across
PB / MedCTA / HAB. Does NOT force the agent protocol — it classifies the existing action/result dicts.

  CanonicalAction  : ToolCall | GUIAction | FileAction | ControlAction | FinalAnswer
  CanonicalResult  : Success | Failure | StateChange | ArtifactProduced | NoProgress
  CanonicalError   : InvalidAction | ToolError | EnvironmentError | MissingCapability | Timeout | InfrastructureError
"""

GUI_OPS = {"navigate", "click", "type", "select", "check", "scroll", "back", "submit", "snapshot"}
CONTROL_OPS = {"done", "abort", "retry", "wait", "escalate"}
FILE_OPS = {"upload", "download"}


def normalize_target(args):
    """Extensible target: element_id -> role/name -> text -> selector -> coordinates (priority order)."""
    a = args or {}
    tgt = {}
    if a.get("ref") is not None:
        tgt["element_id"] = str(a["ref"])
    for k in ("role", "name", "text", "selector"):
        if a.get(k):
            tgt[k] = a[k]
    if a.get("coordinates"):
        tgt["coordinates"] = a["coordinates"]
    return tgt or None


def canonical_action(raw, env_type):
    if not isinstance(raw, dict):
        return {"action_type": "invalid", "raw": str(raw)[:200]}
    t = raw.get("type")
    if t == "final":
        return {"action_type": "final_answer", "content": raw.get("answer", "")}
    if t in ("tool_call_truncated", "invalid_action", "bad_action_type"):
        return {"action_type": "invalid", "raw": (raw.get("raw") or "")[:200]}
    tool = raw.get("tool")
    if not isinstance(tool, str) or not tool.strip():   # a non-final action with no usable tool name is malformed
        return {"action_type": "invalid", "raw": str(raw)[:200]}
    args = raw.get("args") or {}
    if tool in CONTROL_OPS:
        return {"action_type": "control_action", "operation": tool, "reason": args.get("reason")}
    if env_type in ("gui", "healthadminbench"):
        if tool in FILE_OPS:
            return {"action_type": "file_action", "operation": tool,
                    "file_ref": args.get("file_ref"), "destination": args.get("destination"),
                    "target": normalize_target(args)}
        if tool in GUI_OPS:
            return {"action_type": "gui_action", "operation": tool,
                    "target": normalize_target(args),
                    "value": args.get("value") or args.get("text") or args.get("url")}
    if tool == "write_file":
        return {"action_type": "file_action", "operation": "write",
                "path": args.get("path"), "content_len": len(str(args.get("content") or ""))}
    return {"action_type": "tool_call", "name": tool, "arguments": args}


# Action shapes that represent a SYNTACTICALLY/SEMANTICALLY USABLE action the harness can dispatch. A
# 'final_answer' is a well-formed terminal action; a control_action / gui_action / file_action / tool_call
# all parse into a usable shape. Only 'invalid' (the malformed bucket above) is NOT well-formed.
WELL_FORMED_ACTION_TYPES = ("tool_call", "final_answer", "control_action",
                            "gui_action", "file_action")


def action_valid(raw, env_type="tool_sandbox"):
    """PURE protocol/schema validity: is this parsed action WELL-FORMED (a usable tool_call/final/etc.)?
    True for any action that classifies into a dispatchable canonical shape; False ONLY for the 'invalid'
    bucket — i.e. a non-dict action, a truncated tool call, a bad/unknown action type, or a tool_call with
    no tool name. This is INDEPENDENT of whether a well-formed action later FAILED at execution: a tool
    that ran and returned an error is still a valid (well-formed) action.

    Note the malformed agent_error markers run.py emits (invalid_action / bad_action_type /
    truncated_tool_call) classify here as action_type=='invalid' -> action_valid == False."""
    return canonical_action(raw, env_type).get("action_type") != "invalid"


def classify_error(err):
    e = str(err).lower()
    if "unknown" in e or "invalid" in e:
        return "InvalidAction"
    if "timeout" in e:
        return "Timeout"
    if "missing_capability" in e or "capability" in e:
        return "MissingCapability"
    if "upload_failed" in e or "download_failed" in e or "tool" in e:
        return "ToolError"
    if "infrastructure" in e or "connection" in e or "502" in e or "503" in e:
        return "InfrastructureError"
    return "EnvironmentError"


def canonical_result(res):
    """Classify into Success / Failure / StateChange / ArtifactProduced / NoProgress.
    KEY: API success != semantic progress (state_changed=False -> NoProgress)."""
    if not isinstance(res, dict):
        return {"status": "success", "kind": "Success"}
    if res.get("error"):
        return {"status": "failure", "kind": "Failure", "error_type": classify_error(res["error"]),
                "detail": str(res["error"])[:200]}
    if res.get("downloaded") or res.get("uploaded") or res.get("written"):   # a WRITTEN file is a first-class artifact
        return {"status": "success", "kind": "ArtifactProduced",
                "artifact": res.get("downloaded") or res.get("uploaded") or res.get("written")}
    sc = res.get("state_changed")
    if sc is False:
        return {"status": "success", "kind": "NoProgress", "state_changed": False, "surface_changed": False}
    if sc is True:
        return {"status": "success", "kind": "StateChange", "state_changed": True}
    return {"status": "success", "kind": "Success"}


def canonical_observation(res, env_type):
    if not isinstance(res, dict):
        return {"observation_type": "environment_state", "modalities": {"text": str(res)[:4000]}}
    modalities = {}
    if res.get("observation"):
        modalities["text"] = res["observation"]
    out = res.get("output")
    if out is not None:
        modalities["structured" if not isinstance(out, str) else "text"] = out
    if res.get("image_ref"):
        modalities["image_ref"] = res["image_ref"]
    artifacts = [res[k] for k in ("downloaded", "uploaded", "written") if res.get(k)]
    return {"observation_type": "environment_state", "modalities": modalities,
            "current_url": res.get("url"), "artifacts": artifacts,
            "previous_action_result": canonical_result(res)}
