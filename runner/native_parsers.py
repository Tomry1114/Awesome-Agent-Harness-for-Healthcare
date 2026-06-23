#!/usr/bin/env python3
"""Native-track parsers (Canonical Contract §6-7). When the model runs the NATIVE protocol, these turn
its raw output into the SAME CanonicalAction the unified track produces — so both tracks feed one
CanonicalTrace and share the audit, while the model still sees its native prompt/protocol. raw_model_
output is preserved on every parse.

  parse_react   : MedCTA Lagent ReAct (Thought/Action/Action Input)
  parse_bracket : HAB upstream click([id]) / fill([id],"text") / download([id]) / done() ...
"""
import re, json
from canonical_schema import GUI_OPS, FILE_OPS, CONTROL_OPS


def _wrap(canon, raw):
    return {"canonical_action": canon, "raw_model_output": raw[:600]}


def parse_react(text):
    """MedCTA Lagent ReAct -> CanonicalAction."""
    m_act = re.search(r"Action\s*:\s*([A-Za-z_][A-Za-z0-9_]*)", text)
    if not m_act:
        fa = re.search(r"(?:Final Answer|FinalAnswer)\s*:\s*(.+)", text, re.S)
        return _wrap({"action_type": "final_answer", "content": (fa.group(1).strip() if fa else text.strip())}, text)
    name = m_act.group(1).strip()
    if name.lower() in ("finish", "final", "finalanswer"):
        fa = re.search(r"Action\s*Input\s*:\s*(.+)", text, re.S)
        return _wrap({"action_type": "final_answer", "content": (fa.group(1).strip() if fa else "")}, text)
    m_in = re.search(r"Action\s*Input\s*:\s*(\{.*\}|.+)", text, re.S)
    args = {}
    if m_in:
        blob = m_in.group(1).strip()
        try:
            args = json.loads(blob)
        except Exception:
            args = {} if not blob else {"input": blob[:200]}
    return _wrap({"action_type": "tool_call", "name": name, "arguments": args}, text)


def parse_bracket(text):
    """HAB upstream bracket actions -> CanonicalAction."""
    m = re.search(r"\b(navigate|click|fill|type|select|check|scroll|back|submit|download|upload|done)\s*\(([^)]*)\)", text)
    if not m:
        return _wrap({"action_type": "final_answer", "content": text.strip()}, text)
    op, inner = m.group(1), m.group(2).strip()
    mid = re.search(r"\[([^\]]+)\]", inner)
    eid = mid.group(1).strip() if mid else None
    qs = re.search(r'"([^"]*)"', inner)
    val = qs.group(1) if qs else None
    target = {"element_id": eid} if eid else None
    if op == "done":
        return _wrap({"action_type": "control_action", "operation": "done"}, text)
    if op in FILE_OPS:
        return _wrap({"action_type": "file_action", "operation": op, "target": target, "file_ref": val}, text)
    if op == "scroll":
        return _wrap({"action_type": "gui_action", "operation": "scroll", "value": (inner or "down")}, text)
    if op == "fill":
        op = "type"
    return _wrap({"action_type": "gui_action", "operation": op, "target": target, "value": val}, text)


# ---- conformance test (no network) ----
_CASES = [
    (parse_react, "Thought: I should read the image.\nAction: ImageDescription\nAction Input: {}",
     {"action_type": "tool_call", "name": "ImageDescription"}),
    (parse_react, "Thought: zoom in.\nAction: RegionAttributeDescription\nAction Input: {\"region\": \"liver\"}",
     {"action_type": "tool_call", "name": "RegionAttributeDescription"}),
    (parse_react, "Thought: done.\nFinal Answer: Portal vein thrombosis.",
     {"action_type": "final_answer"}),
    (parse_bracket, "ACTION: click([submit-appeal])", {"action_type": "gui_action", "operation": "click"}),
    (parse_bracket, "ACTION: fill([reason-input], \"missing docs\")", {"action_type": "gui_action", "operation": "type"}),
    (parse_bracket, "ACTION: download([auth-letter])", {"action_type": "file_action", "operation": "download"}),
    (parse_bracket, "ACTION: done()", {"action_type": "control_action", "operation": "done"}),
]
if __name__ == "__main__":
    ok = 0
    for fn, inp, exp in _CASES:
        got = fn(inp)["canonical_action"]
        good = all(got.get(k) == v for k, v in exp.items())
        ok += good
        print(("PASS" if good else "FAIL"), fn.__name__, "->", {k: got.get(k) for k in ("action_type", "operation", "name")})
    print("conformance: %d/%d" % (ok, len(_CASES)))
