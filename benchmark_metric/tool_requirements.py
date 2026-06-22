"""Per-task tool requirements for the metrics layer (HARNESS-DERIVED, not paper-native).

PhysicianBench is OPEN-ENDED by design: the upstream benchmark does NOT pre-specify
required/sufficient tools per task (see https://healthrex.github.io/PhysicianBench/).
These requirements are therefore derived conservatively from each task's OWN goal text
+ environment type, and labelled harness-derived:
  - every clinical (fhir) task needs EHR retrieval         -> fhir_search | fhir_read
  - goal names a workspace/output deliverable              -> write_file
  - goal asks to place an order / prescribe / refer        -> fhir_create
  - admin (gui) workflow is not complete until submitted   -> submit
  - imaging (tool_sandbox) needs a perception tool         -> reference.sufficient_tools

Two outputs:
  sufficient_tools     : flat list, ANY-of   -> functional_tool_use (did agent engage right tooling)
  required_tool_groups : list of OR-groups, ALL must be satisfied -> required_tool_completion
"""
import re

_DELIV = re.compile(r"(?:workspace/)?output/[\w.\-]+")
_ORDER = re.compile(r"\b(order|servicerequest|prescrib\w*|medicationrequest|referral|place an order)\b", re.I)

def derive(task):
    et = (task.get("environment") or {}).get("type")
    ref = task.get("reference") or {}
    g = (task.get("goal") or "") + " " + str((task.get("context") or {}).get("text") or "")
    if et == "fhir":
        groups = [["fhir_search", "fhir_read"]]
        if _DELIV.search(g): groups.append(["write_file"])
        if _ORDER.search(g): groups.append(["fhir_create"])
    elif et == "gui":
        groups = [["submit"]]
    elif et == "tool_sandbox":
        suff = ref.get("sufficient_tools") or ["ImageDescription", "RegionAttributeDescription", "OCR"]
        groups = [list(suff)]
    else:
        groups = []
    sufficient = ref.get("sufficient_tools")
    if not sufficient:
        sufficient = sorted({t for grp in groups for t in grp})
    return {"sufficient_tools": sufficient, "required_tool_groups": groups}

def functional_used(task, used_tools):
    """ANY-of: did the agent use at least one task-relevant tool. None if undefined."""
    suff = (task.get("reference") or {}).get("sufficient_tools") or derive(task)["sufficient_tools"]
    if not suff: return None
    return bool(set(suff) & set(used_tools))

def required_complete(task, used_tools):
    """ALL OR-groups satisfied by the tools the agent actually used. None if undefined."""
    groups = (task.get("reference") or {}).get("required_tool_groups") or derive(task)["required_tool_groups"]
    if not groups: return None
    used = set(used_tools)
    return all(set(grp) & used for grp in groups)
