"""Affordance registry — maps a MISSING observation to EXECUTABLE tools drawn from the task manifest.

The judge says only WHAT observation is missing ("no observation of the referenced region"); it never names
a tool. The kernel calls this to turn that need into real, available tool names, so feedback can never
suggest a tool that does not exist / does not accept the region / is the wrong affordance. If nothing
matches, the finding carries NO tool suggestion (better silent than hallucinated)."""
from __future__ import annotations

_PERCEPTION_KW = ("region", "attribute", "describe", "description", "image", "ocr", "segment", "detect",
                  "inspect", "view", "zoom", "crop", "measure", "perceiv", "look")
_EXCLUDE_KW = ("search", "google", "web", "browse", "write", "submit", "create", "update", "delete",
               "file", "finish", "final", "answer", "calculator")


def _name(t):
    return (t.get("name") if isinstance(t, dict) else str(t)) or ""


def _sig(t):
    return (t.get("signature") or "") if isinstance(t, dict) else ""


def is_perception_tool(t):
    s = (_name(t) + " " + _sig(t)).lower()
    if any(k in s for k in _EXCLUDE_KW):
        return False
    return any(k in s for k in _PERCEPTION_KW)


def select_tools(available_tools, region=None, modality=None, prefer_region=True):
    """Executable perception tool names from the manifest, region/attribute-capable ones ranked first. []
    when the substrate exposes no perception affordance (e.g. FORM/FHIR) -> REACQUIRE then suggests no tool."""
    names = [_name(t) for t in (available_tools or []) if is_perception_tool(t)]
    if prefer_region:
        names = sorted(names, key=lambda n: 0 if ("region" in n.lower() or "attribute" in n.lower()) else 1)
    # de-dup, keep order
    seen, out = set(), []
    for n in names:
        if n and n not in seen:
            seen.add(n); out.append(n)
    return out
