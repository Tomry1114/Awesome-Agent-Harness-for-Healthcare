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


def _n2(x):
    return str(x).strip().lower() if x not in (None, "") else ""


# A tool's CAPABILITY (what inputs it actually consumes), inferred from name+signature. "Different
# tool" is only information gain if the tool is RELEVANT to the requirement -- OCR/ImageDescription ignore
# region+attribute, so they are NOT substitutes for a regional anatomical re-look.
def tool_capability(t):
    s = (_name(t) + " " + _sig(t)).lower()
    if "ocr" in s or "text" in s or "label" in s or "annotat" in s:
        return "text"
    if ("region" in s or "attribute" in s or "crop" in s or "zoom" in s or "roi" in s):
        return "region_attribute"
    if "image" in s or "description" in s or "overview" in s or "whole" in s:
        return "global"
    return "region_attribute"                              # default: assume regional if it took a region kw


def requirement_capability(attribute):
    """Which tool capability the requirement needs. A text/label/annotation attribute -> text (OCR); an
    explicit global-context need -> global; else an anatomical regional discrimination -> region_attribute."""
    a = _n2(attribute)
    if any(k in a for k in ("text", "label", "annotation", "caption", "letter", "number", "marking")):
        return "text"
    if any(k in a for k in ("overall", "global", "overview", "whole image", "general appearance")):
        return "global"
    return "region_attribute"


def _contrastive_attribute(attribute):
    """Rewrite an attribute into a DISCRIMINATIVE question so a re-look pursues information gain instead of
    re-describing (a stable VLM re-describing the same region tends to repeat its first error). Generic:
    names no disease / no alternative -- asks for the distinguishing features that confirm or exclude it."""
    a = str(attribute or "").strip()
    if not a:
        return a
    return ("%s -- report the DISCRIMINATING features (precise location, course/orientation, margins, and "
            "relationship to adjacent structures) that CONFIRM or EXCLUDE it; do not merely re-describe" % a)


def select_diversified_tool(available_tools, region=None, attribute=None, prior_attempts=None):
    """Pick a CAPABILITY-RELEVANT perception tool + (possibly rewritten) attribute that pursues
    INFORMATION GAIN over prior_attempts. prior_attempts: list[{tool, region, attribute}] (REQUESTED, incl.
    failed). Returns (tool|None, attribute). Only tools whose capability matches the requirement are candidates
    (a regional anatomical need never routes to OCR/ImageDescription -- a different NAME is not information
    gain). Strategy: (1) an UNTRIED capability-relevant tool for this region; (2) all tried -> primary relevant
    tool + a CONTRASTIVE attribute rewrite (a genuinely new question); (3) contrastive already tried -> None."""
    all_tools = select_tools(available_tools, region=region)
    if not all_tools:
        return None, attribute
    need = requirement_capability(attribute)
    relevant = [t for t in all_tools if tool_capability({"name": t}) == need] or all_tools
    prior = prior_attempts or []
    tried_keys = {(_n2(p.get("tool")), _n2(p.get("region")), _n2(p.get("attribute"))) for p in prior}
    tried_tools_here = {_n2(p.get("tool")) for p in prior if _n2(p.get("region")) == _n2(region)}
    for t in relevant:                                 # 1) an UNTRIED capability-relevant tool
        if _n2(t) not in tried_tools_here and (_n2(t), _n2(region), _n2(attribute)) not in tried_keys:
            return t, attribute
    t = relevant[0]                                    # 2) contrastive rewrite on the primary relevant tool
    contrastive = _contrastive_attribute(attribute)
    if (_n2(t), _n2(region), _n2(contrastive)) not in tried_keys:
        return t, contrastive
    return None, attribute                             # 3) would be an exact repeat -> no productive re-look