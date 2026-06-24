#!/usr/bin/env python3
"""MedCTA (multimodal_clinical_reasoning) BenchmarkPlugin.

ALL MedCTA-specific knowledge lives here: how each perception/tool maps to a semantic role + milestones,
the RESULT-CONDITIONAL resolvers that read the real tool output to decide which milestone was truly earned,
the EvidenceView extractor (localization-aware fidelity), and the dimension policy. The substrate core names
no benchmark/tool; this file does. A 4th dataset drops a sibling module + a spec/registry.json entry; no
core edit. Registered into the substrate registry at import time (see runner/plugins/__init__.py).

Resolvers / extractor consume ONLY the shared helpers exported by substrate (_errored, _result_output,
_hash8, _real_delivery, _source_text) -- never private dimension logic -- so the benchmark/core boundary is
one-directional (plugins -> substrate)."""
import substrate as _S

_errored = _S._errored
_result_output = _S._result_output
_hash8 = _S._hash8
_real_delivery = _S._real_delivery
_source_text = _S._source_text


# ============================================================================= RESOLVERS
def _resolve_region(event, prev_state):
    """RegionAttributeDescription: a successful HTTP call that FELL BACK to the whole image (resolved=False)
    did NOT examine the targeted region -> it must NOT be credited target_region_examined. Reads the real
    localization status from result.output.localization."""
    out = _result_output(event)
    loc = out.get("localization") if isinstance(out, dict) else None
    if _errored(event):
        return {"role": "acquire", "status": "failure", "milestones_added": [], "state_changed": False,
                "obligation_id": "target_region_examined"}
    if isinstance(loc, dict) and loc.get("resolved") is True:
        return {"role": "acquire", "status": "success", "state_changed": True,
                "milestones_added": ["target_region_examined", "relevant_image_evidence_obtained"],
                "obligation_id": "target_region_examined",
                "progress_token": "region:%s:resolved" % _hash8(str(loc.get("requested") or "region"))}
    # fell back to the full image: general image evidence only, the targeted region was NOT examined
    return {"role": "acquire", "status": "partial", "state_changed": False,
            "milestones_added": ["image_overview_obtained"], "obligation_id": "target_region_examined",
            "progress_token": None}


def _resolve_ocr(event, prev_state):
    """MedCTA OCR: empty/blank rendered text -> partial, NO text_evidence_obtained, progress_token=None
    (the page carried no readable text -> no evidence). Non-empty text -> success with a CONTENT-hashed
    evidence token so OCR(page1) and OCR(page2) earn DIFFERENT tokens (new evidence) while a repeated
    identical OCR repeats its token (no progress)."""
    if _errored(event):
        return {"role": "acquire", "status": "failure", "milestones_added": [],
                "obligation_id": "text_evidence_obtained", "state_changed": False, "progress_token": None}
    out = _result_output(event)
    text = out if isinstance(out, str) else (out.get("text") if isinstance(out, dict) else "")
    text = (text or "").strip()
    if not text:
        return {"role": "acquire", "status": "partial", "milestones_added": [],
                "obligation_id": "text_evidence_obtained", "state_changed": False, "progress_token": None}
    return {"role": "acquire", "status": "success", "milestones_added": ["text_evidence_obtained"],
            "obligation_id": "text_evidence_obtained",
            "progress_token": "evidence:ocr:%s" % _hash8(text)}


_GS_EMPTY = ("[no offline result", "no result", "no results found", "[no result")
def _resolve_googlesearch(event, prev_state):
    """MedCTA GoogleSearch: a '[no offline result]' / empty / irrelevant snippet -> partial, no
    external_reference_obtained milestone (the search returned nothing usable). A real snippet -> success
    with a content-hashed external-reference token."""
    if _errored(event):
        return {"role": "acquire", "status": "failure", "milestones_added": [],
                "obligation_id": "external_reference_obtained", "state_changed": False, "progress_token": None}
    out = _result_output(event)
    snippet = out if isinstance(out, str) else (out.get("text") if isinstance(out, dict) else "")
    snippet = (snippet or "").strip()
    low = snippet.lower()
    if (not snippet) or any(low.startswith(m) or m in low for m in _GS_EMPTY):
        return {"role": "acquire", "status": "partial", "milestones_added": [],
                "obligation_id": "external_reference_obtained", "state_changed": False, "progress_token": None}
    return {"role": "acquire", "status": "success", "milestones_added": ["external_reference_obtained"],
            "obligation_id": "external_reference_obtained",
            "progress_token": "evidence:search:%s" % _hash8(snippet)}


# ----------------------------------------------------------------------------- evidence extractor
def _medcta_evidence(trace):
    """MedCTA EvidenceView: real delivery (canonical_observation) refined by localization — a delivered region
    result that FELL BACK to the whole image is delivered but low-fidelity (targeted region not localized)."""
    units = []
    for i, e in enumerate(trace):
        if e.get("event_type") != "tool_call": continue
        d = _real_delivery(e)
        out = _result_output(e)
        loc = out.get("localization") if isinstance(out, dict) else None
        fid = d["fidelity"]
        if d["delivered"] and isinstance(loc, dict) and loc.get("resolved") is False:
            fid = min(fid, 0.5)
        txt = out.get("text") if isinstance(out, dict) else _source_text(e)
        units.append({"id": "%s#%d" % (e.get("tool"), i), "delivered_to_agent": d["delivered"],
                      "delivery_fidelity": fid, "error_visible": d["error_visible"], "payload": str(txt)[:300]})
    return units


# ----------------------------------------------------------------------------- registration
PLUGIN = {
    "benchmark": "MedCTA", "default_tool_role": "acquire",
    "tool_semantics": {
        "ImageDescription": {"role": "acquire", "success_milestones": ["image_overview_obtained", "relevant_image_evidence_obtained"]},
        "RegionAttributeDescription": {"role": "acquire", "success_milestones": ["target_region_examined", "relevant_image_evidence_obtained"]},
        "OCR": {"role": "acquire", "success_milestones": ["text_evidence_obtained"]},
        "GoogleSearch": {"role": "acquire", "success_milestones": ["external_reference_obtained"]},
        "Calculator": {"role": "act", "success_milestones": []}},
    "evidence_extractor": _medcta_evidence,
    "resolvers": {"RegionAttributeDescription": _resolve_region,
                  "OCR": _resolve_ocr,
                  "GoogleSearch": _resolve_googlesearch},
    "dimension_policy": {"required_milestones": ["relevant_image_evidence_obtained"],
                         "required_context_units": ["target_image_evidence"],
                         "governance_policy_id": "MedCTA"}}

_S.register_plugin(PLUGIN)
