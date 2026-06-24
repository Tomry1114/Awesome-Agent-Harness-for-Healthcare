#!/usr/bin/env python3
"""MedCTA (multimodal_clinical_reasoning) BenchmarkPlugin.

ALL MedCTA-specific knowledge lives here: how each perception/tool maps to a semantic role + milestones,
the RESULT-CONDITIONAL resolvers that read the real tool output to decide which milestone was truly earned,
the EvidenceView extractor (localization-aware fidelity), and the dimension policy. The substrate core names
no benchmark/tool; this file does. A 4th dataset drops a sibling module + a spec/registry.json entry; no
core edit. Registered into the substrate registry at import time (see runner/plugins/__init__.py).

Resolvers / extractor consume ONLY the shared helpers exported by substrate (_errored, _result_output,
_hash8, _real_delivery, _source_text) -- never private dimension logic -- so the benchmark/core boundary is
one-directional (plugins -> substrate).

TYPED CONTEXT + SOURCE PROVENANCE (shared cross-benchmark contract):
  * dimension_policy.required_context_units are TYPED {id, type}; the MedCTA semantic TYPE vocabulary is
    target_image_evidence (a region-specific task may carry region_specific_image_evidence).
  * every EvidenceUnit is tagged with context_type (the semantic KIND of context the tool actually
    obtained), source_channel (the SOURCE family -- 'radiology_image' / 'external_web'), source_instance_id
    (the specific instance within that channel: the task IMAGE for every image tool, the query for a web
    search), and extractor (the reader: 'image_vlm' / 'OCR' / 'web_search'). Cross-source corroboration
    counts INDEPENDENT (source_channel, source_instance_id) pairs: two OCR reads of the SAME task image
    share one source_instance_id (one source), not two."""
import re as _re
import substrate as _S

_errored = _S._errored
_result_output = _S._result_output
_hash8 = _S._hash8
_real_delivery = _S._real_delivery
_source_text = _S._source_text


# --- CONTRACT-A: usable_for_context (shared cross-benchmark) ---
def _usable_fields(sem_event):
    """CONTRACT-A: derive (semantic_status, usable_for_context) for an EvidenceUnit from its PRODUCING
    SemanticEvent (the resolved status read from the resolver, NOT the tool name). semantic_status is the
    event status (success|partial|failure); usable_for_context = (status=='success' AND a non-empty
    progress_token). An empty FHIR Bundle / blank OCR / '[no offline result]' search is delivered+partial
    with NO token -> usable_for_context=False (it still feeds Observability/delivery, but is NOT acquired
    context). A degenerate event with no semantic event maps to (None, False)."""
    st = (sem_event or {}).get("status")
    pt = (sem_event or {}).get("progress_token")
    return st, bool(st == "success" and pt)

# --- SOURCE PROVENANCE channels (the unit of INDEPENDENCE for cross-source corroboration) ---
_CH_IMAGE = "radiology_image"
_CH_WEB = "external_web"
# The MedCTA environment binds ONE clinical image per task; every perception tool (ImageDescription /
# RegionAttributeDescription / OCR) reads THAT SAME image. The asset_id is not echoed into each tool_call
# event, so the per-trace image instance is a stable constant -- which is exactly the contract requirement
# that two OCR reads of the SAME image share one source_instance_id (one source, not two).
_IMG_INSTANCE = "image:primary"

# image-perception tools (all read the single task image) -> extractor label for provenance
_IMAGE_EXTRACTORS = {"ImageDescription": "image_vlm", "RegionAttributeDescription": "image_vlm",
                     "OCR": "OCR"}


# V5: a GoogleSearch's SOURCE instance is the RESULT it actually surfaced, not the query that asked for it.
# Two different queries that return the SAME page/domain are ONE web source (so cross-source corroboration
# does not double-count one external page reached by two phrasings); a query with no usable result has no
# result identity and falls back to the query hash, LABELLED so the two are never conflated.
_URL_RE = _re.compile(r"https?://([^/\s)>\]\"']+)(/[^\s)>\]\"']*)?", _re.I)
_DOMAIN_RE = _re.compile(r"\b((?:[a-z0-9-]+\.)+(?:com|org|net|edu|gov|io|ai|co|info|nih\.gov|ncbi\.nlm\.nih\.gov))\b", _re.I)
_GS_NORESULT = ("[no offline result", "[no result", "[search error", "no results found")

def _web_instance(out, query):
    """V5: source_instance_id for a web search, from the RESULT identity when present:
       web:url:<host[/path8]>   when the snippet carries a real URL,
       web:domain:<host>        when only a bare domain appears,
       web:query:<hash8>        FALLBACK (LABELLED) when the result has NO identity (a '[no offline
                                result]' / blank snippet) -- only then keyed by the query."""
    text = out if isinstance(out, str) else (
        (out.get("text") or out.get("snippet") or "") if isinstance(out, dict) else "")
    text = str(text or "")
    low = text.strip().lower()
    if not any(low.startswith(m) or m in low for m in _GS_NORESULT):
        m = _URL_RE.search(text)
        if m:
            host = m.group(1).lower().lstrip("www.")
            path = (m.group(2) or "").rstrip("/")
            return "web:url:%s" % (host if not path else "%s%s" % (host, _hash8(path)[:8]))
        d = _DOMAIN_RE.search(text)
        if d:
            return "web:domain:%s" % d.group(1).lower().lstrip("www.")
    q = str(query or "").strip()
    return "web:query:%s" % _hash8(q) if q else "web:query"


def _provenance(event):
    """(context_type, source_channel, source_instance_id, extractor) for a MedCTA tool_call -- derived from
    what the tool ACTUALLY read. Image-perception tools all read the single task image (one shared
    source_instance_id so repeat reads are ONE source); a web search reads the external web (instance = the
    query). A region tool that truly localized earns the region-specific image type."""
    tool = event.get("tool")
    out = _result_output(event)
    if tool in _IMAGE_EXTRACTORS:
        ctype = "target_image_evidence"
        if tool == "RegionAttributeDescription":
            loc = out.get("localization") if isinstance(out, dict) else None
            if isinstance(loc, dict) and loc.get("resolved") is True:
                ctype = "region_specific_image_evidence"
        return ctype, _CH_IMAGE, _IMG_INSTANCE, _IMAGE_EXTRACTORS[tool]
    if tool == "GoogleSearch":
        args = event.get("args") if isinstance(event.get("args"), dict) else {}
        q = args.get("query") or args.get("q") or ""
        inst = _web_instance(out, q)
        return "external_reference", _CH_WEB, inst, "web_search"
    # Calculator / other non-perception tools acquire no clinical CONTEXT -> no context_type
    return None, None, None, tool


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
    result that FELL BACK to the whole image is delivered but low-fidelity (targeted region not localized).
    Each unit is tagged with context_type + source_channel/source_instance_id/extractor (CONTRACT): every
    image tool shares ONE image source_instance_id so two OCR reads of the same image count as ONE source.
    The semantic progress_token (from the resolver) is carried through so the Context binding/acquisition
    sub-metrics read a per-unit token, not just the trace."""
    sem = _S.map_trace(trace, PLUGIN)        # align each tool_call to its resolved semantic event (token/status)
    sem_by_step = {}
    for s in sem:
        raw = s.get("raw") or {}
        sem_by_step[id(raw)] = s
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
        ctype, channel, instance, extractor = _provenance(e)
        s = sem_by_step.get(id(e)) or {}
        sem_status, usable = _usable_fields(s)
        units.append({"id": "%s#%d" % (e.get("tool"), i), "delivered_to_agent": d["delivered"],
                      "delivery_fidelity": fid, "error_visible": d["error_visible"], "payload": str(txt)[:300],
                      "context_type": ctype, "source_channel": channel,
                      "source_instance_id": instance, "extractor": extractor,
                      "progress_token": s.get("progress_token"),
                      "semantic_status": sem_status, "usable_for_context": usable})
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
                         "required_context_units": [
                             {"id": "target_image_evidence", "type": "target_image_evidence"}],
                         # V1 / CONTRACT-B DEFAULT verification_policy: only claims that GENUINELY need
                         # corroboration are gated to >=2 independent sources. A direct authoritative
                         # single-source read of the ONE task image (one VLM/OCR look) is NOT forced to two
                         # sources -- an unflagged claim is not_applicable to cross_source, never auto-failed.
                         # A web/external medical fact, a high-risk recommendation, or a claim already in
                         # CONFLICT does warrant a second independent source.
                         "verification_policy": {
                             "cross_source_required_for": [
                                 "external_medical_fact", "high_risk_recommendation",
                                 "conflicting_evidence"]},
                         "governance_policy_id": "MedCTA"}}

_S.register_plugin(PLUGIN)
