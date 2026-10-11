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


# --- usable_for_context (shared cross-benchmark) ---
def _usable_fields(sem_event):
    """Derive (semantic_status, usable_for_context) for an EvidenceUnit from its PRODUCING
    SemanticEvent (the resolved status read from the resolver, NOT the tool name). semantic_status is the
    event status (success|partial|failure); usable_for_context = (status=='success' AND a non-empty
    progress_token). An empty FHIR Bundle / blank OCR / '[no offline result]' search is delivered+partial
    with NO token -> usable_for_context=False (it still feeds Clinical Auditability/delivery, but is NOT acquired
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


# A GoogleSearch's SOURCE instance is the RESULT it actually surfaced, not the query that asked for it.
# Two different queries that return the SAME page/domain are ONE web source (so cross-source corroboration
# does not double-count one external page reached by two phrasings); a query with no usable result has no
# result identity and falls back to the query hash, LABELLED so the two are never conflated.
_URL_RE = _re.compile(r"https?://([^/\s)>\]\"']+)(/[^\s)>\]\"']*)?", _re.I)
_DOMAIN_RE = _re.compile(r"\b((?:[a-z0-9-]+\.)+(?:com|org|net|edu|gov|io|ai|co|info|nih\.gov|ncbi\.nlm\.nih\.gov))\b", _re.I)
_GS_NORESULT = ("[no offline result", "[no result", "[search error", "no results found")

def _web_instance(out, query):
    """source_instance_id for a web search, from the RESULT identity when present:
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
    result it surfaced, see _web_instance). A region tool that truly localized earns the region-specific
    image type."""
    tool = event.get("tool")
    out = _result_output(event)
    if tool in _IMAGE_EXTRACTORS:
        ctype = "target_image_evidence"
        if tool == "RegionAttributeDescription":
            # Region-specific evidence is decided by the three-state localization verdict, not by
            # `loc.get("resolved") is True`: the latter only matches bbox calls, while most region calls use a
            # natural-language (semantic) region whose `resolved` is left UNKNOWN by the backend.
            if _localization_verdict(event)[0] is True:
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
    _v, _why = _localization_verdict(event)
    if _v is True:
        return {"role": "acquire", "status": "success", "state_changed": True,
                "milestones_added": ["target_region_examined", "relevant_image_evidence_obtained"],
                "obligation_id": "target_region_examined", "localization_verdict": "localized",
                "localization_basis": _why,
                "progress_token": "region:%s:resolved" % _hash8(str((loc or {}).get("requested") or "region"))}
    # No region credit. THREE-STATE: distinguish "demonstrably did not examine the region" from
    # "we cannot tell" -- the latter is the evaluator's undecidability and must not read as an agent failure.
    # Either way the call still delivered general image evidence.
    return {"role": "acquire", "status": "partial", "state_changed": False,
            "milestones_added": ["image_overview_obtained"], "obligation_id": "target_region_examined",
            "localization_verdict": ("indeterminate" if _v is None else "not_localized"),
            "localization_basis": _why, "progress_token": None}


# The MedCTA OCR engine's documented 'no readable text on this page' sentinel (matched case-insensitively,
# post-strip). Listed alongside its bracket-less form so a future rendering variant is still caught.
_OCR_NO_TEXT = frozenset(("[no text]", "no text"))


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
    # The MedCTA OCR engine renders the literal sentinel '[no text]' for a page that carries no
    # readable text. That sentinel is truthy, so a bare emptiness test would credit a BLANK page with
    # text_evidence_obtained + an evidence token (polluting Patient–Evidence Alignment acquisition / Clinical Auditability). Treat
    # the documented no-text sentinel exactly like the empty case: partial, NO milestone, no token.
    if (not text) or text.lower() in _OCR_NO_TEXT:
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
    Each unit is tagged with context_type + source_channel/source_instance_id/extractor: every
    image tool shares ONE image source_instance_id so two OCR reads of the same image count as ONE source.
    The semantic progress_token (from the resolver) is carried through so the Patient–Evidence Alignment binding/acquisition
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

# ============================================================ Canonical Evaluation Contract (MedCTA)
# Native tool -> the TYPED context unit its success delivers. Mirrors _provenance's vocabulary exactly.
# Calculator acquires no clinical context, so it contributes no unit.
_CONTEXT_TYPE_OF = {"ImageDescription": "target_image_evidence",
                    "OCR": "target_image_evidence",
                    "RegionAttributeDescription": "region_specific_image_evidence",
                    "GoogleSearch": "external_reference",
                    "Calculator": None}


def _medcta_task_contract(task):
    """Compile a MedCTA task into the Canonical Evaluation Contract. PROCESS-ONLY -- NEVER reads
    reference.gold_answer. Care Action Fulfillment: the UNION of milestones the task's required_tool_groups /
    sufficient_tools demand, as ONE CONJUNCTIVE group (must reach the required image/region/text evidence).
    Patient–Evidence Alignment: the typed context units those same tools deliver."""
    ref = task.get("reference") or {}
    rtg = ref.get("required_tool_groups") or []
    if not rtg and ref.get("sufficient_tools"):
        rtg = [list(ref["sufficient_tools"])]
    ts = PLUGIN.get("tool_semantics", {})
    req = []
    region_needed = False
    for grp in rtg:
        for tool in (grp or []):
            sem = ts.get(tool) or {}
            if "target_region_examined" in (sem.get("success_milestones") or []):
                region_needed = True
            for ms in sem.get("success_milestones") or []:
                if ms not in req:
                    req.append(ms)
    execution = {"required_milestone_groups": [req]} if req else {}
    tooling = {"argument_targets": {"region_specific_evidence_required": region_needed}}
    # The SAME native field Care Action Fulfillment already expands (reference.sufficient_tools) also determines WHICH
    # TYPED CONTEXT the task requires. Without a context section, Patient–Evidence Alignment and Closed-Loop Verification would fall back to
    # the static single {target_image_evidence} default: required_evidence_recall would be matched/1 in {0,1},
    # and dim_verification._precondition_coverage -- which CALLS dim_context._acquisition -- would return that
    # same number, so two of seven dimensions would collapse into one indicator. Nothing is authored here --
    # the tool list is native and the type vocabulary is the one _provenance emits.
    units, _seen = [], set()
    for grp in rtg:
        for tool in (grp or []):
            ty = _CONTEXT_TYPE_OF.get(tool)
            if ty and ty not in _seen:
                _seen.add(ty)
                units.append({"id": ty, "type": ty})
    context = {"required_context_units": units} if units else {}
    return {"execution": execution, "tooling": tooling, "context": context,
            "contract_version": "mcta-2"}


def _medcta_localization(raw):
    r = raw.get("result")
    if isinstance(r, dict):
        o = r.get("output")
        if isinstance(o, dict) and isinstance(o.get("localization"), dict):
            return o["localization"]
        if isinstance(r.get("localization"), dict):
            return r["localization"]
    return None


# ============================================================ LOCALIZATION ADJUDICATION (three-state)
# vlm_backend._region_attr records mode="semantic", resolved=None for a natural-language region and says so
# in its own comment: "resolved is UNKNOWN (None), to be decided from the actual text (not auto-True)".
# This is that decision. It reads ONLY what the environment returned (the requested region string + the
# tool's own text), is deterministic, and therefore re-runs offline over already-recorded trajectories.
# The semantic prompt ends with "If you cannot confidently localize that region, SAY SO explicitly", so a
# disclaimer in the text is the tool's own negative verdict rather than our inference.
_LOC_FAIL_RE = _re.compile(
    r"cannot\s+(?:confidently\s+|reliably\s+|precisely\s+)?(?:localiz|localis|identify|determine|pinpoint|isolate|delineat)"
    r"|unable\s+to\s+(?:confidently\s+|reliably\s+|precisely\s+)?(?:localiz|localis|identify|determine|pinpoint|delineat)"
    r"|(?:can|could)\s*not\s+be\s+(?:confidently\s+|reliably\s+)?(?:localiz|localis|identifi|delineat)"
    r"|not\s+able\s+to\s+(?:confidently\s+)?(?:localiz|localis|identify)"
    r"|no\s+(?:clearly\s+|readily\s+)?(?:identifiable|discernible|visible|definable)\s+(?:region|area|structure|lesion)"
    r"|(?:region|area|structure)\s+(?:is\s+)?not\s+(?:clearly\s+)?(?:visible|identifiable|present|discernible|shown)"
    r"|not\s+visible\s+(?:in|on)\s+(?:this|the)\s+(?:image|field|slide)"
    r"|i\s+(?:cannot|can't|am\s+unable\s+to)\s+(?:confidently\s+)?(?:localiz|localis|identify|determine)"
    r"|no\s+arrows?\s+(?:are\s+)?(?:visible|present|seen)"
    r"|cannot\s+confidently\s+state", _re.I)

# A request for the WHOLE image is not a region request at all -- decidable from the request alone, no text
# needed ("entire image", "whole image", "full image", ...).
_WHOLE_IMAGE_RE = _re.compile(
    r"^\s*(?:the\s+)?(?:entire|whole|full|complete|overall|entirety\s+of)\s+(?:the\s+)?"
    r"(?:image|picture|field|slide|frame|photograph|scan)"
    r"|^\s*(?:image|picture|slide)\s+as\s+a\s+whole"
    r"|^\s*all\s+of\s+the\s+(?:image|slide)"
    r"|^\s*(?:general|global|overall)\s+(?:view|impression|appearance)\s*$", _re.I)

# Words that carry no localizing information, so their presence in the text proves nothing about whether the
# requested region was actually addressed.
_LOC_GENERIC = frozenset((
    "image", "images", "region", "regions", "area", "areas", "part", "parts", "portion", "view",
    "focus", "focusing", "focused", "structure", "structures", "tissue", "tissues", "distinctive",
    "shown", "indicated", "visible", "present", "features", "feature", "morphology", "appearance",
    "entire", "whole", "full", "complete", "overall", "general", "main", "primary", "interest",
    "and", "the", "for", "with", "any", "that", "this", "from", "near", "around",
    "central", "center", "centre", "middle", "side", "edge", "corner", "top", "bottom"))


def _loc_tokens(text):
    return [w for w in _re.findall(r"[a-z]{3,}", str(text or "").lower()) if w not in _LOC_GENERIC]


def _localization_verdict(event):
    """(verdict, basis) where verdict is True | False | None.

    True  -- the region was demonstrably examined (real pixel crop, or the returned text addresses the
             requested region and carries no inability disclaimer).
    False -- demonstrably not examined (no region requested, the WHOLE image was requested, the tool itself
             said it could not localize, or it returned nothing).
    None  -- INDETERMINATE. This must never be scored 0: it is the evaluator's undecidability, not the
             agent's failure.
    """
    loc = _medcta_localization(event if isinstance(event, dict) else {})
    if not isinstance(loc, dict):
        return None, "no_localization_block"
    if loc.get("resolved") is True:
        return True, "bbox_crop"
    if loc.get("mode") != "semantic":
        return None, "mode_%s" % loc.get("mode")
    req = str(loc.get("requested") or "").strip()
    if not req:
        return False, "no_region_requested"
    if _WHOLE_IMAGE_RE.search(req):
        return False, "requested_whole_image"
    out = _result_output(event)
    body = out.get("text") if isinstance(out, dict) else None
    if not str(body or "").strip():
        return False, "empty_text"
    if _LOC_FAIL_RE.search(str(body)):
        return False, "tool_declared_cannot_localize"
    toks = _loc_tokens(req)
    if not toks:
        return None, "region_has_no_distinctive_token"
    low = str(body).lower()
    hit = [t for t in toks if t in low]
    if hit:
        return True, "region_terms_addressed:%d/%d" % (len(hit), len(toks))
    return None, "region_terms_absent_no_disclaimer"


def _medcta_argument_semantic_score(call, task=None, policy=None, sem_event=None):
    """PROCESS-only argument semantics for a MedCTA localization tool call. A region tool earns 1.0 when
    it RESOLVED a region (result.localization.resolved) or carried a non-empty semantic region arg; 0.0
    when it silently fell back. Consumes contract argument_targets (region_specific_evidence_required):
    when region-specific evidence IS required, a region tool that did NOT resolve is a hard 0.0. Judges
    only whether the tool LOOKED at a real region -- never the final diagnosis."""
    raw = call if isinstance(call, dict) else {}
    tool = (raw.get("tool") or "")
    args = raw.get("args")
    if isinstance(args, str):
        import json as _j
        try: args = _j.loads(args)
        except Exception: args = {}
    args = args if isinstance(args, dict) else {}
    _targets = ((policy or {}).get("evaluation_contract") or {}).get("tooling", {}).get("argument_targets", {}) or {}
    region_required = bool(_targets.get("region_specific_evidence_required"))
    region = next((args.get(k) for k in ("region", "region_query", "bbox") if args.get(k)), None)
    loc = _medcta_localization(raw)
    if region is None and loc is None:
        return {"status": "not_applicable", "reason": "not_a_localization_call"}
    if isinstance(loc, dict) and "resolved" in loc:
        _v, _why = _localization_verdict(raw)
        if _v is None:
            # INDETERMINATE -> not_applicable, never 0.0. Mapping the adapter's explicit UNKNOWN through
            # `bool(resolved)` would score legitimate semantic-region calls a hard 0.
            return {"status": "not_applicable", "reason": "localization_indeterminate",
                    "localization_basis": _why, "region": str(region)[:120]}
        return {"status": "valid", "score": 1.0 if _v else 0.0, "axis": "localization_adjudicated",
                "localization_basis": _why,
                "region_required": region_required, "region": str(region)[:120]}
    score = 1.0 if region else (0.0 if region_required else 0.0)
    return {"status": "valid", "score": score, "axis": "semantic_region_present",
            "region_required": region_required, "region": str(region)[:120]}

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
                         # DEFAULT verification_policy: only claims that GENUINELY need
                         # corroboration are gated to >=2 independent sources. A direct authoritative
                         # single-source read of the ONE task image (one VLM/OCR look) is NOT forced to two
                         # sources -- an unflagged claim is not_applicable to cross_source, never auto-failed.
                         # A web/external medical fact, a high-risk recommendation, or a claim already in
                         # CONFLICT does warrant a second independent source.
                         "verification_policy": {
                             "cross_source_required_for": [{"type": "high_risk_recommendation", "patterns": ["dose", "dosage", "increase", "decrease", "initiate", "discontinue", "anticoagul", "administer", "mg ", "prescrib", "titrate", "start ", "stop ", "hold "]}, {"type": "external_medical_fact", "patterns": ["guideline", "studies show", "literature", "typically causes", "is known to", "according to", "per uptodate", "recommended per", "class effect"]}, {"type": "conflicting_evidence", "patterns": ["conflict", "inconsistent", "discrepan", "contradict", "disagree", "does not match", "mismatch"]}]},
                         "governance_policy_id": "MedCTA"}}


PLUGIN["task_contract_resolver"] = _medcta_task_contract
PLUGIN["argument_semantic_score"] = _medcta_argument_semantic_score
PLUGIN["contract_version"] = "mcta-2"
_S.register_plugin(PLUGIN)


# ============================================================ Clinical Auditability contract
# MedCTA has ONE final prediction but MANY observable spans -- perception, localisation, text extraction,
# external lookup, computation, terminal. Treating it as a single observability opportunity would
# collapse the denominator to 1; these families keep one opportunity per span family.
def _mcta_observability_family(sem_event):
    if sem_event.get("terminal") in ("final", "escalate"):
        return "terminal"
    cap = str(sem_event.get("capability_id") or "")
    if cap == "ImageDescription":
        return "image_perception"
    if cap == "RegionAttributeDescription":
        return "region_localisation"
    if cap == "OCR":
        return "text_extraction"
    if cap == "GoogleSearch":
        return "external_lookup"
    if cap == "Calculator":
        return "computation"
    return "other:%s" % (sem_event.get("event_role") or "unknown")


def _mcta_target_resolved(sem_event):
    """A perception op is target-resolved when we know WHICH image, and for a region op also WHICH region.
    An ImageDescription call with empty args has no target the operator can resolve, so it must not
    score."""
    e = sem_event.get("raw") or {}
    a = e.get("args") if isinstance(e.get("args"), dict) else {}
    cap = str(sem_event.get("capability_id") or "")
    img = any(str(a.get(k) or "").strip() for k in ("image", "image_path", "img", "path", "file"))
    if cap == "RegionAttributeDescription":
        return bool(img and any(str(a.get(k) or "").strip() for k in ("region", "bbox", "attribute", "query")))
    if cap in ("ImageDescription", "OCR"):
        return bool(img)
    if cap == "GoogleSearch":
        return bool(str(a.get("query") or "").strip())
    if cap == "Calculator":
        return bool(str(a.get("expression") or a.get("expr") or "").strip())
    return False


def _mcta_receipt(sem_event):
    """Perception receipt = the tool returned actual content for the requested target. An error sentinel or
    an empty output is not a receipt -- notably [vlm_api_error] spans must not count."""
    e = sem_event.get("raw") or {}
    out = _result_output(e)
    txt = out.get("text") if isinstance(out, dict) else (out if isinstance(out, str) else None)
    if not isinstance(txt, str) or not txt.strip():
        if isinstance(out, dict) and isinstance(out.get("output"), str):
            txt = out["output"]
        else:
            return False
    low = txt.strip().lower()
    if low.startswith("[") and "error" in low[:32]:
        return False
    return len(low) > 8


PLUGIN["observability_contract"] = {
    "target_resolver": _mcta_target_resolved,
    "receipt_resolver": _mcta_receipt,
    "family_of": _mcta_observability_family,
    "cost_signals": [],
}
