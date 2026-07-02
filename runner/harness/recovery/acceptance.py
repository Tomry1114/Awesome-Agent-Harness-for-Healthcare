"""Bounded Clinical Recovery v3 - Non-regression Acceptance gate (design sec.6a, tail).

The harness does NOT author the revised answer. On the evidence path it ACQUIRES missing perceptual evidence
(read-only) and RETURNS control to the ROOT AGENT, which regenerates answer B itself. This module is the ONLY
thing that then runs: a conservative accept/reject gate over (root answer A, agent-regenerated B, the newly
ACQUIRED evidence). It wraps engines.semantic.evaluate_candidate / adopt_revised and returns exactly one of:

    ACCEPTED        - adopt B (the agent's regenerated answer supersedes A)
    KEPT_ORIGINAL   - keep A (B is not a non-regressive, evidence-supported improvement)

It mints NO MutationAuthorization, performs NO commit, and does NO server read-back.

PERCEPTUAL GROUNDING RULE. A core decision flip A->B on a perceptual task must be grounded in IMAGE-derived
evidence. Evidence acquired from a web/text search channel (GoogleSearch) is NOT image grounding: it cannot
support a claim about what is visible in the clinical image. Such evidence is dropped before the delta-support
check (default-deny: only positively image-sourced evidence counts as grounding), so a B that flips the core
decision on the strength of web text alone is KEPT_ORIGINAL.

EXPERIMENTAL ACCEPTANCE POLICY (not a calibrated correctness guarantee). evaluate_candidate adopts an
evidence-supported core flip only when the delta-support judge reports supported==True with confidence >= 0.8.
That 0.8 threshold is an EXPERIMENTAL non-regression policy chosen to bound T->F harm; it is NOT a validated
probability of correctness. Report replay F->T / T->F / unchanged and the acceptance rate; do not read the
threshold as a correctness claim.

Python 3.8 compatible.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .contracts import ACCEPTED, KEPT_ORIGINAL
from ..engines.semantic import evaluate_candidate, adopt_revised


# Accepted image-grounding provenance signals (perceptual acquisition channels).
_IMAGE_CHANNELS = ("radiology_image", "image")
_IMAGE_EXTRACTORS = ("image_vlm", "ocr")
_IMAGE_TYPES = ("imagedescription", "regionattributedescription", "ocr")
# Web/text provenance is never image grounding for a perceptual claim.
_WEB_MARKERS = ("web", "search", "google", "external")


@dataclass
class AcceptanceResult:
    state: str                                       # ACCEPTED | KEPT_ORIGINAL
    decision: str                                    # ADOPT_B | KEEP_A (from evaluate_candidate)
    reason: str = ""
    adopted: bool = False
    image_evidence: List[Dict[str, Any]] = field(default_factory=list)
    dropped_web_evidence: List[Dict[str, Any]] = field(default_factory=list)


def _is_image_grounded(e):
    """Default-deny: an evidence item grounds a PERCEPTUAL flip only if it is positively image-sourced.
    Any web/text-search provenance is rejected as image grounding."""
    if not isinstance(e, dict):
        return False
    ch = str(e.get("source_channel") or "").lower()
    ex = str(e.get("extractor") or "").lower()
    ty = str(e.get("type") or "").lower()
    st = str(e.get("source_type") or "").lower()
    inst = str(e.get("source_instance_id") or "").lower()
    blob = " ".join((ch, ex, st, inst))
    if any(m in blob for m in _WEB_MARKERS) or "googlesearch" in ty:
        return False
    if (any(c in ch for c in _IMAGE_CHANNELS) or ex in _IMAGE_EXTRACTORS
            or ty in _IMAGE_TYPES or "image" in st):
        return True
    return False


def evaluate(root_answer, candidate_b, goal_spec, all_evidence, new_evidence,
             judge_fn=None, reverify_fn=None, critique=None, itype="ACQUIRE"):
    """Non-regression acceptance of an agent-regenerated B over root answer A, given the newly ACQUIRED
    evidence. Returns an AcceptanceResult (ACCEPTED -> adopt B, else KEPT_ORIGINAL). Fail-safe: any
    uncertainty inside evaluate_candidate keeps A. Web-only new evidence cannot ground a perceptual flip."""
    ne = list(new_evidence or [])
    image_ev = [e for e in ne if _is_image_grounded(e)]
    web_dropped = [e for e in ne if not _is_image_grounded(e)]

    intervention = {"root_answer": root_answer, "type": itype, "critique": critique}
    # Only IMAGE-grounded evidence is handed to the delta-support check; web/text is not image grounding.
    decision, reason = evaluate_candidate(
        intervention, candidate_b, goal_spec, all_evidence, image_ev,
        judge_fn=judge_fn, reverify_fn=reverify_fn)

    if web_dropped and not image_ev:
        reason = "%s;web_only_evidence_rejected_for_perceptual_grounding(n=%d)" % (reason, len(web_dropped))

    accepted = (decision == "ADOPT_B")
    return AcceptanceResult(
        state=ACCEPTED if accepted else KEPT_ORIGINAL,
        decision=decision, reason=reason, adopted=accepted,
        image_evidence=image_ev, dropped_web_evidence=web_dropped)


def would_adopt(comparison, tau=0.15):
    """Thin, documented pass-through to engines.semantic.adopt_revised for the public-dimension A/B margin
    (exposed so callers/tests can reuse the SAME conservative adopt rule the gate applies internally)."""
    return adopt_revised(comparison, tau=tau)
