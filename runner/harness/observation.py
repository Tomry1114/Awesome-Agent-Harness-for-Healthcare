"""Claim-observation coverage — the oracle-blind core of the evidence_coverage gate.

It proves ONE thing only: every PERCEPTUAL claim in the final answer traces to an actually-executed
observation of its target. It is NOT gold tool-path completeness (there is no gold path), so it never
asserts "the agent used all tools a correct decision needs". The honest name is *claim-conditioned
observational coverage* / perceptual traceability.

Pipeline (deterministic FIRST, judge only at the margin):
  1. classify each claim: perceptual | interpretive | background | recommendation   (judge, upstream)
  2. perceptual claims -> deterministic target coverage vs the ledger's normalized observations
       - region never observed            -> unobserved_target        (REACQUIRE_EVIDENCE)
       - region observed, attribute absent -> needs semantic support  -> if unsupported:
                                              unsupported_by_observation (VERIFY_OR_REMOVE)
       - no concrete target                -> untraceable_claim        (EDIT_OR_REMOVE)
  3. interpretive claims require >=1 covered perceptual premise (global, conservative)
  4. background / recommendation claims are NOT gated here
Tool names for REACQUIRE are filled by the KERNEL affordance registry, never by this module or the judge.
"""
from __future__ import annotations
from dataclasses import dataclass, field

from .repair import RepairFinding, RepairOperation, make_finding_id

CLAIM_TYPES = ("perceptual", "interpretive", "background", "recommendation")


@dataclass(frozen=True)
class Observation:
    observation_id: str
    tool_capability: str
    subject: str | None = None
    region: str | None = None
    modality: str | None = None
    attributes_observed: tuple = ()
    result_status: str = "valid"
    content: str = ""              # the tool's ACTUAL output text -- what the support judge must read

    def summary(self):
        """The dict shown to the semantic-support judge: metadata PLUS the actual observed content (without
        content the judge is blind and defaults to 'unsupported' -> false over-correction)."""
        return {"tool": self.tool_capability, "region": self.region, "modality": self.modality,
                "attributes": list(self.attributes_observed), "status": self.result_status,
                "observed_content": (self.content or "")[:600]}


@dataclass(frozen=True)
class Claim:
    claim_id: str
    idx: int
    claim_type: str
    text: str = ""
    subject: str | None = None
    region: str | None = None
    modality: str | None = None
    attribute: str | None = None

    @property
    def path(self):
        return "answer.claims[%d]" % self.idx

    @property
    def stable_key(self):
        """Content-based identity (text+region+modality+attribute) so a finding's id survives re-decomposition
        even when claim ORDER changes between answer revisions (index-based ids were unstable)."""
        import hashlib
        raw = "|".join([(_n(self.text) or ""), (_n(self.region) or ""),
                        (_n(self.modality) or ""), (_n(self.attribute) or "")])
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _n(s):
    return str(s).strip().lower() if s not in (None, "") else None


_GLOBAL_REGIONS = (None, "", "image", "global", "whole", "full", "full-image", "whole-image", "entire")


def _valid(observations):
    return [o for o in observations if o.result_status == "valid"]


def _is_global(region):
    return _n(region) in _GLOBAL_REGIONS


def region_observed(claim, observations):
    """Was the claim's REGION observed by any VALID observation? A WHOLE-IMAGE/global observation counts as
    WEAK coverage of any specific region (the agent did look at the image) -> the specific-attribute check +
    judge then decide whether that look actually supports the claim. Returns the Observation or None
    (None = the agent never looked at all -> REACQUIRE)."""
    cr, cm, cs = _n(claim.region), _n(claim.modality), _n(claim.subject)
    for o in _valid(observations):
        if cs and _n(o.subject) and _n(o.subject) != cs:
            continue
        if cm and _n(o.modality) and _n(o.modality) != cm:
            continue
        if _is_global(o.region):
            return o                       # whole-image look weakly covers any region
        if cr and _n(o.region) != cr:
            continue
        return o
    return None


def attribute_observed(claim, observations):
    """Stronger: was the claim's ATTRIBUTE explicitly in an observation that also covers its region?"""
    ca = _n(claim.attribute)
    if not ca:
        return region_observed(claim, observations)
    cr, cm = _n(claim.region), _n(claim.modality)
    for o in _valid(observations):
        if cr and _n(o.region) != cr:
            continue
        if cm and _n(o.modality) and _n(o.modality) != cm:
            continue
        if ca in {_n(a) for a in o.attributes_observed}:
            return o
    return None


def coverage_findings(claims, observations, task_id, semantic_support=None, rule_id="evidence_coverage"):
    """Deterministic finding generation. `semantic_support`: optional {claim_id: bool} judge verdict, consulted
    ONLY for region-covered-but-attribute-unclear claims (margin). allowed_capabilities is left empty here; the
    kernel fills it from the affordance registry. Conservative: silent unless there is a STRONG gap."""
    claims = list(claims)
    covered_perceptual = [c for c in claims if c.claim_type == "perceptual"
                          and region_observed(c, observations) is not None]
    protected = tuple(c.path for c in claims if c.claim_type in ("perceptual", "interpretive")
                      and region_observed(c, observations) is not None)
    out = []

    def _mk(c, defect, op, change):
        return RepairFinding(
            # id keys on the CONTENT (stable_key), not the index -> survives re-decomposition reordering;
            # target_path stays the positional path for localization in the current answer.
            finding_id=make_finding_id(task_id, rule_id, "claim", c.stable_key, defect),
            rule_id=rule_id, target_type="claim", target_path=c.path, defect_type=defect, operation=op,
            required_change=change, protected_paths=tuple(p for p in protected if p != c.path),
            metadata={"claim_id": c.claim_id, "region": c.region, "modality": c.modality,
                      "attribute": c.attribute})

    for c in claims:
        if c.claim_type == "perceptual":
            if not c.region:
                out.append(_mk(c, "untraceable_claim", RepairOperation.EDIT_OR_REMOVE,
                               "This perceptual claim names no concrete target (region/modality). Tie it to a "
                               "specific observed target, or remove it."))
            elif region_observed(c, observations) is None:
                out.append(_mk(c, "unobserved_target", RepairOperation.REACQUIRE_EVIDENCE,
                               "Inspect the referenced target (%s) before retaining this claim, or remove it." % c.region))
            elif attribute_observed(c, observations) is None:
                # region looked at, but the specific attribute was not explicitly observed -> consult judge
                if semantic_support is not None and semantic_support.get(c.claim_id) is False:
                    out.append(_mk(c, "unsupported_by_observation", RepairOperation.VERIFY_OR_REMOVE,
                                   "The observation of %s does not support this claim. Re-examine it, or remove / "
                                   "soften the claim." % c.region))
                # no verdict or supported -> stay silent (conservative)
        elif c.claim_type == "interpretive":
            # requires at least one covered perceptual premise to exist (global, conservative)
            if not covered_perceptual:
                out.append(_mk(c, "unsupported_by_observation", RepairOperation.VERIFY_OR_REMOVE,
                               "This interpretation rests on no observed perceptual premise. Ground at least one "
                               "supporting observation, or soften the interpretation."))
        # background / recommendation -> NOT gated by this layer
    return out
