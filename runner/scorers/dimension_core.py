#!/usr/bin/env python3
"""How a dimension turns sub-metrics into a score — the single implementation.

This module is the only definition of the applicable-only aggregation; `lifecycle_exec`, `dim_execution`,
`dim_lifecycle`, `dim_tooling` and `dim_context` import it rather than carrying their own copy, so there is no
fallback path through which an ImportError could silently select a different implementation.

Three aggregation forms exist across the panel and they are genuinely different constructs, not duplication:

  * **applicable-only unweighted mean** — Care Action Fulfillment, Clinical Resource Competence, Patient–Evidence Alignment, Clinical Workflow Continuity. Defined here.
  * **CORE-weighted mean with a SHADOW remainder** — Closed-Loop Verification. Defined in `dim_verification`, because
    the weight table is part of that dimension's construct rather than a shared mechanism.
  * **layered composition** — Clinical Auditability composes a headline diagnostic score over four axes and keeps
    the delivery layers as diagnostics. Defined in `dim_observability`.

Only the first is shared, and only the first lives here.

Why applicable-only, and why unweighted
---------------------------------------
A sub-metric with no *opportunity* to be exercised carries no evidence. Scoring it as 1.0 would reward a
trajectory for not being tested, so a clean run would drift upward for the wrong reason; scoring it as 0.0
would punish it for the same non-event. Both are vacuous. The rule is therefore: a sub-metric with no
opportunity is marked NOT_APPLICABLE and is removed from the denominator entirely, so a dimension score is
always a mean over evidence that actually exists. The consequence — deliberate and worth stating in any
write-up — is that a sub-metric's effective weight is 1/(number applicable on *that* trajectory), so the
denominator moves between trajectories. Any A/B comparison must therefore report `n_applicable` alongside
the score, which is why it is part of the returned record rather than a debug aside.

Unweighted, because a weight table is a claim about the relative importance of two constructs, and there is
no evidence that would justify one. Closed-Loop Verification is the exception: its CORE set is a declared contract about
which axes must be present for the dimension to be reportable at all, not a fine-tuned weighting.

Status vocabulary
-----------------
Two spellings for "this sub-metric applies" are live on the wire:

    APPLICABLE      = "valid"        emitted by Care Action Fulfillment / Clinical Resource Competence / Patient–Evidence Alignment / Clinical Workflow Continuity
    APPLICABLE_ALT  = "applicable"   emitted by Clinical Auditability / Closed-Loop Verification

They are **not** interchangeable in `aggregate_applicable`, which counts only `"valid"`. That is exactly
what the four dimensions using it emit, so the strict predicate matches their output. Broadening it would
silently change scores for any future dimension that switched spelling, so it stays strict and the
divergence is documented here.
"""

APPLICABLE = "valid"                 # this sub-metric had an opportunity and was scored
APPLICABLE_ALT = "applicable"        # same meaning, spelling used by Clinical Auditability / Closed-Loop Verification
NOT_APPLICABLE = "not_applicable"    # no opportunity on this trajectory -> excluded from the denominator
ERROR = "error"                      # the sub-metric could not be computed


def submetric(score, status=APPLICABLE, opportunities=None, **extra):
    """One sub-metric record.

    `opportunities` is the count of chances the trajectory gave this sub-metric; it is omitted rather than
    set to None when unknown, so a consumer can distinguish "not reported" from "reported as zero".
    """
    record = {"score": score, "status": status}
    if opportunities is not None:
        record["opportunities"] = opportunities
    record.update(extra)
    return record


def is_applicable(record):
    """Applicable == status APPLICABLE *and* a numeric score. A non-numeric score is never counted.

    `bool` is deliberately NOT excluded even though it is an `int` subclass. If a sub-metric ever
    reports a bool it is a bug in that sub-metric, to be fixed there and not masked here.
    """
    return record.get("status") == APPLICABLE and isinstance(record.get("score"), (int, float))


def aggregate_applicable(submetrics):
    """Mean over applicable sub-metrics only; never a vacuous 1.0.

    Returns the standard dimension record:
      score                  rounded mean, or None when nothing was applicable (NOT 0.0 — absence of
                             evidence is not evidence of failure)
      submetrics             every sub-metric, applicable or not, so the exclusion is auditable
      applicable_submetrics  sorted names that entered the mean
      n_applicable           the denominator, reported because it varies between trajectories
      zero_variance          True when every applicable sub-metric returned the same value — a flag that
                             the dimension could not discriminate on this trajectory
    """
    applicable = {name: rec for name, rec in submetrics.items() if is_applicable(rec)}
    values = [rec["score"] for rec in applicable.values()]
    score = round(sum(values) / len(values), 3) if values else None
    return {"score": score,
            "submetrics": submetrics,
            "applicable_submetrics": sorted(applicable),
            "n_applicable": len(applicable),
            "zero_variance": (len(set(values)) == 1) if values else None}
