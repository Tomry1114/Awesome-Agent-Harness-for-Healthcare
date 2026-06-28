"""Harness mechanism -> LAYER classification (the evolvability substrate). Every governance decision the
harness can emit is tagged infrastructure | compensation | amplification (see HARNESS_DESIGN.md). The layer
report aggregates fire-rates per mechanism so the harness can SELF-REPORT which parts are durable, which are
vestigial (compensation that no longer fires -> retire), and which are provisional (amplification to re-tune).
Pure data + helpers; no runtime dependency."""

INFRASTRUCTURE = "infrastructure"
COMPENSATION = "compensation"
AMPLIFICATION = "amplification"

# reason_code / rule_id  ->  (layer, human mechanism name). Keyed primarily by reason_code; rule_id fallbacks.
LAYER_OF = {
    # --- INFRASTRUCTURE: environment-facing integrity (durable; a stronger model does not obviate it) ---
    "subject_unspecified":            (INFRASTRUCTURE, "subject binding"),
    "foreign_subject":                (INFRASTRUCTURE, "subject binding"),
    "missing_prerequisite":           (INFRASTRUCTURE, "commit prerequisite gating"),
    "process_output_inconsistency":   (INFRASTRUCTURE, "commit integrity (failed/unknown write)"),
    "redundant_commit":               (INFRASTRUCTURE, "pre-commit redundant-write block"),
    "violated_commit":                (INFRASTRUCTURE, "commit execution/postcondition"),
    "unverifiable_commit":            (INFRASTRUCTURE, "commit verifiability"),
    "unmapped_action":                (INFRASTRUCTURE, "fail-closed unmapped action"),
    "unjudgeable":                    (INFRASTRUCTURE, "fail-closed high-risk action"),
    # --- COMPENSATION: patches a model weakness (fragile; retire when fire-rate -> 0) ---
    "unsupported_claim":              (COMPENSATION, "grounding / claim-support veto"),
    "insufficient_grounding":         (COMPENSATION, "grounding under-coverage"),
    "evidence_contradiction":         (COMPENSATION, "grounding contradiction (must-resolve)"),
    "no_new_progress":                (COMPENSATION, "loop / no-progress detection"),
    "unjudgeable_low_confidence":     (COMPENSATION, "low-confidence grounding"),
    # --- AMPLIFICATION: a better way to use existing capability (provisional; re-measure each generation) ---
    "repairable_gap":                 (AMPLIFICATION, "selective epistemic repair (candidate)"),
    "process_gap":                    (AMPLIFICATION, "in-process behavior guidance"),
    # rule_id fallbacks (older bundles surfaced rule_id but not reason_code)
    "commit_execution_failed":        (INFRASTRUCTURE, "commit integrity (failed/unknown write)"),
    "commit_state_unknown":           (INFRASTRUCTURE, "unknown commit state"),
    "state_transition":               (INFRASTRUCTURE, "commit postcondition (state transition)"),
    "post_commit":                    (INFRASTRUCTURE, "commit postcondition"),
    "uncovered_irreversible_action":  (INFRASTRUCTURE, "fail-closed uncovered irreversible action"),
    "unjudgeable_high_risk":          (INFRASTRUCTURE, "fail-closed high-risk action"),
    "claim_supported_by_evidence":    (COMPENSATION, "grounding / claim-support veto"),
    "semantic_low_confidence":        (COMPENSATION, "low-confidence grounding"),
    "final_requires_obligations":     (INFRASTRUCTURE, "commit prerequisite gating"),
}

# trajectory final_disposition -> layer activity (amplification adoption is the outcome-facing signal)
DISPOSITION_LAYER = {
    "revised_commit_adopted":         (AMPLIFICATION, "repair adopted (B)"),
    "kept_original":                  (AMPLIFICATION, "repair declined (kept A)"),
    "kept_original_no_candidate":     (AMPLIFICATION, "repair: no candidate produced"),
    "abstained_unresolved_violation": (INFRASTRUCTURE, "safe abstention (unresolved hard violation)"),
}


def layer_of(reason_code=None, rule_id=None):
    """(layer, mechanism) for a harness decision; None if unclassified (so new mechanisms surface, not hide)."""
    if reason_code and reason_code in LAYER_OF:
        return LAYER_OF[reason_code]
    if rule_id and rule_id in LAYER_OF:
        return LAYER_OF[rule_id]
    return (None, rule_id or reason_code or "unclassified")
