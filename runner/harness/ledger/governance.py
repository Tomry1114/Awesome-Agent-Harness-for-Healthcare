"""Harness-specific metrics — computed from the ledger + harness events at the end of a run (§12).

These are the numbers that show the harness's effect, reported ALONGSIDE Native Outcome + the 7 dims:
  wrong_scope_action_rate, missing_prerequisite_rate, unverified_commit_rate, repair_success_rate,
  over_block_rate (needs an oracle of action legality -> left None in P0), escalation_rate.
All are derived deterministically from recorded interventions / commit_history — no model call.
"""


def summarize(ledger, harness_events, mode=None):
    """Each rate = numerator / its OWN opportunity set (never / task-count). A rate is None when its
    opportunity set is empty (no opportunity -> undefined, not 0)."""
    interventions = ledger.interventions or []
    proposed = ledger.proposed_actions or []
    commits = ledger.commit_history or []
    opp = ledger.opportunities or {}

    def _count_rc(code):
        # count by STRUCTURED reason_code (rule_id substrings would double-count, e.g. a rule named
        # 'final_requires_obligations' contains both 'requires' and 'obligation').
        return sum(1 for iv in interventions if iv.get("reason_code") == code)

    def _rate(num, denom):
        return round(num / denom, 3) if denom else None

    wrong_scope = _count_rc("wrong_scope")
    missing_prereq = _count_rc("missing_prerequisite")
    unverified = sum(1 for c in commits if c.get("verified") is False)
    escalations = sum(1 for iv in interventions if iv.get("effective") == "ESCALATE"
                      or iv.get("decision") == "ESCALATE")
    # opportunity denominators
    n_subject_actions = opp.get("subject_bearing_action", 0)   # actions that operated on some subject
    n_commit_proposals = opp.get("commit_proposal", 0)         # proposed R2+ actions
    n_commits = len(commits)
    # repairs: a REVISE whose obligation is later satisfied + retry accepted (harness_resolution).
    resolutions = [e for e in (harness_events or []) if e.get("event_type") == "harness_resolution"]
    n_eligible_revise = sum(1 for iv in interventions
                            if iv.get("decision") == "REVISE" and iv.get("effective") == "REVISE")
    n_repaired = sum(1 for r in resolutions if r.get("resolution") == "repaired")

    return {
        "mode": mode,
        "n_proposed_actions": len(proposed), "n_commits": n_commits,
        "n_interventions": len(interventions),
        # rate -> opportunity denominator (None when no opportunity).
        "wrong_scope_action_rate": _rate(wrong_scope, n_subject_actions),
        "wrong_scope_opportunities": n_subject_actions,
        "missing_prerequisite_rate": _rate(missing_prereq, n_commit_proposals),
        "missing_prerequisite_opportunities": n_commit_proposals,
        "unverified_commit_rate": _rate(unverified, n_commits),
        "repair_success_rate": _rate(n_repaired, n_eligible_revise),
        "repair_opportunities": n_eligible_revise,
        "escalation_rate": _rate(escalations, len(proposed)),
        "over_block_rate": None,    # needs a legality oracle (held-out); not computable in-run
        "unresolved_risks": len(ledger.unresolved_risks or []),
    }
