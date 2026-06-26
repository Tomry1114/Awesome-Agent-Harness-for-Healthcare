"""Harness-specific metrics — computed from the ledger + harness events at the end of a run (§12).

These are the numbers that show the harness's effect, reported ALONGSIDE Native Outcome + the 7 dims:
  wrong_scope_action_rate, missing_prerequisite_rate, unverified_commit_rate, repair_success_rate,
  over_block_rate (needs an oracle of action legality -> left None in P0), escalation_rate.
All are derived deterministically from recorded interventions / commit_history — no model call.
"""


def summarize(ledger, harness_events, mode=None):
    interventions = ledger.interventions or []
    proposed = ledger.proposed_actions or []
    commits = ledger.commit_history or []
    n_actions = max(1, len(proposed))

    def _count(rule_substr=None, decision=None):
        n = 0
        for iv in interventions:
            if decision and iv.get("decision") != decision and iv.get("effective") != decision:
                continue
            if rule_substr and rule_substr not in str(iv.get("rule_id") or ""):
                continue
            n += 1
        return n

    wrong_scope = _count(rule_substr="subject_scope")
    missing_prereq = _count(rule_substr="requires") + _count(rule_substr="obligation")
    unverified = sum(1 for c in commits if c.get("verified") is False)
    n_commits = max(1, len(commits))
    escalations = sum(1 for iv in interventions if iv.get("effective") == "ESCALATE"
                      or iv.get("decision") == "ESCALATE")
    # repairs: a REVISE followed by a harness_resolution(resolution='repaired')
    resolutions = [e for e in (harness_events or []) if e.get("event_type") == "harness_resolution"]
    n_revise = sum(1 for iv in interventions if iv.get("decision") == "REVISE")
    n_repaired = sum(1 for r in resolutions if r.get("resolution") == "repaired")

    return {
        "mode": mode,
        "n_proposed_actions": len(proposed), "n_commits": len(commits),
        "n_interventions": len(interventions),
        "wrong_scope_action_rate": round(wrong_scope / n_actions, 3),
        "missing_prerequisite_rate": round(missing_prereq / n_actions, 3),
        "unverified_commit_rate": round(unverified / n_commits, 3),
        "repair_success_rate": (round(n_repaired / n_revise, 3) if n_revise else None),
        "escalation_rate": round(escalations / n_actions, 3),
        "over_block_rate": None,    # needs a legality oracle (held-out); not computable in-run
        "unresolved_risks": len(ledger.unresolved_risks or []),
    }
