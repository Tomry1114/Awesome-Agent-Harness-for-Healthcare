"""Evidence-State Ledger — the harness's EXTERNAL working state, shared by Modules A/B/C.

It is the single source of truth for: who the active subject is, what evidence has been observed (and
from where), the status of every obligation, the workflow state, proposed actions, interventions the
harness raised, the commit history, and unresolved risks. The agent can READ harness feedback derived
from the ledger but cannot mutate the ledger directly — so trustworthiness state never depends on the
agent's own (fallible) memory. The ledger also makes context-loss recovery + final audit deterministic.
"""

# obligation states
PENDING, SATISFIED, VIOLATED, WAIVED, UNKNOWN = "PENDING", "SATISFIED", "VIOLATED", "WAIVED", "UNKNOWN"
OBLIGATION_STATES = (PENDING, SATISFIED, VIOLATED, WAIVED, UNKNOWN)


class Ledger:
    def __init__(self):
        self.active_subject = None          # {"type","id"}
        self.evidence = []                  # [EvidenceRecord dict]
        self.obligations = {}               # id -> {"state","kind","requires","satisfied_by","note","event_ids"}
        self.workflow_state = {}            # free-form per-substrate (stages reached, etc.)
        self.proposed_actions = []          # [{"id","action","risk","step"}]
        self.interventions = []             # [WINNER harness_decision per hook + its effective outcome]
        self.findings = []                  # [EVERY non-ALLOW capability finding, deduped] — metric numerators
                                            # come from here so a lower-priority finding (e.g. missing_prereq)
                                            # is not erased when a higher one (e.g. wrong_scope) wins the hook.
        self.commit_history = []            # [{"action","step","verified":bool,...}]
        self.unresolved_risks = []          # [{"rule_id","reason","risk"}]
        # per-metric OPPORTUNITY counts (denominators): each metric is rate = numerator / its own
        # opportunity set, never / task-count. e.g. commit_proposal, subject_bearing_action, eligible_revise.
        self.opportunities = {}
        self._evk = 0

    def bump_opportunity(self, key, n=1):
        self.opportunities[key] = self.opportunities.get(key, 0) + n

    # ---- subject -------------------------------------------------------------
    def set_subject(self, subject):
        self.active_subject = dict(subject) if subject else None

    def subject_id(self):
        return (self.active_subject or {}).get("id")

    # ---- evidence ------------------------------------------------------------
    def add_evidence(self, type, value, subject_id=None, source_event=None, source_type=None, extra=None):
        self._evk += 1
        rec = {"evidence_id": "ev-%d" % self._evk, "type": type, "value": value,
               "subject_id": subject_id, "source_event": source_event, "source_type": source_type}
        if extra:
            rec.update(extra)
        self.evidence.append(rec)
        return rec

    def evidence_for(self, subject_id):
        return [e for e in self.evidence if e.get("subject_id") == subject_id]

    # ---- obligations ---------------------------------------------------------
    def declare_obligation(self, oid, kind="evidence", requires=None, satisfied_by=None, state=PENDING):
        self.obligations[oid] = {"id": oid, "state": state, "kind": kind,
                                 "requires": list(requires or []), "satisfied_by": satisfied_by,
                                 "note": None, "event_ids": []}

    def set_obligation(self, oid, state, note=None, event_id=None):
        if state not in OBLIGATION_STATES:
            raise ValueError("bad obligation state %r" % (state,))
        ob = self.obligations.setdefault(oid, {"id": oid, "state": PENDING, "kind": None,
                                               "requires": [], "satisfied_by": None, "note": None,
                                               "event_ids": []})
        ob["state"] = state
        if note is not None:
            ob["note"] = note
        if event_id is not None:
            ob["event_ids"].append(event_id)
        return ob

    def obligation_state(self, oid):
        return (self.obligations.get(oid) or {}).get("state", UNKNOWN)

    def pending_prerequisites(self, oids):
        """Of the given obligation ids, which are NOT yet SATISFIED/WAIVED."""
        out = []
        for oid in (oids or []):
            if self.obligation_state(oid) not in (SATISFIED, WAIVED):
                out.append(oid)
        return out

    # ---- interventions / commits / risks ------------------------------------
    def record_intervention(self, decision_dict):
        self.interventions.append(decision_dict)

    def record_finding(self, finding):
        """Record one capability finding (deduped by proposal+reason_code+capability) for metric numerators."""
        key = (finding.get("action_key"), finding.get("reason_code"), finding.get("capability"))
        if any((f.get("action_key"), f.get("reason_code"), f.get("capability")) == key for f in self.findings):
            return
        self.findings.append(finding)

    def record_proposed(self, action, risk, step):
        pid = "action-%d" % (len(self.proposed_actions) + 1)
        self.proposed_actions.append({"id": pid, "action": action, "risk": risk, "step": step})
        return pid

    def record_commit(self, action, step, verified=None, detail=None):
        self.commit_history.append({"action": action, "step": step, "verified": verified, "detail": detail})

    def add_unresolved_risk(self, rule_id, reason, risk=None):
        self.unresolved_risks.append({"rule_id": rule_id, "reason": reason, "risk": risk})

    # ---- audit ---------------------------------------------------------------
    def to_dict(self):
        return {"active_subject": self.active_subject, "evidence": self.evidence, "findings": self.findings,
                "obligations": self.obligations, "workflow_state": self.workflow_state,
                "proposed_actions": self.proposed_actions, "interventions": self.interventions,
                "commit_history": self.commit_history, "unresolved_risks": self.unresolved_risks,
                "opportunities": self.opportunities}
