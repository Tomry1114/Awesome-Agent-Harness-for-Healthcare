"""Bounded Clinical Recovery v3 - HAB benchmark adapter (interactive-GUI admin portal).

Normalizes the interactive-GUI administrative-portal tasks into the recovery vocabulary: builds ctx
(authoritative referral/EMR state, binding schema, live observation, portal form layout), derives the
goals the ROOT AGENT committed to (oracle-blind, from the agent's own trajectory/declared commitments),
and maps logical state names to the concrete persisted-state paths - including the payer-portal paths
(payer_a_state / payer_b_state) the recovery read-back verifies against.

This is a benchmark ADAPTER (layer 4): it is allowed to name task fields and state paths. It carries NO
gold/checkpoint/reference material at runtime (oracle-blind) and does NOT change kernel/workflow rules.

Python 3.8 compatible.
"""
from ..contracts import CommittedGoal


# ---- goal-type vocabulary this adapter emits (matched by the workflow modules) --------------------
G_PRIOR_AUTH = "submit_prior_auth"
G_APPEAL = "submit_appeal"
G_DOCUMENT = "document_decision"

# ---- binding schema -------------------------------------------------------------------------------
# Clinical/business payload of an admin submission is SEMANTIC (must trace to a semantic source; can
# never bind from system_metadata). The supporting-evidence ATTACHMENT is an acquired-evidence handle
# (operational/evidence class): if the agent never acquired it, it is missing -> BLOCKED_MISSING_EVIDENCE.
_SEMANTIC_FIELDS = (
    "requestType", "patientLastName", "patientFirstName", "patientSearch", "patientDOB",
    "diagnosisCodes", "cptCodes", "subscriberId", "providerNPI", "providerName", "dateOfService",
    "clinicalIndication", "appealRationale", "disposition", "claimId",
)
_OPERATIONAL_FIELDS = (
    "attachmentEvidenceRef", "confirmationId", "requestId", "idempotency_key", "recovery_tag",
    "episode_id", "correlation_id",
)

# ---- logical -> concrete persisted-state paths (RAW portals_state shape the substrate reads back) --
# The substrate read_state returns {full_state (=emr), payer_a_state (=payerA), payer_b_state (=payerB)}.
# A landed prior-auth => payer_<x>_state.submissions is non-empty; a submitted appeal =>
# payer_<x>_state.appealActions.submittedAppeal is truthy; documentation => emr.agentActions.*.
_STATE_PATHS = {
    "prior_auth_landed": "payer_a_state.submissions",
    "prior_auth_landed_a": "payer_a_state.submissions",
    "prior_auth_landed_b": "payer_b_state.submissions",
    "appeal_submitted": "payer_a_state.appealActions.submittedAppeal",
    "appeal_submitted_a": "payer_a_state.appealActions.submittedAppeal",
    "appeal_submitted_b": "payer_b_state.appealActions.submittedAppeal",
    "appeal_rationale": "payer_a_state.appealActions.submittedRationale",
    "decision_documented": "full_state.agentActions.documentedAppealInEpic",
    "auth_note_added": "full_state.agentActions.addedAuthNote",
}

# ---- lifecycle events that engage recovery --------------------------------------------------------
_TRIGGER_EVENTS = frozenset({"before_final", "deliverable_confirmed"})

# ---- verify specs per goal_type (RAW read-back shape) ---------------------------------------------
def _prior_auth_verify(payer):
    root = "payer_b_state" if str(payer).lower() in ("b", "payerb", "payer_b") else "payer_a_state"
    return {"root": root, "path": ["submissions"], "check": "nonempty", "min_len": 1}


def _appeal_verify(payer):
    root = "payer_b_state" if str(payer).lower() in ("b", "payerb", "payer_b") else "payer_a_state"
    return {"root": root, "path": ["appealActions", "submittedAppeal"], "check": "truthy"}


def _document_verify():
    return {"root": "full_state", "path": ["agentActions", "documentedAppealInEpic"], "check": "truthy"}


# markers used by the oracle-blind trajectory fallback (agent's OWN words, not gold).
_PA_MARKERS = ("submit the prior auth", "submit prior authorization", "file the prior auth",
               "prior authorization request", "submit_auth")
_APPEAL_MARKERS = ("file an appeal", "submit the appeal", "appeal the denial", "dispute the claim")
_DOC_MARKERS = ("document the appeal", "document in epic", "documented the disposition",
                "document the disposition", "record the determination")


class HabBenchmarkAdapter(object):
    """Layer-4 adapter for the interactive-GUI administrative portal (HAB)."""

    def context(self, task):
        task = task or {}
        meta = task.get("metadata") or {}
        authoritative = dict(task.get("authoritative_state") or {})
        # fold selected structured metadata into authoritative_state (portal/EMR-derived, not gold).
        clin = meta.get("clinical_context") or {}
        pat = meta.get("patient") or {}
        for key, val in (
            ("diagnosisCodes", clin.get("diagnosis_code")),
            ("cptCodes", clin.get("cpt_code")),
            ("providerName", clin.get("provider")),
            ("patientLastName", (pat.get("name") or "").split(",")[0].strip() or None),
            ("patientDOB", pat.get("dob")),
        ):
            if val is not None and key not in authoritative:
                authoritative[key] = val

        schema = {
            "semantic_fields": list(_SEMANTIC_FIELDS) + list(task.get("extra_semantic_fields", [])),
            "operational_fields": list(_OPERATIONAL_FIELDS) + list(task.get("extra_operational_fields", [])),
        }
        ctx = {
            "observation": task.get("observation"),
            "authoritative_state": authoritative,
            "schema": schema,
            "system_metadata": dict(task.get("system_metadata") or {}),
            "bound_evidence": dict(task.get("bound_evidence") or {}),
            "state_paths": dict(_STATE_PATHS),
            # portal-layout hints the workflows consume (optional overrides).
            "form_fields": task.get("form_fields"),
            "submit_target": task.get("submit_target"),
            "form_url": task.get("form_url"),
            "appeal_locator_args": task.get("appeal_locator_args", ["claimId"]),
            "documentation_required": task.get("documentation_required", []),
        }
        # carry any explicit workflow-target overrides through untouched.
        for k in ("claim_search_target", "dispute_form_target", "rationale_target",
                  "attachment_target", "document_target", "appeal_url"):
            if k in task:
                ctx[k] = task[k]
        return ctx

    def resolve_commitments(self, root, trajectory, goal, judge, ctx):
        """Oracle-blind: derive the goals the agent committed to. Prefer explicit agent-extracted
        commitments (root/task['recovery_commitments']); else scan the agent's trajectory text. One
        CommittedGoal per independent episode. Empty list => DECLINED_NO_COMMITMENT."""
        task = root if isinstance(root, dict) else {}
        explicit = task.get("recovery_commitments")
        goals = []
        if isinstance(explicit, list) and explicit:
            for i, c in enumerate(explicit):
                goals.append(self._mk_goal(i, c))
            return goals
        # trajectory fallback (agent's own words).
        text = self._trajectory_text(trajectory, goal)
        low = text.lower()
        idx = 0
        if any(m in low for m in _PA_MARKERS):
            goals.append(self._mk_goal(idx, {"goal_type": G_PRIOR_AUTH})); idx += 1
        if any(m in low for m in _APPEAL_MARKERS):
            goals.append(self._mk_goal(idx, {"goal_type": G_APPEAL})); idx += 1
        # Documentation recovery REMOVED: on the live portal documentedAppealInEpic is set by the SAME
        # submit that records the disposition (atomically coupled), so it is NOT an independently
        # completable gap -- recovering it would mean the harness choosing the clinical disposition.
        # HAB recovers ONLY the two separable payer-portal submissions (prior_auth / appeal).
        return goals

    def should_trigger(self, lifecycle_event):
        ev = lifecycle_event
        if isinstance(lifecycle_event, dict):
            ev = lifecycle_event.get("event") or lifecycle_event.get("type")
        return ev in _TRIGGER_EVENTS

    def state_path(self, logical_name):
        """Map a logical state name to its concrete persisted-state path (RAW portals_state shape,
        including payer_a_state / payer_b_state). Unknown names pass through unchanged."""
        return _STATE_PATHS.get(logical_name, logical_name)

    # -- helpers ----------------------------------------------------------------------------------
    def _mk_goal(self, i, c):
        gtype = c.get("goal_type", G_PRIOR_AUTH)
        payer = c.get("payer", "a")
        if gtype in (G_PRIOR_AUTH, "prior_authorization", "submit_auth", "prior_auth"):
            verify = _prior_auth_verify(payer)
            logical = "prior_auth_landed_b" if str(payer).lower().startswith("b") else "prior_auth_landed_a"
        elif gtype in (G_APPEAL, "appeal", "appeal_submission", "file_appeal"):
            verify = _appeal_verify(payer)
            logical = "appeal_submitted_b" if str(payer).lower().startswith("b") else "appeal_submitted_a"
        else:
            verify = _document_verify()
            logical = "decision_documented"
        raw = {"verify": verify, "verify_paths": [self.state_path(logical)], "payer": payer}
        raw.update(c.get("raw", {}) or {})
        return CommittedGoal(
            goal_id=c.get("goal_id", "%s#%d" % (gtype, i)),
            goal_type=gtype,
            committed_fields=dict(c.get("committed_fields", {}) or {}),
            dedup_key=c.get("dedup_key", "%s:%s" % (gtype, c.get("goal_id", i))),
            provenance="agent_commitment",
            raw=raw)

    def _trajectory_text(self, trajectory, goal):
        parts = []
        if isinstance(goal, str):
            parts.append(goal)
        if isinstance(trajectory, str):
            parts.append(trajectory)
        elif isinstance(trajectory, (list, tuple)):
            for step in trajectory:
                if isinstance(step, str):
                    parts.append(step)
                elif isinstance(step, dict):
                    for k in ("content", "text", "output", "thought", "action"):
                        v = step.get(k)
                        if isinstance(v, str):
                            parts.append(v)
        return "\n".join(parts)
