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


# ---- Scoped Repair finding lifecycle ---------------------------------------------------------------------
from enum import Enum
from dataclasses import dataclass, field as _dc_field


class FindingStatus(str, Enum):
    OPEN = "open"
    DELIVERED = "delivered"        # handed to the agent; awaiting an attempt
    ATTEMPTED = "attempted"        # target projection changed since delivery -> revalidate
    RESOLVED = "resolved"          # delta-validated: defect fixed, protected content preserved
    REGRESSED = "regressed"        # attempt fixed the target but destroyed protected content
    EXHAUSTED = "exhausted"        # repair-attempt budget spent -> stop prompting (anti-churn)


@dataclass
class FindingRecord:
    finding: object                # RepairFinding
    status: "FindingStatus"
    first_step: int
    last_step: int
    delivery_count: int = 0
    repair_attempts: int = 0
    baseline_projection: dict = None
    last_projection: dict = None


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
        self.completed_commits = set()      # keys of irreversible commits that ALREADY succeeded (verified True)
        self.pending_resolution = None      # an OPEN must-resolve violation the next final answer must close
        self.unresolved_risks = []          # [{"rule_id","reason","risk"}]
        self.resolutions = []               # [harness_resolution dicts] — a REVISE later repaired (causal)
        self.repair_findings = {}           # finding_id -> FindingRecord (Scoped Repair lifecycle)
        self.observations = []              # normalized perception observations (evidence_coverage)
        self._obsk = 0
        self.advisories = []                # non-enforced (advisory) findings -- for measurement
        self.acquire_count = 0              # read-only evidence acquisitions this task (ACQUIRE cap)
        # OPERATIONAL NON-DEGRADATION (mutation authorization): semantic feedback never grants write
        # permission. Under a hold, a state mutation must match a scoped single-use authorization.
        self.mutation_hold = False          # set when a non-deterministic semantic finding emits feedback
        self.mutation_hold_origin = None     # {intervention_id, finding_id, capability}
        self.mutation_authorizations = []    # [MutationAuthorization] (scoped, single-use)
        self.pending_authorization = None   # C2: the auth verify_commit MATCHED this action (reserve-eligible); the executor reserves/dispatches it -- capabilities never mutate auth state
        self.terminal_locked = False        # graded commit verified -> no further mutation (rule 1)
        self._auth_seq = 0
        # per-metric OPPORTUNITY counts (denominators): each metric is rate = numerator / its own
        # opportunity set, never / task-count. e.g. commit_proposal, subject_bearing_action, eligible_revise.
        self.opportunities = {}
        self._opp_seen = set()              # (key, step) already counted -> one opportunity per ACTION
        self._evk = 0
        self.evidence_version = 0           # bumped on EVERY add_evidence (incl. ATTEMPTED/foreign)
        self.validated_evidence_version = 0  # bumped ONLY on a NEW validated, non-foreign evidence signature
        self._validated_sigs = set()        # dedup: a repeated identical read is not new progress
        # CONTRACT(3): legacy note kept below; the validated counter is the one the repair budget keys on.
                                            # revision-identity key so a stuck-revision counter RESETS
                                            # when new evidence lands (the agent made progress)

    def set_mutation_hold(self, intervention_id=None, finding_id=None, capability=None):
        """A non-deterministic semantic finding emitted feedback -> writes now require explicit authorization."""
        self.mutation_hold = True
        self.mutation_hold_origin = {"intervention_id": intervention_id, "finding_id": finding_id,
                                     "capability": capability}

    def clear_mutation_hold(self):
        self.mutation_hold = False
        self.mutation_hold_origin = None

    def mint_authorization(self, source, allowed_semantic_type, allowed_tool=None, target_path=None,
                           allowed_effect=None, expected_postcondition=None, intervention_id=None):
        """Mint a scoped, single-use write authorization. source in user_goal|deterministic_gap|
        evidence_supported_plan. Returns the MutationAuthorization (also stored on the ledger)."""
        from .authorization import MutationAuthorization, VALID_SOURCES
        if source not in VALID_SOURCES:
            raise ValueError("invalid authorization source: %r" % (source,))
        self._auth_seq += 1
        auth = MutationAuthorization(
            authorization_id="auth-%d" % self._auth_seq,
            intervention_id=intervention_id or (self.mutation_hold_origin or {}).get("intervention_id") or "iv-?",
            source=source, allowed_semantic_type=allowed_semantic_type, allowed_tool=allowed_tool,
            target_path=target_path, allowed_effect=allowed_effect,
            expected_postcondition=expected_postcondition or {},
            baseline_state_version=self.evidence_version, evidence_version=self.validated_evidence_version)
        self.mutation_authorizations.append(auth)
        return auth

    def find_matching_authorization(self, sem, action):
        """The AVAILABLE authorization whose EXACT scope this action matches AND whose evidence_version still
        equals the current validated version, or None. The version guard (C3.1) makes a stale authorization --
        one minted before new evidence landed -- no longer match, forcing a fresh re-evaluation."""
        from .authorization import exact_scope_match, AUTH_AVAILABLE
        for auth in self.mutation_authorizations:
            if (auth.status == AUTH_AVAILABLE
                    and auth.evidence_version == self.validated_evidence_version
                    and exact_scope_match(auth, sem, action)):
                return auth
        return None

    # -- authorization lifecycle transitions (Commit C1; C3.1 = STRICT pre-state, return bool) --
    # Legal edges ONLY: reserve AVAILABLE->RESERVED; release RESERVED->AVAILABLE; dispatch RESERVED->DISPATCHED;
    # verify/unknown/fail DISPATCHED->terminal; cancel AVAILABLE|RESERVED->CANCELLED. Any illegal edge -> False
    # (no state change) so a mis-sequenced caller cannot fabricate a VERIFIED/DISPATCHED out of thin air.
    def _transition(self, auth, allowed_from, to):
        if auth is not None and auth.status in allowed_from:
            auth.status = to
            return True
        return False

    def reserve_authorization(self, auth):
        from .authorization import AUTH_AVAILABLE, AUTH_RESERVED
        return self._transition(auth, (AUTH_AVAILABLE,), AUTH_RESERVED)

    def release_authorization(self, auth):
        from .authorization import AUTH_RESERVED, AUTH_AVAILABLE
        return self._transition(auth, (AUTH_RESERVED,), AUTH_AVAILABLE)

    def dispatch_authorization(self, auth):
        from .authorization import AUTH_RESERVED, AUTH_DISPATCHED
        return self._transition(auth, (AUTH_RESERVED,), AUTH_DISPATCHED)

    def verify_authorization(self, auth):
        from .authorization import AUTH_DISPATCHED, AUTH_VERIFIED
        return self._transition(auth, (AUTH_DISPATCHED,), AUTH_VERIFIED)

    def unknown_authorization(self, auth):
        from .authorization import AUTH_DISPATCHED, AUTH_UNKNOWN
        return self._transition(auth, (AUTH_DISPATCHED,), AUTH_UNKNOWN)

    def fail_authorization(self, auth):
        from .authorization import AUTH_DISPATCHED, AUTH_FAILED
        return self._transition(auth, (AUTH_DISPATCHED,), AUTH_FAILED)

    def cancel_authorization(self, auth):
        from .authorization import AUTH_AVAILABLE, AUTH_RESERVED, AUTH_CANCELLED
        return self._transition(auth, (AUTH_AVAILABLE, AUTH_RESERVED), AUTH_CANCELLED)

    def consume_authorization(self, auth):
        """LEGACY compat (the effect_completion inline block until C5): force-spend to DISPATCHED regardless of
        pre-state. NOT part of the strict machine -- new code uses reserve()->dispatch()."""
        from .authorization import AUTH_DISPATCHED
        if auth is not None:
            auth.status = AUTH_DISPATCHED

    def bump_opportunity(self, key, step=None, n=1):
        """Count one opportunity. When a step is given, the same (key, step) counts ONCE — so a single
        action that is examined in both before_action and after_action is ONE opportunity, keeping every
        rate (numerator/this) a valid probability <= 1."""
        if step is not None:
            tok = (key, step)
            if tok in self._opp_seen:
                return
            self._opp_seen.add(tok)
        self.opportunities[key] = self.opportunities.get(key, 0) + n

    # ---- subject -------------------------------------------------------------
    def set_subject(self, subject):
        self.active_subject = dict(subject) if subject else None

    def subject_id(self):
        """Return the TYPED reference 'type/id' (not the bare id) so downstream comparison keeps the type:
        Patient/123 must not match Encounter/123. _same_subject / _eq handle the untyped-on-one-side case."""
        s = self.active_subject or {}
        if not s.get("id"):
            return None
        i = str(s["id"])
        if "/" in i:                # id already a typed ref (e.g. 'Patient/123') -> don't double-prefix
            return i
        return ("%s/%s" % (s["type"], i)) if s.get("type") else i

    def subject_ref(self):
        return self.active_subject

    # ---- evidence ------------------------------------------------------------
    def add_evidence(self, type, value, subject_id=None, source_event=None, source_type=None, extra=None):
        self._evk += 1
        self.evidence_version += 1          # every event (even a failed/foreign read) bumps this
        _x = extra or {}
        if _x.get("status") == "VALIDATED" and _x.get("scope_relation") != "foreign":
            import hashlib as _hl
            _sig = (str(subject_id), str(source_type or _x.get("source_class")), str(_x.get("resource")),
                    str(_x.get("modality")),
                    _hl.sha1(str(_x.get("value_full") or value or "").encode("utf-8", "replace")).hexdigest()[:16])
            if _sig not in self._validated_sigs:   # only a NEW validated evidence signature is genuine progress
                self._validated_sigs.add(_sig)
                self.validated_evidence_version += 1   # a repeated identical read can no longer refresh the repair budget
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

    def record_commit(self, action, step, verified=None, detail=None, semantic_type=None):
        self.commit_history.append({"action": action, "step": step, "verified": verified, "detail": detail,
                                    "semantic_type": semantic_type})

    def unresolved_operational_commit(self):
        """An OPERATIONAL write (create/update/submit) that was ATTEMPTED but whose LATEST attempt did not
        verify True (failed or unknown) and was never later resolved. Finalizing over it = process-output
        inconsistency (the answer would report success over a write that did not land)."""
        latest = {}
        for c in self.commit_history:
            if c.get("semantic_type") in ("create", "update", "submit"):
                latest[c.get("action")] = c          # keep the LAST attempt per capability
        for c in latest.values():
            if c.get("verified") is not True:
                return c
        return None

    def add_unresolved_risk(self, rule_id, reason, risk=None):
        self.unresolved_risks.append({"rule_id": rule_id, "reason": reason, "risk": risk})

    def record_advisory(self, finding_dict):
        """A semantic/uncertain finding that the admission gate did NOT enforce -- recorded only."""
        self.advisories.append(finding_dict)

    # ---- normalized observations (evidence_coverage gate input) -------------
    def record_observation(self, tool_capability, subject=None, region=None, modality=None,
                           attributes_observed=None, result_status="valid", content=""):
        """One perception/read act -> a normalized observation. `content` is the tool OUTPUT text the
        support judge reads (without it the judge is blind -> false unsupported verdicts)."""
        self._obsk += 1
        rec = {"observation_id": "obs-%d" % self._obsk, "tool_capability": tool_capability,
               "subject": subject, "region": region, "modality": modality,
               "attributes_observed": list(attributes_observed or []), "result_status": result_status,
               "content": (content or "")[:1500]}
        self.observations.append(rec)
        return rec

    # ---- scoped repair lifecycle --------------------------------------------
    REPAIR_MAX_ATTEMPTS = 2

    def repair_decision(self, finding, current_projection):
        """Dedup gate. Returns (mode, record) where mode in {new, revalidate, suppress}. 'new' => first
        sighting (open+deliver). 'suppress' => already resolved/exhausted, or delivered but the agent has
        not changed the target since (don't re-nag). 'revalidate' => the agent changed the target; run
        delta validation on the attempt instead of re-prompting the same text."""
        rec = self.repair_findings.get(finding.finding_id)
        if rec is None:
            return ("new", None)
        if rec.status in (FindingStatus.RESOLVED, FindingStatus.EXHAUSTED):
            return ("suppress", rec)
        if rec.repair_attempts >= self.REPAIR_MAX_ATTEMPTS:
            rec.status = FindingStatus.EXHAUSTED
            return ("suppress", rec)
        if rec.status == FindingStatus.DELIVERED and current_projection == rec.last_projection:
            return ("suppress", rec)        # no new attempt -> silence
        return ("revalidate", rec)

    def open_finding(self, finding, baseline_projection, step):
        rec = FindingRecord(finding=finding, status=FindingStatus.OPEN, first_step=step, last_step=step,
                            baseline_projection=baseline_projection, last_projection=baseline_projection)
        self.repair_findings[finding.finding_id] = rec
        return rec

    def mark_delivered(self, finding_id, projection, step):
        rec = self.repair_findings.get(finding_id)
        if rec is None:
            return None
        rec.status = FindingStatus.DELIVERED
        rec.delivery_count += 1
        rec.last_projection = projection
        rec.last_step = step
        return rec

    def mark_attempted(self, finding_id, projection, step):
        rec = self.repair_findings.get(finding_id)
        if rec is None:
            return None
        rec.status = FindingStatus.ATTEMPTED
        rec.repair_attempts += 1
        rec.last_projection = projection
        rec.last_step = step
        return rec

    def resolve_finding(self, finding_id):
        rec = self.repair_findings.get(finding_id)
        if rec is not None:
            rec.status = FindingStatus.RESOLVED
        return rec

    def regress_finding(self, finding_id):
        rec = self.repair_findings.get(finding_id)
        if rec is not None:
            rec.status = FindingStatus.REGRESSED
        return rec

    # ---- audit ---------------------------------------------------------------
    def to_dict(self):
        # the persisted audit keeps the short `value` preview; the full verification payload (value_full,
        # read by the grounding judge at runtime) is dropped here so result.json stays compact.
        _ev = [{k: v for k, v in e.items() if k != "value_full"} for e in self.evidence]
        return {"active_subject": self.active_subject, "evidence": _ev, "findings": self.findings,
                "obligations": self.obligations, "workflow_state": self.workflow_state,
                "proposed_actions": self.proposed_actions, "interventions": self.interventions,
                "commit_history": self.commit_history, "unresolved_risks": self.unresolved_risks,
                "resolutions": self.resolutions,
                "repair_lifecycle": [{"finding_id": _fid,
                                      "status": (_r.status.value if hasattr(_r.status, "value") else str(_r.status)),
                                      "target_path": _r.finding.target_path, "defect_type": _r.finding.defect_type,
                                      "rule_id": _r.finding.rule_id, "delivery_count": _r.delivery_count,
                                      "repair_attempts": _r.repair_attempts, "first_step": _r.first_step,
                                      "last_step": _r.last_step} for _fid, _r in self.repair_findings.items()],
                "observations": [{"observation_id": _o.get("observation_id"), "tool": _o.get("tool_capability"),
                                  "region": _o.get("region"), "modality": _o.get("modality"),
                                  "status": _o.get("result_status"), "content": (_o.get("content") or "")[:300]}
                                 for _o in self.observations],
                "advisories": self.advisories,
                "opportunities": self.opportunities}
