"""Bounded Clinical Recovery - HAB benchmark adapter (interactive-GUI admin portal; workspace strategy, HAB default workspace).

Normalizes the interactive-GUI administrative-portal tasks into the recovery vocabulary: builds ctx
(authoritative referral/EMR state, binding schema, live observation, portal form layout), derives the
goals the ROOT AGENT committed to (oracle-blind, from the agent's own trajectory/declared commitments),
and maps logical state names to the concrete persisted-state paths - including the payer-portal paths
(payer_a_state / payer_b_state) the recovery read-back verifies against.

This is a benchmark ADAPTER (layer 4): it is allowed to name task fields and state paths. It carries NO
gold/checkpoint/reference material at runtime (oracle-blind) and does NOT change kernel/workflow rules.

Python 3.8 compatible.
"""
import workspace_api as _W
from workspace_api import (action_control, completion_conjunction, conjunct_met, decision_of, evaluate,
                                     obligation_view, readback_portals)

CommittedGoal = _W.load_relative(__file__, "contracts.py").CommittedGoal
# The workspace this strategy was loaded from, which is the run's (the run loads its middleware from its workspace, and
# load_relative keeps a module inside it): the completion content the shell's obligation engine reads for it.
_WORKSPACE = _W.workspace_of(__file__)


# ---- goal-type vocabulary this adapter emits (matched by the workflow modules) --------------------
G_PRIOR_AUTH = "submit_prior_auth"
G_APPEAL = "submit_appeal"
G_DOCUMENT = "document_decision"
G_GUI = "complete_committed_gui_effect"

# The task's completion conjunction (the committed GUI effect and the rest of what completes the task) is not
# defined here: the shell composes it from the workspace's completion_obligations content
# (runner/harness/obligation_gate.py completion_conjunction), the one conjunction the terminal lock and
# ctx.obligations() read too.

# ---- family classification (the PUBLIC challenge type, never gold) --------------------------------
# Workspace code is given the public task only: the environment config's challenge_type
# and payer_portal, never task_type, native_config or policy. The upstream HealthAdminBench harness hands the challenge
# type to its agent prompt. The family is read from it: the denial work queue (challenge denial_triage) is the
# appeal/denial family, the two prior-auth submissions are prior_auth, the DME fax workflow (challenge workflow) is dme.
# (The hidden task_type would split the denial work queue into denial_triage / electronic_appeals / fax_appeal -- the
# appeal family -- and many other denial task types with no family; the challenge type groups them all.)
_APPEAL_CHALLENGES = ("denial_triage",)
# disposition-CHOICE tasks: recovery must NOT author the commit (it would lock in an under-evidenced gold
# disposition). Only these preempt fill/commit on a decision-readiness gap; every other task is a mechanical
# submission recovery may complete. denial_triage is the one HAB disposition choice; submit_auth/dme are form/fax
# submissions.
_DISPOSITION_CHALLENGES = ("denial_triage",)
_PRIOR_AUTH_CHALLENGES = ("submit_auth_aetna", "submit_auth_anthem")
_DME_CHALLENGES = ("workflow",)


def _challenge_type(task):
    """The public challenge type (environment config challenge_type)."""
    cfg = ((task or {}).get("environment") or {}).get("config") or {}
    return str(cfg.get("challenge_type") or "")


def _is_disposition_task(task):
    """DECISION-AUTHORING property, ORTHOGONAL to workflow-family: the task requires CHOOSING a disposition,
    so recovery must never author it. True when the challenge type is denial_triage."""
    return _challenge_type(task) in _DISPOSITION_CHALLENGES


def _family(task):
    """Coarse admin family from the PUBLIC challenge type: appeal (the denial work queue) | prior_auth | dme | ''."""
    ct = _challenge_type(task)
    if ct in _DME_CHALLENGES:
        return "dme"
    if ct in _PRIOR_AUTH_CHALLENGES:
        return "prior_auth"
    if ct in _APPEAL_CHALLENGES:
        return "appeal"
    return ""


def _goal_needs_attachment(task):
    """PUBLIC goal signals that the submission requires a supporting attachment/download."""
    g = str((task or {}).get("goal") or "").lower()
    return any(s in g for s in ("attach", "auth letter", "authorization letter",
                                "download", "supporting document"))


_SUBMIT_SURFACE_HINTS = ("submit", "appeal", "auth", "prior", "fax", "portal", "dispute",
                        "disposition", "denial", "payer", "referral", "note")


def _operational_commitment(trajectory, target_keywords=None):
    """A real OPERATIONAL COMMITMENT is a locally-complete prefix on the TARGET
    submission surface -- the agent REACHED a submission surface (a navigate/click whose route/label
    identifies that surface) AND entered >=1 field WHILE ON THAT SAME surface. 'Clicked somewhere and typed
    somewhere' on unrelated surfaces is NOT a commitment. Distinguishes:
      task_objective       = the public task asks for a submission (metadata only);
      operational_commitment = the agent actually began it on the submission surface (trajectory evidence);
      committed_payload    = the field VALUES the agent actually entered on that surface.
    Oracle-blind: uses the agent's own trajectory + the PUBLIC target keywords, never gold. Completing the
    committed effect should drive the irreversible submit only with such a verifiable commitment."""
    hints = tuple(str(k).lower() for k in (target_keywords or ())) + _SUBMIT_SURFACE_HINTS

    def _surface_id(args, label=""):
        for k in ("url", "route", "path", "form", "surface", "page", "view", "target"):
            v = args.get(k)
            if v:
                return str(v).lower()
        lab = str(label or args.get("label") or args.get("ref") or "").lower()
        return lab or None

    def _is_submit_surface(s):
        return bool(s) and any(h in s for h in hints)

    cur = None
    per_surface = {}                                          # surface-id -> {"reached": bool, "fields": {}}
    filled_any = {}
    submitted_ok = False
    for e in (trajectory or []):
        if not isinstance(e, dict):
            continue
        tool = e.get("tool")
        args = e.get("args") or {}
        if tool in ("navigate", "navigate_to", "click"):
            s = _surface_id(args, e.get("label"))
            if s is not None:
                cur = s
                per_surface.setdefault(cur, {"reached": False, "fields": {}})["reached"] = True
            # HAB never emits a `submit` tool -- submission is a click on a submission-looking control.
            # Reuse this module's own surface hints rather than inventing a second notion of submission.
            if tool == "click" and _is_submit_surface(_surface_id(args, e.get("label")) or ""):
                if e.get("status") in (None, "ok", "success") or e.get("ok") is True:
                    submitted_ok = True
        elif tool in ("fill", "type", "select"):
            # `fill` is the field-write tool HAB agents emit (`type` is rare). Without it this loop
            # would record no fields at all and no trajectory would produce a commitment.
            k = args.get("field") or args.get("ref") or args.get("target") or args.get("name")
            v = args.get("text") if args.get("text") is not None else args.get("value")
            if k is not None and v not in (None, ""):
                s = cur or _surface_id(args)
                rec = per_surface.setdefault(s, {"reached": s is not None, "fields": {}})
                rec["fields"][str(k)] = v
                filled_any[str(k)] = v
        elif tool == "submit" and (e.get("status") in ("ok", "success") or e.get("ok") is True):
            submitted_ok = True
    # a locally-complete prefix = ONE surface the agent both REACHED and FILLED (>=1 field on that surface)
    # that looks like the submission target (route/label matches a submission hint). Unrelated
    # click-here-type-there does not qualify.
    committed = False
    payload = {}
    for s, rec in per_surface.items():
        if rec.get("reached") and rec.get("fields") and _is_submit_surface(s):
            committed = True
            payload.update(rec["fields"])
    # a SUCCESSFUL terminal submit is itself a commitment (the agent committed the effect); whether the
    # final submit landed is a SEPARATE fact used downstream to decide re-submit vs document.
    if submitted_ok and filled_any:
        committed = True
        payload.update(filled_any)
    return committed, payload


def agent_commit_refs(trajectory, submissions, state):
    """The controls of the commits the agent itself attempted whose effect has
    not landed -- the only commits the shell's harness action service lets the harness help complete -- minus every control
    a harness commit was already attempted or refused on in this episode (the shell allows one completion attempt and
    refuses a repeat after an unconfirmed one). `submissions` are the composed submission conjuncts (each names the
    decision it is the effect of); `state` the obligation view they are read on. The shell decides; this keeps the strategy
    from emitting a commit the shell refuses."""
    unmet = set(c.get("decision") for c in (submissions or []) if not conjunct_met(c, state))
    refs, tried = [], set()
    for e in (trajectory or []):
        if not isinstance(e, dict):
            continue
        if e.get("event_type") == "tool_call" and e.get("origin") == "agent":
            ref = action_control(e)
            if ref and ref not in refs and decision_of(e, _WORKSPACE) in unmet:
                refs.append(ref)
        elif (e.get("event_type") == "tool_call" and e.get("tool") == "submit"
              and ((e.get("origin") == "harness" and e.get("requester") == "recovery") or e.get("origin") == "recovery")) or \
                (e.get("event_type") == "harness_action_refused" and e.get("stage") == "irreversible_commit"):
            ref = action_control(e)
            if ref:
                tried.add(ref)
    return [r for r in refs if r not in tried]


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
_TRIGGER_EVENTS = frozenset({"before_final"})

# markers used by the oracle-blind trajectory fallback (agent's OWN words, not gold).
_PA_MARKERS = ("submit the prior auth", "submit prior authorization", "file the prior auth",
               "prior authorization request", "submit_auth")
_APPEAL_MARKERS = ("file an appeal", "submit the appeal", "appeal the denial", "dispute the claim")
_DOC_MARKERS = ("document the appeal", "document in epic", "documented the disposition",
                "document the disposition", "record the determination")


def _decision_readiness_gap(task, live_state, decision_made):
    """Decision-readiness detector -- per-FAMILY evidence requirement. Treating EVERY agentActions
    key starting with 'viewed' as required does not work: the native EHR initialises many viewed* fields for ALL
    tasks, which would over-block denial (spurious viewed*=false gaps) and MISS prior-auth's non-'viewed' evidence
    (clicked*Tab / readClinicalNote / downloadedAuthLetter). Instead this keys a small REQUIRED-evidence set on the
    PUBLIC task family/goal. Oracle-blind: never reads the gold disposition/value -- only which decision-input
    sources the portal TRACKS for that family.
    Returns AVAILABLE-but-UNOBSERVED required sources; empty => evidence-complete (a wrong decision here is
    reasoning, not a missing-source gap). Fires ONLY when a decision was actually made. It does not consult the
    task's policy: policy is not given to workspace code, and every HAB task's policy carries the escalation trigger
    insufficient_evidence_for_action, so a policy condition would hold on every task anyway."""
    if not decision_made:                                    # not a produced decision -> no readiness claim
        return []
    aa = {}
    if isinstance(live_state, dict):
        aa = live_state.get("agentActions") or {}
        if not aa:
            aa = ((live_state.get("_live_full_state") or {}) or {}).get("agentActions") or {}
        if not aa:
            aa = ((live_state.get("full_state") or {}) or {}).get("agentActions") or {}
    if not isinstance(aa, dict):
        aa = {}
    fam = _family(task)
    if fam == "appeal":
        # denial / appeal: the environment TRACKS the denial-detail + remittance-image reviews (native
        # checkpoints full_state.agentActions.viewedDenialDetails / viewedRemittanceImage).
        required = ["viewedDenialDetails", "viewedRemittanceImage"]
    elif fam == "prior_auth":
        # submit_auth: the coverages/diagnoses/services tab reviews + the clinical note; the auth-letter
        # download only when the PUBLIC goal needs a supporting attachment.
        required = ["clickedDiagnosesTab", "clickedServicesTab", "clickedCoveragesTab", "readClinicalNote"]
        if _goal_needs_attachment(task):
            required.append("downloadedAuthLetter")
    else:
        # dme is a mechanical workflow (no disposition decision) -> no readiness gate; unknown -> none.
        required = []
    return [k for k in required if not aa.get(k)]             # required but not observed


# ---- PUBLIC portal source that REVEALS each tracked decision-input flag (oracle-blind) --------
# _decision_readiness_gap returns agentActions trackers the portal keeps for a family (viewed*/clicked*/
# read*/downloaded*). To RESOLVE a gap the recovery must ACQUIRE that evidence: navigate to the PUBLIC portal
# control/route that surfaces it and READ it (read-only), then return control to the root agent to re-decide.
# This maps each flag -> the PUBLIC control label keywords + a coarse route the generic GUI workflow can act
# on. It is public portal naming only (never a gold disposition/value); the workflow stays benchmark-neutral
# -- it just consumes {flag, nav_keywords, route}.
_EVIDENCE_SOURCE_BY_FLAG = {
    "viewedDenialDetails":   {"nav_keywords": ["denial detail", "denial reason", "denial", "eob",
                                               "explanation of benefits"], "route": "denial_details"},
    "viewedRemittanceImage": {"nav_keywords": ["remittance", "remittance image", "remittance advice", "remit",
                                               "view image"], "route": "remittance_image"},
    "clickedDiagnosesTab":   {"nav_keywords": ["diagnoses", "diagnosis"], "route": "diagnoses"},
    "clickedServicesTab":    {"nav_keywords": ["services", "procedures"], "route": "services"},
    "clickedCoveragesTab":   {"nav_keywords": ["coverages", "coverage", "benefits"], "route": "coverages"},
    "readClinicalNote":      {"nav_keywords": ["clinical note", "clinical notes", "progress note", "chart note",
                                               "read note"], "route": "clinical_note"},
    "downloadedAuthLetter":  {"nav_keywords": ["authorization letter", "auth letter", "download auth", "letter"],
                              "route": "auth_letter"},
}


def _decision_readiness_sources(gap_flags):
    """Map each decision-readiness gap flag to the PUBLIC portal source that reveals it (nav keywords + route)
    so the generic recovery workflow can ACQUIRE it read-only. Oracle-blind: public portal naming only, no
    gold. Unknown flags pass through with no locator (workflow will fall through to the re-entry handoff)."""
    out = []
    for f in (gap_flags or []):
        src = _EVIDENCE_SOURCE_BY_FLAG.get(f)
        if src is not None:
            out.append({"flag": f, "nav_keywords": list(src["nav_keywords"]), "route": src["route"]})
        else:
            out.append({"flag": f, "nav_keywords": [], "route": None})
    return out


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
            # the shell's one obligation view of the live read-back (entry.py), what the early decline evaluates
            "obligation_state": obligation_view(readback_portals(task.get("recovery_readback"))),
        }
        # carry any explicit workflow-target overrides through untouched.
        for k in ("claim_search_target", "dispute_form_target", "rationale_target",
                  "attachment_target", "document_target", "appeal_url"):
            if k in task:
                ctx[k] = task[k]
        return ctx

    # (helper defined at module scope: _dig_state)
    def resolve_commitments(self, root, trajectory, goal, judge, ctx):
        """Oracle-blind, GENERIC: the task's completion conjunction is not yet realized. Emit ONE generic
        complete_committed_gui_effect goal carrying it: verify_all, the submission conjuncts (the effects of the commits the
        agent itself attempted; verify_spec = the first, None when the agent attempted none) and agent_commit_refs (the
        controls of those commits that did not land, the only commits the harness may help complete). The conjunction
        is the shell's one composition over the workspace's completion_obligations content from the public goal and the
        agent's own decisions in `trajectory` (runner/harness/obligation_gate.py), the same one the terminal lock and
        ctx.obligations() read. The generic workflow decides per-control whether every REQUIRED field can be bound (from
        EMR/commitment/evidence) and BLOCKS if new content is needed."""
        task = root if isinstance(root, dict) else {}
        ct = _challenge_type(task)
        _decisions, conj = completion_conjunction(task, _WORKSPACE, actions=trajectory)
        if not conj:
            return []                                          # no completion conjunction for this task -> decline
        subs = [c for c in conj if c.get("decision")]
        vs = subs[0] if subs else None
        # already realized? (oracle-blind live-state read folded into authoritative_state) -> nothing to do.
        live = (ctx or {}).get("authoritative_state") or {}
        # the conjunction is evaluated against the shell's obligation view of the same read-back (context)
        # Decline ONLY when the FULL conjunction is already satisfied, not just the submit effect. If the agent
        # submitted but a RESIDUAL subgoal (EHR documentation / work-item resolution) is unmet, recovery must
        # still engage to complete the remaining subgoals -- this is what makes submitted-but-not-documented
        # tasks recoverable (expand trigger coverage beyond the single submit; the workflow + the operational-
        # commitment gate keep cold starts / irreversible submits safe).
        if all(evaluate(conj, (ctx or {}).get("obligation_state") or {})):
            return []                                          # FULL conjunction met -> DECLINED (correct)
        # target keywords from the PUBLIC task goal (operational, not gold) drive generic navigation.
        import re as _re
        gtext = str(task.get("goal") or "")
        kws = [w for w in _re.findall(r"[A-Za-z]{4,}", gtext.lower())
               if w not in ("open", "review", "available", "information", "about", "this", "that",
                            "with", "from", "into", "your", "determine", "appropriate", "request")]
        # PARSE the trajectory for a real operational commitment (not a task_type -> agent_commitment mislabel).
        _committed, _payload = _operational_commitment(trajectory, kws)
        # The agent's OWN multi-value collections (diagnosis/CPT/...) accumulated across the trajectory
        # (committed_payload overwrites repeated same-field entries; a SET must keep all of them). Values are
        # strictly the agent's typed codes -- never invented. Fed to the collection executor.
        try:
            from workspace_api import committed_collection as _cc_fn, COLLECTION_CONCEPTS as _CC_SET
            _committed_collections = {c: _cc_fn(trajectory, c) for c in _CC_SET}
            _committed_collections = {c: v for c, v in _committed_collections.items() if v}
        except Exception:
            _committed_collections = {}
        _cf = dict((task.get("metadata") or {}).get("committed_fields") or {})
        _cf.update(_payload)                                  # the agent's ENTERED values are the committed_payload
        _prov = "operational_commitment" if _committed else "task_objective"
        # Decision-readiness signal: decision-relevant sources the agent skipped before deciding (oracle-blind,
        # policy-driven). Surfaced so recovery/reporting distinguishes a recoverable decision-readiness gap from a
        # genuine reasoning ceiling.
        _decision_made = bool(_dig_state(live, "agentActions.selectedDisposition") or _committed)
        _dr_gap = _decision_readiness_gap(task, live, _decision_made)
        return [CommittedGoal(
            goal_id="gui-0",
            goal_type=G_GUI,
            committed_fields=_cf,
            dedup_key="gui:%s" % ct,
            provenance=_prov,
            raw={"verify_spec": vs, "verify_all": conj,
                 "decisions": list(_decisions),
                 "agent_commit_refs": agent_commit_refs(trajectory, subs, (ctx or {}).get("obligation_state") or {}),
                 "challenge_type": ct, "target_keywords": kws[:12],
                 "operational_commitment": _committed, "committed_payload": _payload,
                 "committed_collections": _committed_collections,   # agent's own multi-value sets
                 "decision_readiness_gap": _dr_gap,
                 # only a disposition-CHOICE task lets the readiness gap PREEMPT/BLOCK completion (recovery
                 # never authors a disposition); submission tasks complete mechanically. See _DISPOSITION_TYPES.
                 "authoring_disposition": _is_disposition_task(task),   # decision-authoring, orthogonal to workflow-family
                 # The PUBLIC portal source that reveals each gap flag, so recovery can ACQUIRE the
                 # missing evidence read-only and hand control back to the agent (oracle-blind mapping).
                 "decision_readiness_sources": _decision_readiness_sources(_dr_gap),
                 # Inputs for a SUBMISSION_READY check (recorded in the goal; no gate in this workspace reads them). The prior-auth
                 # evidence set is the SAME public flags _decision_readiness_gap keys on, but exposed as a SUBMISSION precondition
                 # (NOT gated on authoring_disposition). Oracle-blind: portal-tracked flags only, no gold.
                 "family": _family(task),
                 "goal_needs_attachment": _goal_needs_attachment(task),
                 "submission_evidence_required": (
                     (["clickedDiagnosesTab", "clickedServicesTab", "clickedCoveragesTab", "readClinicalNote"]
                      + (["downloadedAuthLetter"] if _goal_needs_attachment(task) else []))
                     if _family(task) == "prior_auth" else []),
                 # Prior-auth READ-evidence acquire sources -- the public surfaces recovery may open READ-ONLY
                 # to flip clicked*Tab/readClinicalNote/downloadedAuthLetter (each reveals real
                 # coverage/clinical data = reading evidence, not a signal-check). Mapped to nav keywords + route
                 # via _decision_readiness_sources (oracle-blind, public portal naming only).
                 "acquire_read_sources": (_decision_readiness_sources(
                     ["clickedDiagnosesTab", "clickedServicesTab", "clickedCoveragesTab", "readClinicalNote"]
                     + (["downloadedAuthLetter"] if _goal_needs_attachment(task) else []))
                     if _family(task) == "prior_auth" else [])})]

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

def _dig_state(state, dotted):
    node = state
    for seg in [p for p in str(dotted or "").split(".") if p]:
        if isinstance(node, dict):
            node = node.get(seg)
        elif isinstance(node, list):
            try:
                node = node[int(seg)]
            except (ValueError, IndexError, TypeError):
                return None
        else:
            return None
    return node
