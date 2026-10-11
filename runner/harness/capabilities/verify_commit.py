"""Module C — Risk-Adaptive Verify-and-Commit. Operates on ctx.risk + the commit point's POSTCONDITION
predicate. Records an EXPLICIT tri-state verification (ctx.verification = True/False/None) so an
unverifiable commit is never recorded as verified. No tool names / dataset checks.

  R3 (declared unjudgeable)         -> before_action ESCALATE
  R2 commit (effect=irreversible)   -> after_action: predicate -> verified True / violated False / unknown
  final answer (a commit)           -> before_final: adequacy audit over SELECTED evidence (selector-filtered);
                                       a localized hard defect is vetoed here, and the audit is recorded read-only
                                       as ctx.final_audit. Repairing a repairable gap is repair policy: workspace
                                       kernel middleware (e.g. MedCTA: workspaces/medcta/h0/middleware/answer_repair.py)
"""
from ..capability import Capability, FinalAudit
from .. import decision as D
from ..risk import at_least, R2, R3
from ..predicates import evaluate as eval_predicate
from ..authorization import has_verifiable_postcondition, MUTATION_TYPES


class VerifyAndCommit(Capability):
    # LAYER (see HARNESS_DESIGN.md): INFRA (process-output/commit integrity) + COMPENSATION (evidence-contradiction veto: the audit's hard defects). The AMPLIFICATION half, Selective Epistemic Repair of a repairable gap, is workspace kernel middleware.
    name = "verify_commit"

    def before_action(self, action, ctx):
        ctx.ledger.pending_authorization = None   # fresh each action; set below only if this action MATCHES an auth
        # FAIL-CLOSED: a tool action that no manifest rule (or default_action) maps is UNKNOWN to the
        # adapter. It must NOT default-allow as R0/none -> escalate so an unmapped/high-risk tool is
        # caught instead of slipping through. (The final answer is always mapped, so it is excluded.)
        if ctx.sem and not ctx.sem.mapped and ctx.sem.semantic_type != "answer":
            return self._decide(D.ESCALATE, rule_id="unmapped_action", reason_code="unmapped_action",
                                deterministic=True,
                                reason="tool action %r is not mapped by the substrate manifest "
                                       "(no action rule or default_action); cannot adjudicate its risk"
                                       % (getattr(ctx.sem, "capability", None),),
                                feedback="This tool is not declared in the substrate manifest, so its "
                                         "risk/semantics are unknown and it cannot be auto-verified; "
                                         "escalating (fail-closed).")
        # OPERATIONAL NON-DEGRADATION -- two guards before the existing commit controls.
        _is_mutation = bool(ctx.sem and getattr(ctx.sem, "semantic_type", None) in MUTATION_TYPES)
        # RULE 1: once the graded commit is verified-complete, any further mutation can only corrupt a correct
        # terminal state. Stop it; the agent should finalize.
        if _is_mutation and getattr(ctx.ledger, "terminal_locked", False):
            # terminal_locked stops the AGENT re-mutating a verified graded commit. But an AUTHORIZED recovery
            # mutation completing a DECLARED remaining subgoal -- e.g. the EHR documentation write-back AFTER a
            # verified payer submit (a different surface/subgoal, not a re-mutation of the submit) -- is not
            # corrupting; let the scoped single-use authorization (RULE 3 below) adjudicate it rather than a
            # blanket block, so the task's conjunction (submit AND documented) can actually complete.
            if not (action or {}).get("_mutation_authorization_id"):
                return self._decide(D.BLOCK, rule_id="post_verified_mutation", reason_code="terminal_locked",
                                    deterministic=True,
                                    reason="the graded commit is already verified-complete; further mutation can only corrupt it",
                                    feedback="The task's commit is already verified complete -- do NOT mutate further; finalize.")
        # RULE 3: PERMISSION ISOLATION. Semantic feedback never grants write permission. Under a mutation_hold,
        # a state mutation must match an explicit, scoped, single-use authorization (provenance: user_goal /
        # deterministic_gap / evidence_supported_plan). The check unit is "does THIS action match an unconsumed
        # authorization", NOT "was the last finding deterministic".
        if _is_mutation and getattr(ctx.ledger, "mutation_hold", False):
            _req_id = (action or {}).get("_mutation_authorization_id")
            if _req_id:   # EXACT-id: a recovery mutation must match the SPECIFIC authorization it was bound to
                from ..authorization import exact_scope_match, AUTH_AVAILABLE
                _byid = ctx.ledger.find_authorization_by_id(_req_id)
                _auth = _byid if (_byid is not None and _byid.status == AUTH_AVAILABLE
                                  and _byid.evidence_version == ctx.ledger.validated_evidence_version
                                  and exact_scope_match(_byid, ctx.sem, action)) else None
            else:
                _auth = ctx.ledger.find_matching_authorization(ctx.sem, action)
            if _auth is None and not (action or {}).get("_recovery") and ctx.contract and ctx.contract.matching_commit_points(ctx.sem):   # recovery mutations get NO user_goal pass-through -- they must match their own deterministic_gap auth
                # USER_GOAL pass-through: a mutation COVERED by a declared commit point is the task's ORIGINAL
                # intended commit (same operation/resource), not a semantic-feedback-induced off-plan write ->
                # auto-mint a user_goal authorization so normal task flow is not blocked by the hold.
                _auth = ctx.ledger.mint_authorization(
                    source="user_goal", allowed_semantic_type=getattr(ctx.sem, "semantic_type", None),
                    allowed_tool=(action or {}).get("tool"), allowed_effect=getattr(ctx.sem, "effect", None),
                    expected_postcondition={"covered_commit_point": True})
            if _auth is None:
                return self._decide(D.BLOCK, rule_id="semantic_feedback_no_write_auth",
                                    reason_code="mutation_requires_authorization", deterministic=True,
                                    reason="semantic feedback does not authorize a state mutation",
                                    feedback="A check raised an UNCERTAIN concern; that does NOT authorize changing the "
                                             "record. Re-read/inspect, or keep your current state.")
            if not has_verifiable_postcondition(_auth):
                return self._decide(D.ESCALATE, rule_id="authorized_mutation_no_postcondition",
                                    reason_code="unverifiable_commit", deterministic=True,
                                    reason="authorized mutation lacks a verifiable postcondition",
                                    feedback="This authorized mutation cannot be verified (no postcondition); escalating.")
            ctx.ledger.pending_authorization = _auth   # RECORD the match; do NOT consume -- the executor reserves it iff the combined winner is ALLOW, then dispatches immediately before the env call
            # authorized + scoped + verifiable -> fall through to the existing commit controls below.
        # PRE-COMMIT CONTROL: an IRREVERSIBLE commit that ALREADY SUCCEEDED must not be re-executed -- a
        # redundant re-submit/re-create risks corrupting the already-landed state (e.g. a second submit that
        # times out). Block it before it runs; the agent should finalize, not re-commit.
        if (ctx.sem and getattr(ctx.sem, "semantic_type", None) in ("create", "update", "submit")
                and getattr(ctx.sem, "effect", None) == "irreversible"
                and commit_identity(ctx.sem, ctx.ledger) in getattr(ctx.ledger, "completed_commits", set())):
            return self._decide(D.BLOCK, rule_id="redundant_commit", reason_code="redundant_commit",
                                deterministic=True,
                                reason="this irreversible commit already succeeded; re-executing risks corrupting it",
                                feedback="You already completed this commit successfully — do NOT re-submit/re-create "
                                         "it; finalize instead.")
        if ctx.risk == R3:
            return self._decide(D.ESCALATE, rule_id="unjudgeable_high_risk", reason_code="unjudgeable",
                                deterministic=True,
                                reason="action is high-risk and cannot be reliably adjudicated",
                                feedback="This action is high-risk and cannot be auto-verified; escalating.")
        # FAIL-CLOSED: an IRREVERSIBLE action that NO commit point covers has no obligations and no
        # postcondition -> it would execute unverified. Escalate instead of silently allowing it.
        if (ctx.sem and ctx.sem.effect == "irreversible" and ctx.contract
                and not ctx.contract.matching_commit_points(ctx.sem)):
            return self._decide(D.ESCALATE, rule_id="uncovered_irreversible_action",
                                reason_code="unverifiable_commit", deterministic=True,
                                reason="irreversible action is not covered by any commit point (no "
                                       "obligations/postcondition) — cannot be verified",
                                feedback="This irreversible action has no governing commit policy; escalating.")
        return None

    def after_action(self, action, result, before_state, after_state, ctx):
        if not at_least(ctx.risk or R2, R2):
            return None
        # the COMMIT TOOL CALL itself failed -> the commit did not land. Do NOT evaluate the postcondition
        # against a coincidental state change and call it verified; this is a failed (not verified) commit.
        if ctx.result_ok is False:
            if getattr(ctx, "result_status", None) == "unknown":
                # TRANSPORT ack != ENVIRONMENT commit state. A timeout/5xx is ambiguous about whether the
                # IRREVERSIBLE commit landed -> check the declared postconditions against the OBSERVED
                # after-state FIRST (the env may already reflect it, e.g. a playwright submit timed out but
                # localStorage changed). Only a genuinely UNOBSERVABLE effect stays unknown.
                _cp0 = ctx.contract.commit_point_for(ctx.sem) if (ctx.contract and ctx.sem) else None
                _posts0 = (_cp0 or {}).get("postconditions") or []
                if _posts0:
                    _vs = [eval_predicate(p, before_state, after_state, ctx.sem) for p in _posts0]
                    if _vs and all(v is True for v in _vs):
                        ctx.verification = True
                        return None                  # the commit LANDED despite the transport timeout -> ALLOW
                    if any(v is False for v in _vs):
                        ctx.verification = False
                        return self._decide(D.REVISE, rule_id="commit_execution_failed", reason_code="violated_commit",
                            deterministic=True, reason="transport failed AND the expected state change did not occur",
                            feedback="The commit did not take effect (transport error and no observable change) — retry.")
                ctx.verification = None
                return self._decide(D.RECONCILE, rule_id="commit_state_unknown", reason_code="unverifiable_commit",
                                    deterministic=True,
                                    reason="commit result is UNKNOWN (timeout/ambiguous; effect not observable)",
                                    feedback="This commit timed out and its effect is NOT observable. Use a READ/"
                                             "inspect tool to read back the authoritative state before any new write.")
            ctx.verification = False
            return self._decide(D.REVISE, rule_id="commit_execution_failed", reason_code="violated_commit",
                                deterministic=True, reason="the commit tool call failed (did not execute)",
                                feedback="The commit did not execute successfully — check the error and retry.")
        cp = ctx.contract.commit_point_for(ctx.sem) if (ctx.contract and ctx.sem) else None
        posts = (cp or {}).get("postconditions") or []
        # AND every merged postcondition: any False -> violated; else any None -> unknown; else verified.
        verdicts = [eval_predicate(p, before_state, after_state, ctx.sem) for p in posts] or [None]
        verdict = (False if any(v is False for v in verdicts)
                   else (None if any(v is None for v in verdicts) else True))
        ctx.verification = verdict      # explicit tri-state -> the kernel records this, not winner==ALLOW
        _bad = next((p for p, v in zip(posts, verdicts) if v is False), None)
        rid = (_bad.get("type") if isinstance(_bad, dict) else _bad) or "post_commit"
        if verdict is False:
            return self._decide(
                D.REVISE, rule_id=rid, reason_code="violated_commit", deterministic=True,
                reason="commit postcondition not satisfied (observable state unchanged / inconsistent)",
                feedback="The commit did not produce the expected state change — re-check and retry.")
        if verdict is None and posts:
            # A declared postcondition that could NOT be evaluated (state unobservable). Route to the
            # read-only RECONCILE state machine (server read-back) INSTEAD of ESCALATE -- with a real readback
            # affordance the commit usually DID land, and after-action ESCALATE kills the run. Only if the
            # reconciliation itself exhausts its budget without confirming does the run terminate.
            return self._decide(
                D.RECONCILE, rule_id="post_commit", reason_code="unverifiable_commit", deterministic=True,
                reason="commit postcondition not observable from current state; reconcile by server read-back",
                feedback="This commit's effect is not observable from the current state; reading it back from the server to confirm before finishing.")
        return None                     # True (verified) -> no block

    def _audit_path(self, answer, ctx, evid, task_goal, public_context, ptype, side_effecting, judge):
        """Layer-2: adequacy audit -> a localized HARD violation (must-resolve veto), else no shell decision.
        The audit is recorded read-only as ctx.final_audit (capability.FinalAudit) for the workspace's kernel
        middleware, which runs after the shell's capabilities in this hook call: whether and how to repair a
        REPAIRABLE gap is workspace repair policy (e.g. MedCTA: workspaces/medcta/h0/middleware/answer_repair.py).
        The judge is called once per audited answer, here, with the check before_final drew as the consumer
        final_verification."""
        from ..engines.semantic import audit_answer
        _gspec = (ctx.contract.meta or {}).get("goal_spec") if (ctx.contract and ctx.contract.meta) else None
        au = audit_answer(task_goal, public_context, answer, evid, judge_fn=judge, goal_spec=_gspec)
        _ex = {"audit": au.to_dict(), "side_effecting": side_effecting}
        hv = au.top_hard()
        ctx.final_audit = FinalAudit(ctx.hook_serial, ptype, side_effecting, au, vetoed=bool(hv))
        if hv:
            pr = ctx.ledger.pending_resolution
            if isinstance(pr, dict) and pr.get("violation_type") == "audit_hard":
                pr["attempts"] = pr.get("attempts", 1) + 1
            else:
                ctx.ledger.pending_resolution = {
                    "resolution_id": "res-au-%d" % ctx.step, "violation_type": "audit_hard",
                    "critical_claim": hv.get("claim"), "evidence_ids": hv.get("evidence_ids") or [],
                    "reason": hv.get("reason"), "confidence": au.confidence, "hard_type": hv.get("type"),
                    "created_at_step": ctx.step, "attempts": 1}
            ctx.verification = False
            _mr = dict(_ex); _mr["must_resolve"] = True; _mr["resolution"] = dict(ctx.ledger.pending_resolution)
            return self._decide(
                D.REVISE, rule_id=ptype, reason_code="evidence_contradiction", deterministic=False, extra=_mr,
                reason="answer has a hard defect (%s): %s" % (hv.get("type"), hv.get("reason")),
                feedback="Your answer has a hard defect (%s) on claim '%s' that the evidence refutes -- "
                         "remove, correct, or qualify it." % (hv.get("type"), hv.get("claim")))
        # No localized hard defect: nothing for the shell to veto. A repairable gap does not enter the shell's
        # verification (True when the auditor says the answer addresses the task, else unknown). A repair REVISE a
        # workspace middleware raises on the gap keeps this final from being accepted at this hook, so no commit is
        # recorded with that verification.
        ctx.ledger.pending_resolution = None
        ctx.verification = True if au.addresses_task else None
        return None

    def before_final(self, answer, ctx):
        # PROCESS-OUTPUT CONSISTENCY (must-resolve, deterministic): if an OPERATIONAL write was attempted and
        # its latest attempt did not verify (failed/unknown), the agent must NOT finalize over it as if
        # complete. Not an answer-quality judgement -- process<->output inconsistency.
        _uoc = ctx.ledger.unresolved_operational_commit()
        if _uoc is not None:
            pr = ctx.ledger.pending_resolution
            if isinstance(pr, dict) and pr.get("violation_type") == "process_output_inconsistency":
                pr["attempts"] = pr.get("attempts", 1) + 1
            else:
                ctx.ledger.pending_resolution = {
                    "resolution_id": "res-po-%d" % ctx.step, "violation_type": "process_output_inconsistency",
                    "critical_claim": _uoc.get("action"), "evidence_ids": [], "confidence": 1.0,
                    "reason": "operational write %r did not verify" % _uoc.get("action"),
                    "created_at_step": ctx.step, "attempts": 1}
            _mr = {"must_resolve": True, "resolution": dict(ctx.ledger.pending_resolution), "side_effecting": False}
            return self._decide(
                D.REVISE, rule_id="process_output_inconsistency", reason_code="process_output_inconsistency",
                deterministic=True, extra=_mr,
                reason="operational write %r did not complete; the answer must not report success" % _uoc.get("action"),
                feedback="Your action '%s' did NOT complete successfully (failed or unknown). Either complete "
                         "it successfully, or correct your answer to state it did not land -- do not report "
                         "success over a write that did not commit." % _uoc.get("action"))
        cp = ctx.contract.commit_point_for(ctx.sem) if (ctx.contract and ctx.sem) else None
        # if the commit's prerequisites are unmet, obligation_lifecycle owns the REVISE — don't add a
        # semantic verdict on top of an answer that isn't even grounded yet.
        if cp and ctx.ledger.pending_prerequisites(cp.get("requires", [])):
            ctx.verification = None
            return None
        # find a claim-support postcondition among ALL merged postconditions for this commit.
        posts = (cp or {}).get("postconditions") or []
        post = next((p for p in posts if "support" in str(p.get("type") if isinstance(p, dict) else p)), None)
        ptype = (post.get("type") if isinstance(post, dict) else post) if post else None
        if not ptype:
            return None
        # SIDE-EFFECTING flag: a terminal answer with effect 'none' (e.g. a read-only
        # "answer") has NO side effect and must NEVER be erased to nothing — run.py delivers it WITH a
        # verification flag instead of aborting. effect != none (perceptual/irreversible diagnosis,
        # create/submit) stays fail-closed. We only STAMP the flag here; run.py owns deliver-vs-abort.
        # OPERATIONAL side effect = a write that mutates external state (create/update/submit). A final
        # ANSWER is an EPISTEMIC commitment (no env mutation) even if its adapter tags an effect -> it is
        # delivered-with-flag on unverifiable, never erased; only operational writes fail-closed.
        _side_effecting = bool(ctx.sem and getattr(ctx.sem, "semantic_type", None) in ("create", "update", "submit")
                               and getattr(ctx.sem, "effect", "none") not in (None, "none"))
        judge = ctx.semantic_check("final_verification")   # from the judge pools the workspace gives the verification check
        if judge is None:
            ctx.ledger.add_unresolved_risk("semantic_claim_support",
                                           "claim<->evidence not verified (no judge / budget spent)")
            ctx.verification = None
            # the contract REQUIRES semantic support but none is available -> UNKNOWN, not a free pass.
            # ESCALATE (observe records, assist revises, enforce terminates safely). run.py aborts ONLY a
            # SIDE-EFFECTING commit; a no-side-effect answer is delivered WITH the unverified_grounding flag.
            return self._decide(
                D.ESCALATE, rule_id=ptype, reason_code="unverifiable_commit", deterministic=True,
                extra={"side_effecting": _side_effecting, "verification_flag": "unverified_grounding"},
                reason="final answer requires evidence-support verification but no judge/budget is available",
                feedback="This answer needs claim<->evidence verification, which is unavailable.")
        # filter the ledger to ONLY the evidence the postcondition selector allows (e.g. perception/image,
        # VALIDATED) — the judge never sees unrelated (e.g. external web) evidence.
        selector = (post.get("evidence_selector") if isinstance(post, dict) else None) or {}
        evid = [e for e in ctx.ledger.evidence if _selected(e, selector)]
        # task_goal/public_context come from contract.meta, filled by the ORACLE-BLIND compiler from the
        # PUBLIC task fields (ALLOWED_TASK_FIELDS, leak-checked) -- NEVER gold_answer/reference.
        _meta = (ctx.contract.meta if (ctx.contract and ctx.contract.meta) else {})
        return self._audit_path(answer, ctx, evid, _meta.get("goal"), _meta.get("public_context"), ptype,
                                _side_effecting, judge)


def commit_identity(sem, ledger):
    """Intent-level identity for an irreversible commit: not just (type, resource, subject) but ALSO the
    target object and a normalized payload hash -> a re-attempt of the SAME intent matches (redundant), while
    another LEGITIMATE intent of the same resource type (a second, different order) does NOT."""
    import hashlib, json
    _raw = getattr(sem, "raw", None) or {}
    if isinstance(_raw, dict):   # harness carrier keys (_mutation_authorization_id, ...) are not the commit's intent
        _raw = {k: v for k, v in _raw.items() if not str(k).startswith("_")}
    _pl = json.dumps(_raw, sort_keys=True, default=str)
    return (getattr(sem, "semantic_type", None), getattr(sem, "resource", None), ledger.subject_id(),
            getattr(sem, "target_entity", None), hashlib.sha1(_pl.encode("utf-8")).hexdigest()[:10])


def _selected(e, selector):
    """Evidence passes the postcondition's selector: VALIDATED + NOT foreign-subject + every declared
    source_class/modality. Foreign-subject evidence is never fed to the claim-support judge."""
    if e.get("scope_relation") == "foreign":
        return False
    if not selector:
        return True
    if e.get("status") != "VALIDATED":          # STRICT: only explicit VALIDATED evidence (must-resolve basis)
        return False
    if selector.get("source_class") and (e.get("source_class") or e.get("source_type")) != selector["source_class"]:
        return False
    if selector.get("modality") and e.get("modality") != selector["modality"]:
        return False
    return True
