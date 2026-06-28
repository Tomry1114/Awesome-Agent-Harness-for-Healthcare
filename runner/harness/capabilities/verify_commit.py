"""Module C — Risk-Adaptive Verify-and-Commit. Operates on ctx.risk + the commit point's POSTCONDITION
predicate. Records an EXPLICIT tri-state verification (ctx.verification = True/False/None) so an
unverifiable commit is never recorded as verified. No tool names / dataset checks.

  R3 (declared unjudgeable)         -> before_action ESCALATE
  R2 commit (effect=irreversible)   -> after_action: predicate -> verified True / violated False / unknown
  final answer (a commit)           -> before_final: claim support over SELECTED evidence (selector-filtered)
"""
from ..capability import Capability
from .. import decision as D
from ..risk import at_least, R2, R3
from ..predicates import evaluate as eval_predicate
import os


class VerifyAndCommit(Capability):
    # LAYER (see HARNESS_DESIGN.md): INFRA (process-output/commit integrity) + COMPENSATION (claim-support contradiction veto) + AMPLIFICATION (Selective Epistemic Repair, MH_REPAIR-gated, provisional)
    name = "verify_commit"

    def before_action(self, action, ctx):
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
                return self._decide(D.ESCALATE, rule_id="commit_state_unknown", reason_code="unverifiable_commit",
                                    deterministic=True,
                                    reason="commit result is UNKNOWN (timeout/ambiguous; effect not observable)",
                                    feedback="This commit timed out and its effect is NOT observable. Read back the "
                                             "authoritative state to confirm whether it landed before any retry.")
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
            # a declared postcondition that could NOT be evaluated (state unobservable). UNKNOWN must not
            # silently pass in enforce -> ESCALATE (observe records, assist revises, enforce terminates safely).
            return self._decide(
                D.ESCALATE, rule_id="post_commit", reason_code="unverifiable_commit", deterministic=True,
                reason="commit postcondition could not be verified (state not observable)",
                feedback="This commit's effect cannot be verified from the available state; escalating.")
        return None                     # True (verified) -> no block

    def _repmode(self):
        """Answer-layer repair mode (ablation): none=safety-only, hard=must-resolve closure (default),
        soft/select/full add Layer-2 candidate repair. Instance attr overrides env (for tests)."""
        return getattr(self, "repair_mode", None) or os.environ.get("MH_REPAIR", "hard")

    def _repair_min_conf(self):
        """SELECTIVE trigger (auto-receding amplification): a repairable gap engages candidate repair only
        when the answer signals a real deficit -- it does not address the task, OR the auditor is at least
        this confident in the gap. A stronger model that confidently addresses tasks trips this less and the
        amplification layer recedes on its own. (Self-consistency sampling is the fuller, costlier variant.)"""
        try:
            return float(os.environ.get("MH_REPAIR_MIN_CONF", "0.7"))
        except Exception:
            return 0.7

    def _audit_path(self, answer, ctx, evid, task_goal, public_context, ptype, side_effecting):
        """Layer-2: adequacy audit -> a localized HARD violation (must-resolve) or a REPAIRABLE gap
        (candidate mode: keep A, ask for revised B; run.py runs the conservative A/B selection)."""
        from ..engines.semantic import audit_answer
        au = audit_answer(task_goal, public_context, answer, evid, judge_fn=ctx.judge_fn)
        _ex = {"audit": au.to_dict(), "side_effecting": side_effecting}
        hv = au.top_hard()
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
        gap = au.top_gap()
        _significant = gap is not None and (au.addresses_task is False or (au.confidence or 0) >= self._repair_min_conf())
        if _significant:
            ctx.ledger.pending_resolution = None
            ctx.verification = None
            _cd = dict(_ex); _cd["process_gap"] = True; _cd["gap_type"] = gap.get("type")
            _cd["critique"] = gap.get("critique") or gap.get("claim"); _cd["verification_flag"] = "unverified_grounding"
            # IN-PROCESS BEHAVIOR GUIDANCE (not a post-hoc answer verdict): point at the specific PROCESS gap so
            # the agent goes back and does the work with its OWN competence, then re-concludes -- capturing
            # (max-competence - first-pass-competence) WITHOUT reading gold. The harness does not say the
            # answer is wrong; it says the work that would support it is not yet done. The agent re-enters the
            # tool loop (REVISE -> continue); if it still cannot, the answer is delivered with a flag.
            return self._decide(
                D.REVISE, rule_id="process_gap", reason_code="process_gap", deterministic=False, extra=_cd,
                reason="process incomplete for this conclusion (%s): %s" % (gap.get("type"), gap.get("critique")),
                feedback="Before you finalize, your PROCESS is incomplete: %s. Go back and DO the work with "
                         "your tools -- examine the distinguishing feature / rule out the alternative / gather "
                         "the missing evidence -- then answer again. You are NOT being told the answer is "
                         "wrong; you are being told the supporting work is not yet done."
                         % (gap.get("critique") or gap.get("claim")))
        ctx.ledger.pending_resolution = None
        ctx.verification = True if au.addresses_task else None
        return None

    def before_final(self, answer, ctx):
        mode = self._repmode()
        # PROCESS-OUTPUT CONSISTENCY (must-resolve, deterministic): if an OPERATIONAL write was attempted and
        # its latest attempt did not verify (failed/unknown), the agent must NOT finalize over it as if
        # complete. Not an answer-quality judgement -- process<->output inconsistency.
        _uoc = ctx.ledger.unresolved_operational_commit() if mode != "none" else None
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
        # SIDE-EFFECTING flag (P0-5 / CONTRACT 5): a terminal answer with effect 'none' (e.g. a read-only
        # "answer") has NO side effect and must NEVER be erased to nothing — run.py delivers it WITH a
        # verification flag instead of aborting. effect != none (perceptual/irreversible diagnosis,
        # create/submit) stays fail-closed. We only STAMP the flag here; run.py owns deliver-vs-abort.
        # OPERATIONAL side effect = a write that mutates external state (create/update/submit). A final
        # ANSWER is an EPISTEMIC commitment (no env mutation) even if its adapter tags an effect -> it is
        # delivered-with-flag on unverifiable, never erased; only operational writes fail-closed.
        _side_effecting = bool(ctx.sem and getattr(ctx.sem, "semantic_type", None) in ("create", "update", "submit")
                               and getattr(ctx.sem, "effect", "none") not in (None, "none"))
        if not ctx.judge_fn or not ctx.spend_semantic():
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
        from ..engines.semantic import verify_claim_support, SUPPORTED, CONTRADICTED, INSUFFICIENT
        # task_goal/public_context come from contract.meta, filled by the ORACLE-BLIND compiler from the
        # PUBLIC task fields (ALLOWED_TASK_FIELDS, leak-checked) — NEVER gold_answer/reference. The judge
        # still only audits evidence-SUPPORT, not general correctness.
        _meta = (ctx.contract.meta if (ctx.contract and ctx.contract.meta) else {})
        _task_goal = _meta.get("goal")
        _public_context = _meta.get("public_context")
        if mode in ("soft", "select", "full"):
            return self._audit_path(answer, ctx, evid, _task_goal, _public_context, ptype, _side_effecting)
        v = verify_claim_support(_task_goal, _public_context, answer, evid, judge_fn=ctx.judge_fn)
        ctx.verification = v.supported
        _extra = {"semantic": v.to_dict(), "relation": v.relation, "side_effecting": _side_effecting}
        if v.supported is True or v.relation == SUPPORTED:
            ctx.ledger.pending_resolution = None        # any prior contradiction is resolved
            return None
        # CONTRADICTED with high confidence -> the ONLY HARD revise (answer conflicts with the evidence).
        # Keep reason_code 'unsupported_claim' so the governance violation metric stays counted.
        if v.relation == CONTRADICTED and (v.confidence or 0) >= 0.8:   # HIGH-confidence contradiction only
            # MUST-RESOLVE v1: a high-confidence contradiction is enforced as a real commit veto ONLY when it
            # is LOCALIZABLE -- it names the specific refuted claim AND cites VALIDATED evidence. The harness
            # does not independently solve the task; it verifies the agent\'s epistemic commitment is
            # consistent with its OWN validated evidence and refuses to deliver an unresolved confirmed
            # conflict. A high-confidence but NON-localizable contradiction is NOT must-resolve (cannot point
            # at a claim) -> it falls through to the low-confidence flag/escalate path below.
            if mode == "none":     # safety-only: revert to the pre-closure advisory REVISE (flagged delivery)
                return self._decide(
                    D.REVISE, rule_id=ptype, reason_code="unsupported_claim", deterministic=False, extra=_extra,
                    reason="final answer is contradicted by the selected evidence: %s" % v.reason,
                    feedback="Your answer conflicts with the evidence you gathered (%s)." % v.reason)
            if v.localizable():
                pr = ctx.ledger.pending_resolution
                if isinstance(pr, dict) and pr.get("violation_type") == "evidence_contradiction":
                    pr["attempts"] = pr.get("attempts", 1) + 1
                    pr["critical_claim"] = v.critical_claim; pr["evidence_ids"] = v.evidence_ids
                else:
                    ctx.ledger.pending_resolution = {
                        "resolution_id": "res-%d" % ctx.step, "violation_type": "evidence_contradiction",
                        "critical_claim": v.critical_claim, "evidence_ids": v.evidence_ids,
                        "reason": v.reason, "confidence": v.confidence, "created_at_step": ctx.step, "attempts": 1}
                _mr = dict(_extra); _mr["must_resolve"] = True
                _mr["resolution"] = dict(ctx.ledger.pending_resolution)
                return self._decide(
                    D.REVISE, rule_id=ptype, reason_code="evidence_contradiction", deterministic=False, extra=_mr,
                    reason="answer claim is refuted by validated evidence: %s" % (v.critical_claim or v.reason),
                    feedback="Your answer makes a claim the evidence REFUTES: '%s'. Remove, correct, or "
                             "explicitly qualify THAT claim so it no longer conflicts with the evidence; do "
                             "not re-submit it unchanged." % (v.critical_claim or v.reason))
            # non-localizable high-confidence contradiction -> cannot enforce must-resolve; fall through.
        # INSUFFICIENT (evidence under-covers the answer) -> a LIMITED revise; run.py permits at most one,
        # then DELIVERS the answer WITH the 'unverified_grounding' flag (the answer is not wrong, only not
        # fully grounded). Under-coverage must NOT be punished like a contradiction.
        if v.relation == INSUFFICIENT:
            ctx.ledger.pending_resolution = None
            ctx.ledger.add_unresolved_risk("semantic_claim_support",
                                           "selected evidence under-covers the answer (insufficient grounding)")
            _ins = dict(_extra); _ins["verification_flag"] = "unverified_grounding"
            return self._decide(
                D.REVISE, rule_id=ptype, reason_code="insufficient_grounding", deterministic=False,
                extra=_ins,
                reason="selected evidence does not fully cover the final answer: %s" % v.reason,
                feedback="The evidence you gathered does not fully support your answer (%s) — gather more "
                         "grounding or qualify the claim." % v.reason)
        # UNKNOWN / low-confidence (incl. a low-confidence contradiction) -> record unresolved risk + flag;
        # do NOT treat as wrong. run.py aborts ONLY a SIDE-EFFECTING commit; a no-side-effect answer is
        # delivered WITH the unresolved_risk flag.
        ctx.ledger.pending_resolution = None
        ctx.ledger.add_unresolved_risk("semantic_claim_support",
                                       "claim<->evidence support is low-confidence/unknown")
        _unk = dict(_extra); _unk["verification_flag"] = "unresolved_risk"
        return self._decide(
            D.ESCALATE, rule_id="semantic_low_confidence", reason_code="unjudgeable", deterministic=False,
            extra=_unk,
            reason="claim<->evidence support is low-confidence/unknown: %s" % v.reason)


def commit_identity(sem, ledger):
    """Intent-level identity for an irreversible commit: not just (type, resource, subject) but ALSO the
    target object and a normalized payload hash -> a re-attempt of the SAME intent matches (redundant), while
    another LEGITIMATE intent of the same resource type (a second, different order) does NOT."""
    import hashlib, json
    _pl = json.dumps(getattr(sem, "raw", None) or {}, sort_keys=True, default=str)
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
