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


class VerifyAndCommit(Capability):
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

    def before_final(self, answer, ctx):
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
        if not ctx.judge_fn or not ctx.spend_semantic():
            ctx.ledger.add_unresolved_risk("semantic_claim_support",
                                           "claim<->evidence not verified (no judge / budget spent)")
            ctx.verification = None
            # the contract REQUIRES semantic support but none is available -> UNKNOWN, not a free pass:
            # ESCALATE (observe records, assist revises, enforce terminates). If the operator wants enforce
            # on perceptual commits, they must supply a judge.
            return self._decide(
                D.ESCALATE, rule_id=ptype, reason_code="unverifiable_commit", deterministic=True,
                reason="final answer requires evidence-support verification but no judge/budget is available",
                feedback="This answer needs claim<->evidence verification, which is unavailable; escalating.")
        # filter the ledger to ONLY the evidence the postcondition selector allows (e.g. perception/image,
        # VALIDATED) — the judge never sees unrelated (e.g. external web) evidence.
        selector = (post.get("evidence_selector") if isinstance(post, dict) else None) or {}
        evid = [e for e in ctx.ledger.evidence if _selected(e, selector)]
        from ..engines.semantic import verify_claim_support
        v = verify_claim_support(answer, evid, judge_fn=ctx.judge_fn)
        ctx.verification = v.supported
        if v.supported is True:
            return None
        if v.supported is False and (v.confidence or 0) >= 0.5:
            return self._decide(
                D.REVISE, rule_id=ptype, reason_code="unsupported_claim", deterministic=False,
                extra={"semantic": v.to_dict()},
                reason="final answer not supported by the selected evidence: %s" % v.reason,
                feedback="Your answer is not supported by the evidence you gathered (%s) — re-examine "
                         "before answering." % v.reason)
        return self._decide(
            D.ESCALATE, rule_id="semantic_low_confidence", reason_code="unjudgeable", deterministic=False,
            extra={"semantic": v.to_dict()},
            reason="claim<->evidence support is low-confidence/unknown: %s" % v.reason)


def _selected(e, selector):
    """Evidence passes the postcondition's selector: VALIDATED + NOT foreign-subject + every declared
    source_class/modality. Foreign-subject evidence is never fed to the claim-support judge."""
    if e.get("scope_relation") == "foreign":
        return False
    if not selector:
        return True
    if e.get("status") not in (None, "VALIDATED"):
        return False
    if selector.get("source_class") and (e.get("source_class") or e.get("source_type")) != selector["source_class"]:
        return False
    if selector.get("modality") and e.get("modality") != selector["modality"]:
        return False
    return True
