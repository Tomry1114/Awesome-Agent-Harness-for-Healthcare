"""evidence_coverage capability — claim-conditioned observational coverage (the ANSWER-substrate gate).

At before_final, it proves every PERCEPTUAL claim in the answer traces to an actually-executed observation
of its target. Deterministic-first (ledger observations vs claim targets); the judge is consulted only to
(a) decompose+classify the answer into typed claims and (b) settle margin claims (region looked at, attribute
unclear). Findings flow through the SAME RepairFinding + ledger lifecycle as the FORM/FHIR Scoped Repair; the
kernel affordance registry fills allowed_capabilities (the judge never names a tool). Re-run each before_final
attempt = its own delta validation: a claim drops out of the fresh finding set => RESOLVED; a previously
covered (protected) claim newly flagged => REGRESSION. Ablatable: MH_REPAIR in soft/select/full.

LAYER: deterministic observation coverage = INFRASTRUCTURE; claim decomposition / semantic support = AMPLIFICATION.
"""
import os
from ..capability import Capability
from .. import decision as D
from ..observation import Observation, region_observed, attribute_observed, coverage_findings
from ..repair import enforceable
from .. import affordance


def _enabled():
    return os.environ.get("MH_REPAIR", "hard") in ("soft", "select", "full")


class EvidenceCoverage(Capability):
    name = "evidence_coverage"

    def before_final(self, answer, ctx):
        # P0-A: claim-coverage runs ONLY when the answer IS the manifest-declared graded commit (perceptual
        # substrate). On form/FHIR substrates the chat answer is a non-commit terminal_response -> OFF.
        if not _enabled() or not ctx.judge_fn or not getattr(ctx, "final_is_commit", False):
            return None
        meta = (ctx.contract.meta or {}) if ctx.contract else {}
        obs = self._observations(ctx.ledger)
        if not answer:
            return None
        led = ctx.ledger
        task_id = str(meta.get("task_id") or "t")

        # ---- judge step 1: decompose + classify the answer (budget-gated) ----
        if not ctx.spend_semantic():
            return None
        from ..engines.semantic import decompose_claims, claim_semantic_support
        claims = decompose_claims(answer, meta.get("public_context"), ctx.judge_fn, task_id)
        if not claims:
            return None

        # ---- judge step 2 (margin only): region looked at but attribute not explicitly observed ----
        margin = [c for c in claims if c.claim_type == "perceptual" and c.region
                  and region_observed(c, obs) is not None and attribute_observed(c, obs) is None]
        support = {}
        if margin and ctx.spend_semantic():
            # pass the ACTUAL observed content (o.summary), not just metadata -- otherwise the judge is blind
            # and defaults to 'unsupported' -> false over-correction (verified on the perceptual substrate).
            support = claim_semantic_support(margin, [o.summary() for o in obs], ctx.judge_fn)

        all_fresh = coverage_findings(claims, obs, task_id, semantic_support=support)
        # ADMISSION GATE: enforce only DETERMINISTIC defects (unobserved_target = never looked); semantic
        # ones (unsupported_by_observation / untraceable) are ADVISORY -> never force a claim deletion.
        for f in all_fresh:
            if not enforceable(f):
                led.record_advisory(f.to_dict())
        fresh = [f for f in all_fresh if enforceable(f)]
        fresh_paths = {f.target_path for f in fresh}

        # ---- lifecycle: resolve delivered findings the agent fixed; detect regression on protected claims ----
        for fid, rec in list(led.repair_findings.items()):
            if rec.finding.rule_id != "evidence_coverage" or rec.status not in ("delivered", "attempted"):
                continue
            if rec.finding.target_path not in fresh_paths:
                led.resolve_finding(fid)                       # claim grounded or removed -> resolved
            else:
                led.mark_attempted(fid, {"flagged": True}, ctx.step)
        # regression: a claim we promised to PRESERVE is now itself flagged (its grounding was destroyed)
        for f in fresh:
            for fid, rec in led.repair_findings.items():
                if rec.finding.rule_id == "evidence_coverage" and f.target_path in (rec.finding.protected_paths or ()):
                    return self._emit([f], "repair_regression", ctx, meta, obs)

        # ---- deliver NEW findings (dedup; affordance fill for REACQUIRE) ----
        deliver = []
        for f in fresh:
            mode, _ = led.repair_decision(f, {"path": f.target_path})
            if mode == "suppress":
                continue
            f = self._with_affordance(f, meta)
            if mode == "new":
                led.open_finding(f, {"path": f.target_path}, ctx.step)
            led.mark_delivered(f.finding_id, {"path": f.target_path}, ctx.step)
            deliver.append(f)
        if deliver:
            return self._emit(deliver, "claim_unobserved", ctx, meta, obs)

        # No DETERMINISTIC defect. If there are UNCERTAIN semantic findings, route them to CANDIDATE mode:
        # keep the original answer A, ask for a revised B, and let the conservative A/B selector adopt B ONLY
        # if it is clearly better -- so external cannot amplify an uncertain finding into a wrong deletion.
        semantic = [f for f in all_fresh if not enforceable(f)]
        if semantic:
            return self._emit_candidate(semantic)
        return None

    def _emit_candidate(self, findings):
        crit = "; ".join(f.required_change for f in findings[:3])
        return self._decide(
            D.REVISE, rule_id="evidence_coverage", reason_code="candidate_review", deterministic=False,
            extra={"candidate": True, "critique": crit},
            reason="uncertain observational support -- candidate A/B review",
            feedback="An automated check is UNCERTAIN whether some claims are fully supported by your own "
                     "observations: %s. If you can ground or refine them, give a revised answer; otherwise "
                     "restate your current answer. Your ORIGINAL is kept unless the revision is clearly better "
                     "-- do NOT delete claims you are confident in." % crit)

    # ---- helpers ----------------------------------------------------------------------------------------
    def _observations(self, ledger):
        out = []
        for o in getattr(ledger, "observations", []) or []:
            out.append(Observation(observation_id=o.get("observation_id"), tool_capability=o.get("tool_capability"),
                                   subject=o.get("subject"), region=o.get("region"), modality=o.get("modality"),
                                   attributes_observed=tuple(o.get("attributes_observed") or []),
                                   result_status=o.get("result_status", "valid")))
        return out

    def _with_affordance(self, f, meta):
        if f.operation.value != "REACQUIRE_EVIDENCE":
            return f
        tools = affordance.select_tools(meta.get("available_tools"), region=f.metadata.get("region"),
                                        modality=f.metadata.get("modality"))
        if not tools:
            return f
        from ..repair import RepairFinding
        return RepairFinding(finding_id=f.finding_id, rule_id=f.rule_id, target_type=f.target_type,
                             target_path=f.target_path, defect_type=f.defect_type, operation=f.operation,
                             required_change=f.required_change, protected_paths=f.protected_paths,
                             preserve_requirements=f.preserve_requirements, evidence_refs=f.evidence_refs,
                             allowed_capabilities=tuple(tools), confidence=f.confidence, metadata=f.metadata)

    def _emit(self, findings, reason_code, ctx, meta, obs):
        rf = [f.to_dict() for f in findings]
        head = findings[0]
        tool_hint = (" Use one of: %s." % ", ".join(head.allowed_capabilities)) if head.allowed_capabilities else ""
        reason = ("evidence coverage (%s): %s at %s -- %s"
                  % (reason_code, head.defect_type, head.target_path, head.required_change))
        return self._decide(
            D.REVISE, rule_id="evidence_coverage", reason_code=head.defect_type, deterministic=False,
            missing_obligations=[f.required_change for f in findings], reason=reason,
            extra={"repair_findings": rf},
            feedback="A stated observation is not grounded in your actual tool observations: %s%s Either acquire "
                     "the observation, or remove/soften the claim. Do NOT invent findings; preserve grounded claims."
                     % (head.required_change, tool_hint))
