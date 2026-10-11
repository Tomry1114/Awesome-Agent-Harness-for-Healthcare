"""evidence_coverage capability — claim-conditioned observational coverage (the ANSWER-substrate gate).

At before_final, it checks that every PERCEPTUAL claim in the answer traces to an actually-executed observation
of its target. Deterministic-first (ledger observations vs claim targets); the judge is consulted only to
(a) decompose+classify the answer into typed claims and (b) settle margin claims (region looked at, attribute
unclear). Findings flow through the SAME RepairFinding + ledger lifecycle as the FORM/FHIR scoped repair; the
kernel affordance registry fills allowed_capabilities (the judge never names a tool). Re-running on each
before_final attempt is its own delta validation: a claim drops out of the fresh finding set => RESOLVED; a
previously covered (protected) claim newly flagged => REGRESSION.

LAYER (see HARNESS_DESIGN.md): deterministic observation coverage = INFRASTRUCTURE; claim decomposition / semantic
support = AMPLIFICATION (their findings are advisory, see the admission gate below).
"""
from workspace_api import Capability, decision as D
from workspace_api import EvidenceObservation as Observation, region_observed, attribute_observed, coverage_findings
from workspace_api import enforceable
from workspace_api import affordance


class EvidenceCoverage(Capability):
    name = "evidence_coverage"

    def before_final(self, answer, ctx):
        # claim-coverage runs ONLY when the answer IS the manifest-declared graded commit (perceptual
        # substrate). On form/FHIR substrates (this PhysicianBench workspace included) the chat answer is a
        # non-commit terminal_response, so the check returns None.
        if not ctx.has_judge() or not getattr(ctx, "final_is_commit", False):
            return None
        meta = (ctx.contract.meta or {}) if ctx.contract else {}
        obs = self._observations(ctx.ledger)
        if not answer:
            return None
        led = ctx.ledger
        task_id = str(meta.get("task_id") or "t")

        # ---- judge step 1: decompose + classify the answer (budget-gated) ----
        judge = ctx.semantic_check()
        if judge is None:
            return None
        from workspace_api import decompose_claims, claim_semantic_support
        claims = decompose_claims(answer, meta.get("public_context"), judge, task_id)
        if not claims:
            return None

        # ---- judge step 2 (margin only): region looked at but attribute not explicitly observed ----
        margin = [c for c in claims if c.claim_type == "perceptual" and c.region
                  and region_observed(c, obs) is not None and attribute_observed(c, obs) is None]
        support = {}
        margin_judge = ctx.semantic_check() if margin else None
        if margin_judge is not None:
            # pass the ACTUAL observed content (o.summary), not just metadata -- otherwise the judge is blind
            # and defaults to 'unsupported' -> false over-correction.
            support = claim_semantic_support(margin, [o.summary() for o in obs], margin_judge)

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
        # if it is clearly better -- so an uncertain finding cannot turn into a wrong deletion.
        semantic = [f for f in all_fresh if not enforceable(f)]
        if semantic:
            return self._emit_candidate(semantic)
        return None

    def _emit_candidate(self, findings):
        crit = "; ".join(f.required_change for f in findings[:3])
        return self._decide(
            D.REVISE, rule_id="evidence_coverage", reason_code="candidate_review", deterministic=False,
            # the agent renderer reads `reason` + `missing_obligations`, NOT feedback/extra.critique.
            # Put the SPECIFIC critique there so the revised candidate B targets the right claims (not a random
            # rewrite).
            missing_obligations=[f.required_change for f in findings[:3]],
            extra={"candidate": True, "critique": crit},
            reason=("A check is UNCERTAIN whether these claims are fully supported by your own observations: %s. "
                    "Ground or refine them, OR keep your answer -- your ORIGINAL is kept unless a revision is "
                    "clearly better; do NOT delete claims you are confident in." % crit),
            feedback="Uncertain observational support; ground the named claims or restate your answer.")

    # ---- helpers ----------------------------------------------------------------------------------------
    def _observations(self, ledger):
        out = []
        for o in getattr(ledger, "observations", []) or []:
            out.append(Observation(observation_id=o.get("observation_id"), tool_capability=o.get("tool_capability"),
                                   subject=o.get("subject"), region=o.get("region"), modality=o.get("modality"),
                                   attributes_observed=tuple(o.get("attributes_observed") or []),
                                   result_status=o.get("result_status", "valid"),
                                   content=o.get("content", "")))   # the support judge must see the tool output
        return out

    def _with_affordance(self, f, meta):
        if f.operation.value != "REACQUIRE_EVIDENCE":
            return f
        tools = affordance.select_tools(meta.get("available_tools"), region=f.metadata.get("region"),
                                        modality=f.metadata.get("modality"))
        if not tools:
            return f
        from workspace_api import RepairFinding
        return RepairFinding(finding_id=f.finding_id, rule_id=f.rule_id, target_type=f.target_type,
                             target_path=f.target_path, defect_type=f.defect_type, operation=f.operation,
                             required_change=f.required_change, protected_paths=f.protected_paths,
                             preserve_requirements=f.preserve_requirements, evidence_refs=f.evidence_refs,
                             allowed_capabilities=tuple(tools), confidence=f.confidence, metadata=f.metadata)

    def _emit(self, findings, reason_code, ctx, meta, obs):
        rf = [f.to_dict() for f in findings]
        head = findings[0]
        # a REACQUIRE finding with a ROUTABLE affordance must ACTUALLY ACQUIRE the missing observation
        # (D.ACQUIRE + next_action), not return a REVISE that merely tells the agent to "look again".
        # Only the ACQUIRE path makes the runner execute a perception tool and so produce new evidence.
        # Mirrors ActiveEvidence's next_action shape {capability, tool, args{region,attribute}, read_only}.
        if head.operation.value == "REACQUIRE_EVIDENCE" and head.allowed_capabilities:
            # BOTH ACQUIRE paths (ActiveEvidence + EvidenceCoverage) route through the SAME
            # capability-aware, attempt-history-aware planner -- not a fixed allowed_capabilities[0], which would
            # repeat the first tool. A None from the planner ("no productive re-look") -> fall through to the
            # REVISE below (soften/remove), NOT a repeat acquisition.
            _reg, _att = head.metadata.get("region"), head.metadata.get("attribute")
            _prior = list(getattr(ctx.ledger, "acquisition_attempts", []) or []) if ctx else []
            _dtool, _dattr = affordance.select_diversified_tool(meta.get("available_tools") if meta else None,
                                                                region=_reg, attribute=_att, prior_attempts=_prior)
            if _dtool:
                _na = {"capability": "inspect_region_attribute", "tool": _dtool,
                       "args": {"region": _reg, "attribute": _dattr or _att}, "read_only": True}
                return self._decide(
                    D.ACQUIRE, rule_id="evidence_coverage", reason_code=head.defect_type, deterministic=True,
                    missing_obligations=[f.required_change for f in findings],
                    reason=("evidence coverage (%s): %s at %s -- acquiring a targeted observation of %s (%s)"
                            % (reason_code, head.defect_type, head.target_path, _reg, _att)),
                    extra={"next_action": _na, "repair_findings": rf},
                    feedback=("A targeted observation of %s (%s) is being acquired; re-derive your answer from "
                              "ALL observations gathered so far." % (_reg, _att)))
            # else: no productive new observation available -> fall through to the REVISE path below.
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
