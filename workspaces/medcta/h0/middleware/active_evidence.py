"""ACQUIRE capability. Instead of rewriting the answer with the SAME evidence (a reactive repair cannot fix
capability errors), the harness DETECTS a recoverable gap and ACQUIRES NEW read-only evidence: it elicits the
agent's two most plausible interpretations + the single discriminating observable, and if that target was never
focus-observed, asks the kernel/runner to acquire it. The agent then re-reasons with strictly MORE information
than its first pass, closing a gap that needs neither a stronger model nor gold. Perception substrates only;
capped per task."""
from workspace_api import Capability, decision as D, affordance

def _tool_name(t):
    return (t.get("name") if isinstance(t, dict) else str(t)) or ""


class ActiveEvidence(Capability):
    # LAYER: AMPLIFICATION (information acquisition) -- adds capability via read-only evidence, never decides the answer
    name = "active_evidence"

    def before_action(self, action, ctx):
        # PROCESS-LEVEL SEARCH-QUALITY intervention (fires DURING exploration, not at the answer): if the agent
        # OCR-ed a SCIENTIFIC ENTITY label (gene/protein/biomarker/genotype/specimen) whose canonical identity
        # and biological context it never established -- not searched, or searched only GENERICALLY as image
        # text -- force ONE entity-focused search so the decisive context (which organ/tissue/model the figure
        # is about) reaches the agent BEFORE it commits a wrong synthesis. Read-only; perception-affordance
        # gated (no-op on GUI/FHIR); respects the acquire cap + semantic budget.
        if not ctx.has_judge():
            return None
        if (action or {}).get("type") != "tool_call":
            return None
        led, ws = ctx.ledger, ctx.workspace_state
        if ws.get("entity_search_done", False) or getattr(led, "acquire_count", 0) >= 3:
            return None
        meta = (ctx.contract.meta or {}) if ctx.contract else {}
        tools = meta.get("available_tools") or []
        if not affordance.select_tools(tools):
            return None                                    # no perception affordance (GUI/FHIR) -> not applicable
        obs_summ = []
        for o in getattr(led, "observations", []):
            if o.get("result_status") in ("valid", "ok") and (o.get("content") or o.get("region")):
                r, c = o.get("region"), (o.get("content") or "")
                obs_summ.append((("%s: %s" % (r, c)) if r else str(c))[:200])
        if len(obs_summ) < 2:
            return None
        # RE-CHECK as observations GROW -- a later OCR label must still be caught, never blocked by a one-shot
        # flag set before any label was read (order-independent). Skip only when no new observation arrived since
        # the last check, so a judge call is not spent redundantly.
        if len(obs_summ) <= ws.get("entity_check_obs_n", 0):
            return None
        judge = ctx.semantic_check()
        if judge is None:
            return None
        ws["entity_check_obs_n"] = len(obs_summ)
        from workspace_api import detect_unexploited_entity_label
        ent = detect_unexploited_entity_label(obs_summ, judge)
        if ent.get("query"):
            _stool = next((_tool_name(t) for t in (tools or [])
                           if any(k in _tool_name(t).lower() for k in ("googlesearch", "google", "search"))), None)
            if _stool:
                ws["entity_search_done"] = True   # fire the entity resolution once it is found
                na = {"capability": "resolve_entity", "tool": _stool, "args": {"query": ent["query"]},
                      "read_only": True}
                # agent_text: what the agent is told while the search runs in place of its action. The wording is the
                # shared record-read message; its "(%s)" slot holds a FHIR resourceType, which a search does not have, so
                # it reads "(None)".
                return self._decide(D.ACQUIRE, rule_id="active_evidence", reason_code="unexploited_entity_label",
                                    reason="resolve the entity label context: %s" % ent["query"][:60],
                                    extra={"next_action": na, "evidence_class": "entity_resolution",
                                           "agent_text": "A required record (None) was read for you; its result "
                                                         "is in your next observation. Use it, then proceed with "
                                                         "your action."})
        return None

    def before_final(self, answer, ctx):
        if not ctx.has_judge() or not getattr(ctx, "final_is_commit", False):
            return None
        meta = (ctx.contract.meta or {}) if ctx.contract else {}
        tools = meta.get("available_tools") or []
        sel0 = affordance.select_tools(tools)
        if not sel0:
            return None                                    # no perception affordance -> not applicable
        led, ws = ctx.ledger, ctx.workspace_state
        judge = ctx.semantic_check()
        if judge is None:
            return None
        from workspace_api import elicit_discriminator
        obs_summ = []
        for o in getattr(led, "observations", []):
            if o.get("result_status") in ("valid", "ok") and (o.get("content") or o.get("region")):
                r, c = o.get("region"), (o.get("content") or "")
                obs_summ.append((("%s: %s" % (r, c)) if r else str(c))[:200])
        # SYNTHESIS GROUNDING GATE -- runs BEFORE the acquisition cap. If the answer is NOT grounded in the
        # evidence the agent ALREADY gathered (it ignored decisive evidence -- e.g. its own search result -- or its
        # conclusion is contradicted by the evidence), the fix is RE-SYNTHESIS, not more acquisition. Point the agent
        # at the UNRECONCILED evidence (never the answer) and require it to re-synthesise from what it already has.
        # Oracle-blind; capped at 1 nudge/task; fires even when the acquisition cap is reached (re-synthesis needs no
        # new evidence). This closes the loop the entity/knowledge search opened: the process step adds information,
        # the synthesis gate confirms the information was actually used.
        if ws.get("resynth_rounds", 0) < 1 and len(obs_summ) >= 2:
            from workspace_api import synthesis_grounding_check
            _sg = synthesis_grounding_check(answer, obs_summ,
                                            meta.get("goal") or meta.get("public_context"), judge)
            if not _sg.get("grounded", True) and _sg.get("unreconciled"):
                ws["resynth_rounds"] = ws.get("resynth_rounds", 0) + 1
                return self._decide(D.REVISE, rule_id="active_evidence", reason_code="ungrounded_synthesis",
                                    reason="answer not grounded in gathered evidence",
                                    feedback=("Your answer does not reconcile evidence you ALREADY gathered: %s. "
                                              "Re-examine ALL of your gathered evidence and RE-SYNTHESISE your answer "
                                              "-- do not restate the same conclusion without accounting for this. You "
                                              "do NOT need new evidence; reason from what you already have."
                                              % str(_sg["unreconciled"])[:400]))
        # Stop on SUCCESSFUL acquisitions (landed validated evidence), not raw attempts -- so a failed / empty /
        # localization-failed read does not permanently halt acquisition; allow one fallback attempt.
        if getattr(led, "successful_acquisitions", 0) >= 2 or getattr(led, "acquire_count", 0) >= 3:
            return None                                    # per-task acquisition cap (anti tool-spam)
        # Route the adequacy audit's UNANSWERED required element into the acquisition target, so a multi-part
        # goal (identify tissue AND determine stain) whose stain is unaddressed acquires the STAIN, not another
        # morphology look. Sourced from the ledger hint the audit stamps; oracle-blind (goal-derived).
        _req_el = ws.get("pending_required_element")
        # EVIDENCE-CLASS ROUTING: a KNOWLEDGE gap (what a stain/marker looks like, rarity, a definition, the
        # clinical meaning of an observed feature) CANNOT be resolved by looking harder -- route it to GoogleSearch
        # with a query GROUNDED in the observed features. The audit->acquire loop chooses the evidence CLASS rather
        # than only re-looking, because perception-only acquisition cannot fix knowledge-dependent tasks.
        # Oracle-blind; perception gaps fall through below.
        if _req_el:
            from workspace_api import classify_evidence_need
            _need, _kq, _expr = classify_evidence_need(_req_el, answer, obs_summ, judge)
            def _pick(*keys):
                return next((_tool_name(t) for t in (tools or [])
                             if any(k in _tool_name(t).lower() for k in keys)), None)
            # verify_commit RE-STAMPS the pending element each round, so a failed acquisition retries -- but it would
            # repeat the SAME class (a failing search re-runs). Track classes already tried this task and SKIP a repeat,
            # so retry falls through to the next class / the visual default instead of looping on the same failure.
            # (Bounded anyway by the workspace's acquire-round allocation, manifest semantic_budget.rounds.)
            _tried = ws.get("ev_classes_tried")
            if _tried is None:
                _tried = set(); ws["ev_classes_tried"] = _tried
            if _need == "knowledge" and _kq and "knowledge" not in _tried:
                _tried.add("knowledge")
                _stool = _pick("googlesearch", "google", "search")
                if _stool:
                    ws["pending_required_element"] = None
                    na = {"capability": "retrieve_knowledge", "tool": _stool,
                          "args": {"query": _kq}, "read_only": True}
                    return self._decide(D.ACQUIRE, rule_id="active_evidence", reason_code="unresolved_knowledge",
                                        reason="retrieve external knowledge: %s" % _kq[:70],
                                        extra={"next_action": na, "required_element": _req_el,
                                               "evidence_class": "knowledge"})
            if _need == "text" and "text" not in _tried:
                _tried.add("text")
                _otool = _pick("ocr")
                if _otool:
                    ws["pending_required_element"] = None
                    na = {"capability": "read_text", "tool": _otool, "args": {"region_query": "whole image"},
                          "read_only": True}
                    return self._decide(D.ACQUIRE, rule_id="active_evidence", reason_code="unresolved_text",
                                        reason="read the printed labels/text (OCR)",
                                        extra={"next_action": na, "required_element": _req_el,
                                               "evidence_class": "text"})
            if _need == "numeric" and _expr and "numeric" not in _tried:
                _tried.add("numeric")
                _ctool = _pick("calculator", "calc")
                if _ctool:
                    ws["pending_required_element"] = None
                    na = {"capability": "calculate", "tool": _ctool, "args": {"expression": _expr},
                          "read_only": True}
                    return self._decide(D.ACQUIRE, rule_id="active_evidence", reason_code="unresolved_numeric",
                                        reason="compute the required value: %s" % _expr[:60],
                                        extra={"next_action": na, "required_element": _req_el,
                                               "evidence_class": "numeric"})
        # GROUNDED-IDENTIFICATION VERIFICATION -- fires BEFORE the perception re-look, for ANY confident
        # identification answer (tissue/organ/plane/diagnosis/stain), regardless of whether the audit flagged a
        # perception gap. Such an answer may be wrong, and RE-LOOKING is unreliable (the VLM re-reads the same
        # finding differently -> a wrong flip that the ambiguity guard then blocks, so nothing improves). Instead
        # cross-check the identification against EXTERNAL knowledge grounded in the observed features + OCR labels;
        # the agent re-reasons with a real reference = INDEPENDENT new evidence, so a supported flip is PERMITTED
        # (not an ambiguity re-read). Fires ONCE/task; an empty query (non-identification answer) falls through to
        # perception; read-only; the selector still gates. Knowledge-gap searches already returned above; this is
        # the identification-verification path for everything else.
        if not ws.get("grounded_verify_done", False):
            from workspace_api import grounded_verification_query
            _vq = grounded_verification_query(answer, meta.get("goal") or meta.get("public_context"),
                                              obs_summ, judge)
            if _vq:
                _stool = next((_tool_name(t) for t in (tools or [])
                               if any(k in _tool_name(t).lower() for k in ("googlesearch", "google", "search"))), None)
                if _stool:
                    ws["grounded_verify_done"] = True
                    na = {"capability": "verify_identification", "tool": _stool,
                          "args": {"query": _vq}, "read_only": True}
                    return self._decide(D.ACQUIRE, rule_id="active_evidence",
                                        reason_code="grounded_identification_check",
                                        reason="verify the identification vs external knowledge: %s" % _vq[:60],
                                        extra={"next_action": na, "evidence_class": "verification"})
        disc = elicit_discriminator(answer, meta.get("goal") or meta.get("public_context"),
                                    observations=obs_summ, judge_fn=judge, required_element=_req_el)
        if not disc:
            return None
        region, attribute = disc.get("region"), disc.get("attribute")
        # A GLOBAL visual requirement (stain identity, scan orientation, modality, label presence) has NO local
        # region. When the audit routed a required_element and the discriminator names a whole-image attribute with
        # no region, acquire it with a WHOLE-IMAGE tool; returning None on a missing region would drop every global
        # goal element (e.g. the stain).
        if not region:
            if attribute and _req_el:
                # A GLOBAL element still needs the ATTRIBUTE in the prompt. ImageDescription / OCR use FIXED prompts and
                # DROP the attribute, so routing there only re-describes the image and never resolves a specific global
                # element (the stain). Route to the ATTRIBUTE-AWARE region tool with a WHOLE-IMAGE region_query -- the only
                # tool that actually writes the required attribute into a targeted prompt. (region_query is honoured by
                # tools_medcta.run_tool -> RegionAttr.)
                _ratool = next((n for n in affordance.select_tools(tools)
                                if "region" in n.lower() or "attribute" in n.lower()), None)
                if _ratool:
                    ws["pending_required_element"] = None   # consume the requirement on dispatch
                    na = {"capability": "inspect_region_attribute", "tool": _ratool,
                          "args": {"region_query": "whole image", "attribute": attribute}, "read_only": True}
                    return self._decide(D.ACQUIRE, rule_id="active_evidence",
                                        reason_code="unresolved_required_element",
                                        reason="acquire the required global observation (%s)" % attribute,
                                        extra={"next_action": na, "discriminator": disc,
                                               "required_element": _req_el})
            return None                                    # certain / non-perceptual -> nothing to acquire
        from workspace_api import normalized_text as _n
        # Coverage is region+ATTRIBUTE (same rule as EvidenceCoverage), not region-only. A discriminator whose
        # region was looked at but whose specific ATTRIBUTE was never observed is STILL an uncovered requirement ->
        # acquire it. Otherwise the two gap detectors disagree (ActiveEvidence: region seen -> stop;
        # EvidenceCoverage: attribute missing -> reacquire) and same-region/new-attribute gaps are lost.
        _na_attr = _n(attribute)
        from workspace_api import observation_reliability, RELIABILITY_CONFIDENT
        def _covers(o):
            if not (o.get("result_status") == "valid" and o.get("region")):
                return False
            if _n(o.get("region")) != _n(region):
                return False
            if not _na_attr:
                return True                                # no specific attribute -> region coverage suffices
            # An UNCERTAIN / localization-failed observation MENTIONS the attribute but does not RELIABLY establish
            # it -> it must not count as covered, so an alternative acquisition can fire.
            if observation_reliability(o.get("content"), attribute,
                                       o.get("localization_resolved")) != RELIABILITY_CONFIDENT:
                return False
            _attrs = {_n(a) for a in (o.get("attributes") or o.get("attributes_observed") or [])}
            return (_na_attr in _attrs) or (_na_attr in _n(o.get("content") or ""))
        observed = any(_covers(o) for o in getattr(led, "observations", []))
        if observed:
            return None                                    # region AND attribute CONFIDENTLY observed
        # Pursue INFORMATION GAIN -- pick a tool/attribute that differs from prior acquisitions of this region,
        # so a second look is not the same tool+region+prompt (a stable VLM would just repeat its first error).
        # Prior attempts come from the ATTEMPT history (REQUESTED tool/region/attribute), NOT observations (whose
        # attributes_observed is []). Failed attempts are included so a repeat is blocked.
        _prior = list(getattr(led, "acquisition_attempts", []) or [])
        _dtool, _dattr = affordance.select_diversified_tool(tools, region=region, attribute=attribute,
                                                            prior_attempts=_prior)
        # The router returning None means "no productive re-look" -> do NOT fall back to the first tool (that
        # would re-issue the exact repeat the router just refused). No acquisition this round.
        if not _dtool:
            return None
        na = {"capability": "inspect_region_attribute", "tool": _dtool,
              "args": {"region": region, "attribute": _dattr or attribute}, "read_only": True}
        ws["pending_required_element"] = None     # consume the requirement on dispatch
        return self._decide(D.ACQUIRE, rule_id="active_evidence", reason_code="unresolved_discriminator",
                            reason="acquire the discriminating observation (%s / %s) before committing the answer"
                                   % (region, attribute),
                            extra={"next_action": na, "discriminator": disc})
