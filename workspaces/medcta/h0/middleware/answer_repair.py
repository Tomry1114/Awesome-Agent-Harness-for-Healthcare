"""answer_repair -- repair policy on the final answer's adequacy audit.

The fixed shell (runner/harness/capabilities/verify_commit.py) audits the final answer once, with one judge call paid
from the verification reserve, vetoes a localized hard defect itself (REVISE evidence_contradiction, must-resolve) and
records the audit read-only as ctx.final_audit (harness.capability.FinalAudit). Whether and how the agent repairs a
REPAIRABLE gap is strategy, and this file holds it. It makes no judge call: it reads only that record.

Only MedCTA's default workspace lists it. The audit runs only for a final whose commit point declares a claim-support
postcondition, and the one such postcondition is runner/policies/clinical/evidence_grounding.yaml, composed only by
runner/adapters/image_tool_sandbox.yaml.

Listed first under kernel_middleware, directly after the shell's capabilities, so it runs at the point of
verify_commit's audit: its REVISE breaks a before_final tie with the other kernel middleware, and the acquisition
target it routes (ledger.pending_required_element) is set before active_evidence reads it.

LAYER: AMPLIFICATION (selective epistemic repair).
"""
from workspace_api import Capability, decision as D

# confidence at which a repairable gap engages repair (see AnswerRepair._repair_min_conf)
REPAIR_MIN_CONF = 0.7


class AnswerRepair(Capability):
    name = "answer_repair"

    def _repair_min_conf(self):
        """SELECTIVE trigger (auto-receding amplification): a repairable gap engages repair only when the answer
        signals a real deficit -- it does not address the task, OR the auditor is at least this confident in the gap.
        A stronger model that confidently addresses tasks trips this less and the amplification layer recedes on its
        own. (Self-consistency sampling is the fuller, costlier variant.)"""
        return REPAIR_MIN_CONF

    def before_final(self, answer, ctx):
        audit = ctx.final_audit
        if audit is None or audit.hook_serial != ctx.hook_serial or audit.vetoed:
            return None                  # no audit of this final, or the shell vetoed its hard defect
        gap = audit.top_gap()
        _significant = gap is not None and (audit.addresses_task is False
                                            or (audit.confidence or 0) >= self._repair_min_conf())
        if not _significant:
            return None
        _cd = {"audit": audit.audit(), "side_effecting": audit.side_effecting}
        _cd["process_gap"] = True; _cd["gap_type"] = gap.get("type")
        _cd["critique"] = gap.get("critique") or gap.get("claim"); _cd["verification_flag"] = "unverified_grounding"
        # Route ONLY acquisition-suitable gaps (an unanswered element or a missed visual discriminator) into
        # the acquisition target. A form_mismatch / reasoning_gap / over_specific gap must NOT become a visual
        # acquisition target (asking a perception tool what region decides the OUTPUT FORMAT is nonsense) -> those
        # stay a plain REVISE.
        try:
            if gap.get("type") in ("unanswered", "missed_discriminator", "under_observed"):
                ctx.workspace_state["pending_required_element"] = _cd["critique"]
        except Exception:
            pass
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
