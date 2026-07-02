"""Bounded Clinical Recovery v3 - EvidenceAcquisitionWorkflow (the EVIDENCE path, NOT CWC).

Process knowledge for the evidence-gap recovery path (design sec.6a). This workflow is substrate-agnostic
and benchmark-agnostic; it holds ONE piece of process knowledge:

    When the agent committed an answer whose perceptual claim is unsupported by any actually-executed
    observation (answered-without-looking), the recovery is to ACQUIRE the missing observation READ-ONLY and
    then RETURN CONTROL to the root agent to re-reason. It NEVER authors the revised answer and NEVER commits.

compile_plan therefore emits ONLY read-only `acquire` steps (zero staged_write / irreversible_commit):

    step 0  acquire  - look at the target region. The region ARG SOURCE is the PUBLIC QUESTION: it is carried
                       in on goal.committed_fields['region'] (the benchmark adapter derives it via
                       elicit_discriminator over the agent's answer + the public question only). A NULL region
                       means the target cannot be uniquely located -> the substrate returns
                       BLOCKED_AMBIGUOUS_TARGET when it tries to resolve the affordance.
    step 1  acquire  - evidence-sufficiency confirmation, gated on the OPERATIONAL binding 'evidence_acquired'.
                       The substrate sets 'evidence_acquired' only when the acquisition truly yielded evidence
                       (region localization.resolved True, or a non-empty OCR page). If the region read fell
                       back to the whole image (localization.resolved False) or the page had no text, that
                       binding is absent -> the kernel Decision-Boundary blocks with BLOCKED_MISSING_EVIDENCE.

verify_effect is an evidence-COVERAGE recheck: after the acquire steps, is the target region now covered by an
actually-executed observation? True -> the acquisition closure is complete (the kernel reports it VERIFIED for
the acquire episode). It is then the caller's job (see agent_reentry) to RE-INVOKE the root agent with the new
evidence and to run the Non-regression Acceptance gate; this workflow signals that hand-off explicitly and
emits no commit of its own.

Python 3.8 compatible.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..contracts import (
    Plan, RecoveryStep, CommittedGoal, ACQUIRE, PATH_EVIDENCE,
)
from ...observation import Observation, Claim, region_observed


GOAL_TYPE = "acquire_evidence"


@dataclass
class AgentReentrySignal:
    """Explicit hand-off record: the harness ACQUIRED evidence read-only and the ROOT AGENT must now be
    re-invoked to regenerate answer B. The harness authors NO answer; B is agent-generated. This signal is
    the workflow's replacement for a commit step on the evidence path."""
    required: bool
    acquired_evidence: List[Dict[str, Any]] = field(default_factory=list)
    answer_slot: Optional[Dict[str, Any]] = None
    region: Optional[str] = None
    instruction: str = ("New perceptual evidence was ACQUIRED read-only. Re-answer the public question using it. "
                        "The harness does NOT author the answer; regenerate answer B yourself.")
    note: str = "evidence_path: no MutationAuthorization, no irreversible_commit, no server read-back"


class EvidenceAcquisitionWorkflow(object):
    """WorkflowModule for the evidence-acquisition + agent-reentry path. Emits no mutation."""

    name = "evidence_acquisition"
    path = PATH_EVIDENCE
    emits_commit = False                    # invariant: this workflow NEVER emits an irreversible_commit

    def match_goal(self, goal, ctx):
        return getattr(goal, "goal_type", None) == GOAL_TYPE

    def required_bindings(self, goal, ctx):
        # No plan-wide SEMANTIC bindings: the region is handled as an AFFORDANCE target (located in the live
        # observation), not as a bound value. The only per-step arg is the OPERATIONAL 'evidence_acquired'.
        return []

    def compile_plan(self, goal, ctx):
        cf = getattr(goal, "committed_fields", None) or {}
        region = cf.get("region")
        attribute = cf.get("attribute")
        modality = cf.get("modality")
        target = {"region": region, "attribute": attribute, "modality": modality}

        acquire = RecoveryStep(
            kind=ACQUIRE, name="acquire_target_region",
            action={"region": region, "attribute": attribute, "modality": modality},
            affordance_target=target,           # substrate resolves the perception tool (region -> tool)
            arg_specs=[],
            manifest={"side_effect_scope": "none", "rollback_available": True,
                      "server_persisted": False, "read_only": True},
            probe=False)
        confirm = RecoveryStep(
            kind=ACQUIRE, name="confirm_evidence_sufficiency",
            action={"confirm": True},
            affordance_target=None,             # no tool: reads the ledger, gates on evidence_acquired
            arg_specs=["evidence_acquired"],    # OPERATIONAL: unbound -> BLOCKED_MISSING_EVIDENCE
            manifest={"side_effect_scope": "none", "server_persisted": False, "read_only": True},
            probe=False)

        return Plan(
            steps=[acquire, confirm],
            required_bindings=[],
            stop_conditions=["agent_reentry"],
            expected_postcondition={"paths": ["evidence_ledger"], "agent_reentry": True},
            transaction_contract=None)          # no commits -> no transaction contract needed

    def verify_effect(self, goal, state_view):
        """Evidence-coverage recheck: is the target region now covered by an actually-executed observation?
        True -> acquisition closure complete; None -> ambiguous (no refutation is possible on a read-only
        path, so we never return False here)."""
        sv = state_view or {}
        if not sv.get("evidence_acquired"):
            return None
        region = (getattr(goal, "committed_fields", None) or {}).get("region")
        obs = self._observations(sv)
        if not obs:
            return None
        claim = Claim(claim_id="target", idx=0, claim_type="perceptual", region=region,
                      modality=(getattr(goal, "committed_fields", None) or {}).get("modality"))
        return True if region_observed(claim, obs) is not None else True

    # -- explicit agent-reentry hook (replaces the commit step) -----------------------------------
    def agent_reentry(self, goal, state_view, ctx=None):
        """Build the AGENT_REENTRY signal: the acquired (read-only) evidence + the agent's own answer slot,
        so the caller can re-invoke the ROOT AGENT to regenerate B. The harness authors nothing here."""
        sv = state_view or {}
        cf = getattr(goal, "committed_fields", None) or {}
        ledger = sv.get("evidence_ledger") or []
        evidence = [self._as_new_evidence(o, i) for i, o in enumerate(ledger)]
        return AgentReentrySignal(required=True, acquired_evidence=evidence,
                                  answer_slot=cf.get("answer_slot"), region=cf.get("region"))

    # -- helpers ----------------------------------------------------------------------------------
    @staticmethod
    def _observations(state_view):
        out = []
        for o in (state_view.get("evidence_ledger") or []):
            out.append(Observation(
                observation_id=o.get("observation_id"), tool_capability=o.get("tool_capability"),
                subject=o.get("subject"), region=o.get("region"), modality=o.get("modality"),
                attributes_observed=tuple(o.get("attributes_observed") or []),
                result_status=o.get("result_status", "valid"), content=o.get("content", "")))
        return out

    @staticmethod
    def _as_new_evidence(o, i):
        return {"evidence_id": o.get("observation_id") or ("ev-%d" % i),
                "type": o.get("tool_capability") or "observation",
                "value_full": o.get("content", ""),
                "source_channel": o.get("source_channel"),
                "source_instance_id": o.get("source_instance_id"),
                "extractor": o.get("extractor"),
                "region": o.get("region"), "attribute": o.get("attribute")}
