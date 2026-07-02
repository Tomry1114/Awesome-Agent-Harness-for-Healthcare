"""Bounded Clinical Recovery v3 - Appeal Submission workflow (CWC with an HONEST decision boundary).

An appeal is NOT a clean execution-closure like a prior-auth: filing an appeal requires net-new content -
a written appeal RATIONALE and a supporting-evidence ATTACHMENT. These are SEMANTIC/evidence dependencies
the ROOT AGENT must have authored/acquired; the harness must NEVER author them. So this workflow declares
those dependencies as required bindings and lets the kernel's Decision-Boundary gate stop the plan when
they are absent:

  - a missing supporting-evidence attachment (an evidence handle the harness would carry from acquired
    bound_evidence) -> the benchmark schema classes it as an evidence/operational dependency -> the gate
    yields BLOCKED_MISSING_EVIDENCE (a CORRECT refusal, reported as correctly_blocked - never FAILED);
  - a missing rationale (net-new clinical content) -> BLOCKED_NEEDS_DECISION.

For a strong agent that DID author the rationale and acquire the attachment (present in agent_commitment /
bound_evidence), the same plan proceeds: navigate -> search claim -> open dispute form -> fill rationale ->
attach the bound document -> one submit -> read-back verify.

Process knowledge only; substrate-agnostic; benchmark-name-free. Python 3.8 compatible.
"""
from ..contracts import (
    Plan, RecoveryStep, NAVIGATE, READ, STAGED_WRITE, IRREVERSIBLE_COMMIT,
)

GOAL_TYPES = ("submit_appeal", "appeal", "appeal_submission", "file_appeal")

# The two content dependencies an appeal cannot be authored without. The harness declares the NEED; the
# benchmark schema decides their binding class (evidence attachment -> operational/evidence handle;
# rationale -> semantic). Order matters: the evidence attachment is checked before the rationale so a
# weak agent that authored NEITHER blocks first on the missing evidence -> BLOCKED_MISSING_EVIDENCE.
_ATTACHMENT_ARG = "attachmentEvidenceRef"
_RATIONALE_ARG = "appealRationale"


def _dig(state_view, root, path):
    node = state_view
    for k in [root] + list(path or []):
        if isinstance(node, dict):
            node = node.get(k)
        elif isinstance(node, list) and isinstance(k, int):
            node = node[k] if -len(node) <= k < len(node) else None
        else:
            return None
    return node


class AppealSubmissionWorkflow(object):
    """Multi-step appeal; BLOCKS honestly when the rationale/attachment were never authored/acquired."""

    name = "appeal_submission"

    def match_goal(self, goal, ctx):
        return getattr(goal, "goal_type", "") in GOAL_TYPES

    def required_bindings(self, goal, ctx):
        ctx = ctx or {}
        # claim/locator args (bindable from authoritative_state) first, then the content dependencies.
        locators = list(ctx.get("appeal_locator_args", []))
        return locators + [_ATTACHMENT_ARG, _RATIONALE_ARG]

    def compile_plan(self, goal, ctx):
        ctx = ctx or {}
        steps = []
        nav = ctx.get("appeal_url")
        steps.append(RecoveryStep(
            kind=NAVIGATE, name="open_payer_portal",
            action={"op": "navigate", "url": nav} if nav else {"op": "navigate"}))
        # locate the denied claim (read/search) - a read-like step, no mutation.
        steps.append(RecoveryStep(
            kind=READ, name="search_claim",
            action={"op": "snapshot"},
            affordance_target=ctx.get("claim_search_target")))
        # open the dispute/appeal form.
        steps.append(RecoveryStep(
            kind=STAGED_WRITE, name="open_dispute_form",
            action={"op": "click"},
            affordance_target=ctx.get("dispute_form_target") or {"role": "button", "label": "Appeal"},
            manifest={"side_effect_scope": "navigation", "rollback_available": True,
                      "server_persisted": False}))
        # fill the appeal rationale (net-new SEMANTIC content the agent must have authored).
        steps.append(RecoveryStep(
            kind=STAGED_WRITE, name="fill_rationale",
            action={"op": "type", "arg": _RATIONALE_ARG},
            affordance_target=ctx.get("rationale_target") or {"role": "textbox", "label": "Rationale"},
            arg_specs=[_RATIONALE_ARG],
            manifest={"side_effect_scope": "form_field", "rollback_available": True,
                      "server_persisted": False}))
        # attach the acquired supporting-evidence document.
        steps.append(RecoveryStep(
            kind=STAGED_WRITE, name="attach_evidence",
            action={"op": "upload", "arg": _ATTACHMENT_ARG, "file_ref": "last"},
            affordance_target=ctx.get("attachment_target") or {"role": "input", "label": "Attachment"},
            arg_specs=[_ATTACHMENT_ARG],
            manifest={"side_effect_scope": "form_field", "rollback_available": True,
                      "server_persisted": False}))
        # exactly ONE irreversible commit.
        steps.append(RecoveryStep(
            kind=IRREVERSIBLE_COMMIT, name="submit_appeal",
            action={"op": "submit_appeal"},
            affordance_target=ctx.get("submit_target") or {"role": "button", "label": "Submit Appeal"},
            manifest={"side_effect_scope": "payer_submission", "rollback_available": False,
                      "server_persisted": True}))

        pc = {"paths": (getattr(goal, "raw", {}) or {}).get("verify_paths", [])}
        return Plan(steps=steps, required_bindings=self.required_bindings(goal, ctx),
                    expected_postcondition=pc)

    def verify_effect(self, goal, state_view):
        spec = (getattr(goal, "raw", {}) or {}).get("verify")
        if not isinstance(spec, dict):
            return None
        node = _dig(state_view or {}, spec.get("root"), spec.get("path"))
        if node is None:
            return None
        if spec.get("check", "truthy") == "nonempty":
            try:
                return len(node) >= int(spec.get("min_len", 1))
            except TypeError:
                return bool(node)
        return bool(node)
