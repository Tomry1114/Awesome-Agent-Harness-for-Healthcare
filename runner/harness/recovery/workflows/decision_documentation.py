"""Bounded Clinical Recovery v3 - Decision Documentation workflow (CWC / execution path).

The already-proven completion: once a disposition has LANDED (the appeal/auth decision was actually made
and persisted), the remaining deterministic act is to document that disposition in the EMR. This is a
1-commit plan, GATED on a landed disposition: the workflow requires the disposition value to be bound
from authoritative_state; if the disposition has not landed, the required binding cannot resolve and the
kernel blocks (BLOCKED_NEEDS_DECISION) - the harness will not document a decision that was never made.

Process knowledge only; substrate-agnostic; benchmark-name-free. Python 3.8 compatible.
"""
from ..contracts import (
    Plan, RecoveryStep, READ, IRREVERSIBLE_COMMIT,
)

GOAL_TYPES = ("document_decision", "documentation", "decision_documentation",
              "document_disposition")

# the landed-disposition gate: this must resolve from authoritative_state (semantic) or the plan blocks.
_DISPOSITION_ARG = "disposition"


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


class DecisionDocumentationWorkflow(object):
    """Document a landed disposition in the EMR (one commit), gated on the disposition being present."""

    name = "decision_documentation"

    def match_goal(self, goal, ctx):
        return getattr(goal, "goal_type", "") in GOAL_TYPES

    def required_bindings(self, goal, ctx):
        extra = list((ctx or {}).get("documentation_required", []))
        return [_DISPOSITION_ARG] + [a for a in extra if a != _DISPOSITION_ARG]

    def compile_plan(self, goal, ctx):
        ctx = ctx or {}
        steps = []
        # 1) read the current case (confirms we are on the right record; the disposition gate is enforced
        #    by required_bindings at the decision boundary before this runs).
        steps.append(RecoveryStep(
            kind=READ, name="read_case",
            action={"op": "snapshot"},
            probe=False))
        # 2) exactly ONE irreversible commit: write the documentation note.
        steps.append(RecoveryStep(
            kind=IRREVERSIBLE_COMMIT, name="document_decision",
            action={"op": "document_decision", "arg": _DISPOSITION_ARG},
            affordance_target=ctx.get("document_target") or {"role": "button", "label": "Document"},
            arg_specs=[_DISPOSITION_ARG],
            manifest={"side_effect_scope": "emr_note", "rollback_available": False,
                      "autosave_possible": False, "server_persisted": True}))

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
