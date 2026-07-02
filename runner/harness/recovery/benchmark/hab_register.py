"""Bounded Clinical Recovery v3 - HAB workflow registration (kept out of registry.py to avoid a race).

register(registry) installs the interactive-GUI administrative-portal workflow modules onto a shared
WorkflowRegistry, in priority order. Additive: it does NOT edit registry.py. Idempotent-friendly - it
skips a workflow class already present so repeated calls do not double-register.
"""
from ..workflows.prior_auth import PriorAuthorizationWorkflow
from ..workflows.appeal_submission import AppealSubmissionWorkflow
from ..workflows.decision_documentation import DecisionDocumentationWorkflow

# Registration order = match priority. Prior-auth (clean recoverable) and decision-documentation (landed
# disposition) come before the appeal (which honestly blocks for weak agents) so a goal is routed to the
# most specific matching process; each match_goal is keyed on goal_type so ordering only breaks ties.
_WORKFLOW_CLASSES = (
    PriorAuthorizationWorkflow,
    DecisionDocumentationWorkflow,
    AppealSubmissionWorkflow,
)


def register(registry):
    """Register the HAB GUI workflow modules on `registry`. Returns the registry."""
    existing = {type(m) for m in (registry.all() if hasattr(registry, "all") else [])}
    for cls in _WORKFLOW_CLASSES:
        if cls not in existing:
            registry.register(cls())
    return registry


def build_registry(registry_factory):
    """Convenience: build a fresh registry via `registry_factory()` and register the HAB workflows."""
    registry = registry_factory()
    return register(registry)
