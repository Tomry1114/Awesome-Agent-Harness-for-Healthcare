"""Bounded Clinical Recovery v3 - HAB workflow registration (kept out of registry.py to avoid a race).

register(registry) installs the interactive-GUI administrative-portal workflow modules onto a shared
WorkflowRegistry, in priority order. Additive: it does NOT edit registry.py. Idempotent-friendly - it
skips a workflow class already present so repeated calls do not double-register.
"""
from ..workflows.prior_auth import PriorAuthorizationWorkflow
from ..workflows.appeal_submission import AppealSubmissionWorkflow

# Registration order = match priority. HAB recovery targets ONLY the two SEPARABLE mechanical steps that
# are independent of the clinical decision: prior-auth submission and appeal submission to the payer
# portal. Decision documentation was REMOVED: on the live portal the "document in Epic" marker is set by
# the SAME submit that records the disposition, so it is not an independently-completable gap -- recovering
# it would mean the harness choosing the disposition (forbidden). Each match_goal is keyed on goal_type.
_WORKFLOW_CLASSES = (
    PriorAuthorizationWorkflow,
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
