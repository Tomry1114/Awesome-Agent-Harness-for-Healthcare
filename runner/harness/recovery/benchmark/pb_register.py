"""Bounded Clinical Recovery v3 - structured-record stack registration (race-free wiring).

registry.get_recovery_stack() is intentionally left a skeleton because MULTIPLE stack agents (structured-record
/ GUI / perceptual) would otherwise all edit that one function and race. Instead, each stack exposes its own
`register(registry)` + `build_stack(...)` here, and the caller composes them. This module does NOT edit
registry.py.

    register(registry)      - append this stack's WorkflowModule(s) to an existing WorkflowRegistry (additive;
                              never removes another stack's modules).
    build_stack(backend=..) - construct a fully-wired RecoveryStack(env_type='fhir') = FhirSubstrateAdapter +
                              WorkflowRegistry(CreateOrderWorkflow) + PbBenchmarkAdapter.
"""
from ..registry import WorkflowRegistry, RecoveryStack
from ..workflows.create_order import CreateOrderWorkflow
from ..substrate.fhir import FhirSubstrateAdapter
from .pb import PbBenchmarkAdapter

ENV_TYPE = "fhir"


def register(registry):
    """Append the structured-record workflow module(s) to `registry` (additive). Returns the registry."""
    if registry is None:
        registry = WorkflowRegistry()
    registry.register(CreateOrderWorkflow())
    return registry


def build_stack(backend=None, driver=None):
    """Fully-wired structured-record RecoveryStack. `backend` is the substrate backend (real driver wrapper or
    an in-memory test backend); None leaves substrate_adapter unwired for callers that inject later."""
    reg = WorkflowRegistry()
    register(reg)
    substrate = FhirSubstrateAdapter(backend) if backend is not None else FhirSubstrateAdapter()
    return RecoveryStack(
        env_type=ENV_TYPE,
        substrate_adapter=substrate,
        workflow_registry=reg,
        benchmark_adapter=PbBenchmarkAdapter(),
        driver=driver,
        notes=["BCR v3 structured-record stack (CreateOrderWorkflow); registry.py untouched (race-free)"])
