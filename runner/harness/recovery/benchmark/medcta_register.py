"""Bounded Clinical Recovery v3 - perceptual (MCTA) stack registration.

Additive registration hook: it does NOT edit registry.py (which is shared and must not be raced). Call
register(registry) to add the EvidenceAcquisitionWorkflow to an existing WorkflowRegistry, or build_registry()
/ build_stack() to construct a fully-wired perceptual (evidence-path) recovery stack.

The evidence path emits NO irreversible_commit and requires agent re-entry for answer B (see
workflows.evidence_acquisition + acceptance).

Python 3.8 compatible.
"""
from ..registry import WorkflowRegistry, RecoveryStack
from ..workflows.evidence_acquisition import EvidenceAcquisitionWorkflow
from ..benchmark.medcta import MedctaBenchmarkAdapter


def register(registry, workflow=None):
    """Register the evidence-acquisition workflow into an existing WorkflowRegistry (first-match priority)."""
    registry.register(workflow or EvidenceAcquisitionWorkflow())
    return registry


def build_registry(workflow=None):
    r = WorkflowRegistry()
    register(r, workflow=workflow)
    return r


def build_stack(env, judge_fn=None, substrate=None):
    """Construct a wired perceptual recovery stack (evidence path). `env` = a ToolSandboxEnv-style adapter."""
    from ..substrate.perceptual import PerceptualSubstrateAdapter
    return RecoveryStack(
        env_type="perceptual",
        substrate_adapter=substrate if substrate is not None else PerceptualSubstrateAdapter(env),
        workflow_registry=build_registry(),
        benchmark_adapter=MedctaBenchmarkAdapter(judge_fn=judge_fn),
        driver=env,
        notes=["evidence-acquisition + agent-reentry + non-regression acceptance",
               "no irreversible_commit; no MutationAuthorization; read-only acquire"])
