"""Bounded Clinical Recovery v3 - workflow registry + recovery-stack wiring (Phase-2 wired).

WorkflowRegistry holds the WorkflowModules available for an episode and selects the one that matches a
committed goal (first match wins; registration order = priority).

build_registry() composes ONE WorkflowRegistry with every stack's WorkflowModule(s) registered, by
calling each stack's own additive register(registry) hook (structured-record / interactive-GUI /
perceptual). It never edits another stack's modules and never hard-codes concrete wiring in the kernel.

get_recovery_stack(env_type, manifest) returns the fully-wired (substrate_adapter, workflow_registry,
benchmark_adapter) triple for an environment type ('fhir' / 'gui' / 'tool_sandbox'), or None if no stack
is registered for that env. The substrate is built UNWIRED (no live backend/driver); the feature-flag
entry point (entry.py) injects the live driver at run time. This keeps kernel/substrate/workflow files
free of benchmark proper names - only the benchmark ADAPTER layer (imported lazily below) carries task
knowledge.
"""
from dataclasses import dataclass, field
from typing import Any, List, Optional


class WorkflowRegistry:
    """Registers WorkflowModules and matches a CommittedGoal to one (substrate-agnostic)."""

    def __init__(self, modules=None):
        self._modules = list(modules or [])

    def register(self, module):
        self._modules.append(module)
        return module

    def match(self, goal, ctx):
        """First registered module whose match_goal(goal, ctx) is truthy; else None (-> NOT_APPLICABLE)."""
        for m in self._modules:
            try:
                if m.match_goal(goal, ctx):
                    return m
            except Exception:
                # a misbehaving module must not break routing; skip it
                continue
        return None

    def all(self):
        return list(self._modules)

    def __len__(self):
        return len(self._modules)


@dataclass
class RecoveryStack:
    """A fully-wired recovery stack for one environment type (kept for callers that want a named record)."""
    env_type: str
    substrate_adapter: Any = None
    workflow_registry: Optional[WorkflowRegistry] = None
    benchmark_adapter: Any = None
    driver: Any = None
    notes: List[str] = field(default_factory=list)


# ---- environment-type normalization ---------------------------------------------------------------
# The three run-level environment types plus their design-doc aliases. Substrate names never leak into
# the kernel; this map lives in the wiring module only.
_FHIR = ("fhir", "record", "structured_record")
_GUI = ("gui", "interactive_gui")
_PERCEPTUAL = ("tool_sandbox", "perceptual")


def _normalize_env(env_type):
    et = (env_type or "").strip().lower() if isinstance(env_type, str) else env_type
    if et in _FHIR:
        return "fhir"
    if et in _GUI:
        return "gui"
    if et in _PERCEPTUAL:
        return "tool_sandbox"
    return None


def build_registry():
    """Compose ONE WorkflowRegistry with all three stacks' WorkflowModules registered (additive).

    Each stack exposes its own register(registry) hook; this calls all three in priority order. Because
    every match_goal is keyed on goal_type, one combined registry routes correctly for any environment.
    """
    from .benchmark import pb_register, hab_register, medcta_register
    reg = WorkflowRegistry()
    pb_register.register(reg)
    hab_register.register(reg)
    medcta_register.register(reg)
    return reg


def get_recovery_stack(env_type, manifest=None):
    """Return the wired (substrate_adapter, workflow_registry, benchmark_adapter) triple for env_type,
    or None for an unknown environment. Substrate is built unwired (backend/driver=None); the entry
    point injects the live driver. `manifest` is accepted for signature-compat / future per-tool wiring.
    """
    et = _normalize_env(env_type)
    if et is None:
        return None
    registry = build_registry()
    if et == "fhir":
        from .substrate.fhir import FhirSubstrateAdapter
        from .benchmark.pb import PbBenchmarkAdapter
        return (FhirSubstrateAdapter(), registry, PbBenchmarkAdapter())
    if et == "gui":
        from .substrate.gui import GuiSubstrateAdapter
        from .benchmark.hab import HabBenchmarkAdapter
        return (GuiSubstrateAdapter(), registry, HabBenchmarkAdapter())
    if et == "tool_sandbox":
        from .substrate.perceptual import PerceptualSubstrateAdapter
        from .benchmark.medcta import MedctaBenchmarkAdapter
        return (PerceptualSubstrateAdapter(None), registry, MedctaBenchmarkAdapter())
    return None
