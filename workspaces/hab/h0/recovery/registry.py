"""Bounded Clinical Recovery - workflow registry and recovery-stack wiring (workspace strategy, HealthAdminBench default workspace).

WorkflowRegistry holds the WorkflowModules available for an episode and selects the one that matches a
committed goal (first match wins; registration order = priority).

build_registry() composes ONE WorkflowRegistry with the interactive-GUI workflow modules registered.

get_recovery_stack(env_type) returns the fully-wired (substrate_adapter, workflow_registry, benchmark_adapter) triple
for the 'gui' environment type, or None if no stack is registered for that env. The substrate is built UNWIRED (no
backend); entry.py gives it the invocation's backend. Only the benchmark ADAPTER layer (hab.py) carries task knowledge.
"""
import workspace_api as _W

_gui_completion = _W.load_relative(__file__, "gui_completion.py")
_gui_substrate = _W.load_relative(__file__, "gui_substrate.py")
_hab = _W.load_relative(__file__, "hab.py")


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


# ---- environment-type normalization ---------------------------------------------------------------
# The run-level environment type plus its alias. Substrate names never leak into the kernel;
# this map lives in the wiring module only.
_GUI = ("gui", "interactive_gui")


def _normalize_env(env_type):
    et = (env_type or "").strip().lower() if isinstance(env_type, str) else env_type
    if et in _GUI:
        return "gui"
    return None


# Registration order = match priority. HAB recovery targets ONLY the SEPARABLE mechanical steps that are independent
# of the clinical decision. Decision documentation is not a recovery target: on the live portal the "document in Epic"
# marker is set by the SAME submit that records the disposition, so it is not an independently-completable gap --
# recovering it would mean the harness choosing the disposition (forbidden). Each match_goal is keyed on goal_type.
_WORKFLOW_CLASSES = (
    _gui_completion.GenericGuiCompletionWorkflow,
)


def build_registry():
    """Compose ONE WorkflowRegistry with the interactive-GUI workflow modules registered (in priority order)."""
    reg = WorkflowRegistry()
    for cls in _WORKFLOW_CLASSES:
        reg.register(cls())
    return reg


def get_recovery_stack(env_type):
    """Return the wired (substrate_adapter, workflow_registry, benchmark_adapter) triple for env_type, or None for an
    environment with no stack. The substrate is built unwired (driver=None); entry.py wires the invocation's backend."""
    if _normalize_env(env_type) is None:
        return None
    return (_gui_substrate.GuiSubstrateAdapter(), build_registry(), _hab.HabBenchmarkAdapter())
