"""Per-DATASET adapters (dataset adaptation) + registry.

Each dataset's concrete adapter subclasses `harness.dataset_adapter.HarnessAdapter` and wires its runtime
behaviour to its OWN substrate modules (adapters/{pb,hab,medcta}/) + manifest yaml -- through whichever of the
contract's surfaces fits, NOT a forced-uniform shape:
  * HABAdapter    -> declares its runtime gates; its recovery, obligation nudge, grounding audit, read-acquire,
                     terminal gates and write verification are workspace middleware (workspaces/hab/h0/middleware);
  * PBAdapter     -> no overrides (its deliverable lifecycle is workspace middleware: workspaces/pb/h0/middleware);
  * MedctaAdapter -> identity only (its answer-candidate strategy is workspace middleware:
                     workspaces/medcta/h0/middleware/answer_candidate.py).
The harness contract itself (HarnessAdapter) lives under harness/. This module is dataset adaptation only.

Import is side-effect-free: every substrate dependency is imported lazily inside a method.
"""
from harness.dataset_adapter import HarnessAdapter

# env_type <-> the substrate each dataset runs on
FHIR = "fhir"
GUI = "gui"
TOOL_SANDBOX = "tool_sandbox"


class PBAdapter(HarnessAdapter):
    env_type = FHIR
    name = "PhysicianBench"
    manifest_yaml = "runner/adapters/hapi_fhir.yaml"


class HABAdapter(HarnessAdapter):
    env_type = GUI
    name = "HealthAdminBench"
    manifest_yaml = "runner/adapters/admin_portal.yaml"
    # the GUI-substrate runtime gates (enabled for the GUI substrate only)
    gates = frozenset({
        "obligation_gate", "collection_nudge",
        # harness-origin actions (the shell's harness action service) get the additive reserve max_steps // 6.
        # Whether recovery runs is decided by the workspace: the HAB h0 workspace lists middleware/recovery.py.
        "harness_action_reserve",
    })

class MedctaAdapter(HarnessAdapter):
    env_type = TOOL_SANDBOX
    name = "MedCTA"
    manifest_yaml = "runner/adapters/image_tool_sandbox.yaml"


# -- registry -------------------------------------------------------------------------------------
_BY_ENV = {FHIR: PBAdapter, GUI: HABAdapter, TOOL_SANDBOX: MedctaAdapter}
_BY_BENCH = {cls.name: cls for cls in (PBAdapter, HABAdapter, MedctaAdapter)}


def get_adapter(env_type=None, bench=None):
    """Resolve the dataset adapter. Explicit benchmark name wins (two datasets may share a substrate);
    otherwise fall back to env_type. Returns a fresh instance, or None if nothing matches."""
    cls = None
    if bench is not None and bench in _BY_BENCH:
        cls = _BY_BENCH[bench]
    elif env_type is not None and env_type in _BY_ENV:
        cls = _BY_ENV[env_type]
    return cls() if cls is not None else None


def all_adapters():
    """One instance of every registered dataset adapter (registration order = PB, HAB, MedCTA)."""
    return [PBAdapter(), HABAdapter(), MedctaAdapter()]
