"""Policy engine — composes the effective policy from THREE separated layers (none of which the harness
core knows the names of):

  ENVIRONMENT ADAPTER  adapters/<name>.yaml          raw tool -> canonical semantics (manifest) + which
                                                     substrate this env is + which clinical modules apply.
                                                     The ONLY layer that names tools / observation fields.
  SUBSTRATE POLICY     policies/substrate/<name>.yaml generic invariants for the whole class of environment
                                                     (no tool names, no domain resources).
  CLINICAL MODULES     policies/clinical/<name>.yaml  reusable domain rules (medication safety, evidence
                                                     grounding, ...), activated per-adapter; scoped to the
                                                     resources/modalities they name.

`env_type` selects a DEFAULT adapter; a second dataset on the same substrate passes `adapter=` (or
MH_HARNESS_ADAPTER) — different tool names, SAME substrate + clinical modules, zero core change.
"""
import os, json

_RUNNER_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ADAPTER_DIR = os.path.join(_RUNNER_DIR, "adapters")
_SUBSTRATE_DIR = os.path.join(_RUNNER_DIR, "policies", "substrate")
_CLINICAL_DIR = os.path.join(_RUNNER_DIR, "policies", "clinical")

# env_type -> DEFAULT adapter (and the substrate it implies). env_type is a convenience selector, NOT an
# identity: override the adapter to run another dataset of the same class.
_DEFAULT_ADAPTER_BY_ENV = {"fhir": "hapi_fhir", "gui": "admin_portal", "tool_sandbox": "image_tool_sandbox"}
_SUBSTRATE_BY_ENV = {"fhir": "structured_record", "gui": "interactive_gui", "tool_sandbox": "perceptual"}


def _read(dirpath, name):
    if not name:
        return None
    base = os.path.join(dirpath, name)
    for ext, loader in ((".yaml", _load_yaml), (".yml", _load_yaml), (".json", _load_json)):
        p = base + ext
        if os.path.exists(p):
            try:
                return loader(p) or {}
            except Exception:
                return {}
    return None


def _load_yaml(path):
    try:
        import yaml
    except Exception:
        return None
    with open(path) as f:
        return yaml.safe_load(f)


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def substrate_of(env_type):
    return _SUBSTRATE_BY_ENV.get(env_type)


def load_policy(adapter=None, substrate=None, env_type=None):
    """Compose ADAPTER (manifest) + SUBSTRATE (generic invariants) + CLINICAL modules into one policy."""
    adapter_name = adapter or os.environ.get("MH_HARNESS_ADAPTER") or _DEFAULT_ADAPTER_BY_ENV.get(env_type)
    adoc = _read(_ADAPTER_DIR, adapter_name) or {}
    sub_name = adoc.get("substrate") or substrate or _SUBSTRATE_BY_ENV.get(env_type)
    sdoc = _read(_SUBSTRATE_DIR, sub_name) or {}
    clinical_names = list(adoc.get("clinical_modules") or [])
    cdocs = [(_read(_CLINICAL_DIR, c) or {}) for c in clinical_names]

    def _concat(key, docs):
        out = []
        for d in docs:
            out.extend(d.get(key) or [])
        return out

    return {
        "manifest": adoc.get("manifest", {}),
        "evidence_obligations": _concat("evidence_obligations", [sdoc] + cdocs),
        "workflow_obligations": _concat("workflow_obligations", [sdoc] + cdocs),
        # clinical (specific) commit points FIRST -> commit_point_for matches them before the substrate's
        # generic invariant (e.g. {create, MedicationRequest} wins over the bare {effect: irreversible}).
        "commit_points": _concat("commit_points", cdocs) + (sdoc.get("commit_points") or []),
        "_adapter": adapter_name, "_substrate": sub_name, "_clinical_modules": clinical_names,
    }
