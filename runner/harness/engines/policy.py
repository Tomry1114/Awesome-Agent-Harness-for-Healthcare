"""Policy engine — loads the public policy pack for a benchmark/env and merges it over `common`.

A policy pack (runner/policy_packs/<name>.yaml) declares the PUBLIC, oracle-free rules the compiler +
capabilities use: subject spec, subject_arg_keys, read/reversible/commit action lists, and the
obligation / commit-point templates. Packs are deliberately NOT per-task gold — they are environment
policy. YAML is used if available, else a .json sidecar, else the built-in COMMON defaults.
"""
import os, json

_PACK_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "policy_packs")

# SUBSTRATE packs — keyed by the environment SUBSTRATE, never by a benchmark name. The harness does not
# know which benchmark it is running; the environment adapter declares its substrate via env_type. A new
# dataset reuses an existing substrate pack (or adds one substrate) -> zero harness change.
_SUBSTRATE_BY_ENV = {
    "fhir": "structured_record",        # any structured record system (EHR / DB / API resources)
    "gui": "interactive_gui",           # any web / desktop GUI workflow
    "tool_sandbox": "perceptual_tool",  # any perceptual tool environment (image / pathology / video)
}

# minimal safe defaults if no pack file is present (keeps the kernel runnable out of the box).
COMMON_DEFAULTS = {
    "subject": {"type": None, "id_context_keys": []},
    "subject_arg_keys": [], "read_actions": [], "reversible_actions": [], "commit_actions": [],
    "evidence_obligations": [], "workflow_obligations": [], "commit_points": [],
    "final_risk": "R2",
}


def _read_pack_file(name):
    base = os.path.join(_PACK_DIR, name)
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


def _deep_merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def substrate_of(env_type):
    return _SUBSTRATE_BY_ENV.get(env_type)


def load_policy(substrate=None, env_type=None):
    """Resolve a SUBSTRATE pack (explicit substrate, else mapped from env_type), merge common <- pack.
    Takes no benchmark name."""
    name = substrate or _SUBSTRATE_BY_ENV.get(env_type)
    common = _read_pack_file("common")
    policy = _deep_merge(COMMON_DEFAULTS, common or {})
    if name:
        pack = _read_pack_file(name)
        if pack:
            policy = _deep_merge(policy, pack)
    policy["_substrate"] = name
    return policy
