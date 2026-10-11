"""run_config: the experimental CONDITIONS of a run, resolved, in every artifact it produces.

Why. Unless the judge models, the agent model and the harness mode are recorded with the run, two runs made
under different conditions are indistinguishable afterwards and would pool. The harness judge is required
(runner/judges.py); the full judge binding of a run is result.json run_config.judges, this snapshot is the
kernel's own view of its process.

RESOLVED, not raw: `harness_judge_bound` is a boolean, not a missing string, so an unbound judge is
visible as a fact rather than as an absent field.

This module records and decides NOTHING: a run's conditions must be describable without changing what the
run is.
"""


def _env(name):
    import os
    v = os.environ.get(name)
    v = v.strip() if isinstance(v, str) else v
    return v or None


def snapshot():
    """The conditions of THIS process, as plain data."""
    import os

    out = {}

    # --- who judged ---------------------------------------------------------------------------------
    # harness_judge_model is the kernel's semantic judge; required (runner/judges.py), recorded with its token limit.
    out["harness_judge_model"] = _env("MH_HARNESS_JUDGE_MODEL")
    out["harness_judge_bound"] = bool(out["harness_judge_model"])
    try:
        import judges as _J
        out["harness_judge_max_tokens"] = _J.spec("harness")["max_tokens"]
    except Exception:
        out["harness_judge_max_tokens"] = None
    out["benchmark_judge_model"] = _env("LLM_JUDGE_MODEL")
    # recorded as set in the environment; no process judge runs in this harness
    out["process_judge_model"] = _env("MH_PROCESS_JUDGE_MODEL")
    out["agent_model"] = _env("MH_API_MODEL") or _env("MH_TARGET_MODEL")

    # --- how the run was bounded --------------------------------------------------------------------
    out["harness_mode"] = "enforce"   # the harness's one mode
    out["pid"] = os.getpid()
    return out


def render(cfg):
    """One block a reader can scan without knowing the field names."""
    if not isinstance(cfg, dict):
        return "(no run_config recorded)"
    L = ["run_config:"]
    for k in ("harness_judge_model", "benchmark_judge_model", "process_judge_model", "agent_model",
              "harness_mode"):
        if k in cfg:
            L.append("  %-22s %s" % (k + ":", cfg[k]))
    if not cfg.get("harness_judge_bound", True):
        L.append("  !! no harness judge bound -- no semantic checks run")
    return "\n".join(L)
