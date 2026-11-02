"""Shared environment contract for benchmark adapters."""

class EnvironmentAdapter:
    """BENCHMARK ADAPTER CONTRACT. Every benchmark registers ONE env class in ENV_REGISTRY
    and the runner drives it through this fixed surface — adding a benchmark = register a class, not
    edit the loop's execution path:
        reset()                      -> initialize episode (load_task handled by load_task() upstream)
        available_tools()            -> list[str] tools offered to the agent this task
        call_tool(name, args)        -> CanonicalResult-shaped dict (execute one action)
        capabilities()               -> four-state manifest {implemented/available/authorized/healthy}
        teardown()                   -> release resources
        initial_observation()        -> optional first CanonicalObservation (envs that have one)
    Source-metric evaluation (evaluate_source_metric) and artifact collection live in the
    scoring layer (scoring.run_checkpoint dispatch), so the contract spans env + scorer. Conformance is
    enforced by tests/test_conformance.test_benchmark_adapter_contract over every ENV_REGISTRY class."""
    type = "base"
    # {tool: {"path": arg, "content": arg}} -- the tools whose EXECUTION writes the task's deliverable, and
    # which argument carries each part. Declared here because the adapter is what performs the write. None =
    # this substrate declares no written deliverable, so "is the deliverable absent" is UNKNOWN on it, never
    # guessed. Read by the runtime (kernel) through harness.substrate_catalog.deliverable_writes.
    DELIVERABLE_WRITES = None
    # The tools with which the agent ENDS the episode on this substrate, as the upstream benchmark defines them
    # (HealthAdminBench done(): harness/environment.py step sets done and the episode ends). run.py applies such a call
    # as the agent's final (the shell's one episode-end path) instead of executing it. () = the agent ends only by
    # answering.
    EPISODE_END_TOOLS = ()
    def __init__(self, task, **kw): self.task = task; self.cfg = (task.get("environment") or {}).get("config", {})
    def reset(self): ...
    def available_tools(self): return [t["name"] for t in self.task.get("available_tools", [])]

    def call_tool(self, name, args): raise NotImplementedError

    def reconcile_write(self, name, args, result):
        """ACTIVE READ-BACK (infrastructure): after a write, RE-READ the env to confirm the intended resource
        actually landed -- do not trust the write's own result envelope. Returns {confirmed: True|False|None,
        detail}. Default: this env cannot reconcile -> None (no-op; falls back to snapshot-based checks)."""
        return {"confirmed": None, "detail": "reconcile_unsupported"}

    def _healthy(self):
        """Best-effort runtime health of this env's backing service. Subclasses override with a real
        reachability check. A tool that is implemented+available but NOT healthy (service
        DOWN) means a failure is NOT the agent's fault and must not be scored as agent incompetence."""
        return True

    # Tools the runner has executable code for. Subclasses set their real surface; an empty
    # tuple means "trust available_tools" (every offered tool counts as implemented).
    IMPLEMENTED_TOOLS = ()

    def capabilities(self):
        """Four-state capability manifest per tool: implemented (runner has code) /
        available (offered to this task) / authorized (allowed) / healthy (backing service up).

        PURPOSE: a tool that is implemented+available+authorized but NOT healthy means the backing
        SERVICE is down, so a resulting tool failure is NOT the agent's fault and must not be scored
        as agent incompetence. Read `healthy` before attributing an EnvironmentError to the agent."""
        deny = set(getattr(self, "_denied_tools", []) or [])
        healthy = self._healthy()
        impl = set(self.IMPLEMENTED_TOOLS)
        avail = set(self.available_tools())
        names = impl | avail if impl else avail  # empty IMPLEMENTED_TOOLS -> offered == implemented
        out = {}
        for n in sorted(names):
            is_impl = (n in impl) if impl else True
            out[n] = {
                "implemented": is_impl,
                "available": n in avail,
                "authorized": n not in deny,
                # only a tool actually backed by THIS env carries the live health signal
                "healthy": bool(healthy) if is_impl else False,
            }
        return out
    def teardown(self): ...
