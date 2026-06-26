"""Clinical Process Contract Compiler — with a HARD benchmark-leak firewall.

The compiler may read ONLY:
    goal · context · environment type · available capabilities · info the agent has already observed ·
    the public policy pack.
It may NEVER read: gold answer · reference trajectory · checkpoint results · expected tool sequence ·
dimension scores · native outcome. This is enforced in CODE (a whitelisted CompilerInputs object +
a forbidden-key guard), not merely asserted in the paper — so a contract can't be a disguised oracle.
"""
from .contract import ClinicalProcessContract

# keys that, if present in any structure handed to the compiler, indicate an oracle leak.
FORBIDDEN_KEYS = frozenset({
    "checkpoints", "reference", "reference_trajectory", "reference_traj", "gold", "gold_answer",
    "answer_key", "expected_tools", "expected_tool_calls", "sufficient_tools", "tool_chain",
    "dimension_scores", "native_outcome", "outcome", "success", "whitelist_ref", "hidden_reference",
    "expected_subject_answer", "gacc", "pi", "U",
})
# task fields the compiler is ALLOWED to see.
ALLOWED_TASK_FIELDS = ("task_id", "goal", "context", "environment", "available_tools", "source_benchmark")


class LeakError(Exception):
    pass


def _assert_no_leak(obj, path="inputs"):
    """Recursively reject any forbidden oracle key. Cheap, runs once per task at compile time."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in FORBIDDEN_KEYS:
                raise LeakError("benchmark leak: forbidden key %r at %s" % (k, path))
            _assert_no_leak(v, path + "." + str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _assert_no_leak(v, "%s[%d]" % (path, i))


class CompilerInputs:
    """The ONLY view of a task the compiler receives. Constructed by whitelisting task fields; the
    constructor raises LeakError if any forbidden key slips into the allowed subset."""

    __slots__ = ("task_id", "goal", "context", "env_type", "capabilities", "observed", "policy",
                 "source_benchmark")

    def __init__(self, task, env_type=None, capabilities=None, observed=None, policy=None):
        safe = {k: task.get(k) for k in ALLOWED_TASK_FIELDS if k in task}
        _assert_no_leak(safe, "task")
        _assert_no_leak(observed or [], "observed")
        _assert_no_leak(policy or {}, "policy")
        self.task_id = safe.get("task_id")
        self.goal = safe.get("goal")
        self.context = safe.get("context") or {}
        self.env_type = env_type or ((safe.get("environment") or {}).get("type"))
        self.capabilities = list(capabilities or [])
        self.observed = list(observed or [])     # info the agent has already seen (events/values)
        self.policy = dict(policy or {})          # public policy pack
        self.source_benchmark = safe.get("source_benchmark")


class ContractCompiler:
    """Compiles a CompilerInputs into a ClinicalProcessContract. P0: a deterministic, policy-pack-driven
    template compiler — it instantiates the subject + obligation/commit templates the policy pack
    declares for this env type and resolves the subject from task-visible context. Per-dataset semantic
    compilation (richer obligation graphs, evidence binding rules) is layered in P1–P3 via policy packs;
    the COMPILER code itself stays oracle-blind."""

    def compile(self, inputs):
        if not isinstance(inputs, CompilerInputs):
            raise TypeError("ContractCompiler.compile requires a CompilerInputs (leak firewall)")
        policy = inputs.policy or {}
        subject = self._resolve_subject(inputs, policy)
        ev_obs = list(policy.get("evidence_obligations", []))
        wf_obs = list(policy.get("workflow_obligations", []))
        commits = list(policy.get("commit_points", []))
        return ClinicalProcessContract(
            contract_id="%s-%s" % (inputs.source_benchmark or inputs.env_type or "task", inputs.task_id),
            subject=subject, evidence_obligations=ev_obs, workflow_obligations=wf_obs,
            commit_points=commits,
            meta={"env_type": inputs.env_type, "compiled_from": "policy_pack",
                  "source_benchmark": inputs.source_benchmark})

    def _resolve_subject(self, inputs, policy):
        """Subject = the operation target, read from task-visible context via the policy pack's
        `subject` spec (type + which context key holds the id). Oracle-blind: only reads context."""
        spec = policy.get("subject") or {}
        stype = spec.get("type")
        ctx = inputs.context or {}
        sid = None
        for key in (spec.get("id_context_keys") or []):
            if ctx.get(key):
                sid = ctx.get(key); break
        if sid is None and isinstance(inputs.observed, list):
            # fall back to the first explicitly-assigned subject the agent has already observed
            for ev in inputs.observed:
                if isinstance(ev, dict) and ev.get("assigned_subject"):
                    sid = ev["assigned_subject"]; break
        return {"type": stype, "id": sid} if (stype or sid) else None


def build_contract(task, env_type=None, capabilities=None, observed=None, policy=None):
    """Convenience: whitelist -> compile, in one call. Raises LeakError on any oracle key."""
    inputs = CompilerInputs(task, env_type=env_type, capabilities=capabilities,
                            observed=observed, policy=policy)
    return ContractCompiler().compile(inputs)
