"""Clinical Process Harness — the kernel and the fixed shell's capabilities.

Method in one line: COMPILE OBLIGATIONS, BIND EVIDENCE, VERIFY BEFORE COMMITMENT.

Public entry point: build_kernel(task, env_type, observed, ...) -> HarnessKernel. The harness has one mode,
enforce. The kernel sits between agent.act and env.call_tool and emits ALLOW / REVISE / BLOCK / ESCALATE (see
decision.py).
"""
import os

from .decision import ALLOW, REVISE, BLOCK, ESCALATE, HarnessDecision, combine   # noqa: F401
from .contract import ClinicalProcessContract                                    # noqa: F401
from .compiler import build_contract, CompilerInputs, ContractCompiler, LeakError  # noqa: F401
from .kernel import HarnessKernel, Effective                                     # noqa: F401
from .risk import classify_risk                                                  # noqa: F401

from .capabilities.scope_evidence import ScopeEvidenceBinding
from .capabilities.obligation_lifecycle import ObligationLifecycle
from .capabilities.verify_commit import VerifyAndCommit
from .capabilities.reference_integrity import ReferenceIntegrity
from .engines.policy import load_policy, PolicyError                              # noqa: F401

# The fixed shell's own capabilities. They run first, in this order, at every hook; the workspace's kernel
# middleware runs after them. Order is behaviour: resolve() breaks a tie
# by it, and an earlier capability can change the ledger or the payload a later one sees.
SHELL_CAPABILITIES = (("scope_evidence", ScopeEvidenceBinding), ("obligation_lifecycle", ObligationLifecycle),
                      ("reference_integrity", ReferenceIntegrity), ("verify_commit", VerifyAndCommit))


def compose_capabilities(env_type, workspace):
    """The kernel's capabilities for a run: the fixed shell's own, then the run's workspace's kernel middleware in
    manifest order. `workspace` is the run's (runner/run.py resolve_workspace); a run with no workspace (None) runs the
    shell's alone. A middleware that takes a shell capability's name, or repeats a name, is refused."""
    from workspace import Workspace, WorkspaceError
    caps = [cls() for _name, cls in SHELL_CAPABILITIES]
    if not workspace:
        return caps
    ws = Workspace(workspace)
    caps += [cls() for cls in ws.kernel_middleware()]
    names = [c.name for c in caps]
    repeated = sorted(set(n for n in names if names.count(n) > 1))
    if repeated:
        raise WorkspaceError("%s: capability names repeat (the shell's are %s): %s"
                             % (ws.path, [n for n, _cls in SHELL_CAPABILITIES], repeated))
    return caps


def build_kernel(task, env_type=None, observed=None, *, workspace,
                 budget=None, substrate=None, adapter=None, agent_model=None, tool_model=None, instrumentation=None):
    """Compile a contract (oracle-blind) and build the kernel. Composes the
    policy from the ENVIRONMENT ADAPTER (explicit `adapter` / env_type default) + its
    SUBSTRATE + CLINICAL modules — NO benchmark name is passed in or used. A contract-leak attempt
    (LeakError) is re-raised to the caller rather than building a contract that could carry an oracle — fail safe.

    `workspace` is required: the run's workspace path (runner/run.py resolve_workspace), or None for a run with no
    workspace. The kernel never picks one of its own."""
    policy = load_policy(adapter=adapter, substrate=substrate, env_type=env_type, workspace=workspace)
    # dynamic admission against the task's actual tools (ambiguous tool->rule mapping is fatal).
    _tools = [t.get("name") if isinstance(t, dict) else t for t in (task.get("available_tools") or [])]
    if _tools:
        from .engines.policy import admission_errors
        policy.setdefault("_errors", []).extend(admission_errors(policy.get("manifest"), _tools))
    # FAIL-CLOSED: an incomplete/invalid policy (missing module, dangling obligation) must not silently run
    # with rules dropped: the kernel refuses to build.
    if policy.get("_errors"):
        raise PolicyError("policy composition failed (enforce): %s" % (policy["_errors"],))
    try:
        contract = build_contract(task, env_type=env_type,
                                  capabilities=[t.get("name") if isinstance(t, dict) else t
                                                for t in (task.get("available_tools") or [])],
                                  observed=observed, policy=policy)
    except LeakError:
        raise
    caps = compose_capabilities(env_type, workspace)
    risk_of = (lambda a: classify_risk(a, contract, policy))
    judge_fn, judge_model = _build_judge()
    # Judge independence is RECORDED, never enforced: there is no refusal and no bypass switch. A judge that shares
    # the agent's or the tool backend's model is self-scoring: the run goes ahead and says
    # so, in the kernel record and in result.json run_config, so a report can state it.
    judge_independence = "unknown"
    if judge_model:
        _norm = lambda m: str(m or "").strip().lower().split("/")[-1]
        _agent = _norm(agent_model or os.environ.get("MH_API_MODEL") or os.environ.get("MH_OPENAI_MODEL", "gpt-5.5"))
        _tool = _norm(tool_model)
        _j = _norm(judge_model)
        if _j and (_j == _agent or (_tool and _j == _tool)):
            judge_independence = "shared_model_with_agent_or_tool"
        elif _j and (_agent or _tool):
            judge_independence = "independent"
    from .semantic_budget import manifest_section
    kernel = HarnessKernel(contract, caps, policy=policy, env_type=env_type, risk_of=risk_of,
                           budget=budget, judge_fn=judge_fn, judge_model=judge_model, instrumentation=instrumentation,
                           semantic_allocation=manifest_section(workspace))
    kernel.workspace = workspace
    kernel.judge_independence = judge_independence
    return kernel


def _build_judge():
    """The harness's semantic judge, bound from configuration (runner/judges.py role `harness`:
    MH_HARNESS_JUDGE_MODEL, MH_HARNESS_JUDGE_MAX_TOKENS). Routed through the gateway as a JUDGE call
    (judge=True -> MH_JUDGE_KEY), so the operator points it at an INDEPENDENT model (different from the agent
    model + tool backend).

    Required: a missing or invalid configuration raises JudgeConfigError rather than running the kernel with no
    semantic layer (run_task refuses such a run before its first agent step, and build_kernel fails closed if
    called directly). max_tokens comes from configuration."""
    import judges as _J
    spec = _J.spec("harness")
    import gateway
    model, max_tokens = spec["model"], spec["max_tokens"]

    def judge_fn(prompt):
        r = gateway.chat([{"role": "user", "content": prompt}], model=model, judge=True,
                         max_tokens=max_tokens, timeout=60)
        return (r or {}).get("content") if isinstance(r, dict) else r
    return judge_fn, model
