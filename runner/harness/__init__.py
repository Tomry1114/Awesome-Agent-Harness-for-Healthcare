"""Clinical Process Harness (Part 2) — kernel + Modules A/B/C, gated by MH_HARNESS_MODE.

Method in one line: COMPILE OBLIGATIONS, BIND EVIDENCE, VERIFY BEFORE COMMITMENT.

Public entry point: build_kernel(task, bench, env_type, mode, observed) -> HarnessKernel | None.
Returns None when mode == 'off' so the runner pays nothing. The kernel sits between agent.act and
env.call_tool and emits ALLOW / REVISE / BLOCK / ESCALATE (see decision.py).
"""
import os

from .decision import ALLOW, REVISE, BLOCK, ESCALATE, HarnessDecision, combine   # noqa: F401
from .contract import ClinicalProcessContract                                    # noqa: F401
from .compiler import build_contract, CompilerInputs, ContractCompiler, LeakError  # noqa: F401
from .kernel import HarnessKernel, MODES, Effective                              # noqa: F401
from .risk import classify_risk                                                  # noqa: F401

from .capabilities.scope_evidence import ScopeEvidenceBinding
from .capabilities.obligation_lifecycle import ObligationLifecycle
from .capabilities.verify_commit import VerifyAndCommit
from .engines.policy import load_policy

DEFAULT_CAPABILITIES = ("scope_evidence", "obligation_lifecycle", "verify_commit")


def _make_capabilities(names):
    reg = {"scope_evidence": ScopeEvidenceBinding, "obligation_lifecycle": ObligationLifecycle,
           "verify_commit": VerifyAndCommit}
    return [reg[n]() for n in names if n in reg]


def resolve_mode(explicit=None):
    m = (explicit or os.environ.get("MH_HARNESS_MODE", "off")).strip().lower()
    return m if m in MODES else "off"


def build_kernel(task, env_type=None, mode=None, observed=None, capabilities=None,
                 budget=None, substrate=None):
    """Compile a contract (oracle-blind) and build the kernel. Returns None for mode 'off'. Selects the
    SUBSTRATE policy pack from `substrate` or env_type — NO benchmark name is passed in or used.
    On a contract-leak attempt (LeakError) we DISABLE the harness for that task rather than risk an
    oracle — fail safe."""
    mode = resolve_mode(mode)
    if mode == "off":
        return None
    policy = load_policy(substrate=substrate, env_type=env_type)
    try:
        contract = build_contract(task, env_type=env_type,
                                  capabilities=[t.get("name") if isinstance(t, dict) else t
                                                for t in (task.get("available_tools") or [])],
                                  observed=observed, policy=policy)
    except LeakError:
        raise
    caps = _make_capabilities(capabilities or DEFAULT_CAPABILITIES)
    risk_of = (lambda a: classify_risk(a, contract, policy))
    judge_fn, judge_model = _build_judge()
    return HarnessKernel(contract, caps, mode=mode, policy=policy, env_type=env_type,
                         risk_of=risk_of, budget=budget, judge_fn=judge_fn, judge_model=judge_model)


def _build_judge():
    """The harness's semantic judge — opt-in via MH_HARNESS_JUDGE_MODEL. Routed through the gateway as a
    JUDGE call (judge=True -> MH_JUDGE_KEY), so the operator points it at an INDEPENDENT model (different
    from the agent brain + tool backend). Returns (None, None) when unset -> semantic checks fail-safe off."""
    model = os.environ.get("MH_HARNESS_JUDGE_MODEL")
    if not model:
        return None, None
    try:
        import gateway
    except Exception:
        return None, None

    def judge_fn(prompt):
        r = gateway.chat([{"role": "user", "content": prompt}], model=model, judge=True,
                         max_tokens=300, timeout=60)
        return (r or {}).get("content") if isinstance(r, dict) else r
    return judge_fn, model
