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


def build_kernel(task, bench=None, env_type=None, mode=None, observed=None, capabilities=None,
                 budget=None):
    """Compile a contract (oracle-blind) and build the kernel. Returns None for mode 'off'.
    On a contract-leak attempt (LeakError) we DISABLE the harness for that task rather than risk an
    oracle — fail safe, and surface the error in the kernel-less path."""
    mode = resolve_mode(mode)
    if mode == "off":
        return None
    policy = load_policy(bench=bench, env_type=env_type)
    try:
        contract = build_contract(task, env_type=env_type,
                                  capabilities=[t.get("name") if isinstance(t, dict) else t
                                                for t in (task.get("available_tools") or [])],
                                  observed=observed, policy=policy)
    except LeakError:
        raise
    caps = _make_capabilities(capabilities or DEFAULT_CAPABILITIES)
    risk_of = (lambda a: classify_risk(a, contract, policy))
    return HarnessKernel(contract, caps, mode=mode, policy=policy, env_type=env_type,
                         risk_of=risk_of, budget=budget)
