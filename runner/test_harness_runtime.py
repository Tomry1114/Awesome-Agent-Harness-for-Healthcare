"""Runtime-integrity conformance for the Clinical Process Harness (fail-closed + honest status).

Guards the hardening done this session:
  - a crashing capability hook ESCALATEs under enforce (fail closed) but stays effective-ALLOW under
    observe (record-only), and is recorded as a capability error;
  - audit() reports status 'degraded' once any capability error has occurred (else 'active');
  - audit()['n_semantic_checks'] increments after a semantic check is actually spent (the previously
    dead counter is now wired to ctx.semantic_remaining).

No model calls; deterministic. Run: python3 runner/test_harness_runtime.py — expects all PASS.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness as H
from harness import decision as D
from harness.kernel import HarnessKernel
from harness.compiler import build_contract
from harness.capabilities.scope_evidence import ScopeEvidenceBinding
from harness.capabilities.obligation_lifecycle import ObligationLifecycle
from harness.capabilities.verify_commit import VerifyAndCommit

# reuse the synthetic substrate policy from test_harness (oracle-free; tool names only in the manifest).
POLICY = {
    "manifest": {
        "subject": {"type": "patient", "id_context_keys": ["patient_ref"], "from_args": ["patient"]},
        "actions": [
            {"match": {"tool": "fhir_search"}, "semantic_type": "read", "effect": "none",
             "produces_evidence": {"source_class": "record", "resource_from_args": ["resource_type"]}},
            {"match": {"tool": "create_medication_request"}, "semantic_type": "create",
             "effect": "irreversible", "resource": "MedicationRequest"},
        ],
    },
}
TASK = {"task_id": "t1", "goal": "g", "context": {"patient_ref": "Patient/123"},
        "environment": {"type": "fhir"}}


class Boom(ScopeEvidenceBinding):
    def before_action(self, action, ctx):
        raise RuntimeError("boom")


def _boom_kernel(mode):
    contract = build_contract(TASK, env_type="fhir", policy=POLICY)
    return HarnessKernel(contract, [Boom()], mode=mode, policy=POLICY, env_type="fhir")


def test_capability_error_escalates_in_enforce():
    k = _boom_kernel("enforce")
    eff = k.before_action({"type": "tool", "tool": "fhir_search", "args": {}}, step=0)
    assert eff.type == D.ESCALATE, eff.type            # fail closed: NOT a silent ALLOW
    assert k._capability_errors, "the capability error must be recorded"


def test_capability_error_allows_in_observe():
    k = _boom_kernel("observe")
    eff = k.before_action({"type": "tool", "tool": "fhir_search", "args": {}}, step=0)
    assert eff.type == D.ALLOW, eff.type               # observe never changes the run
    assert k._capability_errors, "observe still RECORDS the error"
    # the would-be ESCALATE is recorded to the ledger even though effective is ALLOW
    assert any(iv["decision"] == "ESCALATE" for iv in k.ledger.interventions), k.ledger.interventions


def test_audit_status_degraded_on_error():
    clean = _boom_kernel("enforce")           # before any action -> no error yet
    assert clean.audit()["status"] == "active", clean.audit()["status"]
    clean.before_action({"type": "tool", "tool": "fhir_search", "args": {}}, step=0)
    a = clean.audit()
    assert a["status"] == "degraded", a["status"]
    assert a["capability_errors"], a


def test_n_semantic_checks_increments():
    """A semantic check spent via verify_commit decrements ctx.semantic_remaining, which audit() reports
    as n_semantic_checks = max_semantic_checks - semantic_remaining."""
    pol = H.load_policy(env_type="tool_sandbox")
    task = {"task_id": "m", "goal": "finding?", "context": {}, "environment": {"type": "tool_sandbox"}}
    contract = build_contract(task, env_type="tool_sandbox", policy=pol)
    judge = lambda p: '{"supported": true, "confidence": 0.9, "reason": "ok"}'
    k = HarnessKernel(contract, [ScopeEvidenceBinding(), ObligationLifecycle(), VerifyAndCommit()],
                      mode="enforce", policy=pol, env_type="tool_sandbox", judge_fn=judge,
                      budget={"max_semantic_checks": 5})
    assert k.audit()["n_semantic_checks"] == 0, k.audit()["n_semantic_checks"]
    # ground a perception-tool claim, then accept a final answer -> verify_commit spends one semantic check
    k.after_action({"type": "tool", "tool": "ImageDescription", "args": {}},
                   "a 3cm spiculated RUL nodule", None, None, step=0, result_ok=True)
    assert k.before_final("RUL nodule", step=1).type == D.ALLOW
    n = k.audit()["n_semantic_checks"]
    assert n >= 1, ("a semantic check should have been spent", n)
    assert n == k.budget["max_semantic_checks"] - k.ctx.semantic_remaining, n


def _run():
    fns = [v for kk, v in sorted(globals().items()) if kk.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn(); passed += 1; print("PASS", fn.__name__)
        except AssertionError as e:
            print("FAIL", fn.__name__, "->", e)
        except Exception as e:
            print("ERROR", fn.__name__, "->", repr(e))
    print("\nharness runtime conformance: %d/%d passed" % (passed, len(fns)))
    return passed == len(fns)


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
