"""Conformance for the FAIL-CLOSED unmapped-action safety feature.

A tool action that NO manifest rule (and no manifest-level default_action) maps is UNKNOWN to the
substrate adapter. Such an action must NOT silently default-allow (old behaviour: semantic_type=other,
effect=none -> R0 -> ALLOW). Instead verify_commit ESCALATES it. Under `enforce` that surfaces as a real
ESCALATE; under `observe` the kernel records it but the effective decision is ALLOW.

Standalone, like runner/test_harness.py:  python3 runner/test_harness_unmapped.py
No model calls; deterministic.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness as H  # noqa: F401  (ensures the package imports cleanly)
from harness import decision as D
from harness.kernel import HarnessKernel
from harness.compiler import build_contract
from harness.semantics import canonicalize
from harness.capabilities.scope_evidence import ScopeEvidenceBinding
from harness.capabilities.obligation_lifecycle import ObligationLifecycle
from harness.capabilities.verify_commit import VerifyAndCommit

# A minimal substrate policy whose manifest maps ONLY the tool "known" (a read, no commit).
# It deliberately declares NO default_action -> any other tool stays unmapped -> fail-closed.
POLICY = {
    "manifest": {
        "subject": {"type": "patient", "id_context_keys": ["patient_ref"], "from_args": ["patient"]},
        "actions": [
            {"match": {"tool": "known"}, "semantic_type": "read", "effect": "none",
             "produces_evidence": {"source_class": "record", "modality": "record"}},
        ],
    },
    "evidence_obligations": [],
    "workflow_obligations": [],
    "commit_points": [],
}
TASK = {"task_id": "u1", "goal": "do a thing", "context": {"patient_ref": "Patient/1"},
        "environment": {"type": "fhir"}}

KNOWN = {"type": "tool", "tool": "known", "args": {"patient": "Patient/1"}}
UNKNOWN = {"type": "tool", "tool": "mystery_high_risk_tool", "args": {"patient": "Patient/1"}}


def _kernel(mode, policy=POLICY, task=TASK, env="fhir"):
    contract = build_contract(task, env_type=env, policy=policy)
    caps = [ScopeEvidenceBinding(), ObligationLifecycle(), VerifyAndCommit()]
    return HarnessKernel(contract, caps, mode=mode, policy=policy, env_type=env)


def test_known_tool_is_mapped():
    sem = canonicalize(KNOWN, POLICY["manifest"])
    assert sem.mapped is True, sem.to_dict()
    assert sem.semantic_type == "read" and sem.effect == "none", sem.to_dict()
    assert sem.to_dict().get("mapped") is True            # to_dict carries the field


def test_unknown_tool_is_not_mapped():
    sem = canonicalize(UNKNOWN, POLICY["manifest"])
    assert sem.mapped is False, sem.to_dict()
    # old silent-allow shape, but now flagged unmapped:
    assert sem.semantic_type == "other" and sem.effect == "none", sem.to_dict()


def test_final_answer_is_always_mapped():
    sem = canonicalize({"type": "final", "answer": "x"}, POLICY["manifest"])
    assert sem.mapped is True and sem.semantic_type == "answer", sem.to_dict()


def test_default_action_maps_otherwise_unknown_tool():
    pol = {
        "manifest": {
            "subject": POLICY["manifest"]["subject"],
            "actions": POLICY["manifest"]["actions"],
            "default_action": {"semantic_type": "read", "effect": "none",
                               "produces_evidence": {"source_class": "record", "modality": "record"}},
        },
        "evidence_obligations": [], "workflow_obligations": [], "commit_points": [],
    }
    sem = canonicalize(UNKNOWN, pol["manifest"])
    assert sem.mapped is True and sem.semantic_type == "read", sem.to_dict()


def test_enforce_allows_known_escalates_unknown():
    k = _kernel("enforce")
    assert k.before_action(KNOWN, step=0).type == D.ALLOW
    esc = k.before_action(UNKNOWN, step=1)
    assert esc.type == D.ESCALATE, esc.type


def test_enforce_escalation_reason_code_is_unmapped_action():
    k = _kernel("enforce")
    eff = k.before_action(UNKNOWN, step=0)
    assert eff.type == D.ESCALATE
    assert eff.raw is not None and eff.raw.reason_code == "unmapped_action", eff.raw.to_dict()
    assert eff.raw.deterministic is True


def test_observe_records_but_allows_unknown():
    k = _kernel("observe")
    # observe: effective decision is ALLOW (record-only), but the raw would-be is ESCALATE.
    eff = k.before_action(UNKNOWN, step=0)
    assert eff.type == D.ALLOW, eff.type
    assert eff.raw is not None and eff.raw.type == D.ESCALATE, eff.raw.to_dict()
    # and it was recorded as an intervention in the ledger
    interventions = k.ledger.interventions if hasattr(k.ledger, "interventions") else []
    assert any(i.get("reason_code") == "unmapped_action" for i in interventions), interventions


def test_default_action_kernel_allows_unknown_in_enforce():
    pol = {
        "manifest": {
            "subject": POLICY["manifest"]["subject"],
            "actions": POLICY["manifest"]["actions"],
            "default_action": {"semantic_type": "read", "effect": "none",
                               "produces_evidence": {"source_class": "record", "modality": "record"}},
        },
        "evidence_obligations": [], "workflow_obligations": [], "commit_points": [],
    }
    k = _kernel("enforce", policy=pol)
    # with a declared default_action, an otherwise-unknown tool is mapped -> not escalated.
    assert k.before_action(UNKNOWN, step=0).type == D.ALLOW


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]


def _run():
    passed = failed = 0
    for t in TESTS:
        try:
            t()
            passed += 1
            print("PASS %s" % t.__name__)
        except Exception as ex:
            failed += 1
            print("FAIL %s: %r" % (t.__name__, ex))
    print("\n%d passed, %d failed (%d tests)" % (passed, failed, len(TESTS)))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    _run()
