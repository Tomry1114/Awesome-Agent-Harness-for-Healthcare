"""C6 FhirRecoveryAdapter: extraction -> Commitment, effect_key identity, compile_effect shape+tag,
delegated inspect. Confirms the seam preserves the FHIR behavior."""
import sys
sys.path.insert(0, "runner")
from harness.recovery_adapter import get_recovery_adapter, FhirRecoveryAdapter, Commitment, EffectInspection
from harness.recovery_orchestrator import EffectCompletionKey

R = []
def ck(n, c): R.append((n, bool(c))); print(("OK  " if c else "FAIL") + " " + n)

a = get_recovery_adapter("fhir")
ck("factory_fhir", isinstance(a, FhirRecoveryAdapter))
ck("factory_none_for_unmodelled", get_recovery_adapter("tool_sandbox") is None)   # perceptual substrate not modelled yet

ctx = {"subject": "Patient/MRN1", "artifact_hash": "hh", "task_id": "PB-x"}
c = Commitment(text="Order pelvic ultrasound", category="imaging",
               signature="order pelvic ultrasound", effect_type="ServiceRequest")

# effect_key identity: same commitment+context -> equal key (dedup); different signature -> different key
k1 = a.effect_key(c, ctx)
k2 = a.effect_key(c, ctx)
c2 = Commitment(text="Order CBC", category="lab", signature="order cbc", effect_type="ServiceRequest")
k3 = a.effect_key(c2, ctx)
ck("effect_key_stable", isinstance(k1, EffectCompletionKey) and k1 == k2)
ck("effect_key_per_order", k1 != k3)

# compile_effect: fhir_create + scope + hygiene tag + irreversible
plan = a.compile_effect(c, ctx, None)
tags = (((plan.resource or {}).get("meta") or {}).get("tag") or []) if plan else []
ck("compile_effect_fhir_create", plan is not None and plan.mutation_action["tool"] == "fhir_create")
ck("compile_effect_scope_irreversible", plan.scope.get("allowed_tool") == "fhir_create"
   and plan.scope.get("expected_postcondition", {}).get("verify") == "server_readback")
ck("compile_effect_hygiene_tag", any(t.get("code") == "harness-recovery-created" for t in tags))

# inspect_effect delegates to the driver and normalizes to EffectInspection
class DriverStub:
    def inspect_effect(self, rtype, subject):
        assert rtype == "ServiceRequest" and subject == "Patient/MRN1"
        return {"state": "ABSENT", "texts": [], "matched_ids": []}
insp = a.inspect_effect(c, DriverStub(), ctx)
ck("inspect_delegates_absent", isinstance(insp, EffectInspection) and insp.state == "ABSENT")

n = sum(1 for _, x in R if x)
print("\n%d/%d recovery_adapter tests passed" % (n, len(R)))
assert n == len(R), "FAILURES"
