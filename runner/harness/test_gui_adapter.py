"""C6 GuiRecoveryAdapter (HAB substrate): commitment from agent select/type, effect inspection over the portal
emr full_state via manifest repair_targets, mechanical-submit compile_effect, effect_key. Uses the REAL
admin_portal manifest + emr state shape."""
import sys, yaml
sys.path.insert(0, "runner")
from harness.recovery_adapter import get_recovery_adapter, GuiRecoveryAdapter, Commitment, EffectInspection

R = []
def ck(n, c): R.append((n, bool(c))); print(("OK  " if c else "FAIL") + " " + n)

MANIFEST = yaml.safe_load(open("runner/adapters/admin_portal.yaml"))
a = get_recovery_adapter("gui", MANIFEST)
ck("factory_gui", isinstance(a, GuiRecoveryAdapter))

task = {"goal": "Review denied claim APL-4471 and file an appeal if warranted.", "id": "HAB-x"}
ctx = a.context(task); ctx["artifact_hash"] = "hh"
ck("context_case_id", ctx["case_id"] == "APL-4471")

# agent SELECTED a disposition + TYPED a note, but NEVER submitted -> a committed-but-unsubmitted appeal
traj = [
    {"event_type": "tool_call", "origin": "agent", "tool": "select", "args": {"field": "selectedDisposition", "value": "appeal"}},
    {"event_type": "tool_call", "origin": "agent", "tool": "type", "args": {"field": "authNote", "text": "Medical necessity met per policy."}},
    {"event_type": "tool_call", "origin": "agent", "tool": "snapshot", "args": {}},
]
coms = a.extract_commitments(None, traj, task["goal"], None)
ck("extract_one_commitment", len(coms) == 1 and coms[0].effect_type == "submittedAppeal")
c = coms[0]

# no agent decision -> nothing to complete (never fabricate a disposition)
ck("no_commitment_without_agent_decision", a.extract_commitments(None, [{"event_type": "tool_call", "origin": "agent", "tool": "snapshot", "args": {}}], "", None) == [])

# effect_key identity + per-case scoping
k1 = a.effect_key(c, ctx); k2 = a.effect_key(c, ctx)
ctx2 = dict(ctx); ctx2["case_id"] = "APL-9999"
ck("effect_key_stable", k1 == k2)
ck("effect_key_per_case", a.effect_key(c, ctx2) != k1)

# inspect_effect over the emr full_state (manifest paths)
absent = a.inspect_effect(c, None, {**ctx, "state_view": {"appealActions": {"submittedAppeal": False}, "agentActions": {"selectedDisposition": "appeal"}}})
ck("inspect_absent_not_submitted", absent.state == "ABSENT")
present = a.inspect_effect(c, None, {**ctx, "state_view": {"appealActions": {"submittedAppeal": True}}})
ck("inspect_present_submitted", present.state == "PRESENT" and a.is_realized(c, present.texts))
unknown = a.inspect_effect(c, None, {**ctx, "state_view": None})
ck("inspect_unknown_no_state", unknown.state == "UNKNOWN")

# compile_effect: mechanical submit only, irreversible, verifies the manifest appeal paths
plan = a.compile_effect(c, ctx, MANIFEST)
ck("compile_effect_submit_only", plan.mutation_action["tool"] == "submit" and plan.scope["allowed_effect"] == "irreversible")
ck("compile_effect_verifies_appeal_path", any("submittedAppeal" in p for p in plan.scope["expected_postcondition"]["paths"]))
# it must NOT be a select/type (never chooses disposition or writes note)
ck("compile_effect_not_disposition_or_note", plan.mutation_action["tool"] not in ("select", "type"))

n = sum(1 for _, x in R if x)
print("\n%d/%d gui_adapter tests passed" % (n, len(R)))
assert n == len(R), "FAILURES"
