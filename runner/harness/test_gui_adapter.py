"""H0 GuiRecoveryAdapter re-grounded to the ACTUAL HAB checkpoint target: full_state.agentActions.
documentedAppealInEpic (mechanical, decision-INDEPENDENT), gated on a landed selectedDisposition. Harness
completes only the documentation; never chooses the disposition."""
import sys, yaml
sys.path.insert(0, "runner")
from harness.recovery_adapter import get_recovery_adapter, GuiRecoveryAdapter, FhirRecoveryAdapter

R = []
def ck(n, c): R.append((n, bool(c))); print(("OK  " if c else "FAIL") + " " + n)

MAN = yaml.safe_load(open("runner/adapters/admin_portal.yaml"))
a = get_recovery_adapter("gui", MAN)
task = {"goal": "Open denial DEN-001 for Martinez. Determine the triage disposition.", "id": "HAB-x"}
ctx0 = a.context(task); ctx0["artifact_hash"] = "hh"
ck("context_case", ctx0["case_id"] == "DEN-001")
ck("marker_is_documentedAppealInEpic", a.commit_marker.endswith("documentedAppealInEpic"))

SEL = lambda v: {"event_type": "tool_call", "origin": "agent", "tool": "select", "args": {"field": "selectedDisposition", "value": v}, "action_id": "a-sel"}
def fs(disp=None, documented=False, case="DEN-001"):
    return {"fields": {"caseId": case}, "agentActions": {"selectedDisposition": disp, "documentedAppealInEpic": documented}}

# should_trigger
ck("gui_before_final", a.should_trigger("before_final") and not a.should_trigger("deliverable_confirmed"))
ck("fhir_deliverable", FhirRecoveryAdapter().should_trigger("deliverable_confirmed"))

# VALID: agent selected a disposition that LANDED, documentedAppealInEpic False, case matches -> one commitment
c = {**ctx0, "state_view": fs(disp="Route to Clinical Appeals", documented=False)}
coms = a.extract_commitments(None, [SEL("Route to Clinical Appeals")], task["goal"], None, c)
ck("valid_commitment", len(coms) == 1 and coms[0].payload["disposition"] == "Route to Clinical Appeals"
   and coms[0].effect_type == "decision_documentation" and coms[0].origin_action_ids == ["a-sel"])

# DECISION-INDEPENDENT: even a non-appeal disposition (Write Off) -> commitment (we only document, never decide)
c = {**ctx0, "state_view": fs(disp="Write Off", documented=False)}
ck("any_landed_disposition_commits", len(a.extract_commitments(None, [SEL("Write Off")], "", None, c)) == 1)

# NO decision selected (no select) -> NO commitment (never fabricate a disposition)
c = {**ctx0, "state_view": fs(disp=None, documented=False)}
ck("no_disposition_no_commitment", a.extract_commitments(None, [{"event_type":"tool_call","origin":"agent","tool":"type","args":{"field":"note","text":"x"}}], "", None, c) == [])

# no disposition landed in state -> NO commitment (state is the source of truth)
c = {**ctx0, "state_view": fs(disp=None, documented=False)}
ck("landed_empty_no_commitment", a.extract_commitments(None, [SEL("Route to Clinical Appeals")], "", None, c) == [])

# REAL-PORTAL: disposition landed in state with NO `select` action (typed -> state) -> STILL a commitment
c = {**ctx0, "state_view": fs(disp="Route to Clinical Appeals", documented=False)}
_typed = [{"event_type": "tool_call", "origin": "agent", "tool": "type", "args": {"ref": 13, "text": "Route to Clinical Appeals ..."}}]
ck("landed_via_type_commits", len(a.extract_commitments(None, _typed, "", None, c)) == 1)

# state is the source of truth: the LANDED disposition is what commits (whatever a select action said)
c = {**ctx0, "state_view": fs(disp="Transfer to Patient", documented=False)}
_coms = a.extract_commitments(None, [SEL("Route to Clinical Appeals")], "", None, c)
ck("landed_state_is_truth", len(_coms) == 1 and _coms[0].payload["disposition"] == "Transfer to Patient")

# already documented -> NO commitment
c = {**ctx0, "state_view": fs(disp="Write Off", documented=True)}
ck("already_documented_no_commitment", a.extract_commitments(None, [SEL("Write Off")], "", None, c) == [])

# active case mismatch -> NO commitment
c = {**ctx0, "state_view": fs(disp="Write Off", documented=False, case="DEN-999")}
ck("case_mismatch_no_commitment", a.extract_commitments(None, [SEL("Write Off")], "", None, c) == [])

# no state -> fail-closed
ck("no_state_fail_closed", a.extract_commitments(None, [SEL("Write Off")], "", None, {"case_id": "DEN-001"}) == [])

# inspect_effect: marker ONLY
com = a.extract_commitments(None, [SEL("Write Off")], task["goal"], None, {**ctx0, "state_view": fs(disp="Write Off")})[0]
ck("documented_true_present", a.inspect_effect(com, None, {**ctx0, "state_view": fs(disp="Write Off", documented=True)}).state == "PRESENT")
ck("documented_false_absent", a.inspect_effect(com, None, {**ctx0, "state_view": fs(disp="Write Off", documented=False)}).state == "ABSENT")
ck("documented_true_string_present", a.inspect_effect(com, None, {**ctx0, "state_view": fs(disp="Write Off", documented="True")}).state == "PRESENT")
ck("marker_missing_unknown", a.inspect_effect(com, None, {**ctx0, "state_view": {"agentActions": {}, "fields": {"caseId": "DEN-001"}}}).state == "UNKNOWN")
ck("inspect_case_mismatch_unknown", a.inspect_effect(com, None, {**ctx0, "state_view": fs(disp="Write Off", documented=True, case="DEN-999")}).state == "UNKNOWN")

# compile_effect: dynamic document affordance, never a select, marker false->true
plan = a.compile_effect(com, ctx0, MAN)
ck("compile_prepare_snapshot", any(s["tool"] == "snapshot" for s in plan.prepare_actions))
ck("compile_affordance_document", plan.mutation_action is None and plan.commit_affordance["target_key"] == "documentedAppealInEpic")
ck("compile_not_select", plan.commit_affordance["tool"] != "select")
ck("compile_postcondition_marker", plan.scope["expected_postcondition"]["path"].endswith("documentedAppealInEpic")
   and plan.scope["expected_postcondition"]["equals"] is True)

n = sum(1 for _, x in R if x)
print("\n%d/%d gui H0 (re-grounded) tests passed" % (n, len(R)))
assert n == len(R), "FAILURES"
