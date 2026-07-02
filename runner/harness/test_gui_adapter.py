"""H0 pre-live hardening for GuiRecoveryAdapter: conservative commitment (agent-decided AND landed), commit-marker
-only effect truth (submittedAppeal exact-true), case-identity, dynamic submit affordance, should_trigger hook."""
import sys, yaml
sys.path.insert(0, "runner")
from harness.recovery_adapter import get_recovery_adapter, GuiRecoveryAdapter, FhirRecoveryAdapter

R = []
def ck(n, c): R.append((n, bool(c))); print(("OK  " if c else "FAIL") + " " + n)

MAN = yaml.safe_load(open("runner/adapters/admin_portal.yaml"))
a = get_recovery_adapter("gui", MAN)
task = {"goal": "Review denied claim APL-4471 and file an appeal if warranted.", "id": "HAB-x"}
ctx0 = a.context(task); ctx0["artifact_hash"] = "hh"
ck("context_case", ctx0["case_id"] == "APL-4471")

def C(state, traj): return {**ctx0, "state_view": state}, traj
SEL = lambda v: {"event_type": "tool_call", "origin": "agent", "tool": "select", "args": {"field": "selectedDisposition", "value": v}, "action_id": "a-sel"}
TYPE = lambda: {"event_type": "tool_call", "origin": "agent", "tool": "type", "args": {"field": "authNote", "text": "necessity met"}, "action_id": "a-typ"}
def emr(disp=None, note=None, submitted=False, case="APL-4471", attach=None):
    return {"caseId": case, "agentActions": {"selectedDisposition": disp, "addedAuthNote": note},
            "appealActions": {"submittedAppeal": submitted, "submittedAttachmentNames": attach}}

# ---- should_trigger hook (no benchmark branch in run.py) ----
ck("gui_triggers_before_final", a.should_trigger("before_final") and not a.should_trigger("deliverable_confirmed"))
ck("fhir_triggers_deliverable", FhirRecoveryAdapter().should_trigger("deliverable_confirmed") and not FhirRecoveryAdapter().should_trigger("before_final"))

# ---- VALID commitment: agent selected appeal, it LANDED, submittedAppeal False, case matches ----
c, t = C(emr(disp="appeal", submitted=False), [SEL("appeal")])
coms = a.extract_commitments(None, t, task["goal"], None, c)
ck("valid_commitment", len(coms) == 1 and coms[0].payload["disposition"] == "appeal"
   and coms[0].target_entity == "APL-4471" and coms[0].origin_action_ids == ["a-sel"])

# ---- note-only -> NO commitment (never defaults disposition="appeal") ----
c, t = C(emr(disp=None, note="administrative note", submitted=False), [TYPE()])
ck("note_only_no_commitment", a.extract_commitments(None, t, "", None, c) == [])

# ---- disposition = do_not_appeal -> NO commitment ----
c, t = C(emr(disp="do_not_appeal", submitted=False), [SEL("do_not_appeal")])
ck("do_not_appeal_no_commitment", a.extract_commitments(None, t, "", None, c) == [])

# ---- select attempted but NOT landed in state -> NO commitment ----
c, t = C(emr(disp=None, submitted=False), [SEL("appeal")])          # agent selected, state shows nothing landed
ck("select_not_landed_no_commitment", a.extract_commitments(None, t, "", None, c) == [])

# ---- agent typed a note but it did NOT land -> NO commitment (attempted-but-unlanded edit) ----
c, t = C(emr(disp="appeal", note=None, submitted=False), [SEL("appeal"), TYPE()])
ck("typed_note_unlanded_no_commitment", a.extract_commitments(None, t, "", None, c) == [])

# ---- active case mismatch -> NO commitment (never submit for the wrong case) ----
c, t = C(emr(disp="appeal", submitted=False, case="APL-9999"), [SEL("appeal")])
ck("case_mismatch_no_commitment", a.extract_commitments(None, t, "", None, c) == [])

# ---- already submitted -> NO commitment ----
c, t = C(emr(disp="appeal", submitted=True), [SEL("appeal")])
ck("already_submitted_no_commitment", a.extract_commitments(None, t, "", None, c) == [])

# ---- no state_view -> fail-closed (no commitment) ----
ck("no_state_fail_closed", a.extract_commitments(None, [SEL("appeal")], "", None, {"case_id": "APL-4471"}) == [])

# ===== inspect_effect: commit marker ONLY =====
com = a.extract_commitments(None, [SEL("appeal")], task["goal"], None, C(emr(disp="appeal"), [])[0])[0]
# attachment present but submittedAppeal False -> ABSENT (not "present")
insp_attach = a.inspect_effect(com, None, {**ctx0, "state_view": emr(submitted=False, attach=["doc.pdf"])})
ck("attachment_without_submit_is_absent", insp_attach.state == "ABSENT")
ck("submitted_true_is_present", a.inspect_effect(com, None, {**ctx0, "state_view": emr(submitted=True)}).state == "PRESENT")
ck("marker_missing_is_unknown", a.inspect_effect(com, None, {**ctx0, "state_view": {"appealActions": {}}}).state == "UNKNOWN")
ck("no_state_inspect_unknown", a.inspect_effect(com, None, {**ctx0, "state_view": None}).state == "UNKNOWN")
# inspect fails closed on case mismatch even if marker true
ck("inspect_case_mismatch_unknown", a.inspect_effect(com, None, {**ctx0, "state_view": emr(submitted=True, case="APL-9999")}).state == "UNKNOWN")

# ===== compile_effect: dynamic affordance, submit-only, false->true postcondition =====
plan = a.compile_effect(com, ctx0, MAN)
ck("compile_prepare_snapshot", any(s["tool"] == "snapshot" for s in plan.prepare_actions))
ck("compile_affordance_not_static", plan.mutation_action is None and plan.commit_affordance["tool"] == "submit"
   and plan.commit_affordance["match"]["labels"])
ck("compile_postcondition_false_to_true", plan.scope["expected_postcondition"]["path"].endswith("submittedAppeal")
   and plan.scope["expected_postcondition"]["equals"] is True)
ck("compile_irreversible_submit", plan.scope["allowed_effect"] == "irreversible" and plan.scope["allowed_tool"] == "submit")

n = sum(1 for _, x in R if x)
print("\n%d/%d gui H0 tests passed" % (n, len(R)))
assert n == len(R), "FAILURES"
