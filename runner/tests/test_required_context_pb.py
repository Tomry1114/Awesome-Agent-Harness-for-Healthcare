# -*- coding: utf-8 -*-
"""required_context on PhysicianBench, through the real run.run_task (runner/tests/inline_loop_rig.py).

A MedicationRequest create requires medication_safety_review -> check_current_medications (evidence resource
MedicationRequest; runner/policies/clinical/medication_safety.yaml). required_context, a kernel middleware of the PB
H0 workspace, acquires that record before the irreversible create when it has not been read for the assigned patient.

Two properties keep it from refusing every PB medication create:
- the evidence's subject id (as the agent named it, e.g. `MRN…` or `Patient/MRN…`) is matched to
  Ledger.subject_id() (`patient/MRN…`) after normalization, not by string equality, so a read resolves the
  obligation;
- the acquire uses a tool PB exposes (runner/adapters/hapi_fhir.yaml), not the generic `fhir_search`.
If either breaks, required_context ESCALATEs required_context_unavailable and both tests fail.
"""
import inline_loop_rig as L

PB = "PhysicianBench"
PB_TASK = "PB-pretransplant_covid_clearance"     # names no deliverable: the PB deliverable middleware stays inert
MED_SEARCH = "fhir_medication_request_search_orders"
MED_CREATE = "fhir_medication_request_create"


def _pb(monkeypatch, tmp_path):
    rig = L.Rig(monkeypatch, tmp_path, PB, PB_TASK, protocol="function_calling")
    rig.mrn = rig.task()["context"]["patient_ref"]

    def tool_result(name, args):
        if name == MED_SEARCH:
            return {"entries": [{"resourceType": "MedicationRequest", "id": "mr-current-1", "status": "active",
                                 "subject": {"reference": "Patient/" + rig.mrn},
                                 "medicationCodeableConcept": {"text": "tacrolimus 1 mg capsule"}}], "total": 1}
        if name == MED_CREATE:
            return {"resourceType": "MedicationRequest", "id": "mr-new-1", "status": "active",
                    "subject": {"reference": "Patient/" + rig.mrn}}
        return {"entries": [], "total": 0}
    rig.tool_result = tool_result
    return rig


def _create(rig):
    return L.fc_tool(MED_CREATE, {"patient_reference": "Patient/" + rig.mrn, "medication_display": "nirmatrelvir",
                                  "intent": "order", "status": "active"})


def _required_context_decisions(rig):
    return [(d["stage"], d["decision"], d.get("rule_id")) for d in rig.events("harness_decision")
            if d.get("capability") == "required_context"]


def test_pb_medication_create_after_current_medications_were_read_is_allowed(monkeypatch, tmp_path):
    rig = _pb(monkeypatch, tmp_path)
    # the agent names the patient by its bare MRN, as PB agents commonly do
    rig.replies = [L.fc_tool(MED_SEARCH, {"patient": rig.mrn}), _create(rig), L.fc_final("ordered")]
    rig.run(max_steps=8)

    assert _required_context_decisions(rig) == []
    assert rig.events("harness_escalation") == []
    assert [c[0] for c in rig.env_calls][:2] == [MED_SEARCH, MED_CREATE]
    assert rig.kernel.ledger.obligations["check_current_medications"]["state"] == "SATISFIED"


def test_pb_medication_create_without_current_medications_acquires_them_with_the_pb_tool(monkeypatch, tmp_path):
    rig = _pb(monkeypatch, tmp_path)
    rig.replies = [_create(rig), _create(rig), L.fc_final("ordered")]
    rig.run(max_steps=8)

    patient = "Patient/" + rig.mrn
    assert _required_context_decisions(rig)[0] == ("before_action", "ACQUIRE", "required_context")
    pre = rig.events("harness_acquire_predispatch")
    assert [(e["stage"], e["tool"], e["args"]) for e in pre][:1] == [("before_action", MED_SEARCH, {"patient": patient})]
    assert rig.events("harness_acquire_refused") == []
    # the deferred create did not run; the harness read ran first, through the environment, then the create
    assert rig.env_calls[0] == (MED_SEARCH, {"patient": patient})
    assert [c[0] for c in rig.env_calls][:2] == [MED_SEARCH, MED_CREATE]
    assert [(r["capability"], r["tool"], r["in_vocabulary"], r["executed"], r["succeeded"], r["evidence_added"],
             r["failure_reason"]) for r in rig.audit["actuation"]["records"]["acquire"]][:1] == \
        [("required_context", MED_SEARCH, True, True, True, True, None)]
    assert rig.kernel.ledger.obligations["check_current_medications"]["state"] == "SATISFIED"
    assert rig.events("harness_escalation") == []
