"""Canonical harness trajectory-event builders. Harness events are kept DISTINCT from ordinary tool
events (event_type 'harness_decision' / 'harness_resolution') so the scorer / audit never conflates an
agent tool call with a harness intervention. kernel.py emits compatible shapes; these builders are the
single reference for that schema (and for P1–P3 capabilities that emit their own)."""


def harness_decision(eid, stage, mode, capability, decision, rule_id=None, effective=None,
                     missing_obligations=None, deterministic=None, proposed_action_id=None, extra=None):
    e = {"event_type": "harness_decision", "id": eid, "stage": stage, "mode": mode,
         "capability": capability, "decision": decision, "effective": effective, "rule_id": rule_id,
         "missing_obligations": list(missing_obligations or []), "deterministic": deterministic,
         "proposed_action_id": proposed_action_id}
    if extra:
        e.update(extra)
    return e


def harness_resolution(eid, original_decision_id, resolution, satisfying_event_ids=None):
    return {"event_type": "harness_resolution", "id": eid,
            "original_decision_id": original_decision_id, "resolution": resolution,
            "satisfying_event_ids": list(satisfying_event_ids or [])}
