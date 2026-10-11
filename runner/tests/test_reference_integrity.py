"""Unit tests for ReferenceIntegrity (fabricated_reference). Mock ctx, no gateway. The fixture is a hematology
consult: recipient=Practitioner/primary-oncology-team (invented) and encounter=Encounter/current (placeholder) fail
with FHIR 400, while requester/sender=Practitioner/dr-elena-hayes (named in the goal) is fine."""
import sys
sys.path.insert(0, ".")
from harness.capabilities.reference_integrity import ReferenceIntegrity
from harness import decision as D

REF_PATHS = [
    {"path": "encounter_reference", "resource": "Encounter", "required": False, "on_ungrounded": "strip"},
    {"path": "recipient_reference", "resource": "Practitioner", "required": False, "on_ungrounded": "strip"},
    {"path": "requester_reference", "resource": "Practitioner", "required": False, "on_ungrounded": "revise"},
    {"path": "sender_reference", "resource": "Practitioner", "required": False, "on_ungrounded": "revise"},
]
GOAL = "Hematology consult. You are the attending (Practitioner ID: dr-elena-hayes). Patient MRN2812140441."


class _Sem:
    def __init__(self, commit): self._c = commit
    def is_commit(self): return self._c


class _Ledger:
    def __init__(self, ev, subj="Patient/MRN2812140441"): self.evidence = ev; self._s = subj
    def subject_id(self): return self._s


class _Ctx:
    def __init__(self, commit=True, refs=REF_PATHS, ev=None, goal=GOAL):
        self.sem = _Sem(commit)
        self.manifest = {"reference_paths": refs}
        self.ledger = _Ledger(ev or [])
        self.policy = {"_task": {"goal": goal, "context": ""}}
        self.contract = None


def _ev(value_full, subj="Patient/MRN2812140441"):
    return {"status": "VALIDATED", "value_full": value_full, "subject_id": subj}


def _act(**args):
    return {"type": "tool", "tool": "fhir_service_request_create", "args": args}


def test_auto_strips_placeholder_encounter_and_invented_recipient():
    # encounter+recipient are on_ungrounded=strip -> AUTO-REPAIRED in place, action proceeds (None)
    cap = ReferenceIntegrity()
    ctx = _Ctx()
    a = _act(patient_reference="Patient/MRN2812140441",
             encounter_reference="Encounter/current",
             recipient_reference="Practitioner/primary-oncology-team",
             requester_reference="Practitioner/dr-elena-hayes")
    dec = cap.before_action(a, ctx)
    assert dec is None, dec                                     # all bad refs stripped -> proceed
    assert "encounter_reference" not in a["args"]              # auto-removed
    assert "recipient_reference" not in a["args"]              # auto-removed
    assert a["args"]["requester_reference"] == "Practitioner/dr-elena-hayes"   # grounded -> kept
    reps = {r["value"] for r in ctx.ledger._reference_repairs_pending}
    assert reps == {"Encounter/current", "Practitioner/primary-oncology-team"}


def test_ungrounded_revise_ref_revises():
    # a requester (on_ungrounded=revise) that is invented -> REVISE (not auto-stripped)
    cap = ReferenceIntegrity()
    a = _act(patient_reference="Patient/MRN2812140441",
             requester_reference="Practitioner/dr-nobody-made-up")
    dec = cap.before_action(a, _Ctx())
    assert dec is not None and dec.type == D.REVISE and dec.rule_id == "fabricated_reference", dec
    assert "Practitioner/dr-nobody-made-up" in dec.extra["ungrounded_refs"]
    assert a["args"].get("requester_reference") == "Practitioner/dr-nobody-made-up"   # NOT stripped


def test_grounded_encounter_from_evidence_ok():
    cap = ReferenceIntegrity()
    # agent retrieved Encounter/enc-77 -> referencing it is grounded
    ctx = _Ctx(ev=[_ev('{"resourceType":"Encounter","id":"enc-77","status":"finished"}')])
    a = _act(patient_reference="Patient/MRN2812140441", encounter_reference="Encounter/enc-77")
    assert cap.before_action(a, ctx) is None


def test_all_grounded_no_flag():
    cap = ReferenceIntegrity()
    a = _act(patient_reference="Patient/MRN2812140441", requester_reference="Practitioner/dr-elena-hayes")
    assert cap.before_action(a, _Ctx()) is None


def test_non_commit_noop():
    cap = ReferenceIntegrity()
    ctx = _Ctx(commit=False)
    a = _act(recipient_reference="Practitioner/primary-oncology-team")
    assert cap.before_action(a, ctx) is None


def test_no_reference_paths_noop():
    cap = ReferenceIntegrity()
    ctx = _Ctx(refs=[])
    a = _act(recipient_reference="Practitioner/primary-oncology-team")
    assert cap.before_action(a, ctx) is None


def test_feedback_names_the_fix():
    # revise-path feedback must name the offending ref + a fix
    cap = ReferenceIntegrity()
    a = _act(requester_reference="Practitioner/dr-nobody-made-up")
    dec = cap.before_action(a, _Ctx())
    assert "dr-nobody-made-up" in dec.feedback and "retrieved" in dec.feedback


def test_grounded_in_observation_or_unvalidated_evidence_not_flagged():
    # A practitioner the agent RETRIEVED via a search -- present in ledger observations (or non-VALIDATED
    # evidence), NOT in VALIDATED evidence -- must NOT be flagged fabricated; grounding that only counted VALIDATED
    # evidence would flag retrieved references as invented.
    class _L:
        def __init__(self): self.evidence = []; self.observations = [
            {"content": "Bundle: Practitioner/dr-raymond-simmons  name: Dr Raymond Simmons, attending"}]
        def subject_id(self): return "Patient/MRN1"
    class _S:
        def is_commit(self): return True
    class _C:
        def __init__(self):
            self.sem = _S(); self.ledger = _L(); self.contract = None
            self.manifest = {"reference_paths": [
                {"path": "requester_reference", "resource": "Practitioner", "required": False, "on_ungrounded": "revise"}]}
            self.policy = {"_task": {"goal": "", "context": ""}}
    a = {"type": "tool", "tool": "fhir_service_request_create",
         "args": {"patient_reference": "Patient/MRN1", "requester_reference": "Practitioner/dr-raymond-simmons"}}
    assert ReferenceIntegrity().before_action(a, _C()) is None            # retrieved -> grounded -> not flagged
    # and a truly-absent practitioner is STILL flagged
    a2 = dict(a); a2 = {"type": "tool", "tool": "fhir_service_request_create",
                        "args": {"patient_reference": "Patient/MRN1", "requester_reference": "Practitioner/dr-made-up"}}
    dec = ReferenceIntegrity().before_action(a2, _C())
    assert dec is not None and dec.rule_id == "fabricated_reference"


def test_required_tool_arg_reference_not_stripped():
    # requester_reference is a REQUIRED tool positional arg AND a fabricated one is non-fatal (FHIR accepts it).
    # Stripping it leaves the tool call without a required arg ("missing 1 required positional argument"), so the
    # agent can only retry until the run times out. Design: requester/sender are NOT in reference_paths at all, so
    # the capability never strips them and the required tool arg survives.
    class _L:
        evidence = []; observations = []
        def subject_id(self): return "Patient/MRN1"
    class _S:
        def is_commit(self): return True
    class _C:
        sem = _S(); ledger = _L(); contract = None
        manifest = {"reference_paths": [
            {"path": "encounter_reference", "resource": "Encounter", "required": False, "on_ungrounded": "strip"}]}
        policy = {"_task": {"goal": "", "context": ""}}
    a = {"type": "tool", "tool": "fhir_service_request_create",
         "args": {"patient_reference": "Patient/MRN1", "requester_reference": "Practitioner/dr-invented", "code_display": "X"}}
    assert ReferenceIntegrity().before_action(a, _C()) is None            # requester not listed -> nothing stripped
    assert a["args"]["requester_reference"] == "Practitioner/dr-invented"  # REQUIRED tool arg preserved


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for fn in fns:
        try:
            fn(); ok += 1; print("PASS", fn.__name__)
        except Exception:
            print("FAIL", fn.__name__); traceback.print_exc()
    print("%d/%d passed" % (ok, len(fns)))
