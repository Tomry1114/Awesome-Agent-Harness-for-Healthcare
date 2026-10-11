import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from workspace import H0, Workspace
_GA = sys.modules[Workspace(H0["gui"]).load("middleware/grounding_audit.py", "GroundingAudit").__module__]
extract_salient, grounding_gap, feedback, note_text_from_state = (_GA.extract_salient, _GA.grounding_gap,
                                                                  _GA.feedback, _GA.note_text_from_state)


def test_extract_salient_note_slot_is_clinical_codes_only():
    obs = "Denial CO-50 remark N386 for diagnosis H35.32, procedure 67028. Referral REF-2025-001. Claim CLM-2025-00001."
    s = extract_salient(obs)                                          # default slot='note'
    for tok in ("CO-50", "N386", "H35.32", "67028"):                 # clinical/billing codes kept
        assert any(tok.replace("-", "").lower() == x.replace("-", "").lower() for x in s), (tok, s)
    # generic RECORD ids are NOT required in a clinical note -> dropped (so the audit does not over-fire)
    assert not any("REF" in x or "CLM" in x for x in s), s
    # and a 5-digit FRAGMENT of a longer id is never mistaken for a CPT
    assert "00001" not in extract_salient("CLM-2025-00001")


def test_grounding_gap_flags_unreferenced_observed_facts():
    observed = {"CO-50", "N386", "H35.32", "67028"}
    note = "Denial is CO-50 / N386. Escalating for review."           # mentions CO-50, N386 only
    gap = grounding_gap(note, observed)
    assert "H35.32" in gap and "67028" in gap                          # missing dx + cpt
    assert "CO-50" not in gap and "N386" not in gap                    # referenced -> not flagged


def test_feedback_names_missing_never_supplies_meaning():
    fb = feedback(["H35.32", "67028"])
    assert "H35.32" in fb and "67028" in fb
    assert "supplement" in fb.lower() and "do not invent" in fb.lower()
    assert feedback([]) is None                                        # fully covered -> no nudge


def test_empty_note_flags_all():
    assert grounding_gap("", {"CO-50", "N386"}) == ["CO-50", "N386"]

def test_note_text_from_state_reads_persisted_note_not_answer():
    state = {"full_state": {"triageNotes": "Denial CO-50 / N386. Appeal filed.",
                            "communications": [{"content": "Faxed to Lincare, confirmation FAX-2025-1."}],
                            "agentActions": {"selectedDisposition": "Route to Clinical Appeals"}}}
    note = note_text_from_state(state)
    assert "CO-50" in note and "FAX-2025-1" in note                    # persisted note + communications
    assert "Route to Clinical Appeals" not in note                     # disposition is not note content
    # a code observed but missing from the persisted note is flagged
    gap = grounding_gap(note, extract_salient("diagnosis H35.32 CO-50 N386"))
    assert any("H35.32" in g for g in gap) and not any("CO-50" in g for g in gap)
