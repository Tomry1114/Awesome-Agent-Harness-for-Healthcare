"""Subject identity normalization for scope_evidence (subject_scope_mismatch)."""
import sys
sys.path.insert(0, ".")
from harness.capabilities.scope_evidence import _same_subject


def test_mrn_prefix_normalized():
    # A bare MRN number and the typed patient ref are the SAME patient -> must NOT be a subject_scope_mismatch
    # (otherwise a correct patient read is flagged as out of scope).
    assert _same_subject("8429141456", "patient/MRN8429141456") is True
    assert _same_subject("MRN999", "patient/mrn999") is True
    assert _same_subject("Patient/MRN42", "42") is True


def test_genuine_mismatch_and_type_preserved():
    assert _same_subject("Patient/123", "Encounter/123") is False        # same id, different type
    assert _same_subject("Patient/MRN1", "Patient/MRN2") is False         # different patient


if __name__ == "__main__":
    for fn in [test_mrn_prefix_normalized, test_genuine_mismatch_and_type_preserved]:
        fn(); print("PASS", fn.__name__)
