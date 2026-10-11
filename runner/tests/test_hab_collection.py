import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adapters.hab.collection import collection_delta, present_values, COLLECTION_CONCEPTS


def test_collection_delta_missing_only_order_preserving():
    committed = ["H35.32", "H35.31", "E11.9"]
    present = ["H35.32"]                                   # one chip already added
    assert collection_delta(committed, present) == ["H35.31", "E11.9"]


def test_delta_normalizes_and_dedups_never_invents():
    assert collection_delta(["67028", "67028", "J2778"], ["6 70 28"]) == ["J2778"]   # dup + normalized present
    assert collection_delta([], ["X"]) == []              # nothing committed -> never invents
    assert collection_delta(["A", "B"], []) == ["A", "B"]


def test_present_values_from_chips_or_controls():
    assert present_values(["H35.32", " ", "E11.9"]) == ["H35.32", "E11.9"]
    assert present_values([{"value": "67028"}, {"label": "J2778"}]) == ["67028", "J2778"]


def test_collection_concepts_cover_the_emr_sets():
    assert {"diagnosis", "cpt"} <= COLLECTION_CONCEPTS
