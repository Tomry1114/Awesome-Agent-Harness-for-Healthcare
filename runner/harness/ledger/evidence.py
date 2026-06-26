"""Evidence record helpers. The live evidence list is held in state.Ledger; this is the typed shape +
a constructor so every evidence entry is consistent across capabilities and the audit."""


def make_evidence(evidence_id, type, value, subject_id=None, source_event=None, source_type=None):
    return {"evidence_id": evidence_id, "type": type, "value": value, "subject_id": subject_id,
            "source_event": source_event, "source_type": source_type}


FIELDS = ("evidence_id", "type", "value", "subject_id", "source_event", "source_type")
