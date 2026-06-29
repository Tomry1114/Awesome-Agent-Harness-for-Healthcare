"""Capability wrapper for the substrate-agnostic repair delta validator (pure logic lives in
harness/repair_delta.py). Registerable as a standalone kernel capability; the ScopedRepair capability
also calls validate_repair() directly inside its lifecycle."""
from ..repair_delta import validate_repair, RepairVerdict   # noqa: F401
from ..capability import Capability


class RepairDeltaValidator(Capability):
    # LAYER: INFRASTRUCTURE -- verifies a delivered repair was applied without regression (substrate-agnostic)
    name = "repair_delta"

    def validate(self, finding, before_projection, after_projection, ctx=None):
        return validate_repair(finding, before_projection, after_projection)
