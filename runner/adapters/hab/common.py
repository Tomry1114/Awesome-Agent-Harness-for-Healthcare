"""Shared HAB adapter primitives — deterministic helpers reused by more than one gate, kept in ONE place
so they never drift (same rationale as gui_control_model unifying the control parser).

Scope is intentionally narrow: only helpers that were BYTE-EQUIVALENT across gates live here. Helpers that
merely share a NAME but differ in behaviour (e.g. each gate's local `_norm`, which uses a different
character class / squash policy on purpose) are deliberately NOT hoisted — collapsing them would change
parsing behaviour.
"""
import re


def agent_actions(state_view):
    """The portal's agentActions dict, read from a state view that may be the emr root directly OR wrap it
    under `_live_full_state` / `full_state`. Returns {} when absent."""
    if not isinstance(state_view, dict):
        return {}
    if isinstance(state_view.get("agentActions"), dict):
        return state_view["agentActions"]
    for wrap in ("_live_full_state", "full_state"):
        w = state_view.get(wrap)
        if isinstance(w, dict) and isinstance(w.get("agentActions"), dict):
            return w["agentActions"]
    return {}


# ---- billing / clinical code patterns (shared so collection and grounding_audit can't drift) ----
# ICD-10 diagnosis code, e.g. H35.32 / A00
ICD10_RE = re.compile(r"\b[A-TV-Z]\d{2}(?:\.\d{1,4})?\b")
# CPT (5 digits, e.g. 67028) OR HCPCS (letter + 4 digits, e.g. J2778); standalone, not an id fragment
CPT_HCPCS_RE = re.compile(r"(?<![\w\-])(?:\d{5}|[A-Z]\d{4})(?![\w\-])")
