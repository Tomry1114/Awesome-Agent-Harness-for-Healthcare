"""Bounded Clinical Recovery v3 - GenericStructuredEffectCompletionWorkflow (Layer 3).

ONE generic compiler for a committed-but-unrealized STRUCTURED effect (a record the ROOT AGENT decided to
create/update/send/write but never performed). It knows NO business names -- no "order", no "pelvic
ultrasound", no "medication", no resource type by name. All task specifics arrive as a runtime EFFECT SPEC
that the Benchmark Adapter compiles from the agent's committed effect (goal.raw["effect_spec"]):

    {
      "operation":          "create" | "update" | "send" | "write",   # what to do
      "resource_type":      "<opaque string>",                          # what kind of record
      "payload":            {...} | None,                               # the built record (None -> BLOCK gate)
      "identity":           {"subject": "<ref>"},                       # WHO/WHAT it is about
      "required_bindings":  [ ...arg names... ],                        # decision-boundary gate (kernel BLOCKS if unbound)
      "precondition_reads": [ {"resourceType": "...", "subject": "..."} ],  # generic governance/prereq reads
      "postcondition":      {"resourceType": "...", "subject": "...", "match_text": "..."}  # verify target
    }

The compiled plan is the SAME bounded shape for every structured effect:
    [ precondition_read* ] -> existing-effect PROBE -> ONE irreversible commit -> verify (server read-back).

Operation-driven: the substrate maps op=create/update/send/write to its own primitive; an op the substrate
does not implement returns an honest failure. The clinical decision lives ENTIRELY in the spec (data);
this module is benchmark-name-free and substrate-agnostic. Python 3.8 compatible.
"""
from ..contracts import (
    RecoveryStep, Plan,
    READ, IRREVERSIBLE_COMMIT, VERIFY,
)
# Structured-record text inspection (generic: reads the search read-back's entries/code text; no business
# names). Used only to decide PRESENT/ABSENT/UNKNOWN for the postcondition match.
from ...effect_completion import classify_effect_inspection

_READBACK_KEY = "_bcr_effect_readback"

GOAL_TYPE = "complete_committed_structured_effect"


class GenericStructuredEffectCompletionWorkflow(object):
    """Compile + verify the completion of a committed structured effect from a runtime effect spec."""

    goal_type = GOAL_TYPE

    def _spec(self, goal):
        return (getattr(goal, "raw", None) or {}).get("effect_spec") or {}

    def match_goal(self, goal, ctx):
        gt = getattr(goal, "goal_type", "") or ""
        if gt == GOAL_TYPE:
            return True
        return bool(self._spec(goal))            # tolerate a spec-carrying goal with a legacy type

    def required_bindings(self, goal, ctx):
        # the decision-boundary gate: the Benchmark Adapter declares which args MUST resolve before commit;
        # an unbound SEMANTIC arg -> the kernel BLOCKS (a correct refusal), never a FAILED commit.
        return list(self._spec(goal).get("required_bindings") or [])

    def compile_plan(self, goal, ctx):
        spec = self._spec(goal)
        op = spec.get("operation", "create")
        rt = spec.get("resource_type")
        payload = spec.get("payload")
        subject = (spec.get("identity") or {}).get("subject")
        pc = spec.get("postcondition") or {}
        match_text = pc.get("match_text")

        steps = []
        # 0) generic precondition/governance reads (read-only; harmless if the list is empty).
        for pr in (spec.get("precondition_reads") or []):
            steps.append(RecoveryStep(
                kind=READ, name="precondition_read",
                action={"op": "search", "resourceType": pr.get("resourceType"),
                        "subject": pr.get("subject", subject)}))
        # 1) existing-effect PROBE: if already realized -> kernel short-circuits to ALREADY_REALIZED.
        steps.append(RecoveryStep(
            kind=READ, name="existing_effect_probe", probe=True,
            action={"op": "search", "resourceType": rt, "subject": subject, "match_text": match_text}))
        # 2) exactly ONE irreversible commit; the operation is data, not a hard-coded verb.
        steps.append(RecoveryStep(
            kind=IRREVERSIBLE_COMMIT, name="commit_effect",
            action={"op": op, "resourceType": rt, "resource": payload},
            manifest={"side_effect_scope": "%s_%s" % (op, rt), "server_persisted": True,
                      "rollback_available": False, "autosave_possible": False}))
        # 3) verify by server read-back.
        steps.append(RecoveryStep(
            kind=VERIFY, name="server_read_back",
            action={"op": "search", "resourceType": rt, "subject": subject, "match_text": match_text}))

        postcondition = {"paths": [{"key": _READBACK_KEY, "resourceType": rt,
                                    "subject": subject, "match_text": match_text}]}
        return Plan(steps=steps, required_bindings=[], expected_postcondition=postcondition)

    def verify_effect(self, goal, state_view):
        """True if the committed effect's record is present in the read-back, False if refuted, None if the
        read-back is ambiguous (-> idempotent reconciliation / UNKNOWN). Postcondition is DATA from the spec."""
        state_view = state_view or {}
        raw = state_view.get(_READBACK_KEY)
        if raw is None:
            return None
        insp = classify_effect_inspection(raw)
        st = insp.get("state")
        if st == "UNKNOWN":
            return None
        match_text = str(((self._spec(goal).get("postcondition") or {}).get("match_text")) or "").strip().lower()
        if not match_text:
            # no comparable text -> presence decides: PRESENT -> realized, ABSENT -> not.
            if st == "PRESENT":
                return True
            if st == "ABSENT":
                return False
            return None
        for t in (insp.get("texts") or []):
            tt = str(t or "").strip().lower()
            if tt and (match_text[:40] in tt or tt[:40] in match_text):
                return True
        return False
