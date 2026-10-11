"""FhirEmptyTargetedSearchBroadReacquire -- an over-specific FHIR search that came back EMPTY is
compensated with one broad patient-level read, so an empty targeted query is not mistaken for absent data.

Part of the default harness. It subclasses Capability and emits through _decide(D.ACQUIRE,
extra={"next_action": ...}) like every other base capability.

FHIR-scoped by construction: it returns None unless the tool name starts with `fhir_` and contains
`_search`, so HealthAdminBench (GUI) and MedCTA (tool sandbox) are untouched.
"""
from workspace_api import Capability, decision as D


class FhirEmptyTargetedSearchBroadReacquire(Capability):
    name = "fhir_empty_targeted_search_broad_reacquire"

    def _text(self, obj):
        try:
            if obj is None:
                return ""
            return str(obj)
        except Exception:
            return ""

    def _tool_name(self, action):
        try:
            if isinstance(action, dict):
                return self._text(action.get("tool") or action.get("name") or action.get("action"))
            return self._text(getattr(action, "tool", None) or getattr(action, "name", None) or action)
        except Exception:
            return ""

    def _args(self, action):
        try:
            if isinstance(action, dict):
                a = action.get("args") or action.get("arguments") or {}
                return a if isinstance(a, dict) else {}
            a = getattr(action, "args", None) or getattr(action, "arguments", None) or {}
            return a if isinstance(a, dict) else {}
        except Exception:
            return {}

    def _empty_result(self, result):
        """Should this result be treated as "the targeted query produced nothing to reconcile"?

        TWO propositions, deliberately separated. `result_is_empty` is the shared FACT -- one reader for
        the whole system, so the semantic judge and this capability can never be told different things
        about the same result. The malformed-filter clause below is this capability's own POLICY: a
        search rejected for an unsupported parameter is not evidence of absence either, and it warrants
        the same broad re-read. Folding it into the shared fact would make `result_empty` claim a tool
        ERROR was an empty result, which it is not -- result_status already says "failed".
        """
        try:
            from workspace_api import result_is_empty
            if result_is_empty(result) is True:
                return True
            return self._malformed_filter_error(result)
        except Exception:
            return False

    def _malformed_filter_error(self, result):
        """POLICY, not emptiness: the environment refused the query shape, so absence is unproven."""
        return (isinstance(result, dict) and "error" in result
                and "unexpected keyword" in self._text(result.get("error")).lower())

    def _intervention_count(self, ctx):
        try:
            ints = ctx.ledger.interventions or []
            n = 0
            for it in ints:
                if self.name in self._text(it):
                    n += 1
            return n
        except Exception:
            return 0

    def after_action(self, action, result, before_state, after_state, ctx):
        try:
            if self._intervention_count(ctx) >= 2:
                return None
            tool = self._tool_name(action)
            low_tool = tool.lower()
            if not low_tool.startswith("fhir_") or "_search" not in low_tool:
                return None
            if not self._empty_result(result):
                return None
            args = self._args(action)
            patient = args.get("patient") or args.get("subject") or args.get("identifier")
            if not patient:
                sid = ctx.ledger.subject_id()
                patient = sid if sid else None
            if not patient:
                return None
            # Only compensate over-specific empty searches; a truly broad empty patient search may be valid absence.
            targeted_keys = set(["code", "type", "category", "date", "status", "clinical_status", "authoredon", "intent"])
            if not any(k in args for k in targeted_keys):
                return None
            broad_args = {"patient": patient, "count": 50}
            # Demographics tools generally use identifier rather than patient.
            if "patient_search_demographics" in low_tool:
                broad_args = {"identifier": patient, "count": 50}
            # For typed/over-filtered note searches, broad notes are the useful fallback.
            fallback_tool = tool
            if "document_reference_search" in low_tool:
                fallback_tool = tool
                broad_args = {"patient": patient, "count": 20}
            na = {"tool": fallback_tool, "args": broad_args, "read_only": True}
            # agent_text is what the agent is told when the broad read runs before its next turn. `feedback` is the
            # decision's message: it is recorded, and the agent never sees it.
            return self._decide(
                D.ACQUIRE, rule_id=self.name, reason_code="empty_targeted_search",
                reason="an over-specific FHIR search returned no entries; acquire the broad patient-level view",
                extra={"next_action": na,
                       "agent_text": "A read-only record (%s) was read for you after that action; its result is "
                                     "in your next observation. Use it before deciding what to do next."
                                     % (fallback_tool,)},
                feedback="The last FHIR search returned no entries after using restrictive filters (such as "
                         "code/type/category/date/status). Before relying on that as missing source context "
                         "or moving downstream, perform this broad patient-level fallback retrieval and "
                         "reconcile the broader source data with the empty targeted query.")
        except Exception:
            return None
