"""Module A — Clinical Scope & Evidence Binding.

Addresses "did the right thing, but to the WRONG subject or from the WRONG evidence". Maintains the
active subject + the evidence ledger; blocks actions that operate on a foreign subject and binds
observed evidence to the subject it came from. P0: deterministic subject-scope check driven by the
policy pack's `subject_arg_keys` (which action arg holds the operated-on id). Richer per-dataset
evidence binding (FHIR resource.subject / case id / image region) is layered in P1–P3.
"""
from ..capability import Capability
from .. import decision as D


class ScopeEvidenceBinding(Capability):
    name = "scope_evidence"

    def before_action(self, action, ctx):
        active = ctx.ledger.subject_id()
        if not active:
            return None                                   # no assigned subject yet -> nothing to enforce
        target = self._action_target(action, ctx.policy)
        if target is not None and _norm(target) != _norm(active):
            return self._decide(
                D.BLOCK, rule_id="subject_scope_mismatch", deterministic=True,
                reason="action operates on %s but the active subject is %s" % (target, active),
                feedback="This action targets %s; the assigned subject is %s. Operate only on %s."
                         % (target, active, active),
                extra={"target": target, "active_subject": active})
        return None

    def after_action(self, action, result, before_state, after_state, ctx):
        active = ctx.ledger.subject_id()
        # OBSERVATION-based scope (GUI / route substrates like HAB): the case actually DISPLAYED after this
        # action, read from canonical_observation via the SAME structured keys the scorer uses
        # (case_identity / patient_id / subject_token). A foreign displayed case -> REVISE (go back).
        if active and ctx.observation and ctx.policy.get("subject_observation_keys"):
            shown = _displayed_subject(ctx.observation, ctx.policy["subject_observation_keys"])
            if shown is not None and _norm(shown) != _norm(active):
                return self._decide(
                    D.REVISE, rule_id="subject_scope_mismatch", deterministic=True,
                    reason="page now shows %s but the assigned subject is %s" % (shown, active),
                    feedback="You are viewing %s; the assigned case is %s — return to it." % (shown, active),
                    extra={"shown": shown, "active_subject": active})
        # bind read-derived evidence (perception/search outputs) to the active subject. Bind even when
        # there is no subject id (e.g. MedCTA single-image tasks) -> subject_id=None; the evidence is the
        # perception tool output the grounding / semantic checks rely on.
        name = action.get("tool") if isinstance(action, dict) else None
        if name and _looks_read(name, ctx.policy):
            ctx.ledger.add_evidence(type=name, value=_summarize(result), subject_id=active,
                                    source_event="step-%d" % ctx.step, source_type=ctx.env_type)
        return None

    def _action_target(self, action, policy):
        """Which subject id this action operates on, read from its args via policy.subject_arg_keys."""
        if not isinstance(action, dict):
            return None
        args = action.get("args") or {}
        if not isinstance(args, dict):
            return None
        for k in (policy.get("subject_arg_keys") or []):
            v = args.get(k)
            if v:
                return str(v)
        return None


def _displayed_subject(observation, keys):
    """The subject currently DISPLAYED, from canonical_observation. Structured-first: top-level keys
    (case_identity / patient_id / subject_token), then page_state.<section>.patient_id — mirrors the
    scorer's _page_disclosed_subjects so the harness and the scorer read the SAME signal."""
    if not isinstance(observation, dict):
        return None
    for k in keys:
        v = observation.get(k)
        if v:
            return str(v)
    ps = observation.get("page_state")
    if isinstance(ps, dict):
        for sect in ps.values():
            if isinstance(sect, dict) and sect.get("patient_id"):
                return str(sect["patient_id"])
    return None


def _looks_read(name, policy):
    reads = set(policy.get("read_actions", []))
    if name in reads:
        return True
    low = (name or "").lower()
    return any(h in low for h in ("search", "read", "get", "view", "ocr", "describe", "lookup"))


def _norm(x):
    return str(x or "").strip().lower().split("/")[-1]


def _summarize(result):
    s = result if isinstance(result, str) else str(result)
    return s[:200]
