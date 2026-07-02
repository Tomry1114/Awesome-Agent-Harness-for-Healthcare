"""RecoveryAdapter (Commit C6, HAB-hardened H0) -- the substrate seam for COMPLETE-effect / ACQUIRE recovery.

The recovery CORE is substrate-agnostic: RecoveryOrchestrator, MutationAuthorization, ActionExecutor,
EvidenceState, EffectCompletionKey, the enforce-gate, the tool budget, action_id. Everything substrate-specific
about "what did the agent commit to", "does the effect already exist", "what mutation completes it", and "what
identity dedups it" lives behind ONE adapter, so a GUI-portal and a perceptual substrate plug the same core with
their own adapter instead of forking the harness.

Interface:
    should_trigger(lifecycle_event)                 -> bool   (WHEN this substrate's completion fires)
    context(task)                                   -> dict   (substrate refs: subject/case/artifact scope)
    extract_commitments(root_content, trajectory, goal, judge, context) -> [Commitment]  (agent-origin ONLY)
    effect_key(commitment, context)                 -> EffectCompletionKey
    inspect_effect(commitment, driver, context)     -> EffectInspection   (PRESENT|ABSENT|UNKNOWN)
    is_realized(commitment, texts)                  -> bool
    compile_effect(commitment, context, manifest)   -> EffectPlan | None
No benchmark names.
"""
from dataclasses import dataclass, field

from .recovery_orchestrator import EffectCompletionKey
from .effect_completion import (context_refs, resource_type_for_category, build_order_resource)
from .effect_reconciliation import is_realized as _fhir_is_realized
from .engines.semantic import extract_committed_orders
from .semantics import canonicalize
from .authorization import action_target_path


@dataclass
class Commitment:
    """An action the AGENT committed to (in its deliverable/plan/trajectory) but may not have executed."""
    text: str
    category: str
    signature: str                 # normalized identity for per-order dedup (feeds EffectCompletionKey)
    effect_type: str               # substrate effect kind (e.g. a FHIR resource type / "appeal_submission")
    origin: str = "agent"
    target_entity: str = None      # the subject/case the effect binds to
    payload: dict = None           # structured landed values (e.g. {disposition, note}) -- agent's own, verified
    origin_action_ids: list = field(default_factory=list)   # the agent action(s) that formed this commitment


@dataclass
class EffectInspection:
    state: str                     # "PRESENT" | "ABSENT" | "UNKNOWN"
    texts: list = field(default_factory=list)
    matched_ids: list = field(default_factory=list)


@dataclass
class EffectPlan:
    scope: dict                    # authorization scope (semantic_type/tool/effect/target_path/postcondition)
    effect_type: str               # kind for reporting
    mutation_action: dict = None   # the commit action (may be None until an affordance is resolved at execution)
    resource: dict = None
    prepare_actions: list = field(default_factory=list)   # read-only steps (e.g. [snapshot]) run before commit
    commit_affordance: dict = None # {tool, match:{labels,role}} for the driver to resolve to a concrete ref


class RecoveryAdapter:
    substrate = None

    def should_trigger(self, lifecycle_event):
        return lifecycle_event == "deliverable_confirmed"

    def context(self, task):
        raise NotImplementedError

    def extract_commitments(self, root_content, trajectory, goal, judge, context=None):
        raise NotImplementedError

    def effect_key(self, commitment, context):
        raise NotImplementedError

    def inspect_effect(self, commitment, driver, context):
        raise NotImplementedError

    def is_realized(self, commitment, texts):
        raise NotImplementedError

    def compile_effect(self, commitment, context, manifest):
        raise NotImplementedError

    def compile_evidence_plan(self, unit, target, observation):
        return None


_RECOVERY_TAG = {"system": "https://medical-harness/recovery", "code": "harness-recovery-created"}


class FhirRecoveryAdapter(RecoveryAdapter):
    """Structured-record substrate. Wraps the existing FHIR effect-completion logic UNCHANGED, so the cp3
    flip still holds -- this is the pure-refactor extraction, not a behavior change."""
    substrate = "fhir"

    def should_trigger(self, lifecycle_event):
        return lifecycle_event == "deliverable_confirmed"

    def context(self, task):
        refs = dict(context_refs(task) or {})
        refs["task_id"] = str(task.get("id") or task.get("task_id") or "task")
        return refs

    def extract_commitments(self, root_content, trajectory, goal, judge, context=None):
        out = []
        for u in (extract_committed_orders(root_content, goal, judge) or []):
            text = u.get("text"); cat = u.get("category") or "other"
            out.append(Commitment(text=text, category=cat,
                                  signature=(text or "")[:120].strip().lower(),
                                  effect_type=resource_type_for_category(cat)))
        return out

    def effect_key(self, commitment, context):
        return EffectCompletionKey(context["subject"], context["artifact_hash"],
                                   commitment.signature, commitment.effect_type)

    def inspect_effect(self, commitment, driver, context):
        insp = driver.inspect_effect(commitment.effect_type, context["subject"])
        return EffectInspection(state=insp.get("state"), texts=(insp.get("texts") or []),
                                matched_ids=(insp.get("matched_ids") or []))

    def is_realized(self, commitment, texts):
        return _fhir_is_realized(commitment.text, texts)

    def compile_effect(self, commitment, context, manifest):
        rtb, resource = build_order_resource({"text": commitment.text, "category": commitment.category}, context)
        if not resource:
            return None
        tag = dict(_RECOVERY_TAG); tag["display"] = "harness_recovery:%s" % context.get("task_id", "task")
        resource.setdefault("meta", {}).setdefault("tag", []).append(tag)
        mact = {"type": "tool_call", "tool": "fhir_create", "args": {"resource": resource}}
        fsem = canonicalize(mact, manifest or {})
        scope = {"allowed_semantic_type": fsem.semantic_type, "allowed_tool": "fhir_create",
                 "allowed_effect": fsem.effect, "target_path": action_target_path(fsem, mact),
                 "expected_postcondition": {"resource": rtb, "status": "active", "verify": "server_readback"}}
        return EffectPlan(scope=scope, effect_type=rtb, mutation_action=mact, resource=resource)


def _get_state_path(state_view, path):
    """Walk a dotted writable_path into the portal emr state_view. A leading 'emr.' is dropped (state_view IS emr)."""
    if not isinstance(state_view, dict):
        return None
    parts = [x for x in str(path or "").split(".") if x]
    if parts and parts[0] == "emr":
        parts = parts[1:]
    cur = state_view
    for seg in parts:
        if isinstance(cur, dict) and seg in cur:
            cur = cur[seg]
        else:
            return None
    return cur


def _active_case(state_view):
    """The case currently displayed in the portal state (several declared shapes), or None if indeterminate."""
    if not isinstance(state_view, dict):
        return None
    for p in ("caseId", "case_id"):
        if state_view.get(p):
            return str(state_view[p])
    for holder in ("fields", "signals"):
        h = state_view.get(holder) or {}
        for p in ("caseId", "case_id"):
            if isinstance(h, dict) and h.get(p):
                return str(h[p])
    return None


class GuiRecoveryAdapter(RecoveryAdapter):
    """Interactive-GUI portal substrate (stateful). COMPLETE-effect target: the agent DECIDED the outcome (a
    submit-allowed disposition it SELECTED, plus any note it TYPED -- reversible actions that ALREADY LANDED in
    the authoritative EMR state) but never fired the IRREVERSIBLE submit. The harness completes ONLY the
    mechanical submit; it NEVER chooses the disposition or writes the note.

    H0 conservatism -- a commitment forms ONLY when ALL hold:
      1) the agent explicitly selected a submit-allowed disposition (do_not_appeal etc. -> no commitment);
      2) that EXACT disposition has landed in emr.agentActions.selectedDisposition;
      3) if the agent typed a note, that note has landed (an attempted-but-unlanded edit -> no commitment);
      4) the commit marker emr.appealActions.submittedAppeal is currently False;
      5) the active displayed case == the assigned case;
      6) every value comes from an agent action, never harness-generated.
    Effect truth = the commit marker EXACT-TRUE only (never any_path_nonempty; an attachment is not a submit)."""
    substrate = "gui"

    def __init__(self, manifest=None):
        self.manifest = manifest or {}
        root = self.manifest.get("manifest") if isinstance(self.manifest.get("manifest"), dict) else self.manifest
        effs = (root.get("recovery_effects") or [{}])
        e = effs[0] or {}
        self.effect_type = e.get("effect_type", "appeal_submission")
        self.commit_marker = e.get("commit_marker", "emr.appealActions.submittedAppeal")
        self.disposition_path = e.get("disposition_path", "emr.agentActions.selectedDisposition")
        self.note_path = e.get("note_path", "emr.agentActions.addedAuthNote")
        self.allowed_dispositions = set(e.get("allowed_dispositions") or ["appeal", "dispute", "file_appeal"])
        self.submit_labels = e.get("submit_labels") or ["Submit Appeal", "File Appeal", "Submit"]

    def should_trigger(self, lifecycle_event):
        # HAB fires at the agent-done boundary: all reversible edits are settled, and we never race the agent's
        # own submit. (No benchmark branch in run.py -- the adapter declares its trigger.)
        return lifecycle_event == "before_final"

    def context(self, task):
        import re
        goal = task.get("goal") if isinstance(task.get("goal"), str) else ""
        m = re.search(r"([A-Z]{2,5}-\d+)", goal or "")
        return {"case_id": (m.group(1) if m else None),
                "task_id": str(task.get("id") or task.get("task_id") or "task")}

    def extract_commitments(self, root_content, trajectory, goal, judge, context=None):
        ctx = context or {}
        state_view = ctx.get("state_view")
        assigned = ctx.get("case_id")
        if not isinstance(state_view, dict) or not assigned:
            return []                                  # cannot validate against authoritative state -> fail-closed
        # (5) active displayed case must equal the assigned case
        active = _active_case(state_view)
        if not active or active != str(assigned):
            return []
        # (4) not already submitted
        if _get_state_path(state_view, self.commit_marker):
            return []
        # scan agent actions for the disposition the agent SELECTED + a note it TYPED (with action ids)
        sel_val, sel_id, typed_note, type_id = None, None, False, None
        for ev in (trajectory or []):
            if ev.get("event_type") != "tool_call" or ev.get("origin") == "recovery":
                continue
            tool, args = ev.get("tool"), (ev.get("args") or {})
            field = str(args.get("field") or args.get("target") or "").lower()
            if tool == "select" and "disposition" in field:
                sel_val = args.get("value"); sel_id = ev.get("action_id") or ev.get("step")
            elif tool == "type" and any(k in field for k in ("note", "appeal", "rationale", "reason")):
                typed_note = True; type_id = ev.get("action_id") or ev.get("step")
        # (1) an explicit submit-allowed disposition
        if sel_val is None or str(sel_val).lower() not in self.allowed_dispositions:
            return []
        # (2) that EXACT disposition landed in authoritative state
        landed_disp = _get_state_path(state_view, self.disposition_path)
        if landed_disp is None or str(landed_disp).lower() != str(sel_val).lower():
            return []
        # (3) if the agent typed a note, it must have landed (attempted-but-unlanded -> block)
        landed_note = _get_state_path(state_view, self.note_path)
        if typed_note and not (landed_note and str(landed_note).strip()):
            return []
        ids = [x for x in (sel_id, type_id) if x is not None]
        payload = {"disposition": str(landed_disp), "note": (str(landed_note) if landed_note else None)}
        sig = ("appeal-submit:%s:%s" % (assigned, str(landed_disp))).lower()
        return [Commitment(text="submit appeal (%s)" % landed_disp, category="appeal", signature=sig,
                           effect_type=self.effect_type, target_entity=str(assigned),
                           payload=payload, origin_action_ids=ids)]

    def effect_key(self, commitment, context):
        return EffectCompletionKey(str(commitment.target_entity or context.get("case_id")),
                                   context.get("artifact_hash", ""), commitment.signature, commitment.effect_type)

    def inspect_effect(self, commitment, driver, context):
        """Effect truth = the commit marker ONLY: EXACT-True -> PRESENT, False -> ABSENT, missing/unreadable ->
        UNKNOWN. Never any_path_nonempty (an attachment is not a completed submit)."""
        state_view = (context or {}).get("state_view")
        if not isinstance(state_view, dict):
            return EffectInspection(state="UNKNOWN")
        # case consistency is a precondition for trusting the marker
        active = _active_case(state_view)
        if commitment.target_entity and active and str(active) != str(commitment.target_entity):
            return EffectInspection(state="UNKNOWN")
        v = _get_state_path(state_view, self.commit_marker)
        if v is True:
            return EffectInspection(state="PRESENT", texts=["submitted"])
        if v is False:
            return EffectInspection(state="ABSENT")
        return EffectInspection(state="UNKNOWN")       # missing / non-boolean -> cannot tell

    def is_realized(self, commitment, texts):
        return bool(texts)

    def compile_effect(self, commitment, context, manifest):
        """A PLAN, not a static click: snapshot the page, then submit the UNIQUELY-resolved affordance (by
        label/role) -- the driver resolves the ref from the snapshot; ambiguous/absent -> the driver blocks.
        NEVER a select/type (never chooses disposition or writes note)."""
        scope = {"allowed_semantic_type": "submit", "allowed_tool": "submit", "allowed_effect": "irreversible",
                 "target_path": "%s/submitAppeal" % (commitment.target_entity or context.get("case_id")),
                 "expected_postcondition": {"path": self.commit_marker, "equals": True,
                                            "verify": "state_marker_false_to_true"}}
        return EffectPlan(scope=scope, effect_type=commitment.effect_type,
                          prepare_actions=[{"type": "tool_call", "tool": "snapshot", "args": {}}],
                          commit_affordance={"tool": "submit", "match": {"labels": self.submit_labels, "role": "button"}})


def get_recovery_adapter(env_type, manifest=None):
    """The recovery adapter for this substrate, or None if recovery is not modelled for it yet."""
    if env_type == "fhir":
        return FhirRecoveryAdapter()
    if env_type == "gui":
        return GuiRecoveryAdapter(manifest)
    return None
