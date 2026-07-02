"""RecoveryAdapter (Commit C6) -- the substrate seam for COMPLETE-effect / ACQUIRE recovery.

The recovery CORE is substrate-agnostic: RecoveryOrchestrator, MutationAuthorization, ActionExecutor,
EvidenceState, EffectCompletionKey, the enforce-gate, the tool budget, action_id. Everything FHIR-specific
about "what did the agent commit to", "does the effect already exist", "what mutation completes it", and
"what identity dedups it" lives behind ONE adapter, so a GUI-portal and a perceptual substrate plug the same
core with a GuiRecoveryAdapter / PerceptualRecoveryAdapter instead of forking the harness.

Interface (the methods the effect-completion driver in run.py calls):
    context(task)                                   -> dict   (substrate refs: subject/case/artifact scope)
    extract_commitments(root_content, trajectory, goal, judge) -> [Commitment]   (agent-origin ONLY)
    effect_key(commitment, context)                 -> EffectCompletionKey        (per-order identity)
    inspect_effect(commitment, driver, context)     -> EffectInspection           (PRESENT|ABSENT|UNKNOWN)
    is_realized(commitment, texts)                  -> bool
    compile_effect(commitment, context, manifest)   -> EffectPlan | None          (the mutation to complete)

compile_evidence_plan(...) (the ACQUIRE prerequisite as a possibly-multi-step plan) is part of the interface
for GUI/perceptual substrates; for FHIR the prerequisite acquisition is already driven by RequiredContext +
AdapterCompiler through the orchestrator, so FhirRecoveryAdapter leaves it to that existing path.
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
    effect_type: str               # substrate effect kind (e.g. a FHIR resource type) for key + reporting
    origin: str = "agent"


@dataclass
class EffectInspection:
    state: str                     # "PRESENT" | "ABSENT" | "UNKNOWN"
    texts: list = field(default_factory=list)
    matched_ids: list = field(default_factory=list)


@dataclass
class EffectPlan:
    mutation_action: dict          # the action the orchestrator authorizes + executes
    scope: dict                    # authorization scope (semantic_type/tool/effect/target_path/postcondition)
    effect_type: str               # resource kind for reporting (may differ from the key's effect_type)
    resource: dict = None


class RecoveryAdapter:
    substrate = None

    def context(self, task):
        raise NotImplementedError

    def extract_commitments(self, root_content, trajectory, goal, judge):
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
        return None                # default: prerequisite acquisition handled by the existing ACQUIRE path


_RECOVERY_TAG = {"system": "https://medical-harness/recovery", "code": "harness-recovery-created"}


class FhirRecoveryAdapter(RecoveryAdapter):
    """Structured-record substrate. Wraps the existing FHIR effect-completion logic UNCHANGED, so the
    still flips F->T -- this is the pure-refactor extraction, not a behavior change."""
    substrate = "fhir"

    def context(self, task):
        refs = dict(context_refs(task) or {})
        refs["task_id"] = str(task.get("id") or task.get("task_id") or "task")
        return refs

    def extract_commitments(self, root_content, trajectory, goal, judge):
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
        insp = driver.inspect_effect(commitment.effect_type, context["subject"])   # #7: through the executor
        return EffectInspection(state=insp.get("state"), texts=(insp.get("texts") or []),
                                matched_ids=(insp.get("matched_ids") or []))

    def is_realized(self, commitment, texts):
        return _fhir_is_realized(commitment.text, texts)

    def compile_effect(self, commitment, context, manifest):
        rtb, resource = build_order_resource({"text": commitment.text, "category": commitment.category}, context)
        if not resource:
            return None
        # HYGIENE: tag recovery-created resources so a paired-run cleanup can DELETE them by tag.
        tag = dict(_RECOVERY_TAG); tag["display"] = "harness_recovery:%s" % context.get("task_id", "task")
        resource.setdefault("meta", {}).setdefault("tag", []).append(tag)
        mact = {"type": "tool_call", "tool": "fhir_create", "args": {"resource": resource}}
        fsem = canonicalize(mact, manifest or {})
        scope = {"allowed_semantic_type": fsem.semantic_type, "allowed_tool": "fhir_create",
                 "allowed_effect": fsem.effect, "target_path": action_target_path(fsem, mact),
                 "expected_postcondition": {"resource": rtb, "status": "active", "verify": "server_readback"}}
        return EffectPlan(mutation_action=mact, scope=scope, effect_type=rtb, resource=resource)


def _get_state_path(state_view, path):
    """Walk a dotted writable_path into the portal emr state_view. A leading 'emr.' is dropped (state_view IS emr)."""
    if not isinstance(state_view, dict):
        return None
    parts = [x for x in path.split(".") if x]
    if parts and parts[0] == "emr":
        parts = parts[1:]
    cur = state_view
    for seg in parts:
        if isinstance(cur, dict) and seg in cur:
            cur = cur[seg]
        else:
            return None
    return cur


class GuiRecoveryAdapter(RecoveryAdapter):
    """Interactive-GUI portal substrate (stateful). COMPLETE-effect target: the agent DECIDED the outcome (a
    disposition it SELECTED and/or a note it TYPED -- reversible actions that already landed in state) but never
    fired the IRREVERSIBLE submit, so the appeal was never submitted. The harness completes ONLY the mechanical
    submit; it never chooses the disposition or writes the note. Effect truth = the portal full_state (emr), read
    as harness-internal authoritative state (NOT gold). Effect paths come from the manifest repair_targets (portal
    schema) -- no field hard-coded in core."""
    substrate = "gui"

    def __init__(self, manifest=None):
        self.manifest = manifest or {}
        root = self.manifest.get("manifest") if isinstance(self.manifest.get("manifest"), dict) else self.manifest
        self._targets = (root.get("repair_targets") or [])

    def _writable_paths(self, keyword):
        for t in self._targets:
            kws = (t.get("match") or {}).get("field_keywords") or []
            if any(keyword in k or k in keyword for k in kws):
                return t.get("writable_paths") or []
        return []

    def context(self, task):
        import re
        goal = task.get("goal") if isinstance(task.get("goal"), str) else ""
        m = re.search(r"([A-Z]{2,5}-\d+)", goal or "")
        return {"case_id": (m.group(1) if m else None),
                "task_id": str(task.get("id") or task.get("task_id") or "task")}

    def extract_commitments(self, root_content, trajectory, goal, judge):
        """Agent-origin ONLY: a disposition the agent SELECTED and/or a note it TYPED implies it decided to submit
        the appeal. The value is the agent's own -- never chosen here."""
        disposition, noted = None, False
        for ev in (trajectory or []):
            if ev.get("event_type") != "tool_call" or ev.get("origin") == "recovery":
                continue
            tool, args = ev.get("tool"), (ev.get("args") or {})
            field = str(args.get("field") or args.get("target") or "").lower()
            if tool == "select" and "disposition" in field:
                disposition = args.get("value")
            elif tool == "type" and any(k in field for k in ("note", "appeal", "rationale", "reason")):
                noted = True
        if disposition is None and not noted:
            return []
        value = str(disposition or "appeal")
        return [Commitment(text="submit appeal (%s)" % value, category="appeal",
                           signature=("submit-appeal:%s" % value).lower(), effect_type="submittedAppeal")]

    def effect_key(self, commitment, context):
        return EffectCompletionKey(str(context.get("case_id")), context.get("artifact_hash", ""),
                                   commitment.signature, commitment.effect_type)

    def inspect_effect(self, commitment, driver, context):
        """PRESENT if the submit effect already landed, ABSENT if state readable but not, UNKNOWN if unreadable.
        The emr state_view is provided by the caller in context['state_view'] (a snapshot recovery read)."""
        state_view = context.get("state_view")
        if not isinstance(state_view, dict):
            return EffectInspection(state="UNKNOWN")
        for path in self._writable_paths("appeal"):
            v = _get_state_path(state_view, path)
            if v:
                return EffectInspection(state="PRESENT", texts=[str(v)])
        return EffectInspection(state="ABSENT")

    def is_realized(self, commitment, texts):
        return bool(texts)   # PRESENT (a nonempty submit-effect value) == already submitted

    def compile_effect(self, commitment, context, manifest):
        """Only the MECHANICAL submit (irreversible). NEVER a disposition/note write."""
        mact = {"type": "tool_call", "tool": "submit", "args": {"target": "submitAppeal"}}
        scope = {"allowed_semantic_type": "submit", "allowed_tool": "submit", "allowed_effect": "irreversible",
                 "target_path": "%s/submitAppeal" % context.get("case_id"),
                 "expected_postcondition": {"paths": self._writable_paths("appeal"),
                                            "verify": "state_readback_nonempty"}}
        return EffectPlan(mutation_action=mact, scope=scope, effect_type=commitment.effect_type)


def get_recovery_adapter(env_type, manifest=None):
    """The recovery adapter for this substrate, or None if recovery is not modelled for it yet."""
    if env_type == "fhir":
        return FhirRecoveryAdapter()
    if env_type == "gui":
        return GuiRecoveryAdapter(manifest)
    return None
