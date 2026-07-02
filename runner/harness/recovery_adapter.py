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


def get_recovery_adapter(env_type):
    """The recovery adapter for this substrate, or None if recovery is not modelled for it yet."""
    if env_type == "fhir":
        return FhirRecoveryAdapter()
    return None
