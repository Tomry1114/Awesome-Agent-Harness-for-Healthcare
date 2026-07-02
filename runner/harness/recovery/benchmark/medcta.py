"""Bounded Clinical Recovery v3 - benchmark adapter for the perceptual clinical-reasoning benchmark (MCTA).

Task/field/lifecycle/state-path normalization for the EVIDENCE-acquisition path. This is a BENCHMARK-adapter
layer: it holds task-shape knowledge (how to read the public question, the available perception tools, the
agent's own answer + its own prior observations), and it derives - ORACLE-BLIND - the committed AnswerSlot and
the evidence gap. It reads only the agent's OWN output/trajectory and the PUBLIC question; never gold, never a
checkpoint, never a reference trace.

resolve_commitments emits ONE CommittedGoal (goal_type='acquire_evidence') iff the agent answered a perceptual
question WITHOUT looking - i.e. a perceptual claim in its answer is not covered by any actually-executed
observation. The target REGION for that goal is derived from the PUBLIC QUESTION only, via elicit_discriminator
over the agent's answer + question (never from the answer's own asserted region). A null region is carried
through so the substrate can correctly refuse an unlocatable target (BLOCKED_AMBIGUOUS_TARGET). No perceptual
claim / already-covered -> empty list -> the kernel reports DECLINED_NO_COMMITMENT (a correct non-engagement).

Python 3.8 compatible.
"""
from ..contracts import CommittedGoal
from ...observation import Observation, region_observed
from ...engines.semantic import decompose_claims, elicit_discriminator


# Perceptual milestones this adapter can normalize. Operational fields the workflow/substrate whitelist so
# they may bind from system_metadata / authoritative_state (never a clinical value).
_OPERATIONAL_FIELDS = ("evidence_acquired", "region_localized", "observation_id", "evidence_id",
                       "asset_id", "image_instance")
_SEMANTIC_FIELDS = ("finding", "diagnosis", "answer_choice", "confidence")


class MedctaBenchmarkAdapter(object):
    """BenchmarkAdapter for the perceptual clinical-reasoning tasks (evidence-acquisition path)."""

    name = "perceptual_clinical_reasoning"
    env_type = "tool_sandbox"

    def __init__(self, judge_fn=None):
        # judge_fn is the INDEPENDENT judge used for oracle-blind claim decomposition + discriminator
        # elicitation (region from the public question). None -> conservative (no gap derivable -> DECLINE).
        self.judge_fn = judge_fn

    # -- BenchmarkAdapter surface -----------------------------------------------------------------
    def context(self, task):
        text = ((task.get("context") or {}).get("text")) or task.get("goal") or ""
        tools = task.get("available_tools") or []
        return {
            "goal": text,
            "public_context": text,
            "observation": {"available_tools": tools, "question": text},
            "authoritative_state": {"evidence_ledger": []},
            "system_metadata": {},
            "bound_evidence": {},
            "schema": {"operational_fields": list(_OPERATIONAL_FIELDS),
                       "semantic_fields": list(_SEMANTIC_FIELDS)},
            "task_id": task.get("task_id") or "t",
        }

    def resolve_commitments(self, root, trajectory, goal, judge, ctx):
        answer = self._answer(root, trajectory, ctx)
        if not answer:
            return []
        question = (ctx or {}).get("goal") or self._question(root, ctx)
        task_id = (ctx or {}).get("task_id") or "t"
        jf = self.judge_fn or judge

        prior = self._trajectory_observations(trajectory)
        prior_obs = self._as_observations(prior)

        # oracle-blind decomposition of the agent's OWN answer into typed claims
        claims = decompose_claims(answer, question, jf, task_id)
        perceptual = [c for c in claims if c.claim_type == "perceptual"]
        uncovered = [c for c in perceptual if region_observed(c, prior_obs) is None]

        # discriminator region derived from the PUBLIC QUESTION + answer only (never the answer's own region)
        disc = elicit_discriminator(answer, question,
                                    [o.get("content", "") for o in prior], jf) or {}
        slot = self._answer_slot(answer, jf)
        if isinstance(ctx, dict):
            ctx["discriminator"] = disc
            ctx["answer_slot"] = slot

        # evidence gap = answered-without-looking: a perceptual claim uncovered by any executed observation,
        # OR a perceptual answer with no image observation at all while the discriminator names a checkable
        # feature. No perceptual claim + no discriminator target -> no gap -> DECLINE.
        gap = bool(uncovered) or (bool(perceptual) and not prior_obs)
        if not gap and disc.get("region") and not prior_obs:
            gap = True
        if not gap:
            return []

        region = disc.get("region")
        if region is None and uncovered:
            region = uncovered[0].region
        modality = disc.get("modality") or (uncovered[0].modality if uncovered else None)
        attribute = disc.get("attribute") or (uncovered[0].attribute if uncovered else None)

        cf = {
            "answer_slot": slot,
            "region": region, "modality": modality, "attribute": attribute,
            "uncovered_claims": [c.text for c in uncovered],
            "hypotheses": list(disc.get("hypotheses") or []),
        }
        return [CommittedGoal(
            goal_id="evgap:%s" % task_id, goal_type="acquire_evidence",
            committed_fields=cf, dedup_key="evgap:%s" % task_id,
            provenance="agent_commitment",
            raw={"answer": answer, "question": question})]

    def should_trigger(self, lifecycle_event):
        """Evidence recovery engages when the agent is ABOUT to finalize its answer."""
        ev = lifecycle_event
        if isinstance(ev, dict):
            ev = ev.get("event") or ev.get("lifecycle_event") or ev.get("name")
        return str(ev) == "before_final"

    def state_path(self, logical_name):
        return {
            "evidence_ledger": "evidence_ledger",
            "evidence_acquired": "evidence_acquired",
            "region_localized": "region_localized",
            "answer_slot": "answer_slot",
        }.get(logical_name, logical_name)

    # -- oracle-blind extraction helpers ----------------------------------------------------------
    def _answer(self, root, trajectory, ctx):
        for src in (trajectory, root, ctx):
            a = self._find_answer(src)
            if a:
                return a
        return None

    @staticmethod
    def _find_answer(src):
        if not src:
            return None
        if isinstance(src, str):
            return src
        if isinstance(src, dict):
            for k in ("final_answer", "answer", "deliverable", "response", "output"):
                v = src.get(k)
                if isinstance(v, dict):
                    return v
                if v:
                    return v
        return None

    @staticmethod
    def _question(root, ctx):
        if isinstance(root, dict):
            t = (root.get("context") or {}).get("text")
            if t:
                return t
            if root.get("goal"):
                return root.get("goal")
        return (ctx or {}).get("public_context") or ""

    def _answer_slot(self, answer, judge_fn):
        """Extract the committed AnswerSlot {finding, diagnosis, answer_choice, confidence} from the agent's
        OWN answer. A structured answer is read directly; a free-text answer maps to a diagnosis slot. Never
        infers a hidden reference."""
        if isinstance(answer, dict):
            return {"finding": answer.get("finding"),
                    "diagnosis": answer.get("diagnosis") or answer.get("conclusion"),
                    "answer_choice": answer.get("answer_choice") or answer.get("choice"),
                    "confidence": answer.get("confidence")}
        return {"finding": None, "diagnosis": str(answer)[:500], "answer_choice": None, "confidence": None}

    @staticmethod
    def _trajectory_observations(trajectory):
        """The agent's OWN executed observations (read from its trajectory), oracle-blind."""
        if not trajectory:
            return []
        obs = None
        if isinstance(trajectory, dict):
            obs = trajectory.get("observations") or trajectory.get("ledger")
        elif isinstance(trajectory, (list, tuple)):
            obs = trajectory
        out = []
        for o in (obs or []):
            if isinstance(o, dict):
                out.append(o)
        return out

    @staticmethod
    def _as_observations(records):
        out = []
        for o in records:
            out.append(Observation(
                observation_id=o.get("observation_id") or o.get("id"),
                tool_capability=o.get("tool_capability") or o.get("tool"),
                subject=o.get("subject"), region=o.get("region"), modality=o.get("modality"),
                attributes_observed=tuple(o.get("attributes_observed") or []),
                result_status=o.get("result_status", "valid"), content=o.get("content", "")))
        return out
