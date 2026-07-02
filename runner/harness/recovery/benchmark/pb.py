"""Bounded Clinical Recovery v3 - structured-record clinical-task Benchmark Adapter (Layer 4).

PbBenchmarkAdapter normalizes a structured-record clinical deliverable task into the recovery vocabulary:

    context()             - resolve the PUBLIC subject/requester/authoredOn (via effect_completion.context_refs),
                            split them into the correct binding classes (subject/requester = SEMANTIC ->
                            authoritative_state; authoredOn/idempotency/tag = OPERATIONAL -> system_metadata),
                            and attach a binding schema.
    resolve_commitments() - oracle-blind: read the AGENT's own deliverable and extract the orders it FIRMLY
                            committed to (engines.semantic.extract_committed_orders already DROPS hedged /
                            conditional / deferred items). One CommittedGoal per firm order -> N episodes.
    should_trigger()      - engage on the 'deliverable_confirmed' lifecycle event.
    state_path()          - map a logical state name to a concrete read-back path.

Nothing here is read from gold/checkpoint. The clinical decision (which order) comes ENTIRELY from the agent's
text via the injected judge; on no-judge / parse-fail the extractor returns [] (nothing is auto-completed).
"""
import datetime

from ..contracts import CommittedGoal
from ...effect_completion import context_refs, build_order_resource, resource_type_for_category
from ...engines.semantic import extract_committed_orders

_GOAL_TYPE = "complete_committed_structured_effect"
# the decision-boundary args the kernel must resolve before the commit (generic categories,
# not clinical names): the committed content + who/what + the operational stamps.
_REQUIRED = ["code_text", "subject", "requester", "authoredOn", "idempotency_key", "recovery_tag"]
_GOVERNANCE_RT = "AllergyIntolerance"


def _now_iso():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


class PbBenchmarkAdapter(object):
    """Structured-record clinical-task normalization (one dataset family; no rule changes)."""

    def context(self, task):
        task = task or {}
        refs = context_refs(task) or {}          # {subject, authoredOn, requester} or {} (no subject)
        subject = refs.get("subject")

        authoritative_state = {}
        if subject:
            authoritative_state["subject"] = subject
        if refs.get("requester"):
            authoritative_state["requester"] = refs.get("requester")

        system_metadata = {
            "authoredOn": refs.get("authoredOn") or _now_iso(),
            "idempotency_key": "bcr-idem:%s" % (subject or "unbound"),
            "recovery_tag": "bcr_v3",
        }
        schema = {
            "semantic_fields": ["code_text", "subject", "requester"],
            "operational_fields": ["authoredOn", "idempotency_key", "recovery_tag"],
        }
        deliverable = self._deliverable(task)
        return {
            "schema": schema,
            "authoritative_state": authoritative_state,
            "system_metadata": system_metadata,
            "bound_evidence": {},
            "observation": None,
            "refs": {"subject": subject, "requester": refs.get("requester"),
                     "authoredOn": system_metadata["authoredOn"]},
            "deliverable": deliverable,
            "goal_text": task.get("goal"),
        }

    def resolve_commitments(self, root, trajectory, goal, judge, ctx):
        ctx = ctx or {}
        content = ctx.get("deliverable") or self._deliverable(root) or self._from_trajectory(trajectory)
        goal_text = ctx.get("goal_text") or goal or (isinstance(root, dict) and root.get("goal")) or ""
        judge_fn = judge or ctx.get("judge")
        orders = extract_committed_orders(content, goal_text, judge_fn=judge_fn) or []
        auth_state = ctx.get("authoritative_state") or {}
        sys_meta = ctx.get("system_metadata") or {}
        subject = auth_state.get("subject")
        requester = auth_state.get("requester")
        authored_on = sys_meta.get("authoredOn")
        out = []
        for i, o in enumerate(orders):
            if not isinstance(o, dict):
                continue
            txt = str(o.get("text") or "").strip()
            if not txt:
                continue
            category = str(o.get("category") or "other")
            # Benchmark-specific record shaping lives in the ADAPTER (data), NOT the workflow. Build the
            # structured payload from the committed order + authoritative refs; None payload (unresolved
            # subject/text) -> the kernel's decision-boundary gate BLOCKS before any commit.
            rt, resource = build_order_resource(
                {"text": txt, "category": category},
                {"subject": subject, "requester": requester, "authoredOn": authored_on})
            if rt is None:
                rt = resource_type_for_category(category)
                resource = None
            effect_spec = {
                "operation": "create",
                "resource_type": rt,
                "payload": resource,
                "identity": {"subject": subject},
                "required_bindings": list(_REQUIRED),
                "precondition_reads": [{"resourceType": _GOVERNANCE_RT, "subject": subject}],
                "postcondition": {"resourceType": rt, "subject": subject, "match_text": txt},
            }
            out.append(CommittedGoal(
                goal_id="effect-%d" % i,
                goal_type=_GOAL_TYPE,
                committed_fields={"code_text": txt, "category": category},
                dedup_key="effect:%s" % txt.lower()[:80],
                provenance="agent_commitment",
                raw={"effect_spec": effect_spec, "order": o}))
        return out

    def should_trigger(self, lifecycle_event):
        return lifecycle_event == "deliverable_confirmed"

    def state_path(self, logical_name):
        return {
            "order_effect": "record_store.orders",
            "governance": "record_store.allergies",
        }.get(logical_name, "record_store.%s" % logical_name)

    # -- helpers ---------------------------------------------------------------------------------
    @staticmethod
    def _deliverable(obj):
        if not isinstance(obj, dict):
            return ""
        for k in ("deliverable", "content", "final_answer", "answer", "response"):
            v = obj.get(k)
            if v:
                return v
        c = obj.get("context")
        if isinstance(c, dict) and c.get("deliverable"):
            return c["deliverable"]
        return ""

    @staticmethod
    def _from_trajectory(trajectory):
        if isinstance(trajectory, list):
            for ev in reversed(trajectory):
                if isinstance(ev, dict) and ev.get("content"):
                    return ev["content"]
        return ""
