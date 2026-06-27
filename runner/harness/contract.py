"""Clinical Process Contract — the per-task, executable description of a CORRECT PROCESS.

It is NOT a gold answer and NOT a reference trajectory. It states: who the subject is, which evidence
must be obtained, which workflow obligations must hold, which actions are high-risk commit points, and
what must be verified before/after a commit. Produced by the Contract Compiler from task-visible info +
the environment policy pack ONLY (see compiler.py for the benchmark-leak firewall).
"""


class ClinicalProcessContract:
    __slots__ = ("contract_id", "subject", "evidence_obligations", "workflow_obligations",
                 "commit_points", "meta")

    def __init__(self, contract_id, subject=None, evidence_obligations=None, workflow_obligations=None,
                 commit_points=None, meta=None):
        self.contract_id = contract_id
        self.subject = dict(subject) if subject else None        # {"type","id"}
        self.evidence_obligations = list(evidence_obligations or [])   # [{"id","satisfied_by":{...}}]
        self.workflow_obligations = list(workflow_obligations or [])   # [{"id","requires":[...]}]
        self.commit_points = list(commit_points or [])                 # [{"action","risk","requires":[...],"postcondition"}]
        self.meta = dict(meta or {})

    # all obligation ids the contract knows about (evidence + workflow)
    def obligation_ids(self):
        return ([o.get("id") for o in self.evidence_obligations if o.get("id")] +
                [o.get("id") for o in self.workflow_obligations if o.get("id")])

    def _cp_matches(self, cp, sem):
        if "match" not in cp:           # match OMITTED -> a bare commit point (any commit)
            return getattr(sem, "is_commit", lambda: False)()
        m = cp.get("match")
        if not m:                       # explicit empty match {} is invalid (flagged by the loader) -> nothing
            return False
        for field in ("semantic_type", "effect", "resource"):
            if field in m and m[field] != getattr(sem, field, None):
                return False
        return True

    def matching_commit_points(self, sem):
        """ALL commit points that constrain this action (a clinical rule AND the substrate's generic
        invariant can both apply)."""
        return [cp for cp in self.commit_points if self._cp_matches(cp, sem)]

    def commit_point_for(self, sem):
        """The COMPOSED commit point for an action: every matching rule's constraints are MERGED, never
        first-match-wins. risk = max; requires = union; postconditions = ALL (AND-ed by the verifier).
        Returns None when no rule matches."""
        cps = self.matching_commit_points(sem)
        if not cps:
            return None
        order = {"R0": 0, "R1": 1, "R2": 2, "R3": 3}
        risk = max((cp.get("risk", "R2") for cp in cps), key=lambda r: order.get(r, 2))
        requires, posts, rule_ids = [], [], []
        for cp in cps:
            for r in (cp.get("requires") or []):
                if r not in requires:
                    requires.append(r)
            if cp.get("postcondition"):
                posts.append(cp["postcondition"])
            if cp.get("requires_rule"):
                rule_ids.append(cp["requires_rule"])
        return {"risk": risk, "requires": requires, "postconditions": posts,
                "postcondition": (posts[0] if posts else None),   # back-compat single view
                "requires_rule": (rule_ids[0] if rule_ids else None), "rule_ids": rule_ids,
                "match": {"composed_from": len(cps)}}

    @classmethod
    def from_dict(cls, d):
        return cls(contract_id=d.get("contract_id"), subject=d.get("subject"),
                   evidence_obligations=d.get("evidence_obligations"),
                   workflow_obligations=d.get("workflow_obligations"),
                   commit_points=d.get("commit_points"), meta=d.get("meta"))

    def to_dict(self):
        return {"contract_id": self.contract_id, "subject": self.subject,
                "evidence_obligations": self.evidence_obligations,
                "workflow_obligations": self.workflow_obligations,
                "commit_points": self.commit_points, "meta": self.meta}
