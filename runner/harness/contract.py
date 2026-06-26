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

    def commit_point_for(self, action_name):
        for cp in self.commit_points:
            if cp.get("action") == action_name:
                return cp
        return None

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
