"""Scoped Repair primitives — a localized, minimal, non-degrading repair finding shared by ALL substrates.

A RepairFinding names exactly ONE concrete defect: WHERE it is (target_path), WHAT is wrong (defect_type),
the SMALLEST fix (operation + required_change), and what must NOT be touched (protected_paths /
preserve_requirements). The harness never says "write a triage note"; it says "field X lacks Y; ADD Y;
preserve A,B,C". This is the unit the kernel's finding lifecycle + delta validation operate on, IDENTICALLY
for form fields, FHIR resource paths, and answer claims alike. See HARNESS_DESIGN.md (Scoped Repair layer).
"""
from __future__ import annotations
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RepairOperation(str, Enum):
    ADD = "ADD"
    EDIT = "EDIT"
    REMOVE = "REMOVE"
    REPLACE = "REPLACE"
    VERIFY = "VERIFY"
    REACQUIRE_EVIDENCE = "REACQUIRE_EVIDENCE"


# defect categories accepted from the judge / deterministic layer
DEFECT_TYPES = ("missing", "insufficient_content", "unsupported", "conflicting", "wrong_operation")
TARGET_TYPES = ("field", "resource_path", "claim", "action")


def make_finding_id(task_id, rule_id, target_type, target_path, defect_type) -> str:
    """Stable id: same task + same target + same defect => same finding id, forever. This is what lets the
    lifecycle suppress duplicate prompting instead of re-nagging the same obligation every step."""
    raw = "|".join([str(task_id), str(rule_id), str(target_type), str(target_path), str(defect_type)])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class RepairFinding:
    finding_id: str
    rule_id: str
    target_type: str            # field | resource_path | claim | action
    target_path: str
    defect_type: str            # missing | insufficient_content | unsupported | conflicting | wrong_operation
    operation: RepairOperation
    required_change: str
    protected_paths: tuple = ()
    preserve_requirements: tuple = ()
    evidence_refs: tuple = ()
    confidence: float | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"finding_id": self.finding_id, "rule_id": self.rule_id, "target_type": self.target_type,
                "target_path": self.target_path, "defect_type": self.defect_type,
                "operation": self.operation.value if isinstance(self.operation, RepairOperation) else str(self.operation),
                "required_change": self.required_change, "protected_paths": list(self.protected_paths),
                "preserve_requirements": list(self.preserve_requirements),
                "evidence_refs": list(self.evidence_refs), "confidence": self.confidence}


def _coerce_op(v):
    try:
        return RepairOperation(str(v).strip().upper())
    except Exception:
        return None


def parse_findings(raw, task_id, rule_id="scoped_repair"):
    """Judge JSON dict -> [RepairFinding]. A finding is KEPT only if it names a concrete target_path, a valid
    operation, a recognized defect_type, AND a required_change. This is the spec's hard rule: if the judge
    cannot localize the target and the change, NO finding is emitted (no vague 'write a triage note' REVISE).
    aligned=True / empty / unparseable -> [] (stay silent)."""
    findings = []
    if not isinstance(raw, dict) or raw.get("aligned") is True:
        return findings
    for f in (raw.get("findings") or []):
        if not isinstance(f, dict):
            continue
        tpath = str(f.get("target_path") or "").strip()
        change = str(f.get("required_change") or "").strip()
        op = _coerce_op(f.get("repair_operation") or f.get("operation"))
        dtype = str(f.get("defect_type") or "").strip().lower()
        ttype = str(f.get("target_type") or "field").strip().lower()
        if not tpath or not change or op is None or dtype not in DEFECT_TYPES:
            continue       # incomplete / non-localized finding -> drop
        if ttype not in TARGET_TYPES:
            ttype = "field"
        conf = f.get("confidence")
        findings.append(RepairFinding(
            finding_id=make_finding_id(task_id, rule_id, ttype, tpath, dtype),
            rule_id=rule_id, target_type=ttype, target_path=tpath, defect_type=dtype,
            operation=op, required_change=change,
            protected_paths=tuple(str(x) for x in (f.get("protected_paths") or [])),
            preserve_requirements=tuple(str(x) for x in (f.get("preserve_requirements") or [])),
            evidence_refs=tuple(str(x) for x in (f.get("evidence_refs") or [])),
            confidence=(float(conf) if isinstance(conf, (int, float)) else None)))
    return findings
