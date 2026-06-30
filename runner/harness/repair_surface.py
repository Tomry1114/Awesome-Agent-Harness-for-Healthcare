"""Repair-surface adapters — the ONLY substrate-specific code in the Scoped Repair layer.

The kernel + delta validator are substrate-agnostic; they speak in PROJECTIONS. An adapter knows how to
turn (state, candidate) into a projection for a finding: the target value, the protected values, and the
navigable root (for reindex-tolerant membership checks). Three adapters map the three repair surfaces:
  FORM   -> portal form fields / note sections (stateful GUI substrate)
  FHIR   -> resource paths / planned action payloads (stateful API substrate)
  ANSWER -> answer claims/findings (single-shot answer substrate)
Path syntax is shared: dotted with optional [i] indices, e.g. 'form.disposition_rationale',
'MedicationRequest.dosageInstruction[0].timing', 'answer.findings[1]'.
"""
from __future__ import annotations
import re


def _split_path(path):
    toks = []
    for part in str(path).split("."):
        m = re.match(r"^([^\[\]]*)((?:\[\d+\])*)$", part)
        if not m:
            toks.append(part)
            continue
        name = m.group(1)
        if name:
            toks.append(name)
        for idx in re.findall(r"\[(\d+)\]", m.group(2) or ""):
            toks.append(int(idx))
    return toks


def _get_toks(root, toks):
    cur = root
    for t in toks:
        if isinstance(t, int):
            if isinstance(cur, list) and 0 <= t < len(cur):
                cur = cur[t]
            else:
                return None
        else:
            if isinstance(cur, dict) and t in cur:
                cur = cur[t]
            else:
                return None
    return cur


def resolve(root, path):
    """get-by-path, with a one-segment namespace fallback so 'form.x' resolves whether `root` is the form
    dict itself or {'form': {...}} (the judge's leading namespace is advisory, not load-bearing)."""
    toks = _split_path(path)
    v = _get_toks(root, toks)
    if v is not None:
        return v
    if len(toks) > 1:
        return _get_toks(root, toks[1:])
    return None


def is_present(v):
    return not (v is None or (isinstance(v, str) and not v.strip())
                or (isinstance(v, (list, dict)) and len(v) == 0))


class RepairSurface:
    """Base adapter. project() yields the projection the delta validator consumes. Subclasses only override
    where the repair target lives (state vs candidate)."""
    name = "generic"
    env_types = ()

    def root(self, state, candidate):
        return state if isinstance(state, dict) else {}

    def project(self, state, candidate, finding):
        r = self.root(state, candidate)
        return {"target": resolve(r, finding.target_path),
                "protected": {p: resolve(r, p) for p in finding.protected_paths},
                "root": r}

    def diff(self, before, after):
        return {"target_changed": before.get("target") != after.get("target"),
                "protected_changed": {p: before.get("protected", {}).get(p) != after.get("protected", {}).get(p)
                                      for p in (before.get("protected") or {})}}

    def can_localize(self, state, candidate, finding):
        """ADMISSIBILITY INVARIANT: a finding is emittable only if its target is addressable in the CURRENT
        surface — the target resolves, OR (for an additive defect) its parent container resolves so there is
        a real place to add. A finding whose path resolves NOWHERE is a hallucinated target -> inadmissible,
        dropped (no churn). Substrate-agnostic: each adapter inherits this; only `root` differs."""
        root = self.root(state, candidate)
        if resolve(root, finding.target_path) is not None:
            return True
        op = getattr(finding.operation, "value", str(finding.operation))
        additive = op in ("ADD",) or finding.defect_type in ("missing", "insufficient_content", "unobserved_target")
        if additive:
            path = finding.target_path
            parent = path.rsplit(".", 1)[0] if "." in path else None
            parent = parent.rsplit("[", 1)[0] if parent and parent.endswith("]") else parent
            if parent and resolve(root, parent) is not None:
                return True
            if "." not in path and "[" not in path and isinstance(root, dict):
                return True   # a top-level new key on a real object
        return False


class FormRepairSurface(RepairSurface):     # HAB: portal form fields / note sections / submission fields
    name = "form"
    env_types = ("gui",)


class FhirRepairSurface(RepairSurface):     # FHIR resource path / planned action payload
    name = "fhir"
    env_types = ("fhir",)

    def root(self, state, candidate):
        """PRE-COMMIT, the editable object is the PROPOSED action payload (candidate), NOT the mutable-state
        digest (which only holds already-created resources / deliverables). Expose BOTH under explicit keys so
        findings can target the candidate the agent is about to submit (candidate.args.*) and protect existing
        state. When there is no candidate payload (delta unit tests), fall back to state for back-compat."""
        if isinstance(candidate, dict) and candidate:
            return {"candidate": candidate, "state": state if isinstance(state, dict) else {}}
        return state if isinstance(state, dict) else {}


class AnswerRepairSurface(RepairSurface):   # claims / observations / evidence spans in a single-shot answer
    name = "answer"
    env_types = ("tool_sandbox",)

    def root(self, state, candidate):
        # the answer is the candidate (before_final), not env state
        if isinstance(candidate, dict):
            return candidate
        return state if isinstance(state, dict) else {}


def path_space(root, max_paths=80):
    """All addressable leaf+container paths actually present in the state. Given to the judge so it picks a
    REAL target_path instead of hallucinating one (the verified churn cause: judge guessed emr.denials.DEN-014
    while the real output path was agentActions.selectedDisposition)."""
    out = []

    def walk(o, p):
        if len(out) >= max_paths:
            return
        if isinstance(o, dict):
            for k, v in o.items():
                np = ("%s.%s" % (p, k)) if p else str(k)
                out.append(np)
                walk(v, np)
        elif isinstance(o, list):
            for i, v in enumerate(o[:6]):
                np = "%s[%d]" % (p, i)
                out.append(np)
                walk(v, np)
    walk(root if isinstance(root, dict) else {}, "")
    return out[:max_paths]


def target_sig(projection):
    """The finding-relevant slice of a projection (target + protected) — used for dedup/'did the agent act on
    THIS finding' comparisons. Excludes `root`, whose every-step churn otherwise defeats dedup."""
    return {"target": projection.get("target"), "protected": projection.get("protected")}


_SURFACES = (FormRepairSurface, FhirRepairSurface, AnswerRepairSurface)


def surface_for(env_type):
    for cls in _SURFACES:
        if env_type in cls.env_types:
            return cls()
    return RepairSurface()
