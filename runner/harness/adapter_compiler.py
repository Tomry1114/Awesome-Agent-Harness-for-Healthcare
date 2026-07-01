"""AdapterCompiler (Commit A) -- compile an ABSTRACT evidence request into a REAL, adapter-specific tool call.

Core reasons in abstract units ("read AllergyIntolerance for this subject"); it must NOT guess the tool name
or the query parameter from the resource name. The adapter manifest declares, per evidence_unit, exactly which
tool + which subject parameter (patient vs subject) + how the subject id is shaped (typed reference vs bare id)
+ what result-semantics classify the result. This is what fixes the live `AllergyIntolerance?subject=` -> HTTP
400 bug: the compiler emits `?patient=Patient/<id>` because the adapter says so.

No benchmark names, no checkpoint knowledge. A different substrate ships a different manifest; core is unchanged.
"""
from .evidence_state import EvidenceRequest


def _manifest_body(manifest):
    m = manifest or {}
    return m.get("manifest") if isinstance(m.get("manifest"), dict) else m


def evidence_affordance(manifest, evidence_unit):
    """The adapter-declared affordance dict for an evidence unit, or None if the adapter does not expose one."""
    for a in (_manifest_body(manifest).get("evidence_affordances") or []):
        if isinstance(a, dict) and a.get("evidence_unit") == evidence_unit:
            return a
    return None


def _subject_type_prefix(manifest):
    t = str((_manifest_body(manifest).get("subject") or {}).get("type") or "").strip()
    return (t[:1].upper() + t[1:]) if t else "Patient"


def format_subject(manifest, target_entity, style):
    """Shape the subject id for a query. `typed_ref` -> 'Patient/<id>'; `bare_id` -> '<id>'. Accepts a target
    that is already typed ('Patient/123') or bare ('123')."""
    s = str(target_entity or "").strip()
    if not s:
        return s
    bare = s.rsplit("/", 1)[-1]
    if style == "bare_id":
        return bare
    if "/" in s:                     # already typed
        return s
    return "%s/%s" % (_subject_type_prefix(manifest), bare)


def compile_evidence_request(manifest, evidence_unit, target_entity, obligation_id=None):
    """(manifest, evidence_unit, target_entity) -> EvidenceRequest with a concrete affordance {tool, args,
    read_only} + expected_result_semantics. Returns None if the adapter declares no affordance for the unit or
    there is no subject to bind (core never fabricates a query)."""
    aff = evidence_affordance(manifest, evidence_unit)
    if not aff or not str(target_entity or "").strip():
        return None
    args = dict(aff.get("static_args") or {})
    subj_arg = aff.get("subject_arg")
    if subj_arg:
        args[subj_arg] = format_subject(manifest, target_entity, aff.get("subject_ref_style", "typed_ref"))
    return EvidenceRequest(
        obligation_id=obligation_id, target_entity=str(target_entity), evidence_unit=evidence_unit,
        affordance={"tool": aff.get("tool"), "args": args, "read_only": True},
        query=dict(args), expected_result_semantics=dict(aff.get("result_semantics") or {}))


def result_semantics_for(manifest, evidence_unit, default=None):
    """The adapter's result-semantics for classifying an evidence_unit's read (for EvidenceState). Falls back
    to `default` (a generic collection spec) when the unit is not individually declared."""
    aff = evidence_affordance(manifest, evidence_unit)
    if aff and aff.get("result_semantics"):
        return dict(aff["result_semantics"])
    return dict(default or {"collection_paths": ["entries"], "absence_when_empty": True})
