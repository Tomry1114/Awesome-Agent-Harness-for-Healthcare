"""Policy engine — composes the effective policy from THREE separated layers (none of which the harness
core knows the names of):

  ENVIRONMENT ADAPTER  adapters/<name>.yaml          raw tool -> canonical semantics (manifest) + which
                                                     substrate this env is + which clinical modules apply.
                                                     The ONLY layer that names tools / observation fields.
  SUBSTRATE POLICY     policies/substrate/<name>.yaml generic invariants for the whole class of environment
                                                     (no tool names, no domain resources).
  CLINICAL MODULES     policies/clinical/<name>.yaml  reusable domain rules (medication safety, evidence
                                                     grounding, ...), activated per-adapter; scoped to the
                                                     resources/modalities they name.

`env_type` selects a DEFAULT adapter; a second dataset on the same substrate passes `adapter=` (or
MH_HARNESS_ADAPTER) — different tool names, SAME substrate + clinical modules, zero core change.
"""
import os, json


class PolicyError(Exception):
    """Raised (in assist/enforce) when the composed policy is incomplete/invalid — a missing adapter,
    substrate, clinical module, parse failure, or a dangling obligation reference. Fail-closed: a typo'd
    `clinical_modules: [medication_saftey]` must NOT silently drop the medication rules."""


_RUNNER_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ADAPTER_DIR = os.path.join(_RUNNER_DIR, "adapters")
_SUBSTRATE_DIR = os.path.join(_RUNNER_DIR, "policies", "substrate")
_CLINICAL_DIR = os.path.join(_RUNNER_DIR, "policies", "clinical")

# env_type -> DEFAULT adapter (and the substrate it implies). env_type is a convenience selector, NOT an
# identity: override the adapter to run another dataset of the same class.
_DEFAULT_ADAPTER_BY_ENV = {"fhir": "hapi_fhir", "gui": "admin_portal", "tool_sandbox": "image_tool_sandbox"}
_SUBSTRATE_BY_ENV = {"fhir": "structured_record", "gui": "interactive_gui", "tool_sandbox": "perceptual"}


def _read(dirpath, name):
    """Returns (doc | None, parse_error: bool). None doc = file not found (distinct from a parse error)."""
    if not name:
        return None, False
    base = os.path.join(dirpath, name)
    for ext, loader in ((".yaml", _load_yaml), (".yml", _load_yaml), (".json", _load_json)):
        p = base + ext
        if os.path.exists(p):
            try:
                doc = loader(p)
                if doc is None:
                    return None, True          # present but unparseable (e.g. PyYAML missing) -> error
                return doc, False
            except Exception:
                return None, True
    return None, False


def _load_yaml(path):
    try:
        import yaml
    except Exception:
        return None
    with open(path) as f:
        return yaml.safe_load(f)


def _load_json(path):
    with open(path) as f:
        return json.load(f)


_SEMANTIC_TYPES = {"read", "inspect", "create", "update", "submit", "answer", "other"}
_EFFECTS = {"none", "reversible", "irreversible"}
_BINDINGS = {"required", "implicit_active", "none"}
_SOURCE_CLASSES = {"record", "perception", "interface", "external", "computation"}
_PREDICATE_TYPES = {"state_transition", "object_exists", "field_equals", "field_not_equals",
                    "no_unexpected_side_effect", "target_consistency", "claim_supported_by_evidence"}


def _validate_enums(manifest, errors):
    """Strict enum validation: a typo'd effect/semantic_type/binding/source_class would otherwise be
    accepted verbatim by the canonicalizer and silently mis-classify risk/scope. Fail it at load."""
    subj = manifest.get("subject") or {}
    if subj.get("binding") and subj["binding"] not in _BINDINGS:
        errors.append("invalid_subject_binding:%s" % subj["binding"])
    rules = manifest.get("actions") or []
    pats = []
    for r in rules:
        m = r.get("match")
        if not m:
            errors.append("action_rule_empty_match")           # a rule MUST declare a non-empty match
        if r.get("semantic_type") and r["semantic_type"] not in _SEMANTIC_TYPES:
            errors.append("invalid_semantic_type:%s" % r["semantic_type"])
        if r.get("effect") and r["effect"] not in _EFFECTS:
            errors.append("invalid_effect:%s" % r["effect"])
        if r.get("subject_binding") and r["subject_binding"] not in _BINDINGS:
            errors.append("invalid_subject_binding:%s" % r["subject_binding"])
        pe = r.get("produces_evidence") or {}
        if pe.get("source_class") and pe["source_class"] not in _SOURCE_CLASSES:
            errors.append("invalid_source_class:%s" % pe["source_class"])
        p = (m or {}).get("tool_pattern")
        if p:
            pats.append((p, r.get("semantic_type"), r.get("effect")))
    # static ambiguity audit: one tool_pattern is a substring of another with DIFFERENT semantics -> the
    # first-match order silently decides; flag it (the maintainer should reorder or disambiguate).
    for i, (p, st, ef) in enumerate(pats):
        for q, st2, ef2 in pats[:i]:
            if (p in q or q in p) and (st, ef) != (st2, ef2):
                errors.append("ambiguous_action_rules:%s~%s" % (q, p))


def admission_errors(manifest, tools):
    """Dynamic admission against the task's ACTUAL available tools (stronger than the static substring
    audit): a tool that matches MORE THAN ONE action rule with DIFFERENT semantics is ambiguous — first-
    match order would silently decide its risk. Fail the build rather than guess."""
    from ..semantics import _match_rule
    errs = []
    rules = (manifest or {}).get("actions") or []
    for t in (tools or []):
        name = t if isinstance(t, str) else (t.get("name") if isinstance(t, dict) else "")
        if not name:
            continue
        matched = [r for r in rules if _match_rule(name, r.get("match") or {})]
        sems = {(r.get("semantic_type"), r.get("effect")) for r in matched}
        if len(matched) > 1 and len(sems) > 1:
            errs.append("ambiguous_action_mapping:%s" % name)
    return errs


def substrate_of(env_type):
    return _SUBSTRATE_BY_ENV.get(env_type)


def load_policy(adapter=None, substrate=None, env_type=None):
    """Compose ADAPTER (manifest) + SUBSTRATE (generic invariants) + CLINICAL modules into one policy.
    Collects any composition errors into policy['_errors'] (build_kernel raises on them in assist/enforce)."""
    errors = []
    adapter_name = adapter or os.environ.get("MH_HARNESS_ADAPTER") or _DEFAULT_ADAPTER_BY_ENV.get(env_type)
    adoc, aerr = _read(_ADAPTER_DIR, adapter_name)
    if adapter_name and adoc is None:
        errors.append("adapter_parse_error:%s" % adapter_name if aerr else "adapter_not_found:%s" % adapter_name)
    adoc = adoc or {}
    sub_name = adoc.get("substrate") or substrate or _SUBSTRATE_BY_ENV.get(env_type)
    sdoc, serr = _read(_SUBSTRATE_DIR, sub_name)
    if sub_name and sdoc is None:
        errors.append("substrate_parse_error:%s" % sub_name if serr else "substrate_not_found:%s" % sub_name)
    sdoc = sdoc or {}
    clinical_names = list(adoc.get("clinical_modules") or [])
    cdocs = []
    for c in clinical_names:
        cd, cerr = _read(_CLINICAL_DIR, c)
        if cd is None:
            errors.append("clinical_parse_error:%s" % c if cerr else "clinical_module_not_found:%s" % c)
        cdocs.append(cd or {})

    def _concat(key, docs):
        out = []
        for d in docs:
            out.extend(d.get(key) or [])
        return out

    manifest = adoc.get("manifest", {})
    _validate_enums(manifest, errors)
    ev = _concat("evidence_obligations", [sdoc] + cdocs)
    wf = _concat("workflow_obligations", [sdoc] + cdocs)
    # clinical (specific) commit points FIRST -> commit_point_for matches them before the substrate's
    # generic invariant (e.g. {create, MedicationRequest} wins over the bare {effect: irreversible}).
    commits = _concat("commit_points", cdocs) + (sdoc.get("commit_points") or [])

    # structural validation: empty manifest, duplicate obligation ids, dangling obligation references.
    if adapter_name and not (manifest.get("actions")):
        errors.append("empty_manifest_actions:%s" % adapter_name)
    ids = [o.get("id") for o in (ev + wf) if o.get("id")]
    if len(ids) != len(set(ids)):
        errors.append("duplicate_obligation_ids")
    known = set(ids)
    for o in wf:
        for r in (o.get("requires") or []):
            if r not in known:
                errors.append("workflow_requires_unknown_obligation:%s" % r)
    for cp in commits:
        for r in (cp.get("requires") or []):
            if r not in known:
                errors.append("commit_requires_unknown_obligation:%s" % r)
        if cp.get("match") == {}:        # an explicit empty match is almost always a typo (matches any commit)
            errors.append("overly_broad_commit_match")
        post = cp.get("postcondition")
        pt = (post.get("type") if isinstance(post, dict) else post)
        if pt and pt not in _PREDICATE_TYPES:    # a postcondition the predicate registry can't evaluate
            errors.append("invalid_postcondition_type:%s" % pt)

    return {
        "manifest": manifest, "evidence_obligations": ev, "workflow_obligations": wf,
        "commit_points": commits,
        "_adapter": adapter_name, "_substrate": sub_name, "_clinical_modules": clinical_names,
        "_errors": errors,
    }
