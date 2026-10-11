"""Policy engine — composes the effective policy from THREE separated layers (none of which the harness
core knows the names of):

  ENVIRONMENT ADAPTER  adapters/<name>.yaml          raw tool -> canonical semantics (manifest) + which
                                                     substrate this env is + which clinical modules apply.
                                                     The ONLY layer that names tools / observation fields.
  SUBSTRATE POLICY     policies/substrate/<name>.yaml generic invariants for the whole class of environment
                                                     (no tool names, no domain resources).
  CLINICAL MODULES     policies/clinical/<name>.yaml  reusable domain rules (medication safety, evidence
                                                     grounding, ...), activated per-adapter; scoped to the
                                                     resources/modalities they name. Fixed shell.
  WORKSPACE POLICIES   <workspace>/<manifest policies> editable clinical rules the run's workspace carries
                                                     (chart review), composed after the adapter's modules
                                                     under the same checks; may not reuse a module name.

`env_type` selects a DEFAULT adapter; a second dataset on the same substrate passes `adapter=` — different
tool names, SAME substrate + clinical modules, zero core change.
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
    for ext in (".yaml", ".yml", ".json"):
        if os.path.exists(base + ext):
            return _read_file(base + ext)
    return None, False


def _read_file(p):
    """(doc | None, parse_error: bool) for one existing file."""
    try:
        doc = _load_json(p) if p.endswith(".json") else _load_yaml(p)
    except Exception:
        return None, True
    if doc is None:
        return None, True                      # present but unparseable (e.g. PyYAML missing) -> error
    return doc, False


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
            # q is the EARLIER rule (first-match wins). Flag ONLY when q is a substring of p
            # (q is more GENERIC than the later p) with different semantics -> the generic q wrongly
            # shadows the later specific p (misordered), or q == p (identical patterns disagree).
            # When p is a substring of q (the earlier rule is the more SPECIFIC one), first-match
            # correctly picks the specific rule first -> intentional specialization, NOT ambiguous.
            if (q in p) and (st, ef) != (st2, ef2):
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
        if len(matched) <= 1:
            continue
        # runtime uses FIRST-match, so the first matched rule is authoritative -- specific-before-
        # generic is deterministic, NOT a guess. Flag only if a LATER matched rule is strictly MORE
        # specific than the first (the first's tool_pattern is contained in the later's) with different
        # semantics -> a specific rule is shadowed by an earlier generic one (misordering).
        def _pat(r):
            m = r.get("match") or {}
            return m.get("tool_pattern") or m.get("tool") or ""
        first = matched[0]
        fp, fsem = _pat(first), (first.get("semantic_type"), first.get("effect"))
        for r in matched[1:]:
            rp = _pat(r)
            if fp and rp and fp in rp and fp != rp and (r.get("semantic_type"), r.get("effect")) != fsem:
                errs.append("ambiguous_action_mapping:%s" % name)
                break
    return errs


def substrate_of(env_type):
    return _SUBSTRATE_BY_ENV.get(env_type)


def load_policy(adapter=None, substrate=None, env_type=None, *, workspace):
    """Compose ADAPTER (manifest) + SUBSTRATE (generic invariants) + CLINICAL modules + the WORKSPACE's policy
    modules into one policy. `workspace` is required: the run's workspace path, or None for no workspace policy modules
    (no default is resolved here). Collects any composition errors into policy['_errors'] (build_kernel raises on them
    in assist/enforce)."""
    errors = []
    adapter_name = adapter or _DEFAULT_ADAPTER_BY_ENV.get(env_type)
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
    # Workspace policy modules come after the shell's. A commit point is merged with every other one that
    # matches (Contract.commit_point_for: risk max, requires union), so a workspace module can add to a shell
    # rule but not take anything away from it.
    ws_path = workspace
    ws_files = []
    if ws_path:
        try:
            from workspace import Workspace
            ws_files = Workspace(ws_path).policy_files()
        except Exception as e:
            errors.append("workspace_unloadable:%s" % e)
    for p in ws_files:
        wd, werr = _read_file(p) if os.path.exists(p) else (None, False)
        if wd is None:
            errors.append("workspace_policy_parse_error:%s" % p if werr else "workspace_policy_not_found:%s" % p)
        name = (wd or {}).get("name") or os.path.splitext(os.path.basename(p))[0]
        if name in clinical_names:
            errors.append("workspace_policy_reuses_module_name:%s" % name)
        clinical_names.append(name)
        cdocs.append(wd or {})

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
