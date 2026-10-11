"""Canonical semantics layer — the ADAPTER boundary.

The harness core never sees raw tool names. A per-substrate `manifest` (the adapter's world-description,
declared in the substrate pack) maps a raw action to a SemanticAction:

    semantic_type   read | inspect | create | update | submit | answer | other
    effect          none | reversible | irreversible        (drives risk + commit detection)
    source_class    record | perception | interface | external | computation   (the evidence it yields)
    modality        record | image | administrative | external_text | computation | ...
    resource        optional domain kind (e.g. AllergyIntolerance / MedicationRequest) — adapter vocabulary
    target_entity   the subject id the action operates on   (manifest-declared extraction)
    capability      the raw tool name (kept for AUDIT only; never branched on by the core)

This module is a GENERIC interpreter of the manifest data. No tool name / benchmark name is hard-coded
here — they live in the substrate pack's `manifest:` section. Capabilities consume SemanticAction.
"""

EFFECT_RISK = {"none": "R0", "reversible": "R1", "irreversible": "R2"}


class SemanticAction:
    __slots__ = ("semantic_type", "effect", "source_class", "modality", "resource", "target_entity",
                 "capability", "raw", "mapped", "subject_binding")

    def __init__(self, semantic_type="other", effect="none", source_class=None, modality=None,
                 resource=None, target_entity=None, capability=None, raw=None, mapped=False,
                 subject_binding="implicit_active"):
        self.semantic_type = semantic_type
        self.subject_binding = subject_binding   # required | implicit_active | none (PER-action)
        self.effect = effect
        self.source_class = source_class
        self.modality = modality
        self.resource = resource
        self.target_entity = target_entity
        self.capability = capability
        self.raw = raw
        self.mapped = mapped   # True iff a manifest rule (or default_action) governs this action, or it is the final answer

    def is_commit(self):
        # a final ANSWER is a commit ONLY when the adapter declares it irreversible (e.g. a perceptual
        # diagnosis); a plain "task done" terminal_response in an EHR/GUI is not a commit.
        return self.effect == "irreversible" or self.semantic_type in ("create", "update", "submit")

    def to_dict(self):
        return {"semantic_type": self.semantic_type, "effect": self.effect,
                "source_class": self.source_class, "modality": self.modality, "resource": self.resource,
                "target_entity": self.target_entity, "capability": self.capability,
                "mapped": self.mapped, "subject_binding": self.subject_binding}


def _action_tool(action):
    if not isinstance(action, dict):
        return ""
    if action.get("type") == "final":
        return "final"
    return action.get("tool") or action.get("action") or ""


_CLASSIFIER_CACHE = {}


def _get_classifier(name):
    """Resolve a named action classifier. Registered by name so a manifest stays declarative while the
    predicate itself lives in ONE place -- no second copy of the rule in YAML to drift out of sync."""
    if name in _CLASSIFIER_CACHE:
        return _CLASSIFIER_CACHE[name]
    fn = None
    if name == "hab_terminal_commit":
        try:
            from adapters.hab.gui_control_model import is_terminal_action

            def fn(tool, ref, observation=None):
                # EVIDENCE REQUIRED. is_terminal_action is accurate when it can resolve the control by ref
                # and read its label / commit flag (the path every HAB gate uses, and the one that honours
                # the 'submit to payer/insurance/portal is navigation' exemption). Without controls it
                # falls back to a bare-ref regex whose `submit` alternative also matches lookup-form submits
                # (eligibility-submit-button) and navigation links (submit-authorizations-link). Asserting
                # `irreversible` on that evidence would set ledger.terminal_locked early and BLOCK the rest of
                # the task. So: no controls, no claim.
                if str(tool or "").lower() == "submit":
                    return True                      # a named submit tool needs no page evidence
                controls = None
                if isinstance(observation, dict):
                    c = observation.get("controls")
                    if isinstance(c, list) and c:
                        controls = c
                ref = str(ref or "").strip().strip("[]").strip()   # an agent copies [ref] from PAGE ELEMENTS
                if not controls or not ref:
                    return False
                from adapters.hab.gui_control_model import _NAV_SUBMIT_RE, _TERMINAL_LABEL_RE
                for c in controls:
                    if str(c.get("ref")) != ref:
                        continue
                    label = str(c.get("label") or "")
                    if _NAV_SUBMIT_RE.search(label):
                        return False                 # 'submit to payer/insurance/portal' is navigation
                    if str(c.get("role") or "").lower() in ("a", "link"):
                        return False                 # a hyperlink navigates; the commit is the button it
                                                     # leads to (on the HAB portal, submit-authorizations-link is
                                                     # followed by submit-auth-button).
                    if c.get("commit") is True:
                        return True
                    return bool(_TERMINAL_LABEL_RE.search(label))
                return False                         # ref not in the control list -> unverifiable, no claim
        except Exception:
            fn = None          # unresolvable -> rule cannot match; it never silently matches everything
    _CLASSIFIER_CACHE[name] = fn
    return fn


def _action_ref(action):
    """The element a GUI action targets. Mirrors the arg names the runner accepts for a bracket id."""
    if not isinstance(action, dict):
        return ""
    args = action.get("args")
    if not isinstance(args, dict):
        args = action
    for k in ("ref", "bid", "target", "selector", "element_id"):
        v = args.get(k)
        if v:
            return str(v)
    return ""


def _match_rule(tool, rule_match, action=None, observation=None):
    """A manifest action rule matches by exact `tool`, substring `tool_pattern`, `tool_any` list, or
    `type` (e.g. 'final'). Empty match {} matches nothing (a rule must declare a match).

    Two optional keys further discriminate by the ELEMENT a GUI action targets, because on an
    interactive_gui substrate the tool name alone cannot separate a commit from a page look:
      ref_pattern  regex against args.ref / bid / target / selector / element_id
      classifier   a name registered in _get_classifier, called as fn(tool, ref, observation)
    Both are ANDed with the tool-level match. With action=None (admission-time ambiguity checking, which
    only has tool names) they are skipped so the rule still reports its tool-level candidacy."""
    if not rule_match:
        return False
    declares_tool = bool(rule_match.get("tool") or rule_match.get("tool_pattern")
                         or rule_match.get("tool_any") or rule_match.get("type"))
    tool_ok = False
    if rule_match.get("tool") and rule_match["tool"] == tool:
        tool_ok = True
    pat = rule_match.get("tool_pattern")
    if not tool_ok and pat and pat in tool:
        tool_ok = True
    if not tool_ok:
        for t in (rule_match.get("tool_any") or []):
            if t == tool or (t and t in tool):
                tool_ok = True
                break
    if not tool_ok and rule_match.get("type") and rule_match["type"] == tool:   # 'final' etc.
        tool_ok = True
    if declares_tool and not tool_ok:
        return False

    ref_pat = rule_match.get("ref_pattern")
    clf_name = rule_match.get("classifier")
    if not ref_pat and not clf_name:
        return tool_ok if declares_tool else False
    if action is None:
        return True if declares_tool else False     # admission-time: tool-level candidate

    ref = _action_ref(action)
    if ref_pat:
        import re as _re
        if not _re.search(ref_pat, ref or ""):
            return False
    if clf_name:
        fn = _get_classifier(clf_name)
        if fn is None or not fn(tool, ref, observation):
            return False
    return True


def canonicalize(action, manifest, observation=None):
    """Raw action + substrate manifest -> SemanticAction. Generic: reads the manifest's declared rules."""
    manifest = manifest or {}
    tool = _action_tool(action)
    is_final = isinstance(action, dict) and action.get("type") == "final"

    sem = SemanticAction(capability=tool, raw=action)
    # subject binding default for the substrate; a matching action rule may override it PER-action (e.g. a
    # generic scratchpad write is `none` even though the substrate default is `required`).
    sem.subject_binding = (manifest.get("subject") or {}).get("binding", "implicit_active")
    if is_final:
        # default: a terminal RESPONSE (not a commit). An adapter's final rule may declare it irreversible
        # (effect: irreversible) to make it a real commit (e.g. a perceptual diagnosis). Always mapped.
        sem.semantic_type, sem.effect = "answer", "none"
        sem.mapped = True

    # first matching action rule wins
    matched = False
    for rule in (manifest.get("actions") or []):
        if _match_rule(tool, rule.get("match") or {}, action, observation) or (is_final and (rule.get("match") or {}).get("type") == "final"):
            _apply_rule_body(sem, action, rule)
            sem.mapped = True
            matched = True
            break

    # FAIL-CLOSED: no rule matched. A manifest MAY declare a manifest-level `default_action` (same shape
    # as a rule body) to govern otherwise-unmapped tools; absent that, the action stays mapped=False so a
    # capability can fail closed (escalate) instead of silently allowing an unknown/high-risk tool.
    if not matched and not is_final:
        da = manifest.get("default_action")
        if isinstance(da, dict) and da:
            _apply_rule_body(sem, action, da)
            sem.mapped = True

    sem.target_entity = _extract_subject(action, manifest, observation)
    return sem


def page_evidence_needed(action, manifest):
    """Whether a manifest rule that needs the page's live controls could govern this action: a rule with a `classifier`
    whose tool-level match (the admission-time match, no page) takes the action's tool."""
    tool = _action_tool(action)
    for rule in ((manifest or {}).get("actions") or []):
        m = rule.get("match") or {}
        if m.get("classifier") and _match_rule(tool, m, None):
            return True
    return False


def canonicalize_on_page(action, manifest, controls):
    """canonicalize(action, manifest) with the live control model of the page the action is issued on as a classifier's
    evidence. The shell's one way to classify a GUI commit: the kernel's before_action (its after_action keeps that
    classification), the harness action service, the reconcile write block and the read channels. Without controls it
    is canonicalize(action, manifest); the evidence carries no observation path, so the subject projection is the same."""
    return canonicalize(action, manifest, observation=({"controls": list(controls)} if controls else None))


def _apply_rule_body(sem, action, rule):
    """Apply a manifest action rule body (or default_action) to the SemanticAction."""
    sem.semantic_type = rule.get("semantic_type", sem.semantic_type)
    sem.effect = rule.get("effect", sem.effect)
    pe = rule.get("produces_evidence") or {}
    sem.source_class = pe.get("source_class", sem.source_class)   # only reads that BIND evidence
    sem.modality = pe.get("modality", sem.modality)
    # resource = the action's domain kind (for commit matching), independent of evidence binding
    sem.resource = rule.get("resource") or _resolve_resource(action, rule, pe)
    if "subject_binding" in rule:                # per-action override of the substrate default
        sem.subject_binding = rule["subject_binding"]


def _resolve_resource(action, rule, pe):
    """Domain resource/kind of the action: a literal in the rule's produces_evidence, OR pulled from a
    declared arg (resource_from_args) — adapter vocabulary, never a hard-coded benchmark check. None otherwise."""
    if pe.get("resource"):
        return pe["resource"]
    args = (action or {}).get("args") or {}
    if isinstance(args, dict):
        for k in (pe.get("resource_from_args") or []):
            if args.get(k):
                return str(args[k])
    return None


def _extract_subject(action, manifest, observation):
    """The operated-on subject id, by the manifest's DECLARED extraction (structured args first, then
    declared observation PATHS). Fully data-driven: the core knows no domain field (no 'patient_id',
    no 'page_state') — every path comes from manifest.subject. Dotted paths are supported."""
    subj = manifest.get("subject") or {}
    args = (action or {}).get("args") or {}
    if isinstance(args, dict):
        for entry in (subj.get("from_args") or []):
            v = _resolve_projection(args, entry)        # dotted paths supported (resource.subject.reference)
            if v:
                return v
    if observation and isinstance(observation, dict):
        for entry in (subj.get("from_observation") or []):
            v = _resolve_projection(observation, entry)
            if v:
                return v
    return None


def _resolve_projection(obj, entry):
    """Resolve a subject projection entry against a dict. An entry is either a (dotted) path string, or a
    declared object {path, regex, type}: regex extracts the id from a route/string, type makes it a TYPED
    ref ('admin_case/DEN-001') so a case id is never confused with a patient id of the same number."""
    if isinstance(entry, dict):
        v = _path_get(obj, entry.get("path"))
        if v is None:
            return None
        v = str(v)
        rgx = entry.get("regex")
        if rgx:
            import re
            m = re.search(rgx, v)
            if not m:
                return None
            v = m.group(1) if m.groups() else m.group(0)
        t = entry.get("type")
        return ("%s/%s" % (t, v)) if (t and "/" not in v) else v
    v = _path_get(obj, entry) if "." in str(entry) else (obj.get(entry) if isinstance(obj, dict) else None)
    return str(v) if v else None


def observed_subject(manifest, observation):
    """The subject DISPLAYED in an observation, via the manifest's declared from_observation paths
    (resolved against the RAW environment result — dotted paths supported). Generic: no portal field is
    hard-coded in the core; the adapter declares where its displayed subject lives."""
    if not isinstance(observation, dict):
        return None
    for entry in ((manifest.get("subject") or {}).get("from_observation") or []):
        v = _resolve_projection(observation, entry)
        if v:
            return v
    return None


def _path_get(d, path):
    cur = d
    for part in str(path).split("."):
        cur = cur.get(part) if isinstance(cur, dict) else None
        if cur is None:
            return None
    return cur


def assigned_subject(manifest, goal=None, context=None, observed=None):
    """The ASSIGNED subject (operand) from task-visible info, per the manifest's declared sources, in order:
    a structured context key, an already-observed assignment, then a goal/context-text regex. Reads the
    operand, never a gold answer."""
    subj = manifest.get("subject") or {}
    ctx = context or {}
    for k in (subj.get("id_context_keys") or []):
        if ctx.get(k):
            return str(ctx[k])
    for ev in (observed or []):
        if isinstance(ev, dict) and ev.get("assigned_subject"):
            return str(ev["assigned_subject"])
    rgx = subj.get("from_goal_regex")
    if rgx:
        import re
        hay = " ".join(str(x) for x in (goal, ctx.get("text")) if x)
        m = re.search(rgx, hay)
        if m:
            return m.group(1) if m.groups() else m.group(0)
    return None


def subject_key(ref):
    """A subject reference -> (type or None, id), the form the harness compares subjects in. 'Patient/123' ->
    ('patient', '123'); '123' -> (None, '123'). The MRN identifier prefix is dropped from the id, so
    'Patient/MRN8429141456', 'MRN8429141456' and '8429141456' name the same patient."""
    t = str(ref or "").strip().lower()
    if "/" in t:
        a, b = t.rsplit("/", 1)
    else:
        a, b = None, t
    if b.startswith("mrn"):
        b = b[3:]
    return (a, b)


def same_subject(a, b):
    """Whether two subject references name the same subject: the ids match and, when BOTH carry a type, the types
    match too (Patient/123 != Encounter/123, Patient/123 == 123). The shell's one subject identity: the scope gate,
    evidence scope_relation, obligation satisfaction and required_context all compare through it."""
    (ta, ia), (tb, ib) = subject_key(a), subject_key(b)
    if ia != ib:
        return False
    return not (ta and tb and ta != tb)


def subject_arg_names(manifest):
    """Every argument name through which an action can name its subject on this substrate: the top-level key
    of each from_args entry plus each to_args arg. An action the harness builds may not set any of them
    itself -- the subject is the task's, not the builder's."""
    subj = (manifest or {}).get("subject") or {}
    names = set()
    for e in subj.get("from_args") or []:
        p = e.get("path") if isinstance(e, dict) else e
        if p:
            names.add(str(p).split(".", 1)[0])
    for e in subj.get("to_args") or []:
        if isinstance(e, dict) and e.get("arg"):
            names.add(str(e["arg"]))
    return names


def bind_subject(manifest, tool_params, bare_id):
    """-> (args_patch, None) or (None, problem). Names the assigned subject in an action the harness builds.

    A harness-built ACQUIRE dispatched with `args: {}` would be REVISEd by the scope gate as
    subject_unspecified and the read would never run; this fills the subject argument. `tool_params` is the
    tool's own parameter list (substrate_catalog.tool_parameters), so the argument chosen is one the tool
    accepts and one the scope gate reads back through from_args. No subject binding required by the
    substrate -> an empty patch."""
    subj = (manifest or {}).get("subject") or {}
    if subj.get("binding") != "required":
        return {}, None
    entries = [e for e in (subj.get("to_args") or []) if isinstance(e, dict) and e.get("arg")]
    if not entries:
        return None, "subject_to_args_not_declared"
    if tool_params is None:
        return None, "tool_parameters_unknown"
    for e in entries:
        if e["arg"] in tool_params:
            if not bare_id:
                return None, "no_assigned_subject"
            return {e["arg"]: str(e.get("value") or "{id}").replace("{id}", str(bare_id))}, None
    return None, "tool_takes_no_subject_arg"
