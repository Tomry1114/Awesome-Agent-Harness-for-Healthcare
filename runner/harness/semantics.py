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
        return self.effect == "irreversible" or self.semantic_type in ("create", "update", "submit", "answer")

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


def _match_rule(tool, rule_match):
    """A manifest action rule matches by exact `tool`, substring `tool_pattern`, `tool_any` list, or
    `type` (e.g. 'final'). Empty match {} matches nothing (a rule must declare a match)."""
    if not rule_match:
        return False
    if rule_match.get("tool") and rule_match["tool"] == tool:
        return True
    pat = rule_match.get("tool_pattern")
    if pat and pat in tool:
        return True
    for t in (rule_match.get("tool_any") or []):
        if t == tool or (t and t in tool):
            return True
    if rule_match.get("type") and rule_match["type"] == tool:   # 'final' etc.
        return True
    return False


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
        # the final answer is the answer commit; manifest may override effect/modality. It is always mapped.
        sem.semantic_type, sem.effect = "answer", "irreversible"
        sem.mapped = True

    # first matching action rule wins
    matched = False
    for rule in (manifest.get("actions") or []):
        if _match_rule(tool, rule.get("match") or {}) or (is_final and (rule.get("match") or {}).get("type") == "final"):
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
    """Domain resource/kind of the action: a literal in the rule, OR pulled from a declared arg, OR the
    tool-name suffix — adapter vocabulary, never a hard-coded benchmark check."""
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
        for k in (subj.get("from_args") or []):
            if args.get(k):
                return str(args[k])
    if observation and isinstance(observation, dict):
        for path in (subj.get("from_observation") or []):
            v = _path_get(observation, path)
            if v:
                return str(v)
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
