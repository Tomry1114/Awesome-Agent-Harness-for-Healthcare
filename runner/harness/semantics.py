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
                 "capability", "raw")

    def __init__(self, semantic_type="other", effect="none", source_class=None, modality=None,
                 resource=None, target_entity=None, capability=None, raw=None):
        self.semantic_type = semantic_type
        self.effect = effect
        self.source_class = source_class
        self.modality = modality
        self.resource = resource
        self.target_entity = target_entity
        self.capability = capability
        self.raw = raw

    def is_commit(self):
        return self.effect == "irreversible" or self.semantic_type in ("create", "update", "submit", "answer")

    def to_dict(self):
        return {"semantic_type": self.semantic_type, "effect": self.effect,
                "source_class": self.source_class, "modality": self.modality, "resource": self.resource,
                "target_entity": self.target_entity, "capability": self.capability}


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
    if is_final:
        # the final answer is the answer commit; manifest may override effect/modality.
        sem.semantic_type, sem.effect = "answer", "irreversible"

    # first matching action rule wins
    for rule in (manifest.get("actions") or []):
        if _match_rule(tool, rule.get("match") or {}) or (is_final and (rule.get("match") or {}).get("type") == "final"):
            sem.semantic_type = rule.get("semantic_type", sem.semantic_type)
            sem.effect = rule.get("effect", sem.effect)
            pe = rule.get("produces_evidence") or {}
            sem.source_class = pe.get("source_class", sem.source_class)   # only reads that BIND evidence
            sem.modality = pe.get("modality", sem.modality)
            # resource = the action's domain kind (for commit matching), independent of evidence binding
            sem.resource = rule.get("resource") or _resolve_resource(action, rule, pe)
            break

    sem.target_entity = _extract_subject(action, manifest, observation)
    return sem


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
    """The operated-on subject id, by the manifest's declared extraction (structured args first, then the
    displayed/observation subject). Pure data-driven; the core never parses ids itself."""
    subj = manifest.get("subject") or {}
    args = (action or {}).get("args") or {}
    if isinstance(args, dict):
        for k in (subj.get("from_args") or []):
            if args.get(k):
                return str(args[k])
    if observation and subj.get("from_observation"):
        if isinstance(observation, dict):
            for k in subj["from_observation"]:
                if observation.get(k):
                    return str(observation[k])
            ps = observation.get("page_state")
            if isinstance(ps, dict):
                for sect in ps.values():
                    if isinstance(sect, dict) and sect.get("patient_id"):
                        return str(sect["patient_id"])
    return None


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
