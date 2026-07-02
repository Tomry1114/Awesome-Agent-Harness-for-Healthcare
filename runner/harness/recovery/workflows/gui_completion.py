"""Bounded Clinical Recovery v3 - GenericGuiCompletionWorkflow (Layer 3) : dynamic closed-loop.

ONE generic GUI-completion state machine. It knows NO business names (no prior-auth / appeal / denial /
claim / CPT / rationale). It drives the environment as a closed loop -- observe -> decide the ONE next
deterministic action -> (kernel executes) -> re-observe -> decide again -- until the target effect lands
or a genuinely new decision/content is required.

Generic phases (data-driven, re-evaluated every step from the LIVE control model + persisted state):
    DONE       target effect already realized (verify_spec path satisfied)
    FILL       a required+empty control has a bindable value -> type/select/upload it
    COMMIT      every fillable field done + a commit control present -> the ONE irreversible submit
    CONFIRM    a post-commit confirm/acknowledge control appeared -> click it
    NAVIGATE   not on the target surface yet -> click the single best control toward the goal
    BLOCK      a required field needs NEW content or a NEW decision the harness must not author

Binding is robust (not naive top-level string match): sources are flattened, keys aliased, values format-
normalized (dates / codes / names), select options semantically matched, already-filled fields preserved.

next_action() returns a Decision(kind, step|reason). compile_plan() is kept for the static-plan kernel path
and unit tests. Python 3.8 compatible.
"""
import re

from ..contracts import (
    RecoveryStep, Plan,
    READ, STAGED_WRITE, IRREVERSIBLE_COMMIT,
)

GOAL_TYPE = "complete_committed_gui_effect"
_NEEDS_DECISION = "__gui_needs_new_content__"

# generic control-role -> gui op.
_SELECTISH = ("select", "combobox", "listbox", "dropdown")
_FILEISH = ("file", "upload")
# confirm/acknowledge controls that appear AFTER a submit (generic verbs, no business names).
_CONFIRM_RE = re.compile(r"\b(confirm|acknowledge|ok|continue|yes|proceed|done|finish|close)\b", re.I)
_COMMIT_RE = re.compile(r"\b(submit|save|send|file|create|add|apply|update|register)\b", re.I)
_NAV_STOP_RE = re.compile(r"\b(cancel|back|logout|sign out|delete|remove|clear|reset)\b", re.I)


def _norm(s):
    s = str(s or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def _norm_code(s):
    """Normalize an identifier/code for comparison: strip spaces/punct, upper."""
    return re.sub(r"[^A-Za-z0-9]+", "", str(s or "")).upper()


def _flatten(obj, prefix="", out=None, depth=0):
    """Flatten a nested dict/list into {dotted_key: scalar} and also index every leaf by its LAST segment,
    so an EMR value buried at authoritative_state.patient.dob is reachable by 'dob'."""
    if out is None:
        out = {}
    if depth > 6:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = "%s.%s" % (prefix, k) if prefix else str(k)
            if isinstance(v, (dict, list)):
                _flatten(v, key, out, depth + 1)
            elif v not in (None, ""):
                out[key] = v
                out.setdefault(str(k), v)                    # last-segment alias
    elif isinstance(obj, list):
        # keep whole small scalar lists joinable (e.g. diagnosisCodes: ["H35.32"])
        scal = [x for x in obj if isinstance(x, (str, int, float))]
        if scal and prefix:
            out.setdefault(prefix, scal[0] if len(scal) == 1 else ", ".join(str(x) for x in scal))
            out.setdefault(prefix.split(".")[-1], out[prefix])
        for i, v in enumerate(obj):
            if isinstance(v, (dict, list)):
                _flatten(v, "%s.%d" % (prefix, i), out, depth + 1)
    return out


# label/name aliases: map common form-field concepts to EMR/state key fragments (generic clinical-admin
# vocabulary, NOT task-specific; extend freely -- this is alignment data, not business logic).
_ALIASES = {
    "diagnosis": ["diagnosiscodes", "diagnosis", "icd", "dx"],
    "procedure": ["cptcodes", "cpt", "procedure", "hcpcs"],
    "date of birth": ["patientdob", "dob", "birthdate", "dateofbirth"],
    "member id": ["subscriberid", "memberid", "insuranceid", "policyid"],
    "patient last name": ["patientlastname", "lastname", "familyname"],
    "patient first name": ["patientfirstname", "firstname", "givenname"],
    "patient name": ["patientname", "name", "patient"],
    "provider": ["providername", "provider", "orderingprovider", "physician"],
    "request type": ["requesttype", "authtype", "type"],
    "claim": ["claimid", "claim", "claimnumber"],
}


def _alias_targets(label, name):
    keys = set()
    for src in (label, name):
        n = _norm(src)
        if not n:
            continue
        keys.add(n.replace(" ", ""))
        keys.add(n)
        for canon, al in _ALIASES.items():
            if canon in n or n in canon or any(a in n.replace(" ", "") for a in al):
                keys.update(al)
    return keys


class GenericGuiCompletionWorkflow(object):
    goal_type = GOAL_TYPE

    def match_goal(self, goal, ctx):
        return (getattr(goal, "goal_type", "") or "") == GOAL_TYPE

    def required_bindings(self, goal, ctx):
        return []

    # ---- robust binding ---------------------------------------------------------------------------
    def _sources(self, goal, ctx):
        merged = {}
        # order = priority: agent_commitment, authoritative_state, bound_evidence, system_metadata
        for src in (getattr(goal, "committed_fields", None) or {},
                    (ctx or {}).get("authoritative_state") or {},
                    (ctx or {}).get("bound_evidence") or {},
                    (ctx or {}).get("system_metadata") or {}):
            flat = _flatten(src)
            for k, v in flat.items():
                merged.setdefault(k, v)
                merged.setdefault(_norm(k).replace(" ", ""), v)
        return merged

    def _bind_value(self, control, sources):
        """Resolve a control's value from the flattened sources by name/label/alias. Returns str|None."""
        targets = _alias_targets(control.get("label"), control.get("name"))
        # 1) direct/alias key hit
        for t in targets:
            if t in sources and sources[t] not in (None, ""):
                return self._format(control, sources[t])
        # 2) normalized substring match against source keys
        for sk, sv in sources.items():
            if sv in (None, "") or isinstance(sv, (dict, list)):
                continue
            nsk = _norm(sk).replace(" ", "")
            if any(t and (t in nsk or nsk in t) for t in targets):
                return self._format(control, sv)
        return None

    def _format(self, control, value):
        """Format a bound value to the control (select-option semantic match; codes/dates left as-is str)."""
        val = value if isinstance(value, str) else str(value)
        opts = control.get("options") or []
        if opts:
            nv = _norm_code(val)
            for o in opts:
                if _norm_code(o) == nv or _norm(o) == _norm(val):
                    return o
            for o in opts:                                   # substring fallback
                if nv and (nv in _norm_code(o) or _norm_code(o) in nv):
                    return o
        return val

    def _op_for(self, control):
        r = str(control.get("role") or "").lower()
        if r in _SELECTISH or (control.get("options")):
            return "select"
        if r in _FILEISH or r == "file":
            return "upload"
        return "type"

    # ---- dynamic closed loop ----------------------------------------------------------------------
    def is_realized(self, goal, state_view):
        spec = (getattr(goal, "raw", None) or {}).get("verify_spec")
        if not isinstance(spec, dict):
            return None
        got = _dig(state_view or {}, spec.get("path"))
        if spec.get("check") == "nonempty":
            return bool(got) and (not hasattr(got, "__len__") or len(got) >= int(spec.get("min_len", 1)))
        return bool(got)

    def next_action(self, goal, ctx, controls, state_view, history):
        """Decide the ONE next deterministic action from the LIVE page. Returns a dict:
            {"kind": "step", "step": RecoveryStep}  |  {"kind": "done"}  |  {"kind": "block", "reason": ...}
        history = list of prior action tags this episode (to avoid loops / know we already committed)."""
        history = history or []
        sources = self._sources(goal, ctx)

        # 0) already there?
        if self.is_realized(goal, state_view):
            return {"kind": "done", "reason": "effect_realized"}

        commit_ctrl = None
        fillable = []                        # (control, value)
        required_unbound = None
        confirm_ctrl = None
        for c in controls or []:
            label = c.get("label", "")
            if c.get("commit") or (c.get("role") == "button" and _COMMIT_RE.search(label or "")):
                if commit_ctrl is None:
                    commit_ctrl = c
                continue
            if c.get("role") == "button" and _CONFIRM_RE.search(label or "") and not _NAV_STOP_RE.search(label or ""):
                confirm_ctrl = confirm_ctrl or c
                continue
            role = str(c.get("role") or "").lower()
            is_field = role in ("input", "textbox", "textarea", "select") or role in _SELECTISH or c.get("options")
            if not is_field:
                continue
            if str(c.get("value") or "").strip():
                continue                                     # preserve already-filled fields
            val = self._bind_value(c, sources)
            if val is None:
                if c.get("required"):
                    required_unbound = required_unbound or c
                continue
            fillable.append((c, val))

        already_committed = any(h.startswith("commit:") for h in history)

        # 1) FILL the next unfilled bindable field (one at a time -> re-observe).
        for c, val in fillable:
            tag = "fill:%s" % _norm(c.get("name") or c.get("label") or c.get("ref"))
            if tag in history:
                continue                                     # already filled this field; skip
            return {"kind": "step", "tag": tag, "step": RecoveryStep(
                kind=STAGED_WRITE, name="fill",
                action={"op": self._op_for(c), "arg": "_direct", "value": val},
                affordance_target={"role": c.get("role"), "label": c.get("label"), "ref": c.get("ref")},
                manifest={"side_effect_scope": "form_field", "rollback_available": True,
                          "server_persisted": False})}

        # 2) a required field with no bindable value -> the agent must author it -> BLOCK.
        if required_unbound is not None and not already_committed:
            return {"kind": "block", "reason": "needs_new_content:%s" % (required_unbound.get("label") or "")}

        # 3) post-commit CONFIRM step (multi-stage submit).
        if already_committed and confirm_ctrl is not None and ("confirm:%s" % confirm_ctrl.get("ref")) not in history:
            return {"kind": "step", "tag": "confirm:%s" % confirm_ctrl.get("ref"), "step": RecoveryStep(
                kind=STAGED_WRITE, name="confirm",
                action={"op": "click"},
                affordance_target={"role": "button", "label": confirm_ctrl.get("label"), "ref": confirm_ctrl.get("ref")},
                manifest={"side_effect_scope": "confirm", "rollback_available": False, "server_persisted": False})}

        # 4) COMMIT once the fillable set is exhausted (all required either filled or preserved).
        if commit_ctrl is not None and not already_committed:
            return {"kind": "step", "tag": "commit:%s" % commit_ctrl.get("ref"), "step": RecoveryStep(
                kind=IRREVERSIBLE_COMMIT, name="submit",
                action={"op": "submit"},
                affordance_target={"role": "button", "label": commit_ctrl.get("label"), "ref": commit_ctrl.get("ref")},
                manifest={"side_effect_scope": "gui_submission", "rollback_available": False,
                          "autosave_possible": False, "server_persisted": True})}

        # 5) NAVIGATE toward the target when we are not on a fillable surface yet.
        nav = self._nav_toward(goal, controls, history)
        if nav is not None:
            return nav

        # 6) nothing actionable left.
        if already_committed:
            return {"kind": "done", "reason": "committed_awaiting_verify"}
        return {"kind": "block", "reason": "no_actionable_control"}

    def _nav_toward(self, goal, controls, history):
        """Generic navigation: click the single control whose label best matches the goal's PUBLIC target
        keywords (data from the task, not gold). Never clicks a stop/cancel control; never repeats."""
        kws = [_norm(k) for k in ((getattr(goal, "raw", None) or {}).get("target_keywords") or []) if k]
        if not kws:
            return None
        best = None
        best_score = 0
        for c in controls or []:
            if str(c.get("role") or "") not in ("button", "link", "a", "tab", "menuitem"):
                continue
            label = c.get("label") or ""
            if not label or _NAV_STOP_RE.search(label):
                continue
            nl = _norm(label)
            score = sum(1 for kw in kws if kw and (kw in nl or nl in kw))
            tag = "nav:%s" % c.get("ref")
            if score > best_score and tag not in history:
                best, best_score = c, score
        if best is None or best_score == 0:
            return None
        return {"kind": "step", "tag": "nav:%s" % best.get("ref"), "step": RecoveryStep(
            kind=READ, name="navigate",
            action={"op": "click"},
            affordance_target={"role": best.get("role"), "label": best.get("label"), "ref": best.get("ref")})}

    # ---- static plan (kernel path + unit tests) ---------------------------------------------------
    def compile_plan(self, goal, ctx):
        ctx = ctx or {}
        controls = list(ctx.get("gui_controls") or [])
        sources = self._sources(goal, ctx)
        steps = [RecoveryStep(kind=READ, name="read_form", action={"op": "snapshot"})]
        commit_ctrl = None
        for c in controls:
            if c.get("commit"):
                commit_ctrl = commit_ctrl or c
                continue
            if not c.get("required") or str(c.get("value") or "").strip():
                continue
            val = self._bind_value(c, sources)
            if val is None:
                return Plan(steps=[], required_bindings=[_NEEDS_DECISION], expected_postcondition={})
            steps.append(RecoveryStep(
                kind=STAGED_WRITE, name="fill_%s" % (c.get("name") or c.get("ref")),
                action={"op": self._op_for(c), "arg": "_direct", "value": val},
                affordance_target={"role": c.get("role"), "label": c.get("label"), "ref": c.get("ref")},
                manifest={"side_effect_scope": "form_field", "rollback_available": True,
                          "server_persisted": False}))
        if commit_ctrl is not None:
            steps.append(RecoveryStep(
                kind=IRREVERSIBLE_COMMIT, name="submit", action={"op": "submit"},
                affordance_target={"role": commit_ctrl.get("role"), "label": commit_ctrl.get("label"),
                                   "ref": commit_ctrl.get("ref")},
                manifest={"side_effect_scope": "gui_submission", "rollback_available": False,
                          "server_persisted": True}))
        return Plan(steps=steps, required_bindings=[], expected_postcondition={"paths": []})

    def verify_effect(self, goal, state_view):
        r = self.is_realized(goal, state_view)
        if r is None:
            return None
        if not (state_view or {}):
            return None
        return bool(r)


def _dig(state, dotted):
    node = state
    for seg in [p for p in str(dotted or "").split(".") if p]:
        if isinstance(node, dict):
            node = node.get(seg)
        elif isinstance(node, list):
            try:
                node = node[int(seg)]
            except (ValueError, IndexError, TypeError):
                return None
        else:
            return None
    return node
