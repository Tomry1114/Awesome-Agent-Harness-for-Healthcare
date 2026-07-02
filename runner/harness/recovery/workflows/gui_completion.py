"""Bounded Clinical Recovery v3 - GenericGuiCompletionWorkflow (Layer 3).

ONE generic compiler for a committed-but-unrealized GUI effect (a form/flow the ROOT AGENT decided to
complete but never submitted). It knows NO business names -- no "prior auth", no "appeal", no "denial",
no "claim", no "CPT", no "rationale", no "document in Epic". It knows only the GENERIC GUI structure that
the substrate reports as a live CONTROL MODEL:

    control = {ref, role, label, name, required, value, options, commit}

and completes the remaining deterministic execution closure:

    1. read the live control model (already injected into ctx by the wiring layer)
    2. for each REQUIRED + EMPTY control: bind a value from the four sources
         (agent_commitment / authoritative_state / bound_evidence / system_metadata) by label/name
       - has value  -> staged type/select/upload
       - no value   -> BLOCK (needs new content or a new decision -- the harness never authors it)
    3. execute the ONE commit control (irreversible submit)
    4. verify via the adapter-supplied verify_spec (a state path + check -- DATA, not process code)

A control already carrying a value is left untouched (idempotent). Task differences come ENTIRELY from the
runtime control model + the verify_spec; this module is benchmark-name-free. Python 3.8 compatible.
"""
from ..contracts import (
    RecoveryStep, Plan,
    READ, STAGED_WRITE, IRREVERSIBLE_COMMIT, VERIFY,
)

_READBACK_KEY = "_bcr_gui_readback"

GOAL_TYPE = "complete_committed_gui_effect"

# marker binding name the kernel cannot resolve -> a clean BLOCKED_NEEDS_DECISION (never a FAILED submit).
_NEEDS_DECISION = "__gui_needs_new_content__"

_WS = None


def _norm(s):
    return " ".join(str(s or "").lower().replace("_", " ").split())


def _op_for_role(role):
    r = str(role or "").lower()
    if r in ("select", "combobox", "listbox", "dropdown"):
        return "select"
    if r in ("file", "upload") or r == "input" and False:
        return "upload"
    return "type"


class GenericGuiCompletionWorkflow(object):
    """Compile + verify completion of a committed GUI effect from the live control model + a verify_spec."""

    goal_type = GOAL_TYPE

    def match_goal(self, goal, ctx):
        return (getattr(goal, "goal_type", "") or "") == GOAL_TYPE

    def required_bindings(self, goal, ctx):
        # decided per-control at compile time from the live page; nothing to declare up front.
        return []

    def _bind_value(self, control, sources):
        """Resolve a control's value from the four sources by matching its name/label. Returns str|None.
        Default-deny: an unmatched required control yields None -> the plan BLOCKS."""
        keys = [control.get("name"), control.get("label")]
        norm_keys = [_norm(k) for k in keys if k]
        for src in sources:                      # ordered: commitment, authoritative, evidence, metadata
            if not isinstance(src, dict):
                continue
            # exact key hit first
            for k in keys:
                if k and k in src and src[k] not in (None, ""):
                    return str(src[k])
            # normalized label/name match against the source's keys
            for sk, sv in src.items():
                if sv in (None, "") or isinstance(sv, (dict, list)):
                    continue
                if _norm(sk) in norm_keys or any(nk and (nk in _norm(sk) or _norm(sk) in nk) for nk in norm_keys):
                    return str(sv)
        return None

    def compile_plan(self, goal, ctx):
        ctx = ctx or {}
        controls = list(ctx.get("gui_controls") or [])
        sources = [
            getattr(goal, "committed_fields", None) or {},      # agent_commitment
            ctx.get("authoritative_state") or {},               # authoritative_state
            ctx.get("bound_evidence") or {},                    # bound_evidence
            ctx.get("system_metadata") or {},                   # system_metadata (operational)
        ]

        steps = [RecoveryStep(kind=READ, name="read_form", action={"op": "snapshot"})]
        commit_ctrl = None
        fills = []
        for c in controls:
            if c.get("commit"):
                if commit_ctrl is None:
                    commit_ctrl = c
                continue
            if not c.get("required"):
                continue
            if str(c.get("value") or "").strip():
                continue                                         # already filled -> idempotent skip
            val = self._bind_value(c, sources)
            if val is None:
                # a REQUIRED field with no bindable value -> needs NEW content/decision -> BLOCK cleanly.
                return Plan(steps=[], required_bindings=[_NEEDS_DECISION], expected_postcondition={})
            fills.append((c, val))

        for c, val in fills:
            steps.append(RecoveryStep(
                kind=STAGED_WRITE, name="fill_%s" % (c.get("name") or c.get("ref")),
                action={"op": _op_for_role(c.get("role")), "arg": "_direct", "value": val},
                affordance_target={"role": c.get("role"), "label": c.get("label"), "ref": c.get("ref")},
                manifest={"side_effect_scope": "form_field", "rollback_available": True,
                          "autosave_possible": False, "server_persisted": False}))

        if commit_ctrl is not None:
            steps.append(RecoveryStep(
                kind=IRREVERSIBLE_COMMIT, name="submit",
                action={"op": "submit"},
                affordance_target={"role": commit_ctrl.get("role"), "label": commit_ctrl.get("label"),
                                   "ref": commit_ctrl.get("ref")},
                manifest={"side_effect_scope": "gui_submission", "rollback_available": False,
                          "autosave_possible": False, "server_persisted": True}))

        # the substrate's read_state returns the whole merged portal state; verify_effect digs the spec path.
        return Plan(steps=steps, required_bindings=[], expected_postcondition={"paths": []})

    @staticmethod
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

    def verify_effect(self, goal, state_view):
        """Generic post-submit verification against the adapter's verify_spec (path + check). True/False/None.
        The path is DATA (which persisted-state key indicates the effect landed); the kernel/workflow do not
        know which HAB task it belongs to. state_view is the substrate's merged portal state."""
        spec = (getattr(goal, "raw", None) or {}).get("verify_spec")
        if not isinstance(spec, dict) or not (state_view or {}):
            return None
        got = self._dig(state_view or {}, spec.get("path"))
        check = spec.get("check", "truthy")
        if check == "nonempty":
            if got is None:
                return False
            try:
                return len(got) >= int(spec.get("min_len", 1))
            except TypeError:
                return bool(got)
        # 'truthy' / 'changed_or_truthy'
        if got is None:
            return False
        return bool(got)
