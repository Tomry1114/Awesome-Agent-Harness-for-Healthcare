"""Bounded Clinical Recovery v3 - Prior Authorization workflow (CWC / execution path).

The CLEAN recoverable population: the agent already committed to submitting a prior-authorization request
whose structured fields (request type, patient identity, DOB, diagnosis codes, CPT codes) are all present
in the authoritative EMR/referral state. There is NO new clinical decision to make; the harness simply
completes the deterministic execution closure - fill the structured form from authoritative_state, then
perform exactly ONE irreversible commit (submit), then verify the landed record by read-back.

Process knowledge only. It is substrate-agnostic (calls the injected substrate for primitives) and
benchmark-name-free (concrete field labels / target / verify path arrive via ctx and goal.raw).

Python 3.8 compatible.
"""
from ..contracts import (
    Plan, RecoveryStep, NAVIGATE, STAGED_WRITE, IRREVERSIBLE_COMMIT,
)

GOAL_TYPES = ("submit_prior_auth", "prior_authorization", "submit_auth", "prior_auth")

# Default structured-form field layout for a prior-auth request. The benchmark adapter may override the
# labels/roles via ctx["form_fields"]; the SHAPE (fill each field -> one submit -> verify) is fixed here.
_DEFAULT_FIELDS = (
    {"arg": "requestType", "label": "Request Type", "role": "input", "op": "select"},
    {"arg": "patientLastName", "label": "Patient Last Name", "role": "input", "op": "type"},
    {"arg": "patientFirstName", "label": "Patient First Name", "role": "input", "op": "type"},
    {"arg": "patientDOB", "label": "Patient Date of Birth", "role": "input", "op": "type"},
    {"arg": "diagnosisCodes", "label": "Diagnosis Code", "role": "input", "op": "type"},
    {"arg": "cptCodes", "label": "CPT Code", "role": "input", "op": "type"},
)


def _dig(state_view, root, path):
    node = state_view
    for k in [root] + list(path or []):
        if isinstance(node, dict):
            node = node.get(k)
        elif isinstance(node, list) and isinstance(k, int):
            node = node[k] if -len(node) <= k < len(node) else None
        else:
            return None
    return node


class PriorAuthorizationWorkflow(object):
    """CWC: complete a committed prior-authorization submission from authoritative state."""

    name = "prior_authorization"

    def _fields(self, ctx):
        ff = (ctx or {}).get("form_fields")
        return list(ff) if ff else list(_DEFAULT_FIELDS)

    def match_goal(self, goal, ctx):
        return getattr(goal, "goal_type", "") in GOAL_TYPES

    def required_bindings(self, goal, ctx):
        # every structured field is required; all are expected to resolve from authoritative_state.
        return [f["arg"] for f in self._fields(ctx)]

    def compile_plan(self, goal, ctx):
        ctx = ctx or {}
        fields = self._fields(ctx)
        steps = []
        # 1) navigate to the prior-auth form (portal URL supplied by the benchmark adapter, optional).
        nav = ctx.get("form_url")
        steps.append(RecoveryStep(
            kind=NAVIGATE, name="open_prior_auth_form",
            action={"op": "navigate", "url": nav} if nav else {"op": "navigate"}))
        # 2) fill each structured field (staged; reversible form input, not yet persisted).
        for f in fields:
            steps.append(RecoveryStep(
                kind=STAGED_WRITE, name="fill_%s" % f["arg"],
                action={"op": f.get("op", "type"), "arg": f["arg"]},
                affordance_target={"role": f.get("role", "input"), "label": f["label"]},
                arg_specs=[f["arg"]],
                manifest={"side_effect_scope": "form_field", "rollback_available": True,
                          "autosave_possible": False, "server_persisted": False}))
        # 3) exactly ONE irreversible commit: submit the request.
        submit_target = ctx.get("submit_target") or {"role": "button", "label": "Submit"}
        steps.append(RecoveryStep(
            kind=IRREVERSIBLE_COMMIT, name="submit_prior_auth",
            action={"op": "submit_prior_auth"},
            affordance_target=submit_target,
            manifest={"side_effect_scope": "payer_submission", "rollback_available": False,
                      "autosave_possible": False, "server_persisted": True}))

        verify = (getattr(goal, "raw", {}) or {}).get("verify") or {}
        pc = {"paths": (getattr(goal, "raw", {}) or {}).get("verify_paths", [])}
        return Plan(steps=steps, required_bindings=self.required_bindings(goal, ctx),
                    expected_postcondition=pc)

    def verify_effect(self, goal, state_view):
        spec = (getattr(goal, "raw", {}) or {}).get("verify")
        if not isinstance(spec, dict):
            return None
        node = _dig(state_view or {}, spec.get("root"), spec.get("path"))
        if node is None:
            return None
        if spec.get("check", "nonempty") == "nonempty":
            try:
                return len(node) >= int(spec.get("min_len", 1))
            except TypeError:
                return bool(node)
        return bool(node)
