"""Bounded Clinical Recovery v3 - HAB (interactive-GUI) stack unit tests.

Standalone: `python3 test_hab_v3.py` -> exit 0 on all-pass, non-zero on any failure. No browser, no model
calls, deterministic. Uses the REAL affordance resolver + REAL kernel/workflows/benchmark adapter with a
STUB GUI substrate and synthetic observation strings.

Coverage:
  (a) resolve_affordance: unique -> AffordanceBinding; zero -> BLOCKED_UNRESOLVED_AFFORDANCE;
      multiple -> BLOCKED_AMBIGUOUS_TARGET; multiple + disambiguating bound id -> AffordanceBinding.
  (b) prior-auth happy path (all structured fields from authoritative_state) -> VERIFIED.
  (c) appeal with NO authored rationale/attachment -> BLOCKED_MISSING_EVIDENCE (correctly_blocked, NOT failed).
  (d) decision-documentation happy path (landed disposition) -> VERIFIED.
"""
import copy
import os
import sys

# import root = runner/ (two dirs up from this file: runner/harness/recovery/)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from harness.recovery.contracts import (
    AffordanceBinding, Outcome,
    READ_LIKE_KINDS, IRREVERSIBLE_COMMIT,
    RESULT_OK, RESULT_FAILED,
    VERIFIED, BLOCKED_MISSING_EVIDENCE,
    BLOCKED_UNRESOLVED_AFFORDANCE, BLOCKED_AMBIGUOUS_TARGET,
)
from harness.recovery.metrics import VERIFIED_RECOVERY, CORRECTLY_BLOCKED, FAILED_RECOVERY
from harness.recovery.kernel import RecoveryKernel
from harness.recovery.registry import WorkflowRegistry
from harness.recovery.substrate.gui import (
    GuiSubstrateAdapter, resolve_affordance_in_observation,
)
from harness.recovery.benchmark.hab import HabBenchmarkAdapter, G_PRIOR_AUTH, G_APPEAL, G_DOCUMENT
from harness.recovery.benchmark import hab_register


_CHECKS = []


def check(name, cond):
    _CHECKS.append((name, bool(cond)))
    print(("PASS" if cond else "FAIL"), name)


# --------------------------------------------------------------------------------------------------
# STUB GUI substrate: resolves affordances via the REAL resolver, executes mutations against an
# in-memory persisted-state tree shaped like the RAW portals_state read-back.
# --------------------------------------------------------------------------------------------------
class StubGui(object):
    def __init__(self, observation, state=None):
        self.observation = observation
        self.state = state or {
            "full_state": {"agentActions": {}},
            "payer_a_state": {"submissions": [], "appealActions": {}},
            "payer_b_state": {"submissions": [], "appealActions": {}},
        }

    def resolve_affordance(self, target_spec, observation):
        return resolve_affordance_in_observation(target_spec, observation or self.observation)

    def execute_primitive(self, kind, action, auth):
        act = (action or {}).get("action") or {}
        op = act.get("op") or kind
        if kind in READ_LIKE_KINDS:
            return Outcome(status=RESULT_OK, result=self.observation)
        if auth is None or not auth.consume():
            return Outcome(status=RESULT_FAILED, reason="unauthorized")
        if kind == IRREVERSIBLE_COMMIT:
            if op in ("submit_prior_auth", "submit_auth"):
                self.state["payer_a_state"]["submissions"].append({"confirmationId": "PA-1"})
                return Outcome(status=RESULT_OK, created_id="PA-1")
            if op == "submit_appeal":
                self.state["payer_a_state"].setdefault("appealActions", {})["submittedAppeal"] = True
                return Outcome(status=RESULT_OK, created_id="AP-1")
            if op == "document_decision":
                self.state["full_state"].setdefault("agentActions", {})["documentedAppealInEpic"] = True
                return Outcome(status=RESULT_OK, created_id="DOC-1")
        return Outcome(status=RESULT_OK)

    def read_state(self, paths):
        return copy.deepcopy(self.state)

    def classify_result(self, result):
        return getattr(result, "status", RESULT_OK)


# --------------------------------------------------------------------------------------------------
# synthetic observations
# --------------------------------------------------------------------------------------------------
AFF_OBS = ("URL: /x\n--- INTERACTIVE ELEMENTS ---\n"
           "[ref=0] button[submit] 'Submit'\n"
           "[ref=1] a[link] 'Home'\n")
MULTI_OBS = ("URL: /x\n--- INTERACTIVE ELEMENTS ---\n"
             "[ref=3] button[button] 'Save'\n"
             "[ref=7] button[button] 'Save'\n")
PA_OBS = ("URL: /payer/priorauth\n--- INTERACTIVE ELEMENTS ---\n"
          "[ref=0] input[text] 'Request Type'\n"
          "[ref=1] input[text] 'Patient Last Name'\n"
          "[ref=2] input[text] 'Patient First Name'\n"
          "[ref=3] input[text] 'Patient Date of Birth'\n"
          "[ref=4] input[text] 'Diagnosis Code'\n"
          "[ref=5] input[text] 'CPT Code'\n"
          "[ref=6] button[submit] 'Submit'\n")
DOC_OBS = ("URL: /emr/case\n--- INTERACTIVE ELEMENTS ---\n"
           "[ref=0] button[button] 'Document'\n"
           "[ref=1] button[button] 'Cancel'\n")


def _registry():
    return hab_register.register(WorkflowRegistry())


def _run(task):
    hab = HabBenchmarkAdapter()
    ctx = hab.context(task)
    stub = StubGui(observation=ctx.get("observation"))
    kernel = RecoveryKernel()
    res = kernel.run_episode(hab, _registry(), stub, None, task, [], task.get("goal"), None)
    return res, stub


# --------------------------------------------------------------------------------------------------
# (a) resolve_affordance
# --------------------------------------------------------------------------------------------------
def test_affordance():
    sub = GuiSubstrateAdapter(None)
    uniq = sub.resolve_affordance({"role": "button", "label": "Submit"}, AFF_OBS)
    check("aff.unique -> AffordanceBinding", isinstance(uniq, AffordanceBinding))
    check("aff.unique.ref == 0", isinstance(uniq, AffordanceBinding) and uniq.ref == 0)

    zero = sub.resolve_affordance({"role": "button", "label": "Delete Everything"}, AFF_OBS)
    check("aff.zero -> BLOCKED_UNRESOLVED_AFFORDANCE", zero == BLOCKED_UNRESOLVED_AFFORDANCE)

    many = sub.resolve_affordance({"role": "button", "label": "Save"}, MULTI_OBS)
    check("aff.multiple -> BLOCKED_AMBIGUOUS_TARGET", many == BLOCKED_AMBIGUOUS_TARGET)

    disambig = sub.resolve_affordance({"role": "button", "label": "Save", "bound_id": 7}, MULTI_OBS)
    check("aff.multiple+bound_id -> AffordanceBinding(ref=7)",
          isinstance(disambig, AffordanceBinding) and disambig.ref == 7)

    # role filtering: a link labelled 'Submit' would NOT satisfy a button target (here only ref0 is a button)
    only_button = resolve_affordance_in_observation({"role": "button", "label": "Submit"}, AFF_OBS)
    check("aff role-filter keeps single button", isinstance(only_button, AffordanceBinding))


# --------------------------------------------------------------------------------------------------
# (b) prior-auth happy path
# --------------------------------------------------------------------------------------------------
def test_prior_auth_verified():
    task = {
        "environment": {"type": "gui"},
        "goal": "submit the prior authorization request",
        "observation": PA_OBS,
        "authoritative_state": {
            "requestType": "Outpatient",
            "patientLastName": "Smith",
            "patientFirstName": "Emily",
            "patientDOB": "1958-06-20",
            "diagnosisCodes": "H35.31",
            "cptCodes": "92014",
        },
        "recovery_commitments": [{"goal_type": G_PRIOR_AUTH, "payer": "a",
                                  "goal_id": "pa-1"}],
    }
    res, stub = _run(task)
    check("prior_auth state == VERIFIED (got %s)" % res.state, res.state == VERIFIED)
    check("prior_auth bucket == verified_recovery", res.metrics_bucket == VERIFIED_RECOVERY)
    check("prior_auth landed a submission", len(stub.state["payer_a_state"]["submissions"]) == 1)
    check("prior_auth created_id captured", "PA-1" in (res.created_ids or []))


# --------------------------------------------------------------------------------------------------
# (c) appeal with no authored rationale/attachment -> BLOCKED_MISSING_EVIDENCE
# --------------------------------------------------------------------------------------------------
def test_appeal_blocks_missing_evidence():
    task = {
        "environment": {"type": "gui"},
        "goal": "file an appeal for the denied claim",
        "observation": PA_OBS,
        # claim locator resolvable, but NO rationale and NO acquired supporting-evidence attachment.
        "authoritative_state": {"claimId": "CLM-778"},
        "recovery_commitments": [{"goal_type": G_APPEAL, "payer": "a", "goal_id": "ap-1"}],
    }
    res, stub = _run(task)
    # THE APPEAL-BLOCK ASSERTION:
    check("appeal state == BLOCKED_MISSING_EVIDENCE (got %s)" % res.state,
          res.state == BLOCKED_MISSING_EVIDENCE)
    check("appeal blocked_argument == attachmentEvidenceRef",
          res.blocked_argument == "attachmentEvidenceRef")
    check("appeal bucket == correctly_blocked", res.metrics_bucket == CORRECTLY_BLOCKED)
    check("appeal is NOT failed_recovery", res.metrics_bucket != FAILED_RECOVERY)
    check("appeal did NOT submit anything",
          stub.state["payer_a_state"].get("appealActions", {}).get("submittedAppeal") is not True)


# --------------------------------------------------------------------------------------------------
# (d) decision-documentation happy path
# --------------------------------------------------------------------------------------------------
def test_decision_documentation_verified():
    task = {
        "environment": {"type": "gui"},
        "goal": "document the appeal disposition in Epic",
        "observation": DOC_OBS,
        "authoritative_state": {"disposition": "appeal_upheld"},
        "recovery_commitments": [{"goal_type": G_DOCUMENT, "goal_id": "doc-1"}],
    }
    res, stub = _run(task)
    check("doc state == VERIFIED (got %s)" % res.state, res.state == VERIFIED)
    check("doc bucket == verified_recovery", res.metrics_bucket == VERIFIED_RECOVERY)
    check("doc set documentedAppealInEpic",
          stub.state["full_state"]["agentActions"].get("documentedAppealInEpic") is True)


def main():
    test_affordance()
    test_prior_auth_verified()
    test_appeal_blocks_missing_evidence()
    test_decision_documentation_verified()
    failed = [n for n, ok in _CHECKS if not ok]
    print("\n%d/%d checks passed" % (len(_CHECKS) - len(failed), len(_CHECKS)))
    if failed:
        print("FAILURES:", failed)
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
