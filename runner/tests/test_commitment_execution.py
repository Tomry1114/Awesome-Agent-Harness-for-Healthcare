"""Unit tests for the general CommitmentExecution capability (committed_not_done self-opened gap).
Uses a MOCK judge_fn -- no gateway. Verifies: fires when a firmly-committed action was never executed;
silent when it WAS executed (via real canonicalize executed-counting); fail-safe off; caching; substrate
with no commit affordances is a no-op."""
import sys
sys.path.insert(0, ".")
from workspace import H0, Workspace
CommitmentExecution = Workspace(H0["fhir"]).load("middleware/commitment_execution.py", "CommitmentExecution")
CE = sys.modules[CommitmentExecution.__module__]
_parse_unmet = CE._parse_unmet
from harness import decision as D

MANIFEST = {
    "subject": {"type": "patient", "binding": "required", "from_args": ["patient", "subject"]},
    "actions": [
        {"match": {"tool_pattern": "service_request_create"}, "semantic_type": "create", "resource": "ServiceRequest"},
        {"match": {"tool_pattern": "medication_request_create"}, "semantic_type": "create", "resource": "MedicationRequest"},
        {"match": {"tool_pattern": "_search"}, "semantic_type": "read"},
    ],
}
GUI_NO_COMMIT = {"subject": {"type": "case", "binding": "implicit_active"},
                 "actions": [{"match": {"tool_pattern": "click"}, "semantic_type": "read"}]}


class _Ledger:
    def __init__(self, commits): self.commit_history = commits


class _Ctx:
    def __init__(self, judge_fn, manifest, commits):
        self.judge_fn = judge_fn
        self.manifest = manifest
        self.ledger = _Ledger(commits)

    def budgeted_judge(self):
        """Mirror HarnessContext.budgeted_judge (runner/harness/capability.py): the metered way to reach the judge,
        which the capability uses so its judge calls count against max_semantic_checks.

        Returns judge_fn, i.e. an unmetered budget -- these tests are about the reconciliation logic, not
        about budget exhaustion. It MUST still return None when no judge is bound, or the
        `failsafe_off_without_judge` case would silently stop testing anything.
        """
        return self.judge_fn


def _mock_judge_calls():
    calls = {"n": 0}
    def judge(prompt):
        calls["n"] += 1
        # realistic judge: flag the ultrasound order ONLY if the environment shows ServiceRequest: 0 executed
        if "ServiceRequest: 0 executed" in prompt and "ultrasound" in prompt.lower():
            return '{"unmet": ["right elbow ultrasound order"]}'
        return '{"unmet": []}'
    return judge, calls


DELIV = "ASSESSMENT.\nRECOMMENDATIONS: 1. Right elbow ultrasound order with Doppler. 2. Ortho referral."


def test_fires_when_committed_order_not_executed():
    judge, calls = _mock_judge_calls()
    cap = CommitmentExecution()
    ctx = _Ctx(judge, MANIFEST, commits=[])          # 0 orders executed
    dec = cap.before_final(DELIV, ctx)
    assert dec is not None and dec.type == D.REVISE and dec.rule_id == "committed_not_done", dec
    assert "ultrasound" in " ".join(dec.extra["unmet_commitments"]).lower()
    assert calls["n"] == 1


def test_silent_when_order_was_executed():
    judge, calls = _mock_judge_calls()
    cap = CommitmentExecution()
    # a VERIFIED ServiceRequest create -> canonicalize -> resource=ServiceRequest -> count 1 -> judge sees "1 executed"
    commits = [{"action": {"type": "tool", "tool": "fhir_service_request_create", "args": {"patient": "P1"}},
                "verified": True, "semantic_type": "create"}]
    ctx = _Ctx(judge, MANIFEST, commits=commits)
    dec = cap.before_final(DELIV, ctx)
    assert dec is None, dec                            # executed -> no committed_not_done


def test_failsafe_off_without_judge():
    cap = CommitmentExecution()
    ctx = _Ctx(None, MANIFEST, commits=[])
    assert cap.before_final(DELIV, ctx) is None


def test_no_commit_affordances_noop():
    judge, calls = _mock_judge_calls()
    cap = CommitmentExecution()
    ctx = _Ctx(judge, GUI_NO_COMMIT, commits=[])      # substrate has no create/submit/update
    assert cap.before_final(DELIV, ctx) is None
    assert calls["n"] == 0


def test_caches_per_deliverable():
    judge, calls = _mock_judge_calls()
    cap = CommitmentExecution()
    ctx = _Ctx(judge, MANIFEST, commits=[])
    cap.before_final(DELIV, ctx)
    cap.before_final(DELIV, ctx)                       # same answer -> cached, judge not called again
    assert calls["n"] == 1


def test_parse_unmet_robust():
    assert _parse_unmet('{"unmet": ["a", "b"]}') == ["a", "b"]
    assert _parse_unmet('noise before {"unmet": ["x"]} after') == ["x"]
    assert _parse_unmet("not json") == []
    assert _parse_unmet({"unmet": ["d"]}) == ["d"]
    assert _parse_unmet(None) == []


def test_executable_filter_drops_false_positives():
    _executable = CE._executable
    for bad in ["assessment completed and saved to disk", "continue linezolid", "saved management plan",
                "updating the clinical note", "reviewed labs"]:
        assert not _executable(bad), bad
    for good in ["submitting aripiprazole prescription", "placing cetirizine prescription",
                 "ordered LH lab", "scheduling follow-up appointment", "placing expedited GI referral"]:
        assert _executable(good), good


def test_judge_false_positive_is_filtered_out():
    # judge returns a non-executable "commitment" -> capability must NOT fire (post-filter kills it)
    def judge(_p): return '{"unmet": ["continue linezolid", "assessment saved to disk"]}'
    class L:
        commit_history = []
    class Ctx:
        judge_fn = staticmethod(judge)
        def budgeted_judge(self): return self.judge_fn
        manifest = {"actions": [{"semantic_type": "create", "resource": "ServiceRequest"}]}
        ledger = L()
    assert CommitmentExecution().before_final("continue linezolid; assessment saved to disk", Ctx()) is None


def test_deterministic_reconcile_drops_already_executed():
    # The judge may disobey the executed-count rule and return commitments the agent ACTUALLY executed.
    # The deterministic reconciliation must drop them (content + generic-type match).
    def judge(_p):
        return ('{"unmet": ["placing CA 19-9 service request", '
                '"placing Hepatitis B surface antigen service request", '
                '"placing gastroenterology follow-up consultation order", '
                '"scheduling GI follow-up appointment"]}')
    def _c(tool, disp):
        # PRODUCTION shape: kernel stores the capability NAME in "action"; the real action dict is in "descriptor".
        return {"action": "verify_commit", "descriptor": {"tool": tool, "args": {"code_display": disp}},
                "verified": True, "semantic_type": "create"}
    commits = [
        _c("fhir_service_request_create", "Cancer Ag 19-9 [Units/volume] in Serum or Plasma"),
        _c("fhir_service_request_create", "Hepatitis B virus surface Ag [Presence] in Serum"),
        _c("fhir_service_request_create", "Gastroenterology consultation, follow-up"),
        _c("fhir_appointment_create", ""),
    ]
    cap = CommitmentExecution()
    assert cap.before_final("Note firmly states it is placing all of the above.", _Ctx(judge, MANIFEST, commits)) is None


def test_reconcile_keeps_genuinely_unmet_of_same_type():
    # Precision: executed a DIFFERENT ServiceRequest (CBC); the committed cryoglobulin order was NOT executed
    # -> type-level coarseness must NOT swallow it; committed_not_done still fires on the real omission.
    def judge(_p):
        return '{"unmet": ["order cryoglobulin testing"]}'
    commits = [{"action": {"tool": "fhir_service_request_create", "args": {"code_display": "CBC with Differential"}},
                "verified": True, "semantic_type": "create"}]
    cap = CommitmentExecution()
    dec = cap.before_final("Note states it is ordering cryoglobulin testing.", _Ctx(judge, MANIFEST, commits))
    assert dec is not None and dec.rule_id == "committed_not_done", dec
    assert "cryoglobulin" in " ".join(dec.extra["unmet_commitments"]).lower()


def test_reconcile_word_order_and_phrasing():
    # The agent DID order cortisol ("Cortisol, serum (morning)"); the judge lists "ordering morning serum cortisol"
    # as unmet. commit_history["action"] holds the capability NAME (a string), so reconciliation must read the action
    # from "descriptor"; token-set matching is word-order independent -> must reconcile.
    def judge(_p):
        return '{"unmet": ["ordering morning serum cortisol", "placing Hepatitis C antibody order"]}'
    def _c(tool, disp):
        return {"action": "verify_commit", "descriptor": {"tool": tool, "args": {"code_display": disp}},
                "verified": True, "semantic_type": "create"}
    commits = [_c("fhir_service_request_create", "Cortisol, serum (morning)"),
               _c("fhir_service_request_create", "Hepatitis C virus antibody")]
    assert CommitmentExecution().before_final("Note states it is ordering all of the above.",
                                              _Ctx(judge, MANIFEST, commits)) is None


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for fn in fns:
        try:
            fn(); ok += 1; print("PASS", fn.__name__)
        except Exception:
            print("FAIL", fn.__name__); traceback.print_exc()
    print("%d/%d passed" % (ok, len(fns)))