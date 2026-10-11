# -*- coding: utf-8 -*-
"""Two boundary rules of the HealthAdminBench harness.

Obligations  completion obligations never copy hidden checkpoints: a conjunct comes from the public goal or from the
    agent's own decision (runner/harness/obligation_gate.py, workspaces/hab/h0/policies/completion_obligations.yaml).
Commits  the harness never makes a new irreversible commit on its own initiative: a harness irreversible commit runs
    only to complete a commit the agent itself attempted and that did not land, once (runner/harness/action_service.py);
    the read channels issue none (refuse_irreversible_read); click submits are classified as commits on the page they
    are issued on (runner/adapters/admin_portal.yaml, harness.semantics.canonicalize_on_page), and the
    observed-submission terminal lock does not depend on the R2 branch (runner/harness/kernel.py).

The loop tests drive run.run_task on the scripted portal of runner/tests/inline_hab_stub.py.
"""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inline_hab_stub as S  # noqa: E402
from inline_hab_stub import click, control, compact, fill, final, set_action  # noqa: E402

import middleware_host as MH  # noqa: E402
import workspace as W  # noqa: E402
from harness import obligation_gate as OG  # noqa: E402

FAX_HARD = "HAB-fax-hard-1"          # goal: "Determine the appropriate action ... document your findings in a progress note"
PRIOR_AUTH = "HAB-emr-medium-1"      # goal: "Submit a prior authorization request ... and clear the referral"
DENIAL = "HAB-denial-easy-1"         # goal: "Open denial DEN-001 ... Document your reasoning in a triage note"
NOTE = "Reviewed the denial; route to clinical appeals."
SUBMIT = ("submit", {"ref": "submit-disposition-button"})


def _harness(monkeypatch, tmp_path, task_id):
    h = S.Harness(monkeypatch, tmp_path)
    h.use_task(S.load_hab_task(task_id))
    return h


def _clean(h):
    assert h.net == [], h.net
    assert h.portal.unscripted == [], h.portal.unscripted
    assert h.runtime_errors() == [], h.runtime_errors()


class Compositions(object):
    """Pass-through spies: the conjunct paths of every composition the shell made during a run, and the committed goals
    recovery resolved (episodes are not executed)."""

    def __init__(self, h):
        entry = S.recovery_entry()
        self.paths, self.recovery = [], []
        spy = self
        compose = OG.completion_conjunction

        def completion_conjunction(task, *a, **k):
            out = compose(task, *a, **k)
            spy.paths.append(tuple(c["path"] for c in (out[1] or [])))
            return out

        class ResolveOnly(entry.RecoveryKernel):
            def run_all_episodes(self, bench, wf_registry, substrate, driver, task, trajectory, goal, judge,
                                 events=None):
                ctx = self._build_ctx(bench, task, driver)
                spy.recovery.append(self._resolve(bench, task, trajectory, goal, judge, ctx))
                return []
        h.mp.setattr(OG, "completion_conjunction", completion_conjunction)
        h.mp.setattr(entry, "RecoveryKernel", ResolveOnly)


def _leaves(paths):
    return sorted(set(p.split(".")[-1] for p in paths))


# =====================================================================================================================
# completion obligations: the composition's sources
# =====================================================================================================================
def test_d1_fax_hard_composes_no_fax_or_clear_obligation_while_the_agent_commits_neither(monkeypatch, tmp_path):
    """fax-hard (its right outcome is to fax nothing and clear nothing): the agent documents its findings and stops.
    Only the goal's progress note is composed, it holds, nothing nudges and recovery declines. A dme_fax conjunction
    that required fax_sent and referral_cleared unconditionally would make the nudge turn the final back and recovery
    engage to fax and clear."""
    h = _harness(monkeypatch, tmp_path, FAX_HARD)
    h.portal.path = "/emr/dme"
    h.portal.state["emr"] = {"agentActions": {"addedProgressNote": True}}
    h.portal.controls = [control("dme-tab-notes", label="Notes")]
    spy = Compositions(h)
    rows = h.spy_ledger()
    h.run([click("dme-tab-notes"), final()], max_steps=12)
    _clean(h)
    assert spy.paths and all(_leaves(p) == ["addedProgressNote"] for p in spy.paths), spy.paths
    assert rows[0] == (0, "click", True, False, False, "ALLOW")
    assert h.events("obligation_nudge") == []
    assert spy.recovery and all(goals == [] for goals in spy.recovery)


def test_d1_the_agents_own_fax_send_composes_the_fax_obligation(monkeypatch, tmp_path):
    """The same task after the agent itself clicks Send Fax (which does not land): from that action on, the fax it chose
    to send must land. The composition gains fax_sent (the submission, first), the nudge reports the agent's decision,
    and recovery gets the send control as the one commit it may help complete."""
    h = _harness(monkeypatch, tmp_path, FAX_HARD)
    h.portal.path = "/fax-portal"
    h.portal.state["emr"] = {"agentActions": {"addedProgressNote": True}}
    h.portal.controls = [control("dme-tab-notes", label="Notes"), control("send-fax-button", label="Send Fax")]
    spy = Compositions(h)
    lock = []
    snapshot = OG.obligation_snapshot

    def obligation_snapshot(task, env, *a, **k):
        out = snapshot(task, env, *a, **k)
        lock.append((tuple(out), getattr(out, "submissions", None)))
        return out
    monkeypatch.setattr(OG, "obligation_snapshot", obligation_snapshot)
    h.run([click("dme-tab-notes"), click("send-fax-button"), final(), final(), final(), final()], max_steps=12)
    _clean(h)
    assert lock[:2] == [((True,), 0), ((True,), 0)]                      # step 0: only the goal's progress note
    assert lock[2:4] == [((False, True), 1), ((False, True), 1)]         # step 1: the agent's send, before and after
    nudges = h.events("obligation_nudge")
    assert nudges and all(e["decisions"] == ["fax_send"] for e in nudges)
    goals = [g for gs in spy.recovery for g in gs]
    assert goals and all(_leaves(p["path"] for p in g.raw["verify_all"]) == ["addedProgressNote", "faxesSent"]
                         for g in goals)
    assert all(g.raw["verify_spec"]["path"] == "fax.faxesSent" and g.raw["agent_commit_refs"] == ["send-fax-button"]
               for g in goals)


def test_d1_an_agent_committed_submission_creates_the_obligation(monkeypatch, tmp_path):
    """Prior auth: before the agent submits, nothing is composed (the goal's tabs, letter and portal hop are not
    required; the lock inputs read an empty composition as not met). The agent's own Submit (which does not land)
    composes the submission on the goal's payer and the authorization note the goal asks for."""
    h = _harness(monkeypatch, tmp_path, PRIOR_AUTH)
    h.portal.path = "/payer-a/prior-auth"
    h.portal.controls = [control("tab-summary", label="Summary"),
                         control("submit-auth-button", label="Submit Authorization Request")]
    spy = Compositions(h)
    rows = h.spy_ledger()
    h.run([click("tab-summary"), click("submit-auth-button"), final()], max_steps=12)
    _clean(h)
    assert spy.paths[0] == () and rows[0] == (0, "click", False, False, False, "ALLOW")
    after = [p for p in spy.paths if p]
    assert after and all(p == ("payer_a_state.submissions", "agentActions.addedAuthNote") for p in after), after
    assert rows[1][:5] == (1, "click", False, False, False)
    assert h.events("obligation_nudge") and all(e["decisions"] == ["prior_auth_submit"]
                                                for e in h.events("obligation_nudge"))


# =====================================================================================================================
# the harness action service's irreversible-commit rule
# =====================================================================================================================
def _script(h, per_invocation):
    """Invocation n issues per_invocation[n] through the real GUI backend (as test_harness_action_service.py)."""
    entry = S.recovery_entry()
    invocations, queue = [], list(per_invocation)

    class ScriptedRecoveryKernel(object):
        def run_all_episodes(self, bench, wf_registry, substrate, driver, task, trajectory, goal, judge, events=None):
            rec = {"step": driver.step, "results": []}
            invocations.append(rec)
            for tool, args in (queue.pop(0) if queue else []):
                rec["results"].append((tool, substrate.driver.call_tool(tool, dict(args))))
            return []
    h.mp.setattr(entry, "RecoveryKernel", ScriptedRecoveryKernel)
    return invocations


def _denial_form(portal, lands_from_click=1):
    """The disposition form of DEN-001; the goal's other requirements hold. The n-th click on Submit Disposition
    (1-based) persists the disposition when n >= lands_from_click (None: never)."""
    portal.state["emr"] = {"agentActions": {"viewedDenialDetails": True, "documentedAppealInEpic": True}}
    portal.controls = [control("triage-note", role="textbox", label="Triage Note"),
                       control("submit-disposition-button", label="Submit Disposition")]
    clicks = []

    def submit(p, ref):
        clicks.append(ref)
        if lands_from_click is not None and len(clicks) >= lands_from_click:
            p.emr().setdefault("agentActions", {})["selectedDisposition"] = "Route to Clinical Appeals"
            p.text = "Disposition submitted"
    portal.on_click = {"submit-disposition-button": submit}
    return clicks


def _refusals(h):
    return [(e["step"], e["tool"], e["stage"], e["reason"]) for e in h.events("harness_action_refused")]


def test_d2_a_harness_submit_without_an_agent_attempt_is_refused(monkeypatch, tmp_path):
    """The agent wrote the note but never pressed Submit: the harness may not submit for it. An agent write on the
    surface does not count as its commitment."""
    h = _harness(monkeypatch, tmp_path, DENIAL)
    clicks = _denial_form(h.portal)
    invs = _script(h, [[SUBMIT]])
    h.run([fill("triage-note", NOTE), final("f0")], max_steps=12)
    _clean(h)
    assert invs[0]["results"] == [("submit", {"error": "irreversible_commit_no_agent_attempt"})]
    assert _refusals(h) == [(invs[0]["step"], "submit", "irreversible_commit", "irreversible_commit_no_agent_attempt")]
    assert clicks == [] and h.kernels[0].ledger.mutation_authorizations == []


def test_d2_after_an_agent_attempt_that_did_not_land_the_harness_completes_it_once(monkeypatch, tmp_path):
    """The agent pressed Submit and it did not land; the harness's submit of the same control lands, and a second one
    is refused because the commit has landed. The agent's unverified click is resolved by the landed completion, so the
    final is accepted."""
    h = _harness(monkeypatch, tmp_path, DENIAL)
    clicks = _denial_form(h.portal, lands_from_click=2)
    invs = _script(h, [[SUBMIT, SUBMIT]])
    h.run([fill("triage-note", NOTE), click("submit-disposition-button"), final("f0")], max_steps=12)
    _clean(h)
    assert invs[0]["results"][0][0] == "submit" and "error" not in invs[0]["results"][0][1]
    assert invs[0]["results"][1] == ("submit", {"error": "irreversible_commit_already_landed"})
    assert clicks == ["submit-disposition-button"] * 2                    # the agent's and the harness's
    assert h.portal.emr()["agentActions"]["selectedDisposition"] == "Route to Clinical Appeals"
    commits = [(c["action"], c["verified"], c.get("action_origin")) for c in h.kernels[0].ledger.commit_history
               if c.get("semantic_type") == "submit"]
    assert commits == [("click", False, None), ("submit", True, "harness")]
    assert h.kernels[0].ledger.unresolved_operational_commit() is None
    assert [e["event_type"] for e in h.traj if e.get("step") == 2][-1] == "final_answer"


def test_d2_a_second_harness_attempt_after_an_unknown_one_is_refused_across_invocations(monkeypatch, tmp_path):
    """The agent's Submit and the harness's completion both time out (unknown, not landed). At the next final a new
    recovery invocation asks again: refused for the rest of the episode (begin() does not reset the memory)."""
    h = _harness(monkeypatch, tmp_path, DENIAL)
    _denial_form(h.portal, lands_from_click=None)
    timeouts = []

    def timeout(p, ref):
        timeouts.append(ref)
        raise RuntimeError("Timeout 8000ms exceeded.")
    h.portal.on_click = {"submit-disposition-button": timeout}
    invs = _script(h, [[SUBMIT], [SUBMIT]])
    h.run([fill("triage-note", NOTE), click("submit-disposition-button"), final("f0"), final("f1"), final("f2"),
           final("f3")], max_steps=16)
    _clean(h)
    assert len(invs) >= 2
    first, second = invs[0]["results"], invs[1]["results"]
    assert len(first) == 1 and "Timeout" in first[0][1].get("error", "")   # dispatched; it timed out on the page
    assert [e["status"] for e in h.events("tool_call") if e.get("tool") == "submit"] == ["error"]
    assert second == [("submit", {"error": "irreversible_commit_after_unconfirmed"})]
    assert timeouts == ["submit-disposition-button"] * 2                  # the agent's and the one harness attempt
    assert ("submit", "irreversible_commit", "irreversible_commit_after_unconfirmed") in [r[1:] for r in _refusals(h)]


# =====================================================================================================================
# commit classification (the manifest rule through the shell's classifier), the read channels, the observed lock
# =====================================================================================================================
def _hab_manifest():
    import harness as H
    return H.build_kernel(S.load_hab_task(DENIAL), env_type="gui", workspace=W.H0["gui"]).manifest


def test_classification_a_login_lookup_or_navigation_click_is_not_a_commit(monkeypatch):
    monkeypatch.setenv("MH_HARNESS_JUDGE_MODEL", S.JUDGE_MODEL)
    from harness.semantics import canonicalize_on_page
    manifest = _hab_manifest()
    page = [control("login-button", label="Sign In", commit=True),                 # a type=submit login button
            control("eligibility-submit-button", label="Submit"),                   # a lookup
            control("submit-authorizations-link", label="Submit Authorizations"),  # the dashboard link
            control("submit-to-payer-button", label="Submit to Payer"),            # a hop into the payer portal
            control("send-fax-button", label="Send Fax"),
            control("submit-auth-button", label="Submit Authorization Request", commit=True),
            control("submit-appeal-button", label="Submit Appeal"),
            control("submit-disposition-button", label="Submit Disposition")]

    def kind(ref, controls=page, tool="click"):
        sem = canonicalize_on_page({"type": "tool_call", "tool": tool, "args": {"ref": ref}}, manifest, controls)
        return sem.semantic_type, sem.effect
    for ref in ("login-button", "eligibility-submit-button", "submit-authorizations-link", "submit-to-payer-button"):
        assert kind(ref) == ("inspect", "none"), ref
    for ref in ("send-fax-button", "submit-auth-button", "submit-appeal-button", "submit-disposition-button",
                "[submit-disposition-button]"):
        assert kind(ref) == ("submit", "irreversible"), ref
    assert kind("send-fax-button", controls=None) == ("inspect", "none")           # no page, no claim
    assert kind("submit-appeal-button", controls=[control("submit-appeal-button", role="a", label="Submit Appeal")]) \
        == ("inspect", "none")                                                     # a link navigates
    emr_record = [control("submit-appeal-button", label="Record Appeal Submission")]
    assert kind("submit-appeal-button", controls=emr_record) == ("inspect", "none")


def test_classification_the_agents_click_submit_is_a_commit_and_a_login_click_is_not(monkeypatch, tmp_path):
    """In the loop: the agent's click on Submit Disposition enters the kernel's commit lifecycle (commit_history, verified
    on the state change); its click on the payer login button does not."""
    h = _harness(monkeypatch, tmp_path, DENIAL)
    _denial_form(h.portal)
    h.portal.controls.append(control("login-button", label="Sign In", commit=True))
    h.run([click("login-button"), fill("triage-note", NOTE), click("submit-disposition-button"), final()],
          max_steps=12)
    _clean(h)
    commits = [(c["action"], c["semantic_type"], c["effect"], c["verified"], (c["descriptor"] or {}).get("args"))
               for c in h.kernels[0].ledger.commit_history if c.get("semantic_type") == "submit"]
    assert commits == [("click", "submit", "irreversible", True, {"ref": "submit-disposition-button"})]


def test_observed_submission_lock_engages_without_r2(monkeypatch, tmp_path):
    """The agent clears the referral with a click the manifest does not type as a commit (inspect, R0). Its own clear is
    a decision: referral_cleared is the submission, it is observed landing and the composition holds, so the lock
    engages and the next mutation is blocked. The lock does not depend on the kernel's R2 branch."""
    h = _harness(monkeypatch, tmp_path, PRIOR_AUTH)
    h.portal.path = "/emr/referral/REF-2025-001"
    h.portal.controls = [control("clear-from-worklist-button", label="Clear from Worklist"),
                         control("note-input", role="textbox", label="Note")]

    def clear(p, ref):
        p.emr()["clearedReferrals"] = ["REF-2025-001"]
        p.text = "Referral cleared"
    h.portal.on_click = {"clear-from-worklist-button": clear}
    rows = h.spy_ledger()
    h.run([click("clear-from-worklist-button"), fill("note-input", "done"), final()], max_steps=12)
    _clean(h)
    assert rows[0] == (0, "click", True, True, True, "ALLOW")
    assert [c for c in h.kernels[0].ledger.commit_history] == []            # no R2 action ran
    blocks = [(e["stage"], e["rule_id"]) for e in h.events("harness_decision") if e.get("decision") == "BLOCK"]
    assert blocks == [("before_action", "post_verified_mutation")]
    assert not [x for x in h.portal.log if x[0] == "fill"]


def test_read_channels_never_issue_an_irreversible_commit(monkeypatch, tmp_path):
    """A workspace's turn-start read (next_read) naming the Send Fax button is refused before dispatch: recorded, never
    clicked, and the agent takes the turn."""
    h = _harness(monkeypatch, tmp_path, FAX_HARD)
    h.portal.path = "/fax-portal"
    h.portal.controls = [control("send-fax-button", label="Send Fax"), control("tab-summary", label="Summary")]
    ra = W.Workspace(W.H0["gui"]).module("middleware/read_acquire.py").ReadAcquire
    asked = []

    def next_read(self, ctx):
        if asked:
            return None
        asked.append(ctx.step)
        return MH.Read("click", {"ref": "send-fax-button"})
    monkeypatch.setattr(ra, "next_read", next_read)
    h.run([click("tab-summary"), final()], max_steps=12)
    _clean(h)
    assert asked == [0]
    assert [compact(e) for e in h.events("harness_action_refused")] == [
        {"step": 0, "event_type": "harness_action_refused", "action_origin": "harness", "action_requester": "next_read",
         "tool": "click",
         "args": {"ref": "send-fax-button"}, "stage": "read_channel", "reason": "irreversible_commit_on_read_channel",
         "status": "refused"}]
    assert h.events("harness_acquire_dispatch") == []
    assert ("click", "send-fax-button") not in h.portal.log
    assert [e["step"] for e in h.events("tool_call") if e.get("origin") == "agent"] == [0]
