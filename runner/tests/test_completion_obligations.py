# -*- coding: utf-8 -*-
"""A HealthAdminBench task's completion conjunction has one composition, in the fixed shell
(runner/harness/obligation_gate.py), over the workspace's completion_obligations content; the terminal lock, the
obligation nudge (ctx.obligations()) and recovery all read it, so they cannot disagree about what "done" means. Its
sources are the public goal and the agent's own decisions, never a hidden checkpoint or the task type, so the harness
does not leak grading information into the run.

The loop tests drive run.run_task on the scripted portal of runner/tests/inline_hab_stub.py and record each reader where
the loop consults it: the lock's per-conjunct snapshot (harness.obligation_gate.obligation_snapshot, as run.py and the
recovery action service call it), the context's obligations() (middleware_host.obligations_of) and recovery's committed
goals (HabBenchmarkAdapter.resolve_commitments inside the real run_recovery_v3; only the episodes are not executed).
"""
import copy
import json
import os
import shutil
import sys

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inline_hab_stub as S  # noqa: E402
from inline_hab_stub import click, control, final  # noqa: E402

import middleware_host as MH  # noqa: E402
import workspace as W  # noqa: E402
from harness import obligation_gate as OG  # noqa: E402
HabBenchmarkAdapter = W.Workspace(W.H0["gui"]).module("recovery/hab.py").HabBenchmarkAdapter
GenericGuiCompletionWorkflow = W.Workspace(W.H0["gui"]).module("recovery/gui_completion.py").GenericGuiCompletionWorkflow

DME_TASK = "HAB-fax-easy-1"            # "Download the required documents, fax them to the DME supplier, add a progress note"
PRIOR_AUTH_TASK = "HAB-emr-medium-1"   # "Submit a prior authorization request ... Record the authorization number in EMR"
APPEAL_B_TASK = "HAB-denial-medium-3"  # an electronic appeal on the Payer B portal
DENIAL_TASK = "HAB-denial-easy-1"      # "Open denial DEN-001 ... Document your reasoning in a triage note"
CONTENT = os.path.join("policies", "completion_obligations.yaml")
COMMITS = {"disposition_submit": "submit-disposition-button", "appeal_submit": "submit-appeal-button",
           "prior_auth_submit": "submit-auth-button", "fax_send": "send-fax-button",
           "referral_clear": "clear-from-worklist-button"}


def _agent_click(ref):
    return {"event_type": "tool_call", "origin": "agent", "tool": "click", "status": "ok", "args": {"ref": ref}}


ALL_DECISIONS = [_agent_click(r) for r in COMMITS.values()]


class Readers(object):
    """What each reader of the completion conjunction returned during one run (pass-throughs)."""

    def __init__(self, h):
        entry = S.recovery_entry()
        self.lock, self.nudge, self.recovery = [], [], []
        readers = self
        snapshot, obligations_of = OG.obligation_snapshot, MH.obligations_of

        def obligation_snapshot(task, env, *a, **k):
            out = snapshot(task, env, *a, **k)
            readers.lock.append(out)
            return out

        def obligations(task, env, *a, **k):
            out = obligations_of(task, env, *a, **k)
            readers.nudge.append(out)
            return out

        class ResolveOnly(entry.RecoveryKernel):
            def run_all_episodes(self, bench, wf_registry, substrate, driver, task, trajectory, goal, judge,
                                 events=None):
                ctx = self._build_ctx(bench, task, driver)
                goals = self._resolve(bench, task, trajectory, goal, judge, ctx)
                readers.recovery.append({"view": copy.deepcopy(ctx.get("authoritative_state")), "goals": goals})
                return []
        h.mp.setattr(OG, "obligation_snapshot", obligation_snapshot)
        h.mp.setattr(MH, "obligations_of", obligations)
        h.mp.setattr(entry, "RecoveryKernel", ResolveOnly)


def _run(monkeypatch, tmp_path, task_id, state, ref="tab-summary", max_steps=12):
    """One agent click on `ref` (not a control on the page, so never a classified commit), then finals until the loop
    ends, on a portal whose persisted state is `state`."""
    h = S.Harness(monkeypatch, tmp_path)
    h.use_task(S.load_hab_task(task_id))
    for slice_, value in state.items():
        h.portal.state[slice_] = copy.deepcopy(value)
    h.portal.controls = [control("tab-summary", label="Summary")]
    readers = Readers(h)
    rows = h.spy_ledger()
    h.run([click(ref), final()], max_steps=max_steps)
    assert h.net == [] and h.portal.unscripted == [] and h.runtime_errors() == []
    return h, readers, rows


def _truth(conjuncts, state):
    wf = GenericGuiCompletionWorkflow()
    return tuple(bool(wf._check_one(c, state)) for c in conjuncts)


# The agent's own commit, and the portal state. `want` is the one conjunction's truth over GuiEnvReal.eval_state, in the
# composition's order (the effect of the agent's decision first, then the goal's requirements).
_SCENARIOS = {
    # fax_sent; progress_note_added ("add a progress note"); fax_attachment ("required documents") and fax_recipient
    # ("fax them to") because the agent sent a fax. referral_cleared is not composed: the agent did not clear.
    DME_TASK: ("send-fax-button", ("fax_send",),
               {"emr": {"agentActions": {"addedProgressNote": True}, "clearedReferrals": ["REF-1"]},
                "fax": {"faxesSent": 1, "faxRecipient": "555-0100"}},
               (True, True, False, True)),
    # prior_auth_submitted on Payer A; auth_note_added ("Record the authorization number in EMR", after the agent's submit)
    PRIOR_AUTH_TASK: ("submit-auth-button", ("prior_auth_submit",),
                      {"payerA": {"submissions": [{"requestType": "x"}]}, "emr": {"agentActions": {"addedAuthNote": True}}},
                      (True, True)),
    # appeal_submitted on Payer B, viewed_denial_details ("Review the ... denial ... in EMR"), viewed_remittance_image,
    # appeal_documented ("in a triage note")
    APPEAL_B_TASK: ("submit-appeal-button", ("appeal_submit",),
                    {"payerB": {"appealActions": {"submittedAppeal": True}},
                     "emr": {"agentActions": {"documentedAppealInEpic": True, "viewedDenialDetails": True,
                                              "viewedRemittanceImage": True}}},
                    (True, True, True, True)),
}


@pytest.mark.parametrize("task_id", sorted(_SCENARIOS))
def test_the_lock_the_nudge_and_recovery_read_one_conjunction(monkeypatch, tmp_path, task_id):
    ref, decisions, state, want = _SCENARIOS[task_id]
    h, readers, rows = _run(monkeypatch, tmp_path, task_id, state, ref=ref)
    met = all(want)
    p = h.portal.state   # what recovery reads back (HarnessActionService.read_recovery_state), shell view built from it
    readback = {"full_state": p["emr"], "payer_a_state": p["payerA"], "payer_b_state": p["payerB"], "fax": p["fax"]}
    # the lock: every snapshot run.py and the recovery action service took is the conjunction's truth
    assert readers.lock and all(s == want and s.submissions == 1 for s in readers.lock), readers.lock
    assert rows[0] == (0, "click", met, False, False, "ALLOW")   # the submission already held before the click
    # the nudge: ctx.obligations() is the same conjunction over the same state, and it nudges exactly when unmet
    assert readers.nudge and all(o == (met, decisions, want) for o in readers.nudge), readers.nudge
    assert bool(h.events("obligation_nudge")) is (not met)
    # recovery: engages with that conjunction as verify_all where it is unmet, declines where it is met
    assert readers.recovery
    for rec in readers.recovery:
        if met:
            assert rec["goals"] == []
        else:
            assert len(rec["goals"]) == 1
            verify_all = rec["goals"][0].raw["verify_all"]
            assert rec["goals"][0].raw["verify_spec"] == verify_all[0] and verify_all[0]["decision"] == decisions[0]
            assert _truth(verify_all, readback) == want


def _copy_h0(tmp_path, edit):
    """A copy of the default HAB workspace (workspace.H0["gui"]) with its manifest and completion content passed
    through `edit`."""
    ws = str(tmp_path / "hab_ws")
    shutil.copytree(W.H0["gui"], ws)
    mp = os.path.join(ws, "manifest.yaml")
    manifest = yaml.safe_load(open(mp, encoding="utf-8"))
    content = yaml.safe_load(open(os.path.join(ws, CONTENT), encoding="utf-8"))
    edit(manifest, content)
    open(mp, "w", encoding="utf-8").write(yaml.safe_dump(manifest, sort_keys=False))
    open(os.path.join(ws, CONTENT), "w", encoding="utf-8").write(yaml.safe_dump(content, sort_keys=False))
    return ws


def _require_remittance(manifest, content):
    content["requires"].append({"conjunct": "viewed_remittance_image", "when": {"goal_mentions": ["triage note"]}})


_TRIAGED = {"emr": {"agentActions": {"selectedDisposition": "Route to Clinical Appeals", "documentedAppealInEpic": True,
                                     "viewedDenialDetails": True}}}   # the remittance image was never viewed


@pytest.mark.parametrize("workspace, want", [("h0", (True, True)), ("edited", (True, True, False))])
def test_the_workspace_content_is_what_the_lock_the_nudge_and_recovery_compose(monkeypatch, tmp_path, workspace, want):
    if workspace == "edited":
        monkeypatch.setitem(W.H0, "gui", _copy_h0(tmp_path, _require_remittance))
    h, readers, rows = _run(monkeypatch, tmp_path, DENIAL_TASK, _TRIAGED)
    met = all(want)
    assert readers.lock and all(s == want for s in readers.lock)
    assert rows[0] == (0, "click", met, False, False, "ALLOW")   # the goal's requirements; no submission composed
    assert readers.nudge and all(o == (met, (), want) for o in readers.nudge)
    assert bool(h.events("obligation_nudge")) is (not met)
    assert readers.recovery and all((rec["goals"] == []) is met for rec in readers.recovery)


def test_a_workspace_without_completion_obligations_runs_with_no_conjunction(monkeypatch, tmp_path):
    def drop(manifest, content):
        del manifest["completion_obligations"]
    monkeypatch.setitem(W.H0, "gui", _copy_h0(tmp_path, drop))
    h, readers, rows = _run(monkeypatch, tmp_path, DENIAL_TASK, _TRIAGED, ref="submit-disposition-button")
    assert readers.lock and all(s is None for s in readers.lock)
    assert rows[0] == (0, "click", None, None, False, "ALLOW")
    assert readers.nudge and all(o == (True, (), None) for o in readers.nudge)
    assert h.events("obligation_nudge") == []
    assert readers.recovery and all(rec["goals"] == [] for rec in readers.recovery)


class _Env(object):
    def __init__(self, state):
        self.state = state

    def eval_state(self):
        return self.state


def _hab_tasks():
    path = os.path.join(W.REPO, "benchmark_dataprocess", "HealthAdminBench", "tasks_unified.jsonl")
    return [json.loads(line) for line in open(path, encoding="utf-8")]


@pytest.mark.parametrize("variant", ["no_manifest_key", "nothing_to_compose"])
def test_no_completion_content_is_no_conjunction_for_every_task(monkeypatch, tmp_path, variant):
    def edit(manifest, content):
        if variant == "no_manifest_key":
            del manifest["completion_obligations"]
        else:
            content["decisions"], content["requires"] = {}, []
    ws = _copy_h0(tmp_path, edit)
    assert W.Workspace(ws).check([]) == []
    adapter = W.Workspace(ws).module("recovery/hab.py").HabBenchmarkAdapter   # the copy's strategy reads the copy
    for task in _hab_tasks():
        want = None if variant == "no_manifest_key" else []
        assert OG.completion_conjunction(task, ws, actions=ALL_DECISIONS) == ((), want)
        assert OG.gui_obligations(task, _Env({}), ws, actions=ALL_DECISIONS) == (True, (), None)
        assert OG.obligation_snapshot(task, _Env({}), ws, actions=ALL_DECISIONS) == (None if want is None else ())
        assert adapter().resolve_commitments(task, ALL_DECISIONS, "", None, {"authoritative_state": {}}) == []


def test_every_reader_gets_the_composed_conjunction_for_every_hab_task():
    n_goal = 0
    for task in _hab_tasks():
        for actions in ([], ALL_DECISIONS):
            dec, conj = OG.completion_conjunction(task, W.H0["gui"], actions=actions)
            goals = HabBenchmarkAdapter().resolve_commitments(task, actions, "", None, {"authoritative_state": {}})
            if not conj:
                assert actions == [] and goals == [] and OG.obligation_snapshot(task, _Env({}), W.H0["gui"]) == ()
                continue
            n_goal += actions == []
            assert dec == (tuple(COMMITS) if actions else ())
            assert goals[0].raw["verify_all"] == conj
            assert goals[0].raw["verify_spec"] == (conj[0] if actions else None)
            assert goals[0].raw["agent_commit_refs"] == (list(COMMITS.values()) if actions else [])
            assert OG.obligation_snapshot(task, _Env({}), W.H0["gui"], actions=actions) == (False,) * len(conj)
            assert OG.gui_obligations(task, _Env({}), W.H0["gui"], actions=actions) == (False, dec, (False,) * len(conj))
    assert n_goal == 74          # tasks whose public goal alone composes a requirement


def test_editing_the_content_changes_the_composition(tmp_path):
    task = [t for t in _hab_tasks() if t["task_id"] == DENIAL_TASK][0]
    submit = [_agent_click("submit-disposition-button")]
    h0 = [c["path"].split(".")[-1] for c in OG.completion_conjunction(task, W.H0["gui"], actions=submit)[1]]
    assert h0 == ["selectedDisposition", "viewedDenialDetails", "documentedAppealInEpic"]

    def edit(manifest, content):
        content["requires"] = [
            {"conjunct": "viewed_remittance_image", "when": {"goal_mentions": ["Triage Note"]}},
            {"conjunct": "referral_cleared", "when": {"goal_mentions": ["no such phrase"]}},
            {"conjunct": "progress_note_added", "when": {"agent_decision": "fax_send"}}]
        content["conjuncts"]["disposition_selected"]["alt_paths"] = []
    ws = _copy_h0(tmp_path, edit)
    dec, conj = OG.completion_conjunction(task, ws, submit)
    assert "triage note" in task["goal"].lower()                 # phrases match case-insensitively
    assert dec == ("disposition_submit",)
    assert [c["path"] for c in conj] == ["agentActions.selectedDisposition",
                                         "full_state.agentActions.viewedRemittanceImage"]
    assert conj[0]["alt_paths"] == [] and conj[0]["decision"] == "disposition_submit"
    assert [c["path"].split(".")[-1] for c in OG.completion_conjunction(task, ws, submit + [_agent_click(
        "send-fax-button")])[1]][-1] == "addedProgressNote"      # the agent's fax decision turns its requirement on
    assert [c["path"].split(".")[-1] for c in OG.completion_conjunction(task, W.H0["gui"], actions=submit)[1]] == h0   # the default workspace is unchanged


def test_the_appeal_submission_is_the_goals_payer_where_eval_state_puts_it():
    appeals = [t for t in _hab_tasks() if t["environment"]["config"]["task_type"] == "electronic_appeals"]
    assert len(appeals) == 8
    for task in appeals:
        own = "b" if task["environment"]["config"]["payer_portal"] == "payer_b" else "a"   # the test's oracle only
        other = "a" if own == "b" else "b"
        sub = OG.completion_conjunction(task, W.H0["gui"], actions=[_agent_click("submit-appeal-button")])[1][0]
        assert [sub["path"]] + sub["alt_paths"] == ["payer_%s_state.appealActions.submittedAppeal" % own,
                                                    "payer_%s_state.full_state.appealActions.submittedAppeal" % own]
        filed = lambda payer: {"payer_%s_state" % payer: {"full_state": {"appealActions": {"submittedAppeal": True}}}}
        raw = lambda payer: {"payer_%s_state" % payer: {"appealActions": {"submittedAppeal": True}}}
        assert OG.conjunct_met(sub, filed(own)) and OG.conjunct_met(sub, raw(own))       # eval_state and raw views
        assert not OG.conjunct_met(sub, filed(other)) and not OG.conjunct_met(sub, raw(other))


@pytest.mark.parametrize("edit, problem", [
    (lambda m, c: c["decisions"]["fax_send"].update(effect="nope"), "effect must name a conjunct id"),
    (lambda m, c: c["decisions"]["fax_send"].update(match={"ref_pattern": "^x$"}), "match must declare a tool"),
    (lambda m, c: c["decisions"]["fax_send"]["match"].update(ref_pattern="("), "ref_pattern is not a regex"),
    (lambda m, c: c["requires"].append("nope"), "unknown conjunct 'nope'"),
    (lambda m, c: c["conjuncts"]["fax_sent"].update(check="exists"), "check must be one of"),
    (lambda m, c: c["conjuncts"]["fax_sent"].update(path="fax.{member}.faxesSent"), "the only placeholder"),
    (lambda m, c: c["requires"].append({"conjunct": "fax_sent", "when": {"goal": ["x"]}}), "when must map some of"),
    (lambda m, c: c["requires"].append({"conjunct": "fax_sent", "when": {"agent_decision": "nope"}}),
     "agent_decision must name a decision id"),
    (lambda m, c: c["requires"].append({"conjunct": "fax_sent", "when": {"goal_matches": ["("]}}), "is not a regex"),
    (lambda m, c: c.update(task_types={}), "unknown key 'task_types'"),
    (lambda m, c: m.update(completion_obligations="policies/missing.yaml"), "missing"),
])
def test_content_the_engine_cannot_compose_refuses_the_workspace(tmp_path, edit, problem):
    ws = _copy_h0(tmp_path, edit)
    problems = W.Workspace(ws).check([])
    assert len(problems) == 1 and problem in problems[0], problems
    with pytest.raises(OG.CompletionObligationsError):
        OG.load_completion_obligations(ws)


def test_the_h0_content_loads_and_the_run_records_it_as_loaded(monkeypatch):
    from instrumentation import Instrumentation, record_components
    monkeypatch.delenv("MH_HARNESS_JUDGE_MODEL", raising=False)
    import harness as H
    monkeypatch.setattr(H, "_build_judge", lambda: (None, None))   # no judge wired (build_kernel requires one)
    assert W.Workspace(W.H0["gui"]).check([]) == []
    task = [t for t in _hab_tasks() if t["task_id"] == DENIAL_TASK][0]
    inst = Instrumentation()
    record_components(inst, None, None, H.build_kernel(task, env_type="gui", workspace=W.H0["gui"]), W.H0["gui"])
    assert [c["path"] for c in inst.components if c["role"] == "completion_obligations"] == [
        "workspaces/hab/h0/policies/completion_obligations.yaml"]
