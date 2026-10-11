# -*- coding: utf-8 -*-
"""Final purity, escalation, tracing, cache and environment fidelity in the shell:

    final purity     a verify_final query records nothing and moves no counter; the delivered verdict is recorded once
    escalation       intervention exhaustion stops issuing advisory verdicts without ending the episode; a safety
                     ESCALATE ends it
    tracing          kernel events carry the step; every hook result the shell applies is one shell_hook event;
                     refusals are events
    cache            the semantic cache is scoped per task, keyed on the judge model and the prompt
    env fidelity     PhysicianBench write_file mode "a" appends (upstream tools/file_tools.py); an episode-end tool
                     (HealthAdminBench done()) ends the episode through the final
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import harness as H
import middleware_host as MH
import workspace as W
from harness import decision as D
from harness.capability import Capability
from harness.compiler import build_contract
from harness.kernel import HarnessKernel

import inline_loop_rig as L

PB = "PhysicianBench"
PB_TASK = "PB-pretransplant_covid_clearance"
FHIR_POLICY = {"manifest": {
    "subject": {"type": "patient", "id_context_keys": ["patient_ref"], "from_args": ["patient"]},
    "actions": [{"match": {"tool": "fhir_search"}, "semantic_type": "read", "effect": "none",
                 "produces_evidence": {"source_class": "record", "resource_from_args": ["resource_type"]}}]}}
FHIR_TASK = {"task_id": "t1", "goal": "g", "context": {"patient_ref": "Patient/123"}, "environment": {"type": "fhir"}}
READ = {"type": "tool_call", "tool": "fhir_search", "args": {"resource_type": "Observation"}}


class Scripted(Capability):
    """A capability whose verdict at each hook is a function of the call count (None: no opinion)."""
    name = "scripted_s2"

    def __init__(self, before_action=None, after_action=None, before_final=None):
        self.fns = {"before_action": before_action, "after_action": after_action, "before_final": before_final}
        self.n = {"before_action": 0, "after_action": 0, "before_final": 0}

    def _call(self, hook):
        self.n[hook] += 1
        fn = self.fns[hook]
        return fn(self.n[hook]) if fn else None

    def before_action(self, action, ctx):
        return self._call("before_action")

    def after_action(self, action, result, before_state, after_state, ctx):
        return self._call("after_action")

    def before_final(self, answer, ctx):
        return self._call("before_final")


def _decision(kind, **extra):
    return D.HarnessDecision(kind, capability=Scripted.name, rule_id="scripted_%s" % kind.lower(),
                             reason_code="scripted", reason="scripted %s" % kind, deterministic=True, extra=extra)


def _final_kernel(cap, **kw):
    """A MedCTA-substrate kernel, whose final answer is a commit point, with `cap` after the shell's capabilities."""
    pol = H.load_policy(env_type="tool_sandbox", workspace=W.H0["tool_sandbox"])
    task = {"task_id": "m", "goal": "finding?", "context": {}, "environment": {"type": "tool_sandbox"}}
    contract = build_contract(task, env_type="tool_sandbox", policy=pol)
    return HarnessKernel(contract, [cap], policy=pol, env_type="tool_sandbox", **kw)


def _fhir_kernel(cap, **kw):
    return HarnessKernel(build_contract(FHIR_TASK, env_type="fhir", policy=FHIR_POLICY), [cap], policy=FHIR_POLICY,
                         env_type="fhir", **kw)


def _state(k):
    return (json.dumps(k.ledger.to_dict(), sort_keys=True, default=str), sorted(map(str, k._rev_for_action.items())),
            k._n_interventions, k._evk)


def _final_commits(k):
    return [c for c in k.ledger.commit_history if c.get("detail") == "final_answer"]


# ---------------------------------------------------------------------------------------------------- final purity
def test_verify_final_queries_leave_ledger_and_counters_unchanged():
    k = _final_kernel(Scripted(before_final=lambda n: _decision("REVISE")))
    traj = []
    services = MH.answer_services(k, 3, traj)
    before = _state(k)
    verdicts = [services["verify_final"]("the same answer") for _ in range(4)]
    assert [v.type for v in verdicts] == ["REVISE"] * 4          # a repeated identical ask does not escalate: a query moves no counter
    assert _state(k) == before

    k2 = _final_kernel(Scripted())                                # ALLOW: a query records no commit either
    s2 = MH.answer_services(k2, 1, [])
    assert [s2["verify_final"]("clean answer").type for _ in range(3)] == ["ALLOW"] * 3
    assert _final_commits(k2) == []
    assert k2.before_final("clean answer", step=2).type == "ALLOW" and len(_final_commits(k2)) == 1


def test_the_delivered_verdict_is_recorded_once_as_the_real_verdict():
    k = _final_kernel(Scripted())
    traj = []
    if hasattr(k, "bind_trajectory"):
        k.bind_trajectory(traj)
    ctx = MH.MiddlewareContext(env=None, task={}, trajectory=traj, step=5, max_steps=10, used_actions=0,
                               judge_factory=lambda consumer: None, errors=[], shell=MH.answer_services(k, 5, traj))
    for _ in range(3):
        ctx.verify_final("B")
    real = MH.delivered_verdict(ctx, "B")
    assert real.type == "ALLOW" and len(_final_commits(k)) == 1
    assert [e["hook"] for e in traj if e.get("event_type") == "shell_hook"] == ["before_final"]


# ---------------------------------------------------------------------------------------------------- escalation
def test_intervention_exhaustion_is_not_fatal_but_a_safety_escalate_is():
    k = _fhir_kernel(Scripted(after_action=lambda n: _decision("REVISE")))
    k._n_interventions = k.budget["max_interventions_per_task"]
    eff = k.after_action(READ, {"entries": []}, None, None, step=4)
    assert (eff.type, eff.note) == ("ALLOW", "budget_exhausted_interventions")
    assert eff.events[0]["effective"] == "ALLOW" and eff.events[0]["note"] == "budget_exhausted_interventions"
    for safety in (_decision("ESCALATE"), _decision("BLOCK"), _decision("REVISE", must_resolve=True)):
        ks = _fhir_kernel(Scripted(after_action=lambda n, d=safety: d))
        ks._n_interventions = ks.budget["max_interventions_per_task"]
        assert ks.after_action(READ, {"entries": []}, None, None, step=4).type == safety.type


def test_in_the_loop_exhaustion_goes_on_and_a_safety_escalate_ends_the_episode(monkeypatch, tmp_path):
    """8 after_action REVISEs spend the intervention budget, the 9th is not issued and the episode goes on, the 10th is
    a safety ESCALATE and ends it."""
    rig = L.Rig(monkeypatch, tmp_path, PB, PB_TASK, protocol="function_calling")
    mrn = rig.task()["context"]["patient_ref"]
    rig.tool_result = lambda name, args: {"entries": [{"resourceType": "Observation", "id": "obs-%d" % len(rig.env_calls),
                                                       "subject": {"reference": "Patient/" + mrn},
                                                       "code": {"text": "lab %d" % len(rig.env_calls)}}], "total": 1}
    n = [0]

    def after(action, ctx, result):
        n[0] += 1
        return (L.decision("ESCALATE", rule_id="scripted_safety", reason="safety") if n[0] == 10 else
                L.decision("REVISE", rule_id="scripted_advice", reason="advice %d" % n[0]))
    rig.hooks.after_action = after
    rig.replies = [L.fc_tool("fhir_observation_search_labs", {"patient": "Patient/" + mrn, "count": i},
                             call_id="c%d" % i) for i in range(1, 12)] + [L.fc_final("answer")]
    rig.run(max_steps=14)
    post = [e for e in rig.events("harness_decision") if e["stage"] == "after_action"]
    assert [e["effective"] for e in post] == ["REVISE"] * 8 + ["ALLOW", "ESCALATE"]
    assert post[8]["note"] == "budget_exhausted_interventions"
    assert len(rig.events("tool_call")) == 10
    assert [(e["stage"], e["status"]) for e in rig.events("harness_escalation")] == [("after_action", "error")]


# ---------------------------------------------------------------------------------------------------- tracing
def test_kernel_events_carry_the_step_and_every_hook_result_is_a_shell_event(monkeypatch):
    k = _fhir_kernel(Scripted(before_action=lambda n: _decision("REVISE")), judge_fn=lambda p: "x",
                     budget={"max_semantic_checks": 0})
    traj = []
    k.bind_trajectory(traj)
    eff = k.before_action(READ, step=7)
    assert eff.events[0]["event_type"] == "harness_decision" and eff.events[0]["step"] == 7
    k.after_action(READ, {"entries": []}, None, None, step=7, origin="harness", requester="recovery")
    k.evaluate_final("a", step=8)                                  # a query: no hook result
    k.before_final("a", step=8)
    hooks = [e for e in traj if e["event_type"] == "shell_hook"]
    assert [(e["layer"], e["hook"], e["step"], e["origin"], e["requester"]) for e in hooks] == [
        ("kernel", "before_action", 7, "agent", None), ("kernel", "after_action", 7, "harness", "recovery"),
        ("kernel", "before_final", 8, "agent", None)]
    assert hooks[0]["result"] == {"effective": "REVISE", "raw": "REVISE", "capability": Scripted.name,
                                  "rule_id": "scripted_revise", "note": None, "decision_event": eff.events[0]["id"]}
    # a semantic unit the meter refuses is an event
    assert k.ctx.semantic_check("middleware") is None
    assert [(e["kind"], e["consumer"], e["reason"]) for e in traj if e["event_type"] == "shell_refusal"] == [
        ("semantic_unit", "middleware", "meter")]

    # host: the one event of a hook result, and the instrumentation summary beside the kind
    import instrumentation as I

    class Nudger(object):
        def before_tool(self, ctx, action):
            return MH.Nudge("before_tool", "scripted nudge", "obs", "CONTINUE")
    monkeypatch.setattr(I.W, "component_path", lambda obj: "/ws/middleware/nudger.py")
    monkeypatch.setattr(I.W, "repo_path", lambda p: "middleware/nudger.py")
    inst = I.Instrumentation()
    htraj = []
    ctx = MH.MiddlewareContext(env=None, task={}, trajectory=htraj, step=4, max_steps=10, used_actions=0,
                               judge_factory=lambda c: None, errors=[])
    assert isinstance(MH.MiddlewareStack([Nudger()], inst).before_tool(ctx, {"tool": "x"}), MH.Nudge)
    assert htraj == [{"event_type": "shell_hook", "layer": "host", "hook": "before_tool", "step": 4, "origin": "harness",
                      "requester": "middleware:Nudger", "call_id": "before_tool@4",
                      "result": {"kind": "Nudge", "stage": "before_tool", "label": "CONTINUE",
                                 "feedback": "scripted nudge"}}]
    assert inst.calls[0]["returned"] == "Nudge" and inst.calls[0]["summary"] == htraj[0]["result"]


def test_tool_allowance_refusal_is_an_event(monkeypatch, tmp_path):
    rig = L.Rig(monkeypatch, tmp_path, PB, PB_TASK, protocol="function_calling")
    mrn = rig.task()["context"]["patient_ref"]
    rig.replies = [L.fc_tool("fhir_observation_search_labs", {"patient": "Patient/" + mrn, "count": i}, call_id="c%d" % i)
                   for i in (1, 2)] + [L.fc_final("answer")]
    rig.run(max_steps=1)
    assert [(e["kind"], e["tool"], e["step"]) for e in rig.full_trajectory if e.get("event_type") == "shell_refusal"] \
        == [("tool_allowance", "fhir_observation_search_labs", 1)]


# ---------------------------------------------------------------------------------------------------- semantic cache
def test_semantic_cache_is_scoped_per_task_and_keyed_on_the_judge(monkeypatch, tmp_path):
    from harness.engines import semantic as S
    from harness.observation import Claim

    def judge(model, reply, calls):
        def j(prompt):
            calls.append(prompt)
            return reply
        j.judge_model = model
        return j
    claims = [Claim(claim_id="claim-0", idx=0, claim_type="perceptual", text="nodule", region="RUL", modality="ct",
                    attribute="nodule")]
    ca, cb = [], []
    ja = judge("judge-a", '{"support": {"claim-0": true}}', ca)
    jb = judge("judge-b", '{"support": {"claim-0": false}}', cb)
    S.sem_cache_reset("T1")
    assert S.claim_semantic_support(claims, ["obs"], ja) == {"claim-0": True}
    assert S.claim_semantic_support(claims, ["obs"], jb) == {"claim-0": False}     # another judge is not a hit
    assert S.claim_semantic_support(claims, ["obs"], ja) == {"claim-0": True} and len(ca) == 1
    S._SEM_CACHE["stale-from-another-task"] = "stale"

    rig = L.Rig(monkeypatch, tmp_path, PB, PB_TASK, protocol="function_calling")
    rig.replies = [L.fc_final("answer")]
    rig.run(max_steps=2)
    assert "stale-from-another-task" not in S._SEM_CACHE and S._SEM_SCOPE["task_id"] == PB_TASK


# ---------------------------------------------------------------------------------------------------- env fidelity
def test_pb_write_file_append_appends(tmp_path):
    import environments as E
    env = E.PbNativeEnv.__new__(E.PbNativeEnv)
    env.workspace, env._written, env._created = str(tmp_path), [], []
    env.call_tool("write_file", {"file_path": "note.md", "content": "part one\n"})
    env.call_tool("write_file", {"file_path": "note.md", "content": "part two\n", "mode": "a"})
    assert (tmp_path / "note.md").read_text() == "part one\npart two\n"
    env.call_tool("write_file", {"file_path": "note.md", "content": "fresh", "mode": "w"})
    assert (tmp_path / "note.md").read_text() == "fresh"

    k = _fhir_kernel(Scripted())                    # the kernel's record of the deliverable follows the same rule
    k._note_deliverable({"tool": "write_file", "args": {"path": "n.md", "content": "one "}}, True)
    k._note_deliverable({"tool": "write_file", "args": {"path": "n.md", "content": "two", "mode": "a"}}, True)
    assert k.ctx.last_write["content"] == "one two"


def test_episode_end_tool_ends_the_episode_through_the_final(monkeypatch, tmp_path):
    import environments as E
    rig = L.Rig(monkeypatch, tmp_path, PB, PB_TASK, protocol="function_calling")
    monkeypatch.setattr(E.PbNativeEnv, "EPISODE_END_TOOLS", ("done",), raising=False)
    rig.replies = [L.fc_tool("done", {}), L.fc_final("the agent is not asked again")]
    rig.run(max_steps=5)
    assert len(rig.states) == 1
    assert rig.types() == ["answer_attempt", "final_answer"]
    assert [e["tool"] for e in rig.full_trajectory if e.get("event_type") == "shell_end_signal"] == ["done"]
    assert rig.events("final_answer")[0]["thought"] == "" and rig.result["budget"]["agent_env_actions"] == 0
    assert E.GuiEnvReal.EPISODE_END_TOOLS == ("done",)          # HealthAdminBench declares done() as its end
