# -*- coding: utf-8 -*-
"""Workspace code is isolated from the shell.

Workspace code (agent programs, host and kernel middleware, the modules they load) gets the public task only (no
checkpoints, reference, expected outcome, scoring or policy), no raw environment (read-only views instead), a read-only
ledger, trajectory and observed history with narrow write services, no raw judge and no route back to the shell's
objects, and it may import from the shell only runner/workspace_api.py. These tests pin each of those boundaries, so
that code placed in a workspace can neither read the hidden grading fields nor alter the shell's own state."""
import os
import shutil
import sys

import pytest
import yaml

_RUNNER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RUNNER not in sys.path:
    sys.path.insert(0, _RUNNER)

import harness as H                                                    # noqa: E402
import middleware_host as MH                                           # noqa: E402
import workspace as W                                                  # noqa: E402
from harness import decision as D                                      # noqa: E402
from harness.capabilities.verify_commit import VerifyAndCommit         # noqa: E402
from harness.capability import WorkspaceContextView                    # noqa: E402
from harness.compiler import build_contract                            # noqa: E402
from harness.kernel import HarnessKernel                               # noqa: E402

HIDDEN = {"checkpoints": [{"id": "cp1", "expected": "gold"}], "reference": {"answer": "gold"},
          "expected_outcome": "gold", "scoring": {"w": 1}, "policy": {"forbidden_actions": ["x"]}}
FULL_TASK = dict(HIDDEN, task_id="T-1", goal="Write your note to output/note.md", context={"text": "ctx"},
                 available_tools=[{"name": "write_file"}],
                 environment={"type": "gui", "config": {"task_type": "denial_triage", "challenge_type": "workflow",
                                                        "payer_portal": "payer_a", "native_config": {"k": "v"},
                                                        "possible": True}})


def _ctx(env=None, task=None, traj=None, shell=None, step=2):
    return MH.MiddlewareContext(env=env, task=task if task is not None else FULL_TASK,
                                trajectory=traj if traj is not None else [], step=step, max_steps=10, used_actions=1,
                                judge_factory=lambda: None, errors=[], shell=shell)


def _workspace(tmp_path, files, manifest=None):
    """A workspace directory holding `files` ({relative path: source}) and a manifest."""
    ws = str(tmp_path / "ws")
    os.makedirs(ws)
    m = dict({"benchmark": "HealthAdminBench", "env_type": "gui"}, **(manifest or {}))
    open(os.path.join(ws, "manifest.yaml"), "w").write(yaml.safe_dump(m))
    for rel, src in files.items():
        p = os.path.join(ws, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").write(src)
    return ws


# --------------------------------------------------------------------------------------------------------- public task
def test_middleware_gets_the_public_task_only(tmp_path):
    ctx = _ctx()
    task = ctx.task
    for key in ("checkpoints", "reference", "expected_outcome", "scoring", "policy"):
        assert key not in task
    assert task["environment"]["config"] == {"challenge_type": "workflow", "payer_portal": "payer_a"}
    assert (task["goal"], task["context"]) == (FULL_TASK["goal"], FULL_TASK["context"])
    task["context"]["text"] = "rewritten"                                  # a copy: the shell's task is unchanged
    assert FULL_TASK["context"]["text"] == "ctx" and ctx.task["context"]["text"] == "ctx"

    ws = _workspace(tmp_path, {"middleware/probe.py": "SEEN = []\n\nclass Probe(object):\n"
                                                      "    def __init__(self, task):\n        SEEN.append(task)\n"},
                    {"middleware": ["middleware/probe.py:Probe"]})
    MH.load_middleware(ws, FULL_TASK)
    seen = W.Workspace(ws).module("middleware/probe.py").SEEN
    assert len(seen) == 1 and "checkpoints" not in seen[0] and "reference" not in seen[0]
    assert "task_type" not in seen[0]["environment"]["config"] and "native_config" not in seen[0]["environment"]["config"]


# --------------------------------------------------------------------------------------------------------- no raw env
class _Env(object):
    """An environment that records every tool call and page action a view might make."""

    def __init__(self, tmp_path):
        self.calls = []
        self.controls = [{"ref": "submit", "role": "button", "label": "Submit", "value": ""}]
        self.state = {"full_state": {"faxPortal": {"attachmentNames": []}}}
        self._download_registry = {"r1": {"path": "/job/workspace/auth_letter.pdf", "hash": "h"}}
        self.workspace = str(tmp_path / "out")
        os.makedirs(self.workspace)
        open(os.path.join(self.workspace, "note.md"), "w").write("the note")

    def call_tool(self, name, args):
        self.calls.append((name, args))
        return {"ok": True}

    def list_controls(self):
        return self.controls

    def eval_state(self):
        return self.state


def test_there_is_no_raw_environment(tmp_path):
    ctx = _ctx(env=_Env(tmp_path))
    with pytest.raises(AttributeError):
        ctx.env
    with pytest.raises(AttributeError):
        getattr(ctx, "env")
    with pytest.raises(AttributeError):
        ctx.env = object()


def test_the_environment_views_are_copies_and_cannot_call_a_tool(tmp_path):
    env = _Env(tmp_path)
    ctx = _ctx(env=env)
    controls, state = ctx.controls(), ctx.persisted_state()
    assert controls == env.controls and controls is not env.controls
    assert state == env.state and state is not env.state
    assert ctx.downloads() == ["auth_letter.pdf"]
    assert ctx.output_files() == {"exists": True, "files": {"note.md": {"size": 8, "text": "the note"}}}
    controls[0]["value"] = "typed"
    state["full_state"]["faxPortal"]["attachmentNames"].append("x")
    assert env.controls[0]["value"] == "" and env.state["full_state"]["faxPortal"]["attachmentNames"] == []
    for view in (ctx.controls(), ctx.persisted_state(), ctx.downloads(), ctx.output_files()):
        assert not hasattr(view, "call_tool")
    assert not hasattr(ctx, "call_tool") and env.calls == []
    # acting needs a harness-action invocation (the shell's service), which a view never opens
    with pytest.raises(RuntimeError):
        ctx.request_action("click", {"ref": "submit"})
    assert env.calls == []


# --------------------------------------------------------------------------------------------------------- read-only state
def test_the_trajectory_is_read_only_and_emit_is_the_record():
    traj = [{"step": 0, "event_type": "tool_call", "tool": "click", "args": {"ref": "a"}}]
    ctx = _ctx(traj=traj)
    view = ctx.trajectory
    assert not hasattr(view, "append") and not hasattr(view, "extend")
    with pytest.raises(TypeError):
        view[0] = {}
    view[0]["args"]["ref"] = "rewritten"
    assert traj[0]["args"]["ref"] == "a" and list(view) == traj
    with pytest.raises(ValueError):
        ctx.emit({"event_type": "tool_call", "tool": "click"})              # a reserved record
    ctx.emit({"event_type": "probe_note", "n": 1})
    assert traj[-1] == {"event_type": "probe_note", "n": 1, "step": 2} and len(view) == 2


class _Ledger(object):
    terminal_locked = False
    pending_resolution = None


def _kernel(tmp_path, source, judge_reply="{}"):
    """MedCTA's policy, verify_commit, and one kernel middleware loaded from a workspace file (so it gets the view)."""
    ws = _workspace(tmp_path, {"middleware/probe.py": source},
                    {"benchmark": "MedCTA", "env_type": "tool_sandbox", "kernel_middleware": ["middleware/probe.py:Probe"]})
    pol = H.load_policy(env_type="tool_sandbox", workspace=W.H0["tool_sandbox"])
    task = {"task_id": "m", "goal": "finding?", "context": {}, "environment": {"type": "tool_sandbox"}}
    contract = build_contract(task, env_type="tool_sandbox", policy=pol)
    probe_cls = W.Workspace(ws).load("middleware/probe.py", "Probe")
    k = HarnessKernel(contract, [VerifyAndCommit(), probe_cls()], policy=pol, env_type="tool_sandbox",
                      judge_fn=lambda p: judge_reply, budget={"max_semantic_checks": 5})
    return k, sys.modules[probe_cls.__module__]


OVERREACH = '''
from workspace_api import Capability

LOG = []


class Probe(Capability):
    name = "probe"

    def before_action(self, action, ctx):
        action["args"]["ref"] = "rewritten"                   # its own copy
        for field, value in (("terminal_locked", True), ("pending_resolution", {"violation_type": "forged"})):
            try:
                setattr(ctx.ledger, field, value)
                LOG.append((field, "written"))
            except AttributeError:
                LOG.append((field, "refused"))
        try:
            ctx.ledger.reserve_authorization(None)
            LOG.append(("reserve_authorization", "called"))
        except AttributeError:
            LOG.append(("reserve_authorization", "refused"))
        ctx.workspace_state["seen"] = ctx.workspace_state.get("seen", 0) + 1
        return None
'''


def test_writing_terminal_locked_or_pending_resolution_through_the_view_fails(tmp_path):
    k, mod = _kernel(tmp_path, OVERREACH)
    action = {"type": "tool_call", "tool": "ImageDescription", "args": {"ref": "img"}}
    k.before_action(action, step=0)
    assert mod.LOG == [("terminal_locked", "refused"), ("pending_resolution", "refused"),
                       ("reserve_authorization", "refused")]
    assert k.ledger.terminal_locked is False and k.ledger.pending_resolution is None
    assert action["args"]["ref"] == "img"                      # the shell's action is not the middleware's copy
    assert k._capability_errors == [] and k._workspace_state == {"seen": 1}
    view = WorkspaceContextView(k.ctx)
    with pytest.raises(AttributeError):
        view.ledger.terminal_locked = True
    with pytest.raises(AttributeError):
        view.ledger.pending_resolution = {"violation_type": "forged"}
    with pytest.raises(AttributeError):
        view.ledger = _Ledger()
    host = MH.LedgerView(k.ledger)
    with pytest.raises(AttributeError):
        host.terminal_locked = True
    assert k.ledger.terminal_locked is False and k.ledger.pending_resolution is None


def test_the_view_leads_nowhere_near_the_raw_context_or_judge(tmp_path):
    k, _mod = _kernel(tmp_path, OVERREACH)
    view = k._ctx_for(k.capabilities[-1])
    assert type(view) is WorkspaceContextView
    for name in ("_ctx", "judge_fn", "semantic_meter", "spend_semantic", "spend_verify_semantic", "semantic_remaining",
                 "n_general_spent",
                 "__dict__"):
        with pytest.raises(AttributeError):
            getattr(view, name)
    for name in ("_ledger", "__dict__", "add_evidence", "mint_authorization", "record_commit"):
        with pytest.raises(AttributeError):
            getattr(view.ledger, name)
    contract = view.contract
    contract.meta["goal"] = "rewritten"
    assert k.ctx.contract.meta["goal"] == "finding?"          # a copy
    ctx = _ctx()
    for name in ("_shell", "_judge_factory", "_errors", "_verdicts", "_forced", "env", "__dict__"):
        with pytest.raises(AttributeError):
            getattr(ctx, name)
    for obj in (MH.LedgerView(k.ledger), ctx.trajectory):
        for name in ("_ledger", "_items", "__dict__"):
            with pytest.raises(AttributeError):
                getattr(obj, name)


# --------------------------------------------------------------------------------------------------------- imports
def test_a_non_allow_listed_import_is_refused_at_load_and_statically(tmp_path):
    ws = _workspace(tmp_path, {
        "a.py": "import json\nfrom harness import state\n",
        "b.py": "import json\n\ndef later():\n    import sys\n    return sys\n\ndef codecs():\n    return json.codecs\n",
        "c.py": "from workspace_api import Revision, load_relative\nimport re\nOK = Revision('s', 'f', 'o')\n",
        "d.py": ("import middleware_host\n\ndef f(ctx, name):\n    g = getattr\n    return (ctx.__class__, eval('1'), "
                 "open('/x'), getattr(ctx, name), g(ctx, 'x'), ctx.gi_frame)\n"),
    })
    with pytest.raises(W.WorkspaceImportError):
        W._module(os.path.join(ws, "a.py"))
    b = W._module(os.path.join(ws, "b.py"))
    with pytest.raises(W.WorkspaceImportError):
        b.later()                                             # the guard covers an import run later, at call time
    with pytest.raises(AttributeError):
        b.codecs()                                            # a standard-library module is a facade: no module attributes
    assert W._module(os.path.join(ws, "c.py")).OK.stage == "s"
    with pytest.raises(W.WorkspaceImportError):
        W._module(os.path.join(ws, "d.py"))
    problems = W.Workspace(ws).check([])
    for expected in ("a.py:2: from harness import is refused", "b.py:4: import sys is refused",
                     "d.py:1: import middleware_host is refused", "d.py:5: the attribute __class__ is refused",
                     "d.py:5: the name eval is refused", "d.py:5: the name open is refused",
                     "d.py:4: getattr is allowed only as a direct call with a literal attribute name",
                     "d.py:5: getattr is allowed only as a direct call with a literal attribute name",
                     "d.py:5: the attribute gi_frame is refused"):
        assert any(p.endswith(expected) for p in problems), (expected, problems)
    assert not any(p.startswith("c.py") for p in problems)


# --------------------------------------------------------------------------------------------------------- H0 runs
def _guarded_loads(monkeypatch):
    """(file, ran under the workspace builtins) for every workspace module loaded from now on."""
    seen, orig = [], W._module

    def module(path):
        mod = orig(path)
        seen.append((os.path.basename(path), mod.__dict__.get("__builtins__") is W.WORKSPACE_BUILTINS))
        return mod
    monkeypatch.setattr(W, "_module", module)
    return seen


def test_h0_workspaces_pass_the_static_check():
    for env_type in ("fhir", "gui", "tool_sandbox"):
        assert W.isolation_problems(W.H0[env_type]) == [], env_type


def test_h0_runs_scripted_on_physicianbench(monkeypatch, tmp_path):
    import inline_loop_rig as L
    loads = _guarded_loads(monkeypatch)
    rig = L.Rig(monkeypatch, tmp_path, "PhysicianBench", "PB-buprenorphine_mood_support", protocol="function_calling")
    mrn = rig.task()["context"]["patient_ref"]
    write = L.fc_tool("write_file", {"file_path": "/workspace/output/management_plan.txt", "content": "the plan"})
    rig.replies = ([L.fc_tool("fhir_observation_search_labs", {"patient": "Patient/" + mrn})]
                   + [L.fc_final("answer %d" % i) for i in range(1, 5)] + [write])
    rig.run(max_steps=10)
    assert W.isolation_problems(rig.workspace) == []
    assert {"deliverable.py", "goal_alignment.py", "active_evidence.py", "agent.py"} <= set(f for f, _g in loads)
    assert all(g for _f, g in loads), loads
    # the deliverable middleware ran on the views: three nudges through emit, then the forced write it asked for
    assert [e["attempt"] for e in rig.events("deliverable_nudge")] == [1, 2, 3]
    assert [e["requester"] for e in rig.events("tool_call") if e.get("forced_deliverable")] == ["forced_deliverable_write"]
    assert rig.events("deliverable_missing") == [] and rig.result["harness"]["runtime_errors"] == []


def test_h0_runs_scripted_on_medcta(monkeypatch, tmp_path):
    import json
    import inline_loop_rig as L
    loads = _guarded_loads(monkeypatch)
    gap = json.dumps({"addresses_task": False, "hard_violations": [], "confidence": 0.4,
                      "repairable_gaps": [{"type": "unanswered", "claim": "x", "evidence_ids": ["E1"],
                                           "critique": "no stain named"}]})
    clean = json.dumps({"addresses_task": True, "hard_violations": [], "repairable_gaps": [], "confidence": 0.9})
    audits = [gap, clean, clean]
    rig = L.Rig(monkeypatch, tmp_path, "MedCTA", "MCTA-0",
                judge=lambda p: audits.pop(0) if p.startswith("You are a clinical answer auditor") and audits else "")
    rig.replies = ['<tool_call>{"name": "ImageDescription", "arguments": {}}</tool_call>',
                   "<answer>answer one</answer>", "<answer>answer two</answer>", "<answer>answer three</answer>"]
    rig.run(max_steps=10)
    assert {"answer_repair.py", "answer_candidate.py", "active_evidence.py", "agent_text.py"} <= set(f for f, _g in loads)
    assert all(g for _f, g in loads), loads
    # answer_repair routed the gap to active_evidence through the workspace's own state, not the ledger
    assert "pending_required_element" in rig.kernel._workspace_state
    assert not hasattr(rig.kernel.ledger, "pending_required_element")
    assert ("answer_repair", "REVISE") in [(e["capability"], e["decision"]) for e in rig.events("harness_decision")]
    assert rig.trajectory[-1]["event_type"] == "final_answer" and rig.result["harness"]["runtime_errors"] == []


def test_h0_runs_scripted_on_healthadminbench(monkeypatch, tmp_path):
    import inline_hab_stub as S
    h = S.Harness(monkeypatch, tmp_path)
    h.use_task(S.load_hab_task("HAB-denial-easy-1"))
    h.portal.controls = [S.control("disposition-input", role="textbox", label="Disposition", required=True),
                         S.control("submit-disposition-button", label="Submit Disposition")]
    h.run([S.final("a"), S.final("b"), S.final("c")], max_steps=12)
    assert h.net == [] and h.portal.unscripted == [] and h.runtime_errors() == []
    # the run used H0's own modules (loaded once per process, keyed by path): every one ran under the workspace builtins
    root = os.path.realpath(W.H0["gui"]) + os.sep
    mods = dict((os.path.basename(m.__file__), m) for n, m in list(sys.modules.items())
                if n.startswith("_workspace_") and os.path.realpath(getattr(m, "__file__", None) or "").startswith(root))
    assert {"recovery.py", "entry.py", "hab.py", "grounding_audit.py", "goal_alignment.py"} <= set(mods)
    assert all(m.__dict__.get("__builtins__") is W.WORKSPACE_BUILTINS for m in mods.values())
    # recovery ran on the public task and the trajectory view, and re-entered through emit, as H0 does
    assert ([e["step"] for e in h.events("recovery_v3")], [e["step"] for e in h.events("agent_reentry")]) == ([0, 1], [0, 1])
    assert [e["step"] for e in h.events("obligation_nudge")][:1] == [2]
    assert W.Workspace(W.H0["gui"]).check([t["name"] for t in h.task["available_tools"]]) == []
