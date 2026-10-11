# -*- coding: utf-8 -*-
"""HealthAdminBench recovery is workspace middleware on the shell's harness action service.

The strategy is workspaces/hab/h0/middleware/recovery.py (first in the HAB manifest) and the modules it loads from
workspaces/hab/h0/recovery/; the shell keeps whether and how each action runs (runner/harness/action_service.py) and
offers it to middleware as context services (runner/middleware_host.py).

Real run.run_task episodes on the stand-in portal and agent (runner/tests/inline_hab_stub.py) show:
  * the workspace strategy reaches the environment only through its context's harness-action services;
  * a workspace without recovery.py runs with no recovery and everything else as when recovery does nothing;
  * editing the workspace copy (the streak stop) changes what the run does;
  * the shell's ceilings bind a workspace middleware that tries to exceed them.
"""
import ast
import os
import shutil
import sys

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inline_hab_stub as S  # noqa: E402
from inline_hab_stub import click, compact, control, fill, final  # noqa: E402

import middleware_host as MH  # noqa: E402
import workspace as W  # noqa: E402

DENIAL_TASK = "HAB-denial-easy-1"
NOTE = "Reviewed the denial; route to clinical appeals."
RECOVERY_ENTRY = "middleware/recovery.py:Recovery"


def _clean(h):
    assert h.net == [], h.net
    assert h.portal.unscripted == [], h.portal.unscripted
    assert h.runtime_errors() == [], h.runtime_errors()


def _harness(monkeypatch, tmp_path):
    h = S.Harness(monkeypatch, tmp_path)
    h.use_task(S.load_hab_task(DENIAL_TASK))
    return h


def _copy_h0(tmp_path, name="hab_ws"):
    ws = str(tmp_path / name)
    shutil.copytree(W.H0["gui"], ws, ignore=shutil.ignore_patterns("__pycache__"))
    return ws


def _set_manifest_middleware(ws, entries):
    mp = os.path.join(ws, "manifest.yaml")
    manifest = yaml.safe_load(open(mp, encoding="utf-8"))
    manifest["middleware"] = entries
    open(mp, "w", encoding="utf-8").write(yaml.safe_dump(manifest, sort_keys=False))


def _recovery_class():
    return W.Workspace(W.H0["gui"]).load("middleware/recovery.py", "Recovery")


def _events(h, *types):
    return [compact(e) for e in h.traj if e.get("event_type") in types]


def _recovery_origin(h):
    return [e for e in h.traj if e.get("origin") == "harness" or e.get("action_origin") == "harness"
            or e.get("event_type") in ("recovery_v3", "agent_reentry", "harness_action_refused")]


# =====================================================================================================================
# the strategy reaches the environment only through ctx.request_action / ctx.read (and the read-only views)
# =====================================================================================================================
_SERVICES = ("begin_harness_actions", "post_stop_allowance", "request_action", "read", "recovery_state", "controls",
             "read_back", "harness_actions_open")
_DENIED = ("env", "judge", "_shell", "_judge_factory", "ledger")


class _Guarded(object):
    """The context the recovery middleware is handed in this test: every read is forwarded, except the raw environment,
    the judge and the shell's internals, which fail the run loudly."""

    def __init__(self, ctx):
        object.__setattr__(self, "_guarded_ctx", ctx)

    def __getattr__(self, name):
        if name in _DENIED:
            raise AssertionError("workspace recovery read ctx.%s" % name)
        return getattr(object.__getattribute__(self, "_guarded_ctx"), name)


def _c3_portal(h):
    """The portal of test_inline_hab_recovery's guard-restore test: the real recovery opens the remittance tab and the
    guard restores the triage note that click cleared."""
    note = "CO-50 medical necessity; appeal with the clinical records."
    p = h.portal
    p.state["emr"] = {"agentActions": {"selectedDisposition": "Route to Clinical Appeals", "viewedDenialDetails": True},
                      "triageNotes": [note]}
    p.controls = [control("tab-remittance", label="Remittance Image"),
                  control("triage-note", role="textbox", label="Triage Note", value="draft: CO-50")]

    def open_remittance(portal, ref):
        portal.emr()["agentActions"]["viewedRemittanceImage"] = True
        portal.emr()["triageNotes"] = [""]
        portal.find("triage-note")["value"] = ""
        portal.text = "Remittance image"
    p.on_click = {"tab-remittance": open_remittance}


def test_workspace_recovery_reaches_the_environment_only_through_its_context_services(monkeypatch, tmp_path):
    h = _harness(monkeypatch, tmp_path)
    _c3_portal(h)
    import environments as E
    state = {"hook": 0, "service": 0}
    touched = []                                  # (what, inside a recovery hook, inside a context service)

    def around(fn, key):
        def wrapped(*a, **k):
            state[key] += 1
            try:
                return fn(*a, **k)
            finally:
                state[key] -= 1
        return wrapped

    Recovery = _recovery_class()
    for hook in ("before_final", "after_episode"):
        inner = getattr(Recovery, hook)

        def guarded(self, ctx, *a, _inner=inner):
            return around(_inner, "hook")(self, _Guarded(ctx), *a)
        monkeypatch.setattr(Recovery, hook, guarded)
    for name in _SERVICES:
        monkeypatch.setattr(MH.MiddlewareContext, name, around(getattr(MH.MiddlewareContext, name), "service"))

    def spy(what, fn):
        def wrapped(*a, **k):
            touched.append((what, state["hook"] > 0, state["service"] > 0))
            return fn(*a, **k)
        return wrapped
    monkeypatch.setattr(E.GuiEnvReal, "call_tool", spy("call_tool", E.GuiEnvReal.call_tool))
    monkeypatch.setattr(E.GuiEnvReal, "restore_persisted_content", spy("restore", E.GuiEnvReal.restore_persisted_content))
    monkeypatch.setattr(E.GuiEnvReal, "reconcile_write", spy("reconcile_write", E.GuiEnvReal.reconcile_write))
    for name in ("evaluate", "click", "fill", "goto"):
        monkeypatch.setattr(S.FakePage, name, spy("page." + name, getattr(S.FakePage, name)))

    h.run([final("done")], max_steps=12)
    _clean(h)                                     # no runtime error: the guarded context was never refused
    in_hook = [(what, svc) for what, hook, svc in touched if hook]
    assert ("call_tool", True) in in_hook and ("page.click", True) in in_hook and ("restore", True) in in_hook
    assert [x for x in in_hook if not x[1]] == []       # every environment touch of the strategy went through the shell
    assert [(e["tool"], e["origin"]) for e in h.events("tool_call") if e.get("step") == 0] == [
        ("snapshot", "harness"), ("snapshot", "harness"), ("click", "harness"), ("type", "harness"),
        ("restore_agent_content", "harness"), ("snapshot", "harness")]


_WORKSPACE_RECOVERY = ["middleware/recovery.py"] + ["recovery/%s" % f for f in (
    "entry.py", "kernel.py", "registry.py", "gui_substrate.py", "gui_completion.py", "hab.py", "contracts.py", "metrics.py")]
_ALLOWED_IMPORTS = ("workspace_api", "re", "json", "hashlib", "dataclasses", "typing")


def test_workspace_recovery_code_names_no_raw_environment_judge_or_budget_internals():
    ws = W.H0["gui"]
    assert sorted(_WORKSPACE_RECOVERY[1:]) == sorted("recovery/" + f for f in os.listdir(os.path.join(ws, "recovery"))
                                                     if f.endswith(".py"))
    for rel in _WORKSPACE_RECOVERY:
        tree = ast.parse(open(os.path.join(ws, rel), encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 0, "%s: a relative import bypasses the workspace loader" % rel
                names = [node.module]
            else:
                names = []
            for n in names:
                assert n in _ALLOWED_IMPORTS, "%s imports %s" % (rel, n)
            if isinstance(node, ast.Attribute):
                assert node.attr not in ("env", "page", "judge", "judge_fn", "budgeted_judge", "ledger", "_harness",
                                         "action_service", "_shell", "_recovery_actions", "_agent_actions"), \
                    "%s reads .%s" % (rel, node.attr)


# =====================================================================================================================
# a workspace without recovery.py: no recovery, the rest as when recovery does nothing
# =====================================================================================================================
def _reentry_scenario(h):
    h.portal.controls = [control("disposition-input", role="textbox", label="Disposition", required=True),
                         control("submit-disposition-button", label="Submit Disposition")]
    return [final("a"), final("b"), final("c")], 12


def _forced_stop_scenario(h):
    h.portal.controls = [control("noop", label="Refresh")]
    return [click("noop")] * 30, 6


@pytest.mark.parametrize("scenario", [_reentry_scenario, _forced_stop_scenario])
def test_a_workspace_without_recovery_runs_with_no_recovery_and_the_rest_unchanged(monkeypatch, tmp_path, scenario):
    runs = {}
    for arm in ("h0", "idle", "removed"):
        with monkeypatch.context() as mp:
            h = S.Harness(mp, tmp_path / arm)
            h.use_task(S.load_hab_task(DENIAL_TASK))
            if arm == "idle":                          # the h0 workspace with the recovery hooks doing nothing
                Recovery = _recovery_class()
                mp.setattr(Recovery, "before_final", lambda self, ctx, answer: None)
                mp.setattr(Recovery, "after_episode", lambda self, ctx: None)
            if arm == "removed":
                ws = _copy_h0(tmp_path)
                manifest = yaml.safe_load(open(os.path.join(ws, "manifest.yaml"), encoding="utf-8"))
                _set_manifest_middleware(ws, [e for e in manifest["middleware"] if e != RECOVERY_ENTRY])
                os.remove(os.path.join(ws, "middleware", "recovery.py"))
                mp.setitem(W.H0, "gui", ws)
                assert W.Workspace(ws).check([]) == []
            script, max_steps = scenario(h)
            h.run(script, max_steps=max_steps)
            _clean(h)
            runs[arm] = ([compact(e) for e in h.traj], [t["harness_feedback"] for t in h.agent.turns],
                         h.result["budget"], _recovery_origin(h))
    assert runs["h0"][3], "the scenario must make H0's recovery act"
    assert runs["removed"][3] == [] and runs["idle"][3] == []
    assert runs["removed"][:3] == runs["idle"][:3]


# =====================================================================================================================
# the workspace copy is what runs: an edit changes the run
# =====================================================================================================================
def test_editing_the_streak_stop_in_a_workspace_copy_changes_the_run(monkeypatch, tmp_path):
    ws = _copy_h0(tmp_path)
    path = os.path.join(ws, "middleware", "recovery.py")
    src = open(path, encoding="utf-8").read()
    assert src.count("MAX_BLOCKED_RECOVERY = 2 ") == 1
    open(path, "w", encoding="utf-8").write(src.replace("MAX_BLOCKED_RECOVERY = 2 ", "MAX_BLOCKED_RECOVERY = 1 "))
    out = {}
    for arm in ("h0", "edited"):
        with monkeypatch.context() as mp:
            h = S.Harness(mp, tmp_path / arm)
            h.use_task(S.load_hab_task(DENIAL_TASK))
            if arm == "edited":
                mp.setitem(W.H0, "gui", ws)
            script, max_steps = _reentry_scenario(h)
            h.run(script, max_steps=max_steps)
            _clean(h)
            out[arm] = {k: [e["step"] for e in h.events(k)] for k in ("recovery_v3", "agent_reentry", "obligation_nudge")}
    # h0 workspace (as in test_inline_hab_recovery's re-entry tests): two blocked invocations re-enter at 0 and 1,
    # the third final is not recovered
    assert (out["h0"]["recovery_v3"], out["h0"]["agent_reentry"], out["h0"]["obligation_nudge"][:1]) == ([0, 1], [0, 1], [2])
    # the copy stops after ONE blocked invocation: no recovery, so no re-entry, from the second final on
    assert (out["edited"]["recovery_v3"], out["edited"]["agent_reentry"], out["edited"]["obligation_nudge"][:1]) == (
        [0], [0], [1])


def test_the_run_records_the_workspace_modules_recovery_loaded():
    from instrumentation import Instrumentation, record_components
    inst = Instrumentation()
    record_components(inst, None, MH.load_middleware(W.H0["gui"], S.load_hab_task(DENIAL_TASK)), None, W.H0["gui"])
    paths = [(c["role"], c["path"]) for c in inst.components]
    i = paths.index(("middleware", "workspaces/hab/h0/middleware/recovery.py"))
    modules = [p for r, p in paths if r == "module"]
    assert set(modules) == set("workspaces/hab/h0/recovery/" + f for f in (
        "entry.py", "registry.py", "gui_completion.py", "contracts.py", "gui_substrate.py", "hab.py", "kernel.py",
        "metrics.py"))
    assert paths[i + 1] == ("module", "workspaces/hab/h0/recovery/entry.py")


def test_workspace_code_loads_only_its_own_workspace_modules(tmp_path):
    ws = _copy_h0(tmp_path)
    by = os.path.join(ws, "middleware", "recovery.py")
    assert W.load_relative(by, "../recovery/contracts.py") is W.Workspace(ws).module("recovery/contracts.py")
    with pytest.raises(W.WorkspaceError):
        W.load_relative(by, "../../elsewhere.py")


# =====================================================================================================================
# the shell's ceilings bind a workspace middleware that tries to exceed them
# =====================================================================================================================
_OVERREACHING = '''
"""A workspace middleware that tries to exceed every ceiling the shell sets on harness actions."""
from workspace_api import Revision

LOG = []


class Recovery(object):
    def __init__(self, task):
        self.finals = 0

    def before_final(self, ctx, answer):
        self.finals += 1
        step = ctx.begin_harness_actions()
        LOG.append(("begin", self.finals, step))
        if step is None:
            try:
                ctx.request_action("click", {"ref": "noop"})
            except RuntimeError as exc:
                LOG.append(("refused_outside_an_invocation", str(exc)[:40]))
            return None
        for op in OPS.get(self.finals, []):
            fn = ctx.read if op[0] == "read" else ctx.request_action
            LOG.append((op[0], op[1], fn(op[1], dict(op[2]))))
        return None

    def after_episode(self, ctx):
        LOG.append(("post_stop", ctx.post_stop_allowance(), ctx.post_stop_allowance()))
        try:
            LOG.append(("after_stop", ctx.request_action("snapshot", {})))
        except RuntimeError:
            LOG.append(("after_stop", "no invocation"))


OPS = {}
'''


def _overreaching(monkeypatch, tmp_path, ops):
    ws = _copy_h0(tmp_path)
    open(os.path.join(ws, "middleware", "recovery.py"), "w", encoding="utf-8").write(
        _OVERREACHING.replace("OPS = {}", "OPS = %r" % (ops,)))
    monkeypatch.setitem(W.H0, "gui", ws)
    return W.Workspace(ws).module("middleware/recovery.py").LOG


def _refusals(h):
    return [(e["step"], e["tool"], e["reason"]) for e in h.events("harness_action_refused")]


def test_the_total_cap_and_the_invocation_rule_bind_an_overreaching_workspace(monkeypatch, tmp_path):
    h = _harness(monkeypatch, tmp_path)
    log = _overreaching(monkeypatch, tmp_path, {1: [("act", "click", {"ref": "noop"})] * 30})
    h.portal.controls = [control("noop", label="Refresh")]
    h.run([final("f0"), final("f1")], max_steps=60)
    _clean(h)
    ran = [r for kind, _t, r in [x for x in log if x[0] == "act"]]
    assert [("error" not in r) for r in ran] == [True] * 20 + [False] * 10
    assert _refusals(h) == [(0, "click", "recovery_total_cap")] * 10
    assert [x for x in h.portal.log if x == ("click", "noop")] == [("click", "noop")] * 20
    assert h.result["budget"]["recovery_env_actions"] == 20
    begins = [x for x in log if x[0] == "begin"]
    assert begins[0] == ("begin", 1, 0) and len(begins) >= 2
    assert all(step is None for _k, _n, step in begins[1:])          # the cap is spent: no further invocation opens
    assert any(x[0] == "refused_outside_an_invocation" and x[1].startswith("request_action needs a harness-action")
               for x in log)


def test_the_irreversible_commit_and_provenance_checks_bind_an_overreaching_workspace(monkeypatch, tmp_path):
    """The agent wrote the note and pressed Submit on this form, so a recovery submit of the same control passes the
    irreversible-commit rule (the agent itself committed) and runs; the portal does not take it (not landed), so the
    shell refuses a second submit. A write of a value with no traced source and a READ that is really a write are
    refused too."""
    h = _harness(monkeypatch, tmp_path)
    h.portal.controls = [control("triage-note", role="textbox", label="Triage Note"),
                         control("member-input", role="textbox", label="Member ID"),
                         control("attach-doc-1", label="Attach"),
                         control("submit-disposition-button", label="Submit Disposition")]
    submit = ("act", "submit", {"ref": "submit-disposition-button"})
    log = _overreaching(monkeypatch, tmp_path, {1: [submit, submit, ("act", "type", {"ref": "member-input", "text": "QQ-123"}),
                                                    ("read", "click", {"ref": "attach-doc-1"})]})
    h.run([fill("triage-note", NOTE), click("submit-disposition-button"), final("f0")], max_steps=12)
    _clean(h)
    results = [(t, r.get("error")) for kind, t, r in [x for x in log if x[0] in ("act", "read")]]
    assert results == [("submit", None), ("submit", "irreversible_commit_after_unconfirmed"),
                       ("type", "value_untraced"), ("click", "read_not_read_only")]
    assert [x for x in h.portal.log if x == ("click", "submit-disposition-button")] == [
        ("click", "submit-disposition-button")] * 2                    # the agent's and the one recovery submit
    assert _refusals(h) == [(2, "submit", "irreversible_commit_after_unconfirmed"), (2, "type", "value_untraced"),
                            (2, "click", "read_not_read_only")]
    assert not [x for x in h.portal.log if x[0] == "fill" and x[1] == "member-input"]


def test_the_post_stop_allowance_is_once_and_only_after_tool_budget_exhausted(monkeypatch, tmp_path):
    h = _harness(monkeypatch, tmp_path)
    log = _overreaching(monkeypatch, tmp_path, {})
    h.portal.controls = [control("noop", label="Refresh")]
    h.run([click("noop")] * 30, max_steps=6)
    _clean(h)
    assert compact(h.traj[6]) == {"step": 10, "event_type": "tool_budget_exhausted", "status": "escalated"}
    assert log[-2][:3] == ("post_stop", 10, None)
    assert "error" not in log[-1][1]
    assert [(e["step"], e["tool"], e["origin"]) for e in h.traj[7:]] == [(10, "snapshot", "harness")]
    assert h.kernels and h.result["budget"]["recovery_env_actions"] == 1


def test_no_post_stop_allowance_after_a_final(monkeypatch, tmp_path):
    h = _harness(monkeypatch, tmp_path)
    log = _overreaching(monkeypatch, tmp_path, {})
    h.run([final("f0")], max_steps=12)
    _clean(h)
    assert log[-2:] == [("post_stop", None, None), ("after_stop", "no invocation")]


# =====================================================================================================================
# the context services at their seam
# =====================================================================================================================
class _Service(object):
    def __init__(self):
        self.calls = []

    def scope(self):
        return object()

    def invocation_open(self, scope=None):
        return True

    def begin(self, step, scope=None):
        self.calls.append(("begin", step))
        return step

    def post_stop_allowance(self, step, scope=None):
        self.calls.append(("post_stop", step))
        return step

    def __getattr__(self, name):
        return lambda *a, **k: self.calls.append((name,) + a) or {"ran": name}


def _ctx(shell, step=3, traj=None):
    return MH.MiddlewareContext(env=None, task={}, trajectory=traj if traj is not None else [], step=step, max_steps=12,
                                used_actions=0, judge_factory=lambda: None, errors=[], shell=shell)


def test_harness_actions_need_an_invocation_the_context_opened():
    svc = _Service()
    ctx = _ctx(MH.harness_action_services(svc, at_step=3))
    for call in (lambda: ctx.request_action("click", {}), lambda: ctx.read("click", {}), ctx.recovery_state,
                 ctx.harness_actions_open, lambda: ctx.read_back("submit", {}, {})):
        with pytest.raises(RuntimeError):
            call()
    assert ctx.controls() == []      # a read-only view of the environment, in every hook: no invocation needed
    with pytest.raises(RuntimeError):
        ctx.post_stop_allowance()                            # before_final context: no post-stop allowance
    assert ctx.begin_harness_actions() == 3
    assert ctx.request_action("click", {"ref": "x"}) == {"ran": "request_action"}
    assert svc.calls == [("begin", 3), ("request_action", "click", {"ref": "x"})]
    with pytest.raises(RuntimeError):
        _ctx({}).begin_harness_actions()                     # a context without the service


def test_the_budget_ceiling_and_the_invocation_fail_closed():
    """The budget gate fails closed: open while budget remains, closed when it is spent, closed when the agent's counter
    raises; an invocation does not open once a per-task cap is spent. The harness counters belong to the action
    channel itself."""
    from harness.action_service import HarnessActionService

    def boom():
        raise RuntimeError("budget infra down")

    def service(agent, recovery, harness=object(), max_steps=12, total=None):
        s = HarnessActionService(harness, None, None, {}, [], max_steps, agent_actions=agent, reserve=lambda: 2,
                                 errors=[], workspace=W.H0["gui"])
        s.n_invocation, s.n_total = recovery, (recovery if total is None else total)   # the channel's own counters
        return s
    assert service(lambda: 3, 10).budget_open() is True           # 13 < 12 + 2
    assert service(lambda: 4, 10).budget_open() is False
    assert service(boom, 0).budget_open() is False
    assert service(lambda: 0, 20).begin(5) is None                   # the per-task recovery cap
    assert service(lambda: 0, 0, total=40).begin(5) is None          # the per-task harness cap
    assert service(lambda: 0, 19).begin(5) == 5
    assert service(lambda: 0, 0, harness=None).begin(5) is None    # no kernel, no harness action
    s = service(lambda: 32, 0, max_steps=30)                         # 32 < 30 + 2 is False
    assert s.reserve() == 2 and not s.budget_open()
    assert s.post_stop_allowance(7) == 7 and s.reserve() == 30 // 6 and s.budget_open()   # the post-stop re-grant
    assert s.post_stop_allowance(7) is None                                  # once per task


def test_the_post_stop_allowance_opens_at_the_stop_step_and_events_carry_it():
    traj = [{"step": 4, "event_type": "tool_call"}, {"step": 9, "event_type": "tool_budget_exhausted"}]
    svc = _Service()
    ended = _ctx(MH.harness_action_services(svc, trajectory=traj, end_reason="tool_budget_exhausted"), step=12, traj=traj)
    with pytest.raises(RuntimeError):
        ended.begin_harness_actions()
    assert ended.post_stop_allowance() == 9 and svc.calls == [("post_stop", 9)]
    ended.emit({"event_type": "recovery_v3", "step": 12})
    assert traj[-1] == {"event_type": "recovery_v3", "step": 9}
    other = _ctx(MH.harness_action_services(_Service(), trajectory=traj, end_reason="final_answer"), step=12, traj=traj)
    assert other.post_stop_allowance() is None
