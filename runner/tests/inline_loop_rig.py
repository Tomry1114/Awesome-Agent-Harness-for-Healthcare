# -*- coding: utf-8 -*-
"""Scripted rig for the run.py loop behaviours that the replay tests do not exercise (inline interventions:
kernel decisions before an action, after an action and before the final answer).

It drives the real run.run_task and replaces only what sits outside the loop, at the model, environment and scoring
seams:

    model      PhysicianBench: transport.chat_fc (the workspace agent program's chat). MedCTA: transport.chat_text
               (the text protocol's gateway call, the MedCTA text program's chat). Each call pops the next scripted
               reply.
    env        PhysicianBench: the upstream registry's dispatch and PbNativeEnv._get (the FHIR network). MedCTA:
               ApiVLM._call (the perception backend). Everything above those seams runs for real.
    judge      none by default (MH_HARNESS_JUDGE_MODEL unset, so harness._build_judge returns no judge). A test that
               needs one passes `judge=fn`: harness._build_judge then returns fn under a stand-in model id.
    scoring    scoring.run_checkpoint is replaced by a skip (the scorer is shell and not under test).
    network    any socket connect is refused and counted; a test fails if one happened.

Kernel decisions come from a scripted workspace kernel middleware: the substrate's default workspace (workspace.H0)
is copied to a temporary directory, one middleware file is added and listed last under kernel_middleware, and
workspace.H0 points at the copy for the test. That middleware forwards its three hooks to plain functions a test sets
on `rig.hooks`, so every decision in a test is visible in the test itself.

The agent object is the real one (WorkspaceAgent, with the default workspace's program for the model's protocol); its
act() is wrapped only to record the state the shell hands it each turn and the action it returned.

Nothing here writes inside the repository: the workspace copy, the job directory and the deliverable live under
pytest's tmp_path, and every patch goes through pytest's monkeypatch.
"""
import copy
import json
import os
import shutil
import socket
import sys

import yaml

TESTS = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.dirname(TESTS)
REPO = os.path.dirname(RUNNER)
for _p in (os.path.join(RUNNER, "scorers"), RUNNER):
    if _p not in sys.path:
        sys.path.insert(0, _p)

HOOKS_MODULE = "_inline_loop_scripted_hooks"
MIDDLEWARE_NAME = "scripted_inline_loop"
MIDDLEWARE_FILE = "middleware/scripted_inline_loop.py"
MIDDLEWARE_ENTRY = MIDDLEWARE_FILE + ":ScriptedInlineLoop"

MIDDLEWARE_SRC = '''# -*- coding: utf-8 -*-
"""Scripted kernel middleware for runner/tests/inline_loop_rig.py: forwards each hook to the test's function."""
import %s as hooks

from workspace_api import Capability


class ScriptedInlineLoop(Capability):
    name = "%s"

    def before_action(self, action, ctx):
        return hooks.call("before_action", action, ctx)

    def after_action(self, action, result, before_state, after_state, ctx):
        return hooks.call("after_action", action, ctx, result)

    def before_final(self, answer, ctx):
        return hooks.call("before_final", answer, ctx)
''' % (HOOKS_MODULE, MIDDLEWARE_NAME)

# The harness configuration a standard run uses, minus the judge: run_task requires every judge's configuration
# (runner/judges.py), so JUDGE_CONFIG_ENV names stand-in models, and without Rig(judge=fn) harness._build_judge is
# replaced by one that returns no judge and no network client.
ENV_SET = {"MH_API_MODEL": "inline-loop-scripted-brain",
           "MH_OPENAI_KEY": "inline-loop-no-key", "MH_VLM_API_MODEL": "inline-loop-perception"}
# Read by the runtime and able to change the paths under test if inherited from the shell.
ENV_UNSET = ("MH_HARNESS_JUDGE_MODEL", "MH_ALLOW_SHARED_HARNESS_JUDGE",
             "MH_RECOVERY_V3", "MH_COMPLETE_EFFECT", "MH_OBLIGATION_GATE", "MH_RECOVERY_EXTRA", "MH_ACTIVE_READBACK",
             "MH_RECONCILE_BUDGET", "MH_DELIV_SCAFFOLD", "MH_SCRIPT", "MH_SCRIPT_FILE", "MH_MODEL_LABEL",
             "MH_OPENAI_MODEL", "MH_OPENAI_BASE", "MH_TOOL_MODE", "LLM_JUDGE_MODEL", "MH_GACC", "MH_GACC_MODEL",
             "MH_RUBRIC_JUDGE", "MH_MM_JUDGE", "MH_MM_JUDGE_MODEL", "FHIR_BASE_URL", "OPENAI_API_KEY",
             "OPENAI_BASE_URL")

JUDGE_MODEL = "inline-loop-judge"
JUDGE_CONFIG_ENV = {"MH_HARNESS_JUDGE_MODEL": JUDGE_MODEL, "LLM_JUDGE_MODEL": JUDGE_MODEL,
                    "MH_JUDGE_MODEL": JUDGE_MODEL, "MH_GACC_MODEL": JUDGE_MODEL}
BENCH_ENV_TYPE = {"PhysicianBench": "fhir", "MedCTA": "tool_sandbox"}


class ScriptedHooks(object):
    """The module the scripted middleware reads. A test sets before_action / after_action / before_final to a
    function(payload, ctx[, result]) -> HarnessDecision | None; every call is recorded in `calls`."""

    def __init__(self):
        self.before_action = None
        self.after_action = None
        self.before_final = None
        self.calls = []

    def call(self, hook, payload, ctx, result=None):
        self.calls.append((hook, copy.deepcopy(payload) if isinstance(payload, (dict, str)) else payload))
        fn = getattr(self, hook)
        if fn is None:
            return None
        return fn(payload, ctx) if hook != "after_action" else fn(payload, ctx, result)


def decision(type, **kw):
    """A HarnessDecision raised by the scripted middleware (capability name filled in)."""
    from harness.decision import HarnessDecision
    kw.setdefault("capability", MIDDLEWARE_NAME)
    return HarnessDecision(type, **kw)


HEAVY_KEYS = ("result", "observation", "agent_visible_text", "canonical_observation", "canonical_result",
              "delivery_record", "state_record", "ts")


def slim(event):
    """An event without the bulky payload copies, for readable comparisons."""
    return {k: v for k, v in event.items() if k not in HEAVY_KEYS}


class NotADecision(object):
    """What a broken kernel middleware might return instead of a HarnessDecision. The kernel's per-capability
    try/except does not cover what it does with the returned value, so this raises out of the kernel hook."""


def fc_tool(name, args, call_id=None):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"id": call_id or "call_%s" % name, "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)}}]}


def fc_final(text):
    return {"role": "assistant", "content": text}


class Rig(object):
    def __init__(self, monkeypatch, tmp_path, bench, task_id, protocol=None, task_edit=None, judge=None):
        self.mp, self.tmp, self.bench, self.task_id = monkeypatch, tmp_path, bench, task_id
        self.env_type = BENCH_ENV_TYPE[bench]
        self.replies = []
        self.model_calls = []
        self.exhausted = 0
        self.states = []          # the state dict the shell handed agent.act, per turn (deep copy)
        self.actions = []         # what agent.act returned, per turn
        self.env_calls = []       # (tool, args) reaching the environment boundary
        self.gets = []            # FHIR GET paths
        self.network = []
        self.kernels = []
        self.hooks = ScriptedHooks()
        self.tool_result = lambda name, args: {"entries": [], "total": 0}
        self.get_result = lambda path: {"error": "inline-loop: no FHIR server"}
        self.vlm_text = lambda prompt: "inline-loop observation: a scripted perception result"
        self.judge_prompts = []
        for k, v in ENV_SET.items():
            monkeypatch.setenv(k, v)
        for k in ENV_UNSET:
            monkeypatch.delenv(k, raising=False)
        import models
        # the stand-in brain's protocol is its model-table entry (runner/models.yaml for a real model); text by default
        monkeypatch.setitem(models.table(), ENV_SET["MH_API_MODEL"], protocol or models.TEXT)
        monkeypatch.setitem(sys.modules, HOOKS_MODULE, self.hooks)
        # The scripted middleware is workspace code, so it may import only allow-listed modules: the test's hooks
        # module is admitted to that import allow-list for this test only, as a facade offering `call`.
        import workspace as W
        monkeypatch.setitem(W.STDLIB_IMPORTS, HOOKS_MODULE, ("call",))
        monkeypatch.delitem(W._FACADES, HOOKS_MODULE, raising=False)
        self._refuse_network()
        import run as R
        self.R = R
        self._install_task(task_edit)
        self._install_scoring()
        self._install_agent_spy()
        self._install_kernel_spy()
        for k, v in JUDGE_CONFIG_ENV.items():
            monkeypatch.setenv(k, v)
        if judge is not None:
            self._install_judge(judge)
        else:
            import harness as H
            monkeypatch.setattr(H, "_build_judge", lambda: (None, None))
        self.workspace = self._install_workspace()
        if bench == "PhysicianBench":
            self._install_pb()
        else:
            self._install_medcta()

    # ------------------------------------------------------------------------------ installation
    def _refuse_network(self):
        rig = self
        orig_connect, orig_connect_ex = socket.socket.connect, socket.socket.connect_ex

        def connect(sock, address):
            if sock.family == getattr(socket, "AF_UNIX", object()):
                return orig_connect(sock, address)
            rig.network.append(str(address))
            raise ConnectionRefusedError(111, "inline-loop rig: network refused")

        def connect_ex(sock, address):
            if sock.family == getattr(socket, "AF_UNIX", object()):
                return orig_connect_ex(sock, address)
            rig.network.append(str(address))
            return 111
        self.mp.setattr(socket.socket, "connect", connect)
        self.mp.setattr(socket.socket, "connect_ex", connect_ex)

    def _install_task(self, task_edit):
        orig = self.R.load_task

        def load_task(bench, task_id):
            t = copy.deepcopy(orig(bench, task_id))
            if task_edit is not None:
                task_edit(t)
            return t
        self.mp.setattr(self.R, "load_task", load_task)

    def _install_scoring(self):
        def run_checkpoint(cp, ctx):
            return {"id": cp["id"], "dimension": cp["dimension"], "subdimension": cp.get("subdimension"),
                    "weight": float(cp.get("weight", 1.0)), "checkpoint_status": "skipped",
                    "failure_mode": None, "skip_reason": "inline_loop_rig"}
        self.mp.setattr(self.R.scoring, "run_checkpoint", run_checkpoint)

    def _install_agent_spy(self):
        rig, orig = self, self.R.agents.make_agent

        def make_agent(name, task, *a, **k):
            agent = orig(name, task, *a, **k)
            real_act = agent.act

            def act(state):
                rig.states.append(copy.deepcopy(state))
                a = real_act(state)
                rig.actions.append(copy.deepcopy(a))
                return a
            agent.act = act
            return agent
        self.mp.setattr(self.R.agents, "make_agent", make_agent)

    def _install_kernel_spy(self):
        import harness as H
        rig, orig = self, H.build_kernel

        def build_kernel(*a, **k):
            kern = orig(*a, **k)
            rig.kernels.append(kern)
            return kern
        self.mp.setattr(H, "build_kernel", build_kernel)

    def _install_judge(self, fn):
        import harness as H
        rig = self
        self.mp.setenv("MH_HARNESS_JUDGE_MODEL", JUDGE_MODEL)

        def build_judge():
            def judge_fn(prompt):
                rig.judge_prompts.append(prompt)
                return fn(prompt)
            return judge_fn, JUDGE_MODEL
        self.mp.setattr(H, "_build_judge", build_judge)

    def _install_workspace(self):
        import workspace as W
        dst = str(self.tmp / "workspace")
        shutil.copytree(W.H0[self.env_type], dst)
        with open(os.path.join(dst, MIDDLEWARE_FILE), "w") as f:
            f.write(MIDDLEWARE_SRC)
        mp = os.path.join(dst, "manifest.yaml")
        manifest = yaml.safe_load(open(mp, encoding="utf-8"))
        manifest["kernel_middleware"] = list(manifest.get("kernel_middleware") or []) + [MIDDLEWARE_ENTRY]
        with open(mp, "w", encoding="utf-8") as f:
            yaml.safe_dump(manifest, f, sort_keys=False)
        self.mp.setitem(W.H0, self.env_type, dst)
        return dst

    def _next_reply(self, messages, tools):
        self.model_calls.append({"messages": copy.deepcopy(messages), "tools": tools is not None})
        if not self.replies:
            self.exhausted += 1
            return None
        return self.replies.pop(0)

    def _install_pb(self):
        import environments as E
        import transport as T
        rig = self
        orig_setup = E.PbNativeEnv._setup_native_registry

        def setup_native_registry(env):
            orig_setup(env)

            def dispatch(name, args):
                rig.env_calls.append((name, copy.deepcopy(args)))
                return copy.deepcopy(rig.tool_result(name, args))
            env._reg.dispatch = dispatch

        def get(env, path):
            rig.gets.append(path)
            return copy.deepcopy(rig.get_result(path))
        self.mp.setattr(E.PbNativeEnv, "_setup_native_registry", setup_native_registry)
        self.mp.setattr(E.PbNativeEnv, "_get", get)
        self.mp.setattr(E.PbNativeEnv, "_healthy", lambda env: True)

        def chat_fc(messages, tools_schema, *a, **k):
            r = rig._next_reply(messages, tools_schema)
            return fc_final("inline-loop: brain script exhausted") if r is None else r
        self.mp.setattr(T, "chat_fc", chat_fc)

    def _install_medcta(self):
        import transport as T
        import adapters.medcta.tools_medcta as TM
        import adapters.medcta.vlm_backend as VB
        rig = self
        VB.get_backend.cache_clear()     # built by run.py from this rig's environment, not a previous test's

        def vlm_call(backend, image, prompt, max_tokens=256):
            rig.env_calls.append(("vlm", prompt))
            return rig.vlm_text(prompt)

        def no_ocr():
            raise RuntimeError("inline-loop rig: OCR is not scripted")

        def google_search(query):
            rig.env_calls.append(("search", query))
            return "[search error] inline-loop rig: search is not scripted"
        self.mp.setattr(VB.ApiVLM, "_call", vlm_call)
        self.mp.setattr(TM, "_ocr_reader", no_ocr)
        self.mp.setattr(TM, "google_search", google_search)

        def chat_text(messages, *a, **k):
            r = rig._next_reply(messages, None)
            return "<answer>inline-loop: brain script exhausted</answer>" if r is None else r
        self.mp.setattr(T, "chat_text", chat_text)

    # ------------------------------------------------------------------------------ running
    def run(self, max_steps, workspace=None):
        """run_task with the rig's stand-ins. `workspace` is the run's workspace (None: the substrate's default
        workspace, which this rig points at its own copy)."""
        kw = {"fhir_base": "http://127.0.0.1:9/fhir"} if self.bench == "PhysicianBench" else {}
        try:
            self.result = self.R.run_task(self.bench, self.task_id, "gpt5", max_steps=max_steps,
                                          job_dir=str(self.tmp / "job"), cleanup=False, workspace=workspace, **kw)
        finally:
            self._forget_process_state(workspace)
        # the loop's events; the shell's trace (shell_* events: runner/harness/origin.py) is in full_trajectory
        self.full_trajectory = self.result["_trajectory"]
        self.trajectory = [e for e in self.full_trajectory if not str(e.get("event_type", "")).startswith("shell_")]
        assert self.network == [], "the rig let a network connection through: %s" % self.network
        return self.result

    def _forget_process_state(self, workspace=None):
        """Process-wide state a run leaves behind: the workspace copies' modules (workspace._module keeps them in
        sys.modules under a path-derived name) and MedCTA's perception-backend singleton, built from this rig's
        stand-in environment variables."""
        roots = [os.path.realpath(p) + os.sep for p in (self.workspace, workspace) if p]
        for name, mod in list(sys.modules.items()):
            f = getattr(mod, "__file__", None) or ""
            if name.startswith("_workspace_") and any(os.path.realpath(f).startswith(r) for r in roots):
                del sys.modules[name]
        if self.bench == "MedCTA":
            import adapters.medcta.vlm_backend as VB
            VB.get_backend.cache_clear()

    # ------------------------------------------------------------------------------ reading
    def task(self):
        return self.R.load_task(self.bench, self.task_id)

    def events(self, *types):
        return [e for e in self.trajectory if not types or e.get("event_type") in types]

    def types(self):
        return [e.get("event_type") for e in self.trajectory]

    def feedback_at(self, turn):
        """The harness_feedback dict the shell handed the agent on its `turn`-th act() call (0-based)."""
        return self.states[turn].get("harness_feedback")

    @property
    def kernel(self):
        assert len(self.kernels) == 1, self.kernels
        return self.kernels[0]

    @property
    def audit(self):
        return self.result["harness"]["audit"]

    def actuation_records(self):
        return self.audit["actuation"]["records"]["actuation"]
