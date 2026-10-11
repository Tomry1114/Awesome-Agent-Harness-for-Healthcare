# -*- coding: utf-8 -*-
"""Scripted HealthAdminBench stand-ins for the inline-intervention tests (test_inline_hab_*.py).

The tests drive the real runner loop, run.run_task(bench="HealthAdminBench"), with the real GuiEnvReal, the real
HABAdapter gates, the real harness kernel and the real recovery entry (the HAB workspace's recovery/entry.py, as its
middleware loads it: recovery_entry()). Only the outer boundaries are replaced:

    browser   playwright's sync_playwright and environments._gui_axtree. The page stand-in here is not a
              recording: it is a portal whose state the test controls -- the localStorage portals_state dict, a
              control table (CONTROL_JS rows with required/value), click and fill handlers, downloadable refs --
              and it logs every page action and every direct localStorage write so a test can tell an executed
              action from a refused one.
    model     a scripted agent at agent.act; each turn records the harness_feedback / last_result /
              last_observation run.py hands it, i.e. what the agent receives next turn.
    judge     harness._build_judge returns a constant stub that records its prompts.
    scoring   scoring.run_checkpoint returns a skipped checkpoint.
    network   any socket connect raises and is recorded; tests assert none happened.

Not a test module (no test_ prefix). Every patch goes through pytest's monkeypatch, so nothing outlives a test.
"""
import copy
import json
import os
import re
import socket

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.dirname(HERE)

BENCH = "HealthAdminBench"
JUDGE_MODEL = "s0b-stub-judge"
JUDGE_REPLY = "NO"
# Mechanism switches and model/judge variables a stray shell value would silently change. They are
# unset for every scripted run, so the tests pin the default (H0) configuration.
UNSET_ENV = ("MH_OBLIGATION_GATE", "MH_RECOVERY_V3", "MH_COMPLETE_EFFECT", "MH_RECOVERY_EXTRA", "MH_ACTIVE_READBACK",
             "MH_RECONCILE_BUDGET", "MH_MODEL_LABEL", "MH_OPENAI_MODEL", "MH_PORTAL_BASE",
             "LLM_JUDGE_MODEL", "MH_GACC", "MH_GACC_MODEL", "MH_RUBRIC_JUDGE", "MH_MM_JUDGE", "MH_MM_JUDGE_MODEL",
             "MH_ALLOW_SHARED_HARNESS_JUDGE")
PORTAL_BASE = "http://portal.s0b.invalid"

_COMMIT_LABEL = re.compile(r"submit|send|transmit|finaliz|\bfile\b", re.I)
_TESTID_SEL = re.compile(r'^\[data-testid="(.*)"\]$')


def control(ref, role="button", label="", value="", required=False, section="", commit=None, checked=None,
            disabled=False, visible=True):
    """One CONTROL_JS row (adapters/hab/gui_control_model.py), with CONTROL_JS's own commit rule by default."""
    if commit is None:
        commit = bool(role == "button" and _COMMIT_LABEL.search(label or ""))
    return {"ref": ref, "role": role, "label": label, "section": section, "name": "", "id": "", "testid": ref,
            "readonly": False, "required": bool(required), "value": value, "disabled": bool(disabled),
            "visible": bool(visible), "checked": checked, "selected_index": None, "selected_text": None,
            "placeholder_selected": None, "aria_selected": None, "options": [], "commit": bool(commit)}


class Portal(object):
    """The state a test controls, behind the fake page."""

    def __init__(self, path="/emr/denied/DEN-001", title="EMR Portal"):
        self.state = {"emr": {}, "payerA": {}, "payerB": {}, "fax": {}}   # localStorage portals_state
        self.controls = []
        self.text = ""                  # page text beyond the controls (what the agent reads: codes, IDs)
        self.path = path
        self.title = title
        self.on_click = {}              # ref -> fn(portal, ref)
        self.on_fill = {}               # ref -> fn(portal, ref, text), after the control value is set
        self.downloads = {}             # ref -> file name
        self.log = []                   # page actions in order: (op, ref[, text])
        self.storage_writes = []        # payloads of direct localStorage writes (restore_persisted_content)
        self.unscripted = []            # page calls this stand-in has no answer for

    # -- lookups
    def find(self, ref):
        for c in self.controls:
            if str(c.get("ref")) == str(ref):
                return c
        return None

    def emr(self):
        return self.state["emr"]

    # -- page actions
    def click(self, ref):
        self.log.append(("click", ref))
        fn = self.on_click.get(ref)
        if fn is not None:
            fn(self, ref)

    def fill(self, ref, text):
        self.log.append(("fill", ref, text))
        c = self.find(ref)
        if c is None:
            raise RuntimeError("s0b-portal: no control %r" % (ref,))
        c["value"] = text
        fn = self.on_fill.get(ref)
        if fn is not None:
            fn(self, ref, text)

    # -- page scripts
    def readback(self, ref):
        c = self.find(ref)
        if c is None:
            return {"found": False}
        return {"found": True, "value": c.get("value") or "", "checked": c.get("checked"),
                "selected_text": c.get("selected_text"), "label": c.get("label") or "",
                "section": c.get("section") or ""}

    def restore(self, payload):
        """The localStorage write of GuiEnvReal.restore_persisted_content, with its script's semantics: create
        missing containers, fill a leaf only when it is absent, null or ''."""
        self.storage_writes.append(copy.deepcopy(payload))
        n = 0
        for top, items in (payload or {}).items():
            node0 = self.state.setdefault(top, {})
            for item in items:
                node, path = node0, [str(p) for p in (item.get("path") or [])]
                for seg in path[:-1]:
                    if isinstance(node, list) and seg.isdigit():
                        node = node[int(seg)]
                    else:
                        if not node.get(seg):
                            node[seg] = {}
                        node = node[seg]
                leaf = path[-1]
                if isinstance(node, list) and leaf.isdigit():
                    cur = node[int(leaf)] if int(leaf) < len(node) else None
                    if cur in (None, ""):
                        node[int(leaf)] = item.get("value")
                        n += 1
                elif node.get(leaf) in (None, ""):
                    node[leaf] = item.get("value")
                    n += 1
        return n

    def axtree(self):
        lines = ["RootWebArea %r" % self.title]
        for c in self.controls:
            if not c.get("visible", True):
                continue
            line = "\t[%s] %s %r" % (c["ref"], c["role"], c.get("label") or "")
            if c.get("value"):
                line += " value=%r" % c["value"]
            if c.get("required"):
                line += ", required"
            lines.append(line)
        if self.text:
            lines.append("\tStaticText %r" % self.text)
        return "\n".join(lines)


class _Download(object):
    def __init__(self, name):
        self.suggested_filename = name

    def save_as(self, dest):
        with open(dest, "wb") as fh:
            fh.write(("s0b download stand-in: %s\n" % self.suggested_filename).encode())


class _DownloadWait(object):
    def __init__(self, page):
        self.page = page
        self.ref = None

    def __enter__(self):
        self.page._download_wait = self
        return self

    def __exit__(self, *exc):
        self.page._download_wait = None
        return False

    @property
    def value(self):
        name = self.page.portal.downloads.get(self.ref)
        if name is None:
            raise RuntimeError("s0b-portal: %r is not a download" % (self.ref,))
        return _Download(name)


class _Mouse(object):
    def __init__(self, page):
        self.page = page

    def wheel(self, dx, dy):
        self.page.portal.log.append(("scroll", dy))


class _Locator(object):
    def __init__(self, page, what):
        self.page, self.what = page, what

    first = property(lambda self: self)
    last = property(lambda self: self)

    def click(self, *a, **k):
        # GuiEnvReal clicks by visible text when an action carries no ref. The stand-in has no text layout: the click
        # fails as a page error and is logged, as a click on a missing element would.
        self.page.portal.log.append(("click_text", self.what))
        raise RuntimeError("s0b-portal: no element with text %r" % (self.what,))

    def fill(self, *a, **k):
        self.page.portal.unscripted.append(("locator.fill", self.what))
        raise RuntimeError("s0b-portal: label locator fill %r" % (self.what,))

    def count(self):
        return 0

    def get_attribute(self, *a, **k):
        return None

    def scroll_into_view_if_needed(self, *a, **k):
        return None

    get_by_role = get_by_text = lambda self, *a, **k: _Locator(self.page, a)


class FakePage(object):
    def __init__(self, portal):
        self.portal = portal
        self.mouse = _Mouse(self)
        self._download_wait = None

    def on(self, event, handler):
        """Playwright's event registration (GuiEnvReal listens for "crash"); a scripted page never crashes."""
        pass

    @staticmethod
    def _ref(selector):
        m = _TESTID_SEL.match(str(selector))
        return m.group(1) if m else None

    # navigation / waiting
    def goto(self, url, **k):
        self.portal.log.append(("goto", url))
        m = re.match(r"^https?://[^/]+(/.*)?$", str(url))
        self.portal.path = (m.group(1) if m else None) or "/"

    def wait_for_load_state(self, *a, **k):
        return None

    def wait_for_timeout(self, *a, **k):
        return None

    @property
    def url(self):
        return PORTAL_BASE + self.portal.path

    def title(self):
        return self.portal.title

    # actions
    def click(self, selector, **k):
        ref = self._ref(selector)
        if ref is None:
            self.portal.unscripted.append(("click", selector))
            raise RuntimeError("s0b-portal: unscripted selector %r" % (selector,))
        if self._download_wait is not None:
            self._download_wait.ref = ref
        self.portal.click(ref)

    def fill(self, selector, text, **k):
        ref = self._ref(selector)
        if ref is None:
            self.portal.unscripted.append(("fill", selector))
            raise RuntimeError("s0b-portal: unscripted selector %r" % (selector,))
        self.portal.fill(ref, text)

    def select_option(self, selector, value, **k):
        self.portal.unscripted.append(("select_option", selector))
        raise RuntimeError("s0b-portal: select is not scripted")

    def press(self, *a, **k):
        self.portal.unscripted.append(("press", a))
        raise RuntimeError("s0b-portal: press is not scripted")

    def go_back(self, *a, **k):
        self.portal.log.append(("back",))

    def set_input_files(self, *a, **k):
        self.portal.unscripted.append(("set_input_files", a))
        raise RuntimeError("s0b-portal: upload is not scripted")

    def expect_download(self, *a, **k):
        return _DownloadWait(self)

    def locator(self, what, **k):
        return _Locator(self, what)

    get_by_text = get_by_label = get_by_role = locator

    # page scripts
    def evaluate(self, script, arg=None):
        from adapters.hab.gui_control_model import CONTROL_JS, READBACK_JS
        s = str(script)
        if "localStorage.setItem('portals_state'" in s:
            return self.portal.restore(arg)
        if "localStorage.getItem('portals_state')" in s:
            return json.dumps(self.portal.state)
        if s == CONTROL_JS:
            return copy.deepcopy(self.portal.controls)
        if s == READBACK_JS:
            return self.portal.readback(arg)
        self.portal.unscripted.append(("evaluate", s[:80]))
        raise RuntimeError("s0b-portal: unscripted page script")


class _Browser(object):
    def __init__(self, portal):
        self.portal = portal

    def new_page(self, *a, **k):
        return FakePage(self.portal)

    def close(self):
        return None


class _Playwright(object):
    def __init__(self, portal):
        self.portal = portal
        self.chromium = self

    def launch(self, *a, **k):
        return _Browser(self.portal)

    def start(self):
        return self

    def stop(self):
        return None


class ScriptedAgent(object):
    """The model seam. `script` is a list of actions; an entry may be a callable(turn_record) -> action. Once the
    script is exhausted every turn answers `final` with EXHAUSTED."""
    name = "s0b-scripted"
    model = None
    EXHAUSTED = "s0b: script exhausted"

    def __init__(self, script, portal=None):
        self.script = list(script)
        self.turns = []
        self.portal = portal

    def act(self, state):
        rec = {"harness_feedback": copy.deepcopy(state.get("harness_feedback")),
               "last_result": copy.deepcopy(state.get("last_result")),
               "last_observation": state.get("last_observation"),
               "page_log_len": (len(self.portal.log) if self.portal is not None else None)}
        self.turns.append(rec)
        if not self.script:
            return {"type": "final", "answer": self.EXHAUSTED}
        nxt = self.script.pop(0)
        return copy.deepcopy(nxt(rec) if callable(nxt) else nxt)


def recovery_entry():
    """The recovery entry module of the HAB workspace a run uses (workspace.H0["gui"]): the one module object its
    middleware/recovery.py loaded, so a patch on it reaches the run."""
    import workspace as W
    return W.Workspace(W.H0["gui"]).module("recovery/entry.py")


def load_hab_task(task_id):
    import run as R
    return copy.deepcopy(R.load_task(BENCH, task_id))


class Harness(object):
    """Everything one scripted run_task needs, installed through monkeypatch."""

    def __init__(self, monkeypatch, tmp_path):
        import harness as H
        import run as R
        import scoring as S
        import playwright.sync_api as PW
        import environments as E
        self.mp, self.tmp = monkeypatch, tmp_path
        self.R, self.H = R, H
        self.portal = Portal()
        self.judge_prompts = []
        self.net = []
        self.kernels = []
        self.recovery_calls = []
        monkeypatch.setenv("MH_HARNESS_JUDGE_MODEL", JUDGE_MODEL)
        for k in UNSET_ENV:
            monkeypatch.delenv(k, raising=False)
        for k in ("LLM_JUDGE_MODEL", "MH_JUDGE_MODEL", "MH_GACC_MODEL"):   # run_task requires every judge's config
            monkeypatch.setenv(k, JUDGE_MODEL)
        monkeypatch.setenv("MH_PORTAL_BASE", PORTAL_BASE)

        cap = self

        def build_judge():
            def judge_fn(prompt):
                cap.judge_prompts.append(prompt)
                return JUDGE_REPLY
            return judge_fn, JUDGE_MODEL
        monkeypatch.setattr(H, "_build_judge", build_judge)

        orig_bk = H.build_kernel

        def build_kernel(*a, **k):
            kern = orig_bk(*a, **k)
            cap.kernels.append(kern)
            return kern
        monkeypatch.setattr(H, "build_kernel", build_kernel)

        def run_checkpoint(cp, ctx):
            return {"id": cp["id"], "dimension": cp["dimension"], "subdimension": cp.get("subdimension"),
                    "weight": float(cp.get("weight", 1.0)), "checkpoint_status": "skipped",
                    "failure_mode": None, "skip_reason": "s0b_scripted"}
        monkeypatch.setattr(S, "run_checkpoint", run_checkpoint)

        monkeypatch.setattr(PW, "sync_playwright", lambda: _Playwright(cap.portal))
        monkeypatch.setattr(E, "_gui_axtree", lambda page: page.portal.axtree())
        downloads = str(tmp_path / "downloads")
        orig_make = E.make_env

        def make_env(task, **kw):
            env = orig_make(task, **kw)
            env.workspace = downloads          # GuiEnvReal's default is /tmp/hab_ws_<pid>
            cap.env = env
            return env
        monkeypatch.setattr(E, "make_env", make_env)

        def refuse(sock, address):
            cap.net.append(str(address))
            raise ConnectionRefusedError(111, "s0b: network refused")
        monkeypatch.setattr(socket.socket, "connect", refuse)

    def use_task(self, task):
        self.task = task
        self.mp.setattr(self.R, "load_task", lambda bench, tid: copy.deepcopy(task))

    def run(self, script, max_steps, workspace=None):
        """run_task on the scripted portal. `workspace` is the run's workspace (None: HAB's H0)."""
        self.agent = ScriptedAgent(script, self.portal)
        self.mp.setattr(self.R.agents, "make_agent", lambda name, task, workspace=None: self.agent)
        res = self.R.run_task(BENCH, self.task["task_id"], agent_name=self.agent.name, max_steps=max_steps,
                              job_dir=str(self.tmp / "job"), cleanup=False, workspace=workspace)
        self.result = res
        # the loop's events; the shell's trace (shell_* events: runner/harness/origin.py) is in full_traj
        self.full_traj = res["_trajectory"]
        self.traj = [e for e in self.full_traj if not str(e.get("event_type", "")).startswith("shell_")]
        return res

    def spy_ledger(self):
        """After every kernel after_action, record (step, tool, public_obligations_met, observed_submission_realized,
        terminal_locked, effective decision). run.py writes the two obligation fields just before after_action and
        the kernel sets terminal_locked inside it, so this is the ledger as the next action will see it. A
        pass-through: the kernel's return value is unchanged."""
        rows = []
        orig_bk = self.H.build_kernel

        def build_kernel(*a, **k):
            kern = orig_bk(*a, **k)
            inner = kern.after_action

            def after_action(action, *args, **kw):
                out = inner(action, *args, **kw)
                led = kern.ledger
                rows.append((kw.get("step"), action.get("tool"), led.public_obligations_met,
                             led.observed_submission_realized, led.terminal_locked,
                             getattr(out, "type", None)))
                return out
            kern.after_action = after_action
            return kern
        self.mp.setattr(self.H, "build_kernel", build_kernel)
        return rows

    def spy_kernel_before_final(self):
        """Steps at which the kernel's before_final ran (a pass-through)."""
        steps = []
        orig_bk = self.H.build_kernel

        def build_kernel(*a, **k):
            kern = orig_bk(*a, **k)
            inner = kern.before_final

            def before_final(answer, *args, **kw):
                steps.append(kw.get("step"))
                return inner(answer, *args, **kw)
            kern.before_final = before_final
            return kern
        self.mp.setattr(self.H, "build_kernel", build_kernel)
        return steps

    def stub_recovery(self, summaries):
        """Replace run_recovery_v3 (the whole recovery episode) with a stub. `summaries` is a list consumed one per
        invocation: a summary dict, or a callable(call_record) -> summary. Each call is recorded with the step,
        root content, lifecycle and whether the recovery budget gate was open at entry."""
        entry = recovery_entry()
        calls = []
        queue = list(summaries)

        def run_recovery_v3(task, root_content, lifecycle, trajectory=None, driver=None, judge=None, goal=None,
                            manifest=None, step=None):
            rec = {"step": step, "root_content": root_content, "lifecycle": lifecycle,
                   "budget_open": driver.can_execute_recovery_action(), "driver": driver}
            calls.append(rec)
            nxt = queue.pop(0) if queue else {"episodes": []}
            return nxt(rec) if callable(nxt) else copy.deepcopy(nxt)
        self.mp.setattr(entry, "run_recovery_v3", run_recovery_v3)
        return calls

    def spy_recovery(self):
        """Record every real run_recovery_v3 invocation (step, root content, lifecycle, budget gate at entry) and
        pass through to it."""
        entry = recovery_entry()
        calls = []
        inner = entry.run_recovery_v3

        def run_recovery_v3(task, root_content, lifecycle, trajectory=None, driver=None, **kw):
            calls.append({"step": kw.get("step"), "root_content": root_content, "lifecycle": lifecycle,
                          "budget_open": driver.can_execute_recovery_action()})
            return inner(task, root_content, lifecycle, trajectory, driver=driver, **kw)
        self.mp.setattr(entry, "run_recovery_v3", run_recovery_v3)
        return calls

    def script_recovery_kernel(self, per_invocation):
        """Keep run_recovery_v3 and its GUI backend real, and replace only the recovery STRATEGY (RecoveryKernel)
        with a script: invocation n issues the primitives in per_invocation[n] ([(tool, args), ...]) through the
        real backend (budget gate, completion allowance, executor, recovery event, budget charge) and records what
        each returned. Returns no episodes."""
        entry = recovery_entry()
        invocations = []
        queue = list(per_invocation)

        class ScriptedRecoveryKernel(object):
            def run_all_episodes(self, bench, wf_registry, substrate, driver, task, trajectory, goal, judge,
                                 events=None):
                ops = queue.pop(0) if queue else []
                rec = {"step": driver.step, "results": []}
                invocations.append(rec)
                for tool, args in ops:
                    rec["results"].append((tool, substrate.driver.call_tool(tool, dict(args))))
                return []
        self.mp.setattr(entry, "RecoveryKernel", ScriptedRecoveryKernel)
        return invocations

    # -- views
    def events(self, *types):
        return [e for e in self.traj if (not types or e.get("event_type") in types)]

    def at(self, step):
        return [compact(e) for e in self.traj if e.get("step") == step]

    def kinds(self):
        return [e.get("event_type") for e in self.traj]

    def feedback(self, turn):
        return self.agent.turns[turn]["harness_feedback"]

    def runtime_errors(self):
        return (self.result.get("harness") or {}).get("runtime_errors")


# Event keys that carry bulky or derived payloads; a test that needs one reads the raw event.
_BULK = ("result", "observation", "agent_visible_text", "canonical_observation", "delivery_record", "canonical_result",
         "canonical_action", "state_record", "ts", "id", "action_key")


def compact(ev):
    return {k: v for k, v in ev.items() if k not in _BULK}


def click(ref):
    return {"type": "tool_call", "tool": "click", "args": {"ref": ref}}


def fill(ref, text, **extra):
    args = {"ref": ref, "text": text}
    args.update(extra)
    return {"type": "tool_call", "tool": "fill", "args": args}


def final(answer="s0b final"):
    return {"type": "final", "answer": answer}


def set_action(key, value=True, text=None):
    """A click handler that records a portal agentActions flag and changes the page (so the env sees a change)."""
    def handler(portal, ref):
        portal.emr().setdefault("agentActions", {})[key] = value
        portal.text = (portal.text + " " + (text or ("%s recorded" % key))).strip()
    return handler


# The prior-auth read-acquire probe (runner/adapters/__init__.py HABAdapter.forced_action). run.py dispatches a
# harness read only while `not _in_repair[0]`, so on a prior-auth task a revealed, unmet "Diagnoses" tab tells a
# gate that unlocks repair (no dispatch on the next turn) from one that does not (dispatch on the next turn).
PROBE_TAB = "tab-diagnoses"


def probe_tab(portal):
    c = control(PROBE_TAB, label="Diagnoses", visible=False)
    portal.controls.append(c)
    portal.on_click[PROBE_TAB] = set_action("clickedDiagnosesTab", text="Diagnoses tab open")
    return c


def reveal_then(portal, action):
    """An agent turn that reveals the probe tab, then emits `action` (the tab is invisible at that turn's start)."""
    def turn(_rec):
        portal.find(PROBE_TAB)["visible"] = True
        return action
    return turn


def probe_dispatch_steps(h):
    return [e["step"] for e in h.events("read_acquire_dispatch")]
