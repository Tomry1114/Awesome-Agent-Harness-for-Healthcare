# -*- coding: utf-8 -*-
"""A portal page whose renderer dies mid-episode ends the episode as a crash result (void, re-run), never as a hang
and never as the agent's failure.

When a task's chromium has no renderer left, a pending page call (e.g. page.evaluate) can keep waiting long after
the page's "crash" event has fired, holding the worker. Closing the page from that event fails the waiting call at
once; every later use of the crashed page raises BrowserPageCrashed, which the step loop's `except Exception`
handlers cannot swallow, and run_task turns it into a harness_error result with no scored checkpoints."""
import pytest

import environments as ENV
import judges
import run as R


class _Page(object):
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True

    def evaluate(self, *a, **k):
        return None


def _env():
    env = ENV.GuiEnvReal.__new__(ENV.GuiEnvReal)
    env.page = _Page()
    env.page_crashed = False
    return env


def test_the_crash_event_closes_the_page_so_a_waiting_call_fails():
    env = _env()
    page = env.page
    env._alive()                                   # a live page: nothing raised
    env._on_page_crash(page)
    assert env.page_crashed and page.closed


@pytest.mark.parametrize("use", [lambda e: e.call_tool("click", {"ref": "x"}), lambda e: e._observe(),
                                 lambda e: e._read_state(), lambda e: e.eval_state(),
                                 lambda e: e.reconcile_write("submit", {}, {})])
def test_every_use_of_a_crashed_page_raises_past_the_step_loops_handlers(use):
    env = _env()
    env._on_page_crash(env.page)
    with pytest.raises(ENV.BrowserPageCrashed):
        use(env)
    assert not issubclass(ENV.BrowserPageCrashed, Exception)   # `except Exception` in the step loop cannot eat it


def test_run_task_turns_a_crashed_page_into_a_crash_result(monkeypatch):
    monkeypatch.setattr(judges, "require", lambda bench: {})
    monkeypatch.setattr(judges, "run_record", lambda binding, result: {})
    monkeypatch.setattr(R, "load_task", lambda bench, tid: {})
    monkeypatch.setattr(R, "resolve_workspace", lambda bench, task, ws: (None, None))
    monkeypatch.setattr(R, "resume_key", lambda *a, **k: "key")

    def crashed(*a, **k):
        raise ENV.BrowserPageCrashed("the portal page's renderer crashed during the episode")
    monkeypatch.setattr(R, "_run_task", crashed)
    res = R.run_task("HealthAdminBench", "HAB-emr-medium-2", agent_name="stub")
    assert res["end_reason"] == "harness_error" and res["outcome_status"] == "harness_error"
    assert res["checkpoints"] == []
