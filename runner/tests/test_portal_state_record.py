# -*- coding: utf-8 -*-
"""runner/portal_state_record.py with a fake GUI env (never a live portal): what it writes per env call, that every
wrapper returns what it wrapped, the hash dedup, the size bound, and that a recorder failure never reaches the run."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.dirname(HERE)
REPO = os.path.dirname(RUNNER)
sys.path.insert(0, RUNNER)

import portal_state_record as PSR  # noqa: E402

H0 = os.path.join(REPO, "workspaces", "hab", "h0")
TASK = {"goal": "Open denial DEN-001 and submit the disposition. Document your reasoning in a triage note."}


class FakePage(object):
    def __init__(self):
        self.state = {"emr": {"agentActions": {}}}
        self.fail = False

    def evaluate(self, script, arg=None):
        if self.fail:
            raise RuntimeError("page gone")
        assert script == PSR.READ_STATE_JS
        return json.dumps(self.state)


class FakeEnv(object):
    type = "gui"

    def __init__(self):
        self.page = FakePage()
        self.calls = []
        self._download_registry = {}
        self._last_download = None

    def call_tool(self, name, args):
        self.calls.append((name, args))
        if name == "click" and args.get("ref") == "submit-disposition-button":
            self.page.state["emr"]["agentActions"]["selectedDisposition"] = "Write Off"
        if name == "download":
            self._download_registry["doc"] = {"path": "/x/doc.pdf", "hash": "h"}
            self._last_download = "/x/doc.pdf"
        out = {"ok": True, "url": "http://p/emr/denied/DEN-001"}
        if name == "fill":
            out["write_receipt"] = {"observed": args.get("text"), "control_label": "Triage note"}
        return out

    def reconcile_write(self, name, args, result):
        return {"confirmed": True, "detail": "fake"}

    def list_controls(self):
        return [{"ref": "submit-disposition-button", "role": "button", "label": "Submit"}]


class FakeCtx(object):
    def __init__(self):
        self.judge_fn = lambda prompt: "YES" if "ok" in prompt else "NO"


class FakeKernel(object):
    def __init__(self):
        self.ctx = FakeCtx()


def _rows(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def test_only_a_gui_env_is_recorded(tmp_path):
    env = FakeEnv()
    env.type = "fhir"
    assert PSR.attach(env, task=TASK, workspace=H0, kernel=FakeKernel(), path=str(tmp_path / "r.jsonl")) is None
    assert not (tmp_path / "r.jsonl").exists()


def test_the_record_holds_state_controls_readback_downloads_obligations_judges_and_final(tmp_path):
    env, kernel, path = FakeEnv(), FakeKernel(), str(tmp_path / "logs" / "portal_state_record.jsonl")
    rec = PSR.attach(env, task=TASK, workspace=H0, kernel=kernel, path=path)
    trajectory = []
    rec.bind_trajectory(trajectory)
    assert kernel.ctx.judge_fn("is it ok?") == "YES"                      # the wrapper returns what it wrapped
    res = env.call_tool("fill", {"ref": "triage-note-input", "text": "note"})
    assert res["write_receipt"]["observed"] == "note" and env.calls[-1][0] == "fill"
    trajectory.append({"event_type": "tool_call", "tool": "click", "origin": "agent",
                       "args": {"ref": "submit-disposition-button"}})
    env.call_tool("click", {"ref": "submit-disposition-button"})
    assert env.reconcile_write("submit", {}, {}) == {"confirmed": True, "detail": "fake"}
    env.call_tool("download", {"ref": "download-doc-1"})
    assert PSR.finish(rec, {"full_state": {"agentActions": {"selectedDisposition": "Write Off"}}}) == path

    rows = _rows(path)
    kinds = [r["kind"] for r in rows]
    assert kinds[0] == "blob" and "attach" in kinds and kinds[-1] == "final"
    calls = [r for r in rows if r["kind"] == "env_call"]
    assert [c["tool"] for c in calls] == ["fill", "click", "download"]
    assert calls[0]["readback"] == {"observed": "note", "control_label": "Triage note"}
    assert calls[0]["judge_calls"][0]["prompt_chars"] == len("is it ok?")   # judge calls since the last env call
    assert [r for r in rows if r["kind"] == "judge"][0]["response"] == "YES"
    assert [r for r in rows if r["kind"] == "reconcile_write"][0]["result"]["confirmed"] is True
    blobs = {r["sha256"]: r for r in rows if r["kind"] == "blob"}
    assert blobs[calls[1]["portals_state"]]["value"]["emr"]["agentActions"]["selectedDisposition"] == "Write Off"
    assert calls[2]["portals_state"] == calls[1]["portals_state"]            # unchanged state: one hash, one blob
    assert sum(1 for r in blobs.values() if r["what"] == "portals_state") == 2
    assert blobs[calls[2]["downloads"]]["value"]["last"] == "/x/doc.pdf"
    assert blobs[calls[0]["controls"]]["value"][0]["ref"] == "submit-disposition-button"
    ob0, ob1 = calls[0]["obligations"], calls[1]["obligations"]
    assert ob0["composed"] and "disposition_selected" not in [c["id"] for c in ob0["conjuncts"]]
    by_id = {c["id"]: c for c in ob1["conjuncts"]}
    assert ob1["decisions"] == ["disposition_submit"]
    assert by_id["disposition_selected"]["met"] is True and by_id["disposition_selected"]["decision"] == \
        "disposition_submit"
    assert by_id["appeal_documented"]["met"] is False and ob1["met"] is False       # per-conjunct truth
    final = rows[-1]
    assert final["env_calls"] == 3 and final["judge_calls"] == 1 and final["recorder_errors"] == 0
    assert blobs[final["eval_state"]]["what"] == "final_eval_state"


def test_the_record_is_size_bounded(tmp_path):
    env, path = FakeEnv(), str(tmp_path / "r.jsonl")
    rec = PSR.attach(env, task=TASK, workspace=H0, kernel=FakeKernel(), path=path, max_bytes=1500)
    rec.bind_trajectory([])
    for i in range(20):
        env.page.state["emr"]["n"] = "x" * 300 + str(i)                   # a new state every call
        env.call_tool("scroll", {})
    PSR.finish(rec, {"big": "y" * 5000})
    rows = _rows(path)
    assert [r["kind"] for r in rows].count("truncated") == 1
    cut = [i for i, r in enumerate(rows) if r["kind"] == "truncated"][0]
    assert not [r for r in rows[cut:] if r["kind"] == "blob"]               # past the bound: hashes only
    assert [r for r in rows[cut:] if r["kind"] == "env_call"] and all(
        r.get("portals_state") for r in rows if r["kind"] == "env_call")
    final = rows[-1]
    assert final["truncated"] is True and final["env_calls"] == 20 and final["dropped_rows"] > 0
    assert os.path.getsize(path) < 2 * 1500 + 3000                          # past 2 * max_bytes only the final line


def test_a_recorder_failure_never_reaches_the_run(tmp_path):
    env, path = FakeEnv(), str(tmp_path / "r.jsonl")
    rec = PSR.attach(env, task=TASK, workspace=H0, kernel=FakeKernel(), path=path)
    env.page.fail = True
    env.list_controls = lambda: (_ for _ in ()).throw(RuntimeError("controls gone"))
    assert env.call_tool("click", {"ref": "x"})["ok"] is True
    PSR.finish(rec, None)
    rows = _rows(path)
    call = [r for r in rows if r["kind"] == "env_call"][0]
    assert "portals_state_error" in call and "controls_error" in call
    assert rows[-1]["recorder_errors"] >= 2


def test_run_batch_copies_the_record_into_the_bundle(tmp_path):
    import run_batch as RB
    job = tmp_path / "job"
    (job / "logs").mkdir(parents=True)
    (job / "logs" / "portal_state_record.jsonl").write_text('{"kind": "final"}\n')
    bdir = RB.write_bundle(str(tmp_path / "out"), "HAB-x", {"_job_dir": str(job), "_trajectory": []}, {})
    assert open(os.path.join(bdir, "portal_state_record.jsonl")).read() == '{"kind": "final"}\n'
