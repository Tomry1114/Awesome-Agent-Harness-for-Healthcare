"""run.score_checkpoints: MedCTA's checkpoints run at once and come back in checkpoint order, equal to scoring them
one by one; other benchmarks stay one at a time."""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import run


def _rig(monkeypatch, delays):
    seen, lock = [], threading.Lock()
    live, peak = [0], [0]

    def fake(cp, ctx):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        time.sleep(delays[cp["id"]])
        with lock:
            live[0] -= 1
            seen.append(cp["id"])
        return {"id": cp["id"], "ctx_q": ctx["q"]}
    monkeypatch.setattr(run.scoring, "run_checkpoint", fake)
    return seen, peak


def test_medcta_runs_at_once_in_checkpoint_order(monkeypatch):
    delays = {"a": 0.3, "b": 0.05, "c": 0.15}
    seen, peak = _rig(monkeypatch, delays)
    cps = [{"id": k} for k in ("a", "b", "c")]
    out = run.score_checkpoints(cps, {"q": 1}, "tool_sandbox")
    assert [r["id"] for r in out] == ["a", "b", "c"] and all(r["ctx_q"] == 1 for r in out)
    assert peak[0] == 3 and seen[0] == "b"


def test_other_benchmarks_one_at_a_time(monkeypatch):
    seen, peak = _rig(monkeypatch, {"a": 0.05, "b": 0.01})
    out = run.score_checkpoints([{"id": "a"}, {"id": "b"}], {"q": 2}, "gui")
    assert [r["id"] for r in out] == ["a", "b"] and peak[0] == 1 and seen == ["a", "b"]
