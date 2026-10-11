# -*- coding: utf-8 -*-
"""Portal-state record of a GUI (HealthAdminBench) run: what a zero-cost replay needs and a recorded trajectory does not
hold. Observation only: it reads the page and the harness, never writes either, and never changes what
the run does. Written beside the run's logs (<job>/logs/portal_state_record.jsonl) and copied into the bundle by
runner/run_batch.py.

    rec = attach(env, task=, workspace=, kernel=, path=)   None unless env.type == "gui"
    rec.bind_trajectory(trajectory)                         the run's trajectory list (obligation composition reads it)
    finish(rec, final_eval_state)                           the eval_state the scorer read

Per env call (one "env_call" line, after the call returns):
    portals_state        localStorage portals_state as the page holds it, by sha256; its value is written once per
                         distinct sha256 (a "blob" line), so an unchanged state costs one hash
    controls             the live control model (env.list_controls, CONTROL_JS), by sha256, value once per sha256
    readback             the call's own write receipt (GuiEnvReal._write_receipt, READBACK_JS), as returned
    downloads            the download registry (env._download_registry and env._last_download), by sha256
    obligations          the completion conjunction (runner/harness/obligation_gate.completion_conjunction over the
                         trajectory so far) and each conjunct's truth over obligation_view(portals_state):
                         {"composed": bool, "met": bool|None, "decisions": [...], "conjuncts": [{id, decision, met}]}
    judge_calls          the harness judge calls made since the previous env call: prompt sha256 and length, response
Also: one "attach" line (the state after reset), "reconcile_write" lines (GuiEnvReal.reconcile_write read-backs),
"judge" lines (prompt sha256 + head, response text) in call order, and "final" (the final eval_state, by sha256).

Obligations are composed over the trajectory as it stands when the call returns: run.py appends the call's own
tool_call event after that, so a decision the call takes shows from the next line on.

Size bound. MAX_BYTES per task. Past it, values are no longer written (blob lines, judge texts): only their sha256, and
one "truncated" line says from where. Past 2 * MAX_BYTES no line is written but the final one, which counts the rows
dropped. A value larger than MAX_VALUE_BYTES is written as its sha256 and length only.
Every failure inside the recorder is caught and counted ("recorder_errors" on the final line); the run never sees it.
"""
import copy
import hashlib
import json
import os

MAX_BYTES = 12 * 1024 * 1024
MAX_VALUE_BYTES = 2 * 1024 * 1024
JUDGE_TEXT_CHARS = 16000
PROMPT_HEAD_CHARS = 240
# GuiEnvReal._read_state's own script (the replay's page stand-in recognizes it by this text)
READ_STATE_JS = "() => { try { return localStorage.getItem('portals_state'); } catch(e){ return null; } }"


def _canon(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _sha(text):
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


class Recorder(object):
    def __init__(self, env, task, workspace, path, max_bytes=MAX_BYTES):
        self.env, self.task, self.workspace, self.path = env, task, workspace, path
        self.max_bytes = max_bytes
        self.bytes = 0
        self.truncated = False
        self.seen = set()
        self.n_env = 0
        self.n_judge = 0
        self.pending_judge = []
        self.errors = 0
        self.dropped = 0
        self.trajectory = None
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        self.fh = open(path, "w", encoding="utf-8")

    # ------------------------------------------------------------------ writing
    def _write(self, row):
        if self.bytes > 2 * self.max_bytes and row.get("kind") != "final":
            self.dropped += 1
            return
        line = json.dumps(row, sort_keys=True, ensure_ascii=False, default=str) + "\n"
        self.fh.write(line)
        self.fh.flush()
        self.bytes += len(line.encode("utf-8"))

    def _room(self, n):
        if self.truncated:
            return False
        if self.bytes + n > self.max_bytes:
            self.truncated = True
            self._write({"kind": "truncated", "at_bytes": self.bytes, "env_call": self.n_env,
                         "detail": "past MAX_BYTES only sha256s are written"})
            return False
        return True

    def _ref(self, what, value):
        """sha256 of the value; the value itself is written once per sha256 while there is room."""
        text = _canon(value)
        sha = _sha(text)
        if sha not in self.seen:
            size = len(text.encode("utf-8"))
            if size > MAX_VALUE_BYTES:
                self._write({"kind": "blob_omitted", "what": what, "sha256": sha, "bytes": size})
                self.seen.add(sha)
            elif self._room(size):
                self._write({"kind": "blob", "what": what, "sha256": sha, "value": value})
                self.seen.add(sha)
        return sha

    # ------------------------------------------------------------------ reading (never writes the env)
    def _portals(self):
        page = getattr(self.env, "page", None)
        if page is None:
            return None
        raw = page.evaluate(READ_STATE_JS)
        st = json.loads(raw) if raw else None
        return st if isinstance(st, dict) else {}

    def _obligations(self, portals):
        from harness.obligation_gate import completion_conjunction, evaluate, obligation_view
        taken, conj = completion_conjunction(self.task, self.workspace, list(self.trajectory or []))
        if conj is None:
            return {"composed": False, "met": None, "decisions": None, "conjuncts": None}
        view = obligation_view({k: (portals or {}).get(k) for k in ("emr", "payerA", "payerB", "fax")})
        truth = evaluate(conj, view) if conj else ()
        return {"composed": bool(conj), "met": (bool(conj) and all(truth)) if conj else None,
                "decisions": list(taken or []),
                "conjuncts": [{"id": c.get("id"), "decision": c.get("decision"), "path": c.get("path"),
                               "met": bool(t)} for c, t in zip(conj, truth)]}

    def snapshot(self, kind, extra=None):
        row = {"kind": kind, "env_call": self.n_env}
        row.update(extra or {})
        try:
            portals = self._portals()
            row["portals_state"] = None if portals is None else self._ref("portals_state", portals)
        except Exception as e:
            portals, row["portals_state_error"] = None, repr(e)[:300]
            self.errors += 1
        try:
            row["controls"] = self._ref("controls", self.env.list_controls())
        except Exception as e:
            row["controls_error"] = repr(e)[:300]
            self.errors += 1
        try:
            reg = copy.deepcopy(self.env.__dict__.get("_download_registry") or {})
            row["downloads"] = self._ref("downloads", {"registry": reg,
                                                       "last": self.env.__dict__.get("_last_download")})
        except Exception as e:
            row["downloads_error"] = repr(e)[:300]
            self.errors += 1
        try:
            row["obligations"] = self._obligations(portals)
        except Exception as e:
            row["obligations_error"] = repr(e)[:300]
            self.errors += 1
        row["judge_calls"], self.pending_judge = self.pending_judge, []
        self._write(row)

    # ------------------------------------------------------------------ hooks
    def env_call(self, name, args, result):
        receipt = result.get("write_receipt") if isinstance(result, dict) else None
        ok = result.get("ok") if isinstance(result, dict) else None
        args_text = _canon(args)
        self.snapshot("env_call", {"tool": name, "args": args if len(args_text) <= 4000 else {"sha256": _sha(args_text)},
                                   "ok": ok, "readback": receipt,
                                   "error": (str(result.get("error"))[:500] if isinstance(result, dict)
                                             and result.get("error") else None)})
        self.n_env += 1

    def reconcile(self, name, args, result):
        self._write({"kind": "reconcile_write", "env_call": self.n_env - 1, "tool": name, "result": result})

    def judge(self, prompt, response):
        p = prompt if isinstance(prompt, str) else _canon(prompt)
        r = response if isinstance(response, str) or response is None else _canon(response)
        entry = {"n": self.n_judge, "prompt_sha256": _sha(p), "prompt_chars": len(p),
                 "response_sha256": None if r is None else _sha(r)}
        self.pending_judge.append(entry)
        row = {"kind": "judge", "env_call": self.n_env, "n": self.n_judge, "prompt_sha256": entry["prompt_sha256"],
               "response_sha256": entry["response_sha256"]}
        text = {"prompt_head": p[:PROMPT_HEAD_CHARS], "response": None if r is None else r[:JUDGE_TEXT_CHARS],
                "response_cut": bool(r is not None and len(r) > JUDGE_TEXT_CHARS)}
        if self._room(len(_canon(text))):
            row.update(text)
        self._write(row)
        self.n_judge += 1

    def close(self, final_eval_state):
        try:
            sha = self._ref("final_eval_state", final_eval_state)
        except Exception as e:
            sha = None
            self.errors += 1
            self._write({"kind": "final_error", "error": repr(e)[:300]})
        self._write({"kind": "final", "env_calls": self.n_env, "judge_calls": self.n_judge, "eval_state": sha,
                     "truncated": self.truncated, "bytes": self.bytes, "dropped_rows": self.dropped,
                     "recorder_errors": self.errors})
        self.fh.close()

    def bind_trajectory(self, trajectory):
        self.trajectory = trajectory


def attach(env, *, task, workspace, kernel, path, max_bytes=MAX_BYTES):
    """Start recording a GUI env's run. Wraps env.call_tool and env.reconcile_write on the instance and the kernel's
    judge (kernel.ctx.judge_fn); each wrapper returns exactly what it wrapped. -> Recorder, or None (not a GUI env, or
    the recorder could not start)."""
    if getattr(env, "type", None) != "gui":
        return None
    try:
        rec = Recorder(env, task, workspace, path, max_bytes=max_bytes)
    except Exception:
        return None
    call_tool, reconcile = env.call_tool, env.reconcile_write

    def recorded_call_tool(name, args):
        res = call_tool(name, args)
        try:
            rec.env_call(name, args, res)
        except Exception:
            rec.errors += 1
        return res

    def recorded_reconcile(name, args, result):
        out = reconcile(name, args, result)
        try:
            rec.reconcile(name, args, out)
        except Exception:
            rec.errors += 1
        return out
    env.call_tool, env.reconcile_write = recorded_call_tool, recorded_reconcile
    ctx = getattr(kernel, "ctx", None)
    judge_fn = getattr(ctx, "judge_fn", None)
    if judge_fn is not None:
        def recorded_judge(prompt, *a, **k):
            out = judge_fn(prompt, *a, **k)
            try:
                rec.judge(prompt, out)
            except Exception:
                rec.errors += 1
            return out
        ctx.judge_fn = recorded_judge
    try:
        rec.snapshot("attach")
    except Exception:
        rec.errors += 1
    return rec


def finish(rec, final_eval_state):
    if rec is None:
        return None
    try:
        rec.close(final_eval_state)
    except Exception:
        return None
    return rec.path
