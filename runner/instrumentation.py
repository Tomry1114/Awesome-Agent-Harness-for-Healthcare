# -*- coding: utf-8 -*-
"""Instrumentation: which workspace components a run loaded, and which of their hooks ran and with what result.

Measurement, not policy. Nothing here changes what the agent or the harness does; the record goes beside the
trajectory as instrumentation.jsonl, never into the trajectory or result.json.

    {"kind": "component_loaded", "role", "path", "sha256"}
        once per workspace file the run loaded: manifest, agent_program, system_prompt, prompt_fragment,
        tool_description, middleware, kernel_middleware, policy, completion_obligations, module (a workspace module a
        middleware or kernel middleware file loaded through workspace.load_relative, listed after that file)
    {"kind": "middleware_call", "path", "name", "hook", "step", "returned", ...}
        every call of a workspace hook. `returned` is the decision type (kernel middleware), the result class
        (middleware: Observation / Read / Accept / Repair / str, Revision / PostRevision with its `stage`, Nudge
        with its `stage` and `label`) or the boolean (exempt_from_budget); null when the hook returned nothing.
        Kernel middleware adds `reason_code`, `raised` when the hook threw, and `won` / `effective` when its
        decision shaped the hook and what the mode made of it, and `query` when the call was made by the kernel's
        evaluate_final (a query, not a hook the shell applied). A middleware's result is always applied by the shell.
        `summary`: a short description of what came back (summarize), beside the kind in `returned`.
    {"kind": "semantic_check", "hook", "step", "granted"}
        one semantic check a middleware asked the shell for through ctx.semantic_check() (on_answer), and whether
        the semantic meter granted it or refused it. Asked only where the harness has a judge.

These calls show where a file with hooks acted: it fired where its hook returned something, and (kernel
middleware) was applied where it won with a non-ALLOW effective type. Always-on components (system prompt, tool
descriptions, agent program, policies) have no moment of firing; they appear only as loaded, and their effect shows
only by comparing trajectories. Shell code is never recorded here.
"""
import hashlib
import os

import workspace as W

_HEAD = 160


def _head(text):
    return None if text is None else str(text)[:_HEAD]


def summarize(out):
    """A short, JSON-shaped description of a hook's result: its kind plus the fields that say what it does. Shared by
    instrumentation (middleware_call.summary) and the shell_hook trajectory events (middleware_host). None -> None."""
    if out is None:
        return None
    kind = type(out).__name__
    if isinstance(out, bool):
        return {"kind": "bool", "value": out}
    if isinstance(out, str):
        return {"kind": "str", "text": _head(out), "chars": len(out)}
    if kind == "HarnessDecision":
        return {"kind": "decision", "type": out.type, "rule_id": out.rule_id, "reason_code": out.reason_code,
                "stage": out.stage}
    s = {"kind": kind}
    for f in ("stage", "label", "disposition", "intervention_type", "reverified", "flag", "tool"):
        if hasattr(out, f) and getattr(out, f, None) is not None:
            s[f] = getattr(out, f)
    if kind == "Read":
        s["arg_keys"] = sorted((out.args or {}).keys()) if isinstance(out.args, dict) else None
    fb = getattr(out, "feedback", None)
    if isinstance(fb, dict):
        s["feedback"] = {k: _head(fb.get(k)) for k in ("decision", "stage", "reason") if fb.get(k) is not None}
    elif fb is not None:
        s["feedback"] = _head(fb)
    if kind == "Accept":
        s["text_sha"] = hashlib.sha1((out.text or "").encode("utf-8", "replace")).hexdigest()[:12]
    if kind == "Observation":
        s["last_obs"] = _head(out.last_obs)
    return s


class Instrumentation(object):
    def __init__(self):
        self.components, self.calls = [], []

    def records(self):
        return self.components + self.calls

    def loaded(self, role, path):
        path = os.path.abspath(path)
        rel = W.repo_path(path)
        if any(c["path"] == rel for c in self.components):
            return
        with open(path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        self.components.append({"kind": "component_loaded", "role": role, "path": rel, "sha256": digest})

    def call(self, obj, hook, step, returned, **extra):
        """Record one call of a workspace hook. -> the record, or None for shell code (not recorded)."""
        path = W.component_path(obj)
        if path is None:
            return None
        rec = {"kind": "middleware_call", "path": W.repo_path(path),
               "name": getattr(obj, "name", None) or type(obj).__name__, "hook": hook, "step": step,
               "returned": returned}
        rec.update(extra)
        self.calls.append(rec)
        return rec

    def semantic_check(self, hook, step, granted):
        """Record one semantic check a middleware asked for at `hook`, paid (granted) or refused by the pool."""
        rec = {"kind": "semantic_check", "hook": hook, "step": step, "granted": bool(granted)}
        self.calls.append(rec)
        return rec


def record_components(inst, agent, middleware, kernel, workspace):
    """The workspace files this run loaded, as the shell loaded them: the agent program and what it reads, the
    middleware host's middleware, the kernel's middleware, the workspace modules each of those loaded, the workspace
    policy modules it composed and the completion-obligation content of its workspace. `workspace` is the run's
    (runner/run.py resolve_workspace; None: no workspace)."""
    if isinstance(agent, W.WorkspaceAgent):
        ws = agent.workspace
        inst.loaded("manifest", os.path.join(ws.path, "manifest.yaml"))
        inst.loaded("agent_program", ws._file(ws.key("agent_program")))
        inst.loaded("system_prompt", ws._file(ws.key("system_prompt")))
        for name in sorted(ws._fragments):
            inst.loaded("prompt_fragment", os.path.join(ws._file("prompt_fragments"), name + ".md"))
        for name in sorted(ws._descriptions):
            inst.loaded("tool_description", os.path.join(ws._file("tool_descriptions"), name + ".yaml"))
    for m in (middleware.middlewares if middleware is not None else []):
        if W.component_path(m):
            inst.loaded("middleware", W.component_path(m))
            for dep in W.modules_loaded_by(W.component_path(m)):
                inst.loaded("module", dep)
    if kernel is not None and workspace:
        inst.loaded("manifest", os.path.join(workspace, "manifest.yaml"))
        for cap in kernel.capabilities:
            if W.component_path(cap):
                inst.loaded("kernel_middleware", W.component_path(cap))
                for dep in W.modules_loaded_by(W.component_path(cap)):
                    inst.loaded("module", dep)
        for p in W.Workspace(workspace).policy_files():
            inst.loaded("policy", p)
        cof = W.Workspace(workspace).completion_obligations_file()
        if cof:
            inst.loaded("completion_obligations", cof)
