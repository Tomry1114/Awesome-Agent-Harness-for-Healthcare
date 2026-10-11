# -*- coding: utf-8 -*-
"""Is a PhysicianBench checkpoint code-graded or judge-graded? Read from the upstream test source.

Each upstream checkpoint is a pytest function in tasks/v1/<task>/tests/test_outputs.py. It is judge-graded when
its body -- or any module-level function it calls, transitively -- calls one of the upstream LLM helpers
(llm_judge, llm_extract*, call_llm). Otherwise it is code-graded ("deterministic"): FHIR queries, regexes and
arithmetic only.

This is the checkpoint's TYPE, identical for every run and configuration, so results can be compared per
checkpoint kind; what one run happened to call is recorded separately by the grader's judge block. Static
reading, no execution, no network.
"""
import ast
import json
import os

LLM_HELPERS = frozenset(("llm_judge", "llm_extract", "llm_extract_value", "llm_extract_decision",
                         "llm_extract_boolean", "llm_extract_finding", "call_llm"))

_TASKS = {}
_FILES = {}


def _task_refs(root):
    if root not in _TASKS:
        refs = {}
        p = os.path.join(root, "benchmark_dataprocess", "PhysicianBench", "tasks_unified.jsonl")
        try:
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    t = json.loads(line)
                    for cp in t.get("checkpoints") or []:
                        if cp.get("native_test_ref"):
                            refs[(t.get("task_id"), cp.get("id"))] = cp["native_test_ref"]
        except OSError:
            pass
        _TASKS[root] = refs
    return _TASKS[root]


def _called_names(fn):
    names = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                names.add(f.id)
            elif isinstance(f, ast.Attribute):
                names.add(f.attr)
    return names


def _file_kinds(path):
    """{function_name: "llm"|"deterministic"} for every module-level function in one test file."""
    if path in _FILES:
        return _FILES[path]
    try:
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=path)
    except (OSError, SyntaxError):
        _FILES[path] = None
        return None
    fns = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    calls = {name: _called_names(fn) for name, fn in fns.items()}
    memo = {}

    def uses_llm(name, stack=()):
        if name in memo:
            return memo[name]
        if name in stack:
            return False
        direct = calls.get(name, set())
        hit = bool(direct & LLM_HELPERS) or any(uses_llm(c, stack + (name,)) for c in direct if c in fns)
        memo[name] = hit
        return hit
    out = {name: ("llm" if uses_llm(name) else "deterministic") for name in fns}
    _FILES[path] = out
    return out


def grading(task_id, cp_id, root):
    """"deterministic" | "llm" | None (no native_test_ref, or the upstream source is not readable here)."""
    ref = _task_refs(root).get((task_id, cp_id))
    if not ref or "::" not in ref:
        return None
    rel, fn = ref.split("::", 1)
    kinds = _file_kinds(os.path.join(root, "benchmark", "PhysicianBench", "PhysicianBench", rel))
    if not kinds:
        return None
    return kinds.get(fn)
