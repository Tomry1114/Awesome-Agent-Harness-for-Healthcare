"""Repo-root pytest config: stop standalone scripts from poisoning collection.

THE PROBLEM. Some files named `test_*.py` are not pytest tests -- they are standalone scripts
(`python path/to/file.py`) that call `sys.exit()` at module level. pytest imports every `test_*.py` during
collection, so that `sys.exit` raises SystemExit *inside collection*, which pytest reports as INTERNALERROR
and aborts the ENTIRE run. One such file silently zeroes out its whole directory, hiding both the passing
and the genuinely failing tests next to it.

**A visible red is strictly better than an invisible blank.** Un-poisoning collection may surface failures
that were previously invisible -- that is progress, not regression. Do not "restore green" by hiding them.

WHY DETECTION RATHER THAN A LIST. A hardcoded list drifts: the next standalone script re-poisons its
directory and nobody notices, which is exactly the failure mode this file exists to end. Detection is
self-maintaining.

SKIPS ARE RECORDED. Silently skipping would recreate the same problem in a new form -- coverage that looks
present and is not. Every skipped file's repo-relative path is recorded in `_skipped`, so "not collected"
can stay visible.
"""
import ast
import os

import pytest

_skipped = []
_cache = {}


def _exits_at_import(path):
    """True iff importing this file would call sys.exit().

    AST, not a regex. A regex on indentation misses `ok=...; print(...); sys.exit(0)` (one line, three
    statements) and cannot tell a real module-level exit from the word appearing in a comment.

    `if __name__ == "__main__": sys.exit(...)` is EXCLUDED -- that is the correct way to write a dual-use
    file, it does not fire on import, and flagging it would push people away from the right pattern.
    """
    hit = _cache.get(path)
    if hit is not None:
        return hit
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            tree = ast.parse(fh.read(), filename=path)
    except (OSError, SyntaxError):
        _cache[path] = False
        return False

    def _is_main_guard(node):
        if not isinstance(node, ast.If):
            return False
        t = node.test
        return (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name)
                and t.left.id == "__name__")

    found = False
    for node in tree.body:                       # TOP-LEVEL statements only
        if _is_main_guard(node):
            continue                             # runs under `python file.py`, never on import
        for sub in ast.walk(node):
            if isinstance(sub, ast.FunctionDef) or isinstance(sub, ast.AsyncFunctionDef):
                continue                         # a body that merely *defines* an exit is fine
            if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr == "exit"):
                found = True
                break
        if found:
            break
    _cache[path] = found
    return found


def pytest_ignore_collect(collection_path, config):
    """Refuse collection BEFORE import. `pytest_collect_file` is too late: the built-in python plugin still
    imports the module, and the sys.exit fires during that import. Returning True here skips it outright."""
    p = str(collection_path)
    if os.path.basename(p).startswith("test_") and p.endswith(".py") and _exits_at_import(p):
        rel = os.path.relpath(p, os.path.dirname(os.path.abspath(__file__)))
        if rel not in _skipped:
            _skipped.append(rel)
        return True
    return None


# ============================================================================================================
# SURGICAL cross-file os.environ isolation.
#
# DO NOT "simplify" this to os.environ.clear() + update(pristine).
#
# A per-test `os.environ.clear(); update(pristine)` fights pytest's own bookkeeping (pytest sets
# PYTEST_CURRENT_TEST during a test and deletes it at teardown; clearing it makes that delete raise KeyError)
# and it also wipes whatever session-scoped fixtures legitimately set.
#
# This removes ONLY the keys that module-level collection introduced, and touches nothing else:
#     pristine        = conftest import time      (before collection)
#     post_collection = collection finished       (after every module-level write has run)
#     leaked          = post_collection - pristine
# Per MODULE (not per test), drop exactly `leaked`. PYTEST_* and anything a session fixture sets later are
# untouched, because they are not in that diff.
#
# MODULE scope, not function scope, and the reason matters: a file that sets an env var in its OWN
# module-scoped fixture is doing something legitimate -- sharing state within one file is that file's
# business. A function-scoped reset runs AFTER that fixture and wipes it (e.g. test_acq_routing relies on a
# module-scoped environment). The target is CROSS-file leakage; module scope removes exactly that and leaves
# intra-file setup alone. This is a SEMANTIC choice, not a performance one: tightening it to function scope
# breaks files with module-scoped env fixtures.
# ============================================================================================================
_PRISTINE_ENV = dict(os.environ)
_LEAKED_KEYS = set()


def pytest_collection_finish(session):
    """Collection is over -> every module-level `os.environ[...] = ...` has already fired. Whatever is new
    relative to import time IS the cross-file leak."""
    for k, v in os.environ.items():
        if k.startswith("PYTEST_"):
            continue
        if k not in _PRISTINE_ENV or _PRISTINE_ENV[k] != v:
            _LEAKED_KEYS.add(k)


@pytest.fixture(autouse=True, scope="module")
def _isolate_env():
    for k in _LEAKED_KEYS:
        if k in _PRISTINE_ENV:
            os.environ[k] = _PRISTINE_ENV[k]
        else:
            os.environ.pop(k, None)
    yield
