# -*- coding: utf-8 -*-
"""Test files that exit at import cannot be collected: the root conftest skips them before import, because
the exit would kill the session. They still test live code, so they run HERE, each in its own process, and
must exit 0; otherwise a broken script-style test (a stale assertion, an import of a deleted module) would go
unnoticed. Discovery uses the same predicate the conftest uses to skip, so a new script-style test cannot be
skipped without also being run.
"""
import importlib.util
import os
import subprocess
import sys

import pytest

_RUNNER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO = os.path.dirname(_RUNNER)


def _predicate():
    spec = importlib.util.spec_from_file_location("_root_conftest", os.path.join(_REPO, "conftest.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._exits_at_import


def _scripts():
    exits = _predicate()
    out = []
    for root, dirs, files in os.walk(_RUNNER):
        dirs[:] = [d for d in dirs if d not in ("__pycache__",) and not d.startswith(".")]
        for f in files:
            if f.startswith("test_") and f.endswith(".py") and exits(os.path.join(root, f)):
                out.append(os.path.relpath(os.path.join(root, f), _RUNNER))
    return sorted(out)


SCRIPTS = _scripts()


def test_the_script_tests_are_found():
    assert SCRIPTS, "discovery found nothing -- the predicate or the walk is broken"


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_exits_zero(script):
    env = dict(os.environ)
    env["PYTHONPATH"] = _RUNNER
    p = subprocess.run([sys.executable, script], cwd=_RUNNER, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, timeout=600)
    tail = p.stdout.decode("utf-8", "replace")[-1500:]
    assert p.returncode == 0, "%s exited %d:\n%s" % (script, p.returncode, tail)
