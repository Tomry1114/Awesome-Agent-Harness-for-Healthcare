"""Every scorer module compiles and imports. A syntax error in a scorer module that no test imports would otherwise go
unnoticed until `score.py` (the Clinical Conduct Boundaries rescore + 7-dim aggregation) fails to start."""
import glob
import importlib
import os
import py_compile
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCORERS = os.path.join(HERE, "scorers")


def test_every_scorer_module_compiles():
    for p in sorted(glob.glob(os.path.join(SCORERS, "*.py"))):
        py_compile.compile(p, doraise=True)


def test_the_scoring_entry_points_import():
    for d in (SCORERS, HERE):
        if d not in sys.path:
            sys.path.insert(0, d)
    for name in ("rescore_judges", "aggregate_report"):        # score.py is a script: it parses argv on import
        importlib.import_module(name)
