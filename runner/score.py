#!/usr/bin/env python3
"""Thin wrapper around runner/scorers/score.py, where all scoring code lives (runner/scorers/).
`python runner/score.py ...` delegates verbatim to the real script; calling runner/scorers/score.py
directly is equivalent."""
import os, runpy
_real = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scorers", "score.py")
runpy.run_path(_real, run_name="__main__")   # __file__=_real -> its own sys.path bootstrap + argparse run
