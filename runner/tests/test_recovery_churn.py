"""Recovery stops UNPRODUCTIVE churn (>=2 consecutive BLOCKED_UNRESOLVED_AFFORDANCE) without cutting a
PRODUCTIVE episode -- a VERIFIED/PARTIAL/acquire episode resets the counter, so recovery that turns an
unmet check into a met one keeps running."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from workspace import H0, Workspace
RecoveryKernel = Workspace(H0["gui"]).module("recovery/kernel.py").RecoveryKernel
C = Workspace(H0["gui"]).module("recovery/contracts.py")


class _R:
    def __init__(self, st):
        self.state = st


def _kernel(states):
    it = iter(states)
    k = RecoveryKernel()
    k._build_ctx = lambda *a, **kw: {}
    k._resolve = lambda *a, **kw: [object() for _ in states]
    k._run_single = lambda *a, **kw: _R(next(it))
    k._finalize = lambda r: r
    return k


def test_stops_after_two_consecutive_blocked():
    B = C.BLOCKED_UNRESOLVED_AFFORDANCE
    out = _kernel([B, B, B, B, B, B]).run_all_episodes(None, None, None, None, {}, [], None, None)
    assert len(out) == 2                                  # a run of blocked episodes is cut after two


def test_productive_episode_resets_and_is_not_cut():
    B, V = C.BLOCKED_UNRESOLVED_AFFORDANCE, C.VERIFIED
    # V,B,V,B,B -> never 2 consecutive blocked until the very end -> all 5 run (productive V never cut)
    out = _kernel([V, B, V, B, B]).run_all_episodes(None, None, None, None, {}, [], None, None)
    assert len(out) == 5
    assert out[0].state == V and out[2].state == V       # both productive episodes preserved


def test_declined_does_not_count_as_churn():
    V, D = C.VERIFIED, C.DECLINED_NO_COMMITMENT
    out = _kernel([V, D, D]).run_all_episodes(None, None, None, None, {}, [], None, None)
    assert len(out) == 3                                  # denial pattern (VERIFIED then cheap DECLINEs) survives
