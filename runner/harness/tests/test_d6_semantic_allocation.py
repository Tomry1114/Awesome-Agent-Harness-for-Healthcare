# -*- coding: utf-8 -*-
"""The shell limits only the total semantic budget; the workspace manifest's semantic_budget section allocates it
(runner/harness/semantic_budget.py).

    an edited allocation changes which consumer is refused
    a pool sum above the total (or an unknown consumer) is refused at Workspace.check
    the total is enforced across all consumers through the one meter
    refusals are counted once, at the meter, whichever consumer is refused
    the default (H0) workspaces declare the default allocation, which matches a reference model of it draw for draw
"""
import io
import itertools
import json
import os
import re
import shutil
import sys

import pytest
import yaml

_RUNNER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _RUNNER not in sys.path:
    sys.path.insert(0, _RUNNER)

import harness as H                                                    # noqa: E402
import middleware_host as MH                                           # noqa: E402
import workspace as W                                                  # noqa: E402
from harness import decision as D                                      # noqa: E402
from harness import semantic_budget as S                               # noqa: E402
from harness.capabilities.obligation_lifecycle import ObligationLifecycle   # noqa: E402
from harness.capabilities.scope_evidence import ScopeEvidenceBinding   # noqa: E402
from harness.capabilities.verify_commit import VerifyAndCommit         # noqa: E402
from harness.capability import HarnessContext                          # noqa: E402
from harness.compiler import build_contract                            # noqa: E402
from harness.kernel import HarnessKernel                               # noqa: E402
from harness.state import Ledger                                       # noqa: E402

TASK = {"task_id": "m", "goal": "finding?", "context": {}, "environment": {"type": "tool_sandbox"}}
CLEAN_AUDIT = json.dumps({"addresses_task": True, "hard_violations": [], "repairable_gaps": [], "confidence": 0.9})


@pytest.fixture
def prompts(monkeypatch):
    """build_kernel binds the harness judge from configuration; here it is a stub recording every prompt it answers."""
    seen = []
    monkeypatch.setattr(H, "_build_judge", lambda: ((lambda p: seen.append(p) or "{}"), "d6-stub-judge"))
    return seen


def _medcta_with(path, section):
    """A copy of the default (H0) MedCTA workspace whose manifest declares `section` as semantic_budget."""
    path = str(path)
    shutil.copytree(W.H0["tool_sandbox"], path)
    mp = os.path.join(path, "manifest.yaml")
    manifest = yaml.safe_load(io.open(mp, encoding="utf-8"))
    manifest["semantic_budget"] = section
    with io.open(mp, "w", encoding="utf-8") as fh:
        yaml.safe_dump(manifest, fh, sort_keys=False)
    return path


def _kernel(workspace, **kw):
    k = H.build_kernel(TASK, env_type="tool_sandbox", workspace=workspace, **kw)
    assert k.ctx.semantic_meter.spent_total() == 0 and k.ctx.semantic_meter.refused_total() == 0
    return k


def _view(k, cls):
    """The context the kernel hands its workspace kernel middleware of class `cls`."""
    return k._ctx_for(next(c for c in k.capabilities if type(c).__name__ == cls))


def _host(k, cls, hook, body):
    """Run `body(ctx)` as the `hook` of a host middleware of class `cls`, through the shell's middleware stack, with the
    judge services run.py wires (budgeted ctx.judge; on_answer's semantic_check)."""
    mw = type(cls, (object,), {hook: lambda self, ctx, *a: body(ctx)})()
    ctx = MH.MiddlewareContext(env=None, task={}, trajectory=[], step=1, max_steps=10, used_actions=0,
                               judge_factory=lambda consumer: k.ctx.budgeted_judge(consumer), errors=[],
                               shell=MH.answer_services(k, 1, []))
    return getattr(MH.MiddlewareStack([mw]), hook)(ctx, "a")


def _refused(k):
    return sorted(c for c, r in k.audit()["semantic_budget"]["consumers"].items() if r["refused"])


def _verify_kernel(prompts, budget=5, section=None):
    """The shell's final verification (verify_commit) on a MedCTA answer grounded by one perception read."""
    pol = H.load_policy(env_type="tool_sandbox", workspace=W.H0["tool_sandbox"])
    contract = build_contract(TASK, env_type="tool_sandbox", policy=pol)

    def judge(p):
        prompts.append(p)
        return CLEAN_AUDIT
    k = HarnessKernel(contract, [ScopeEvidenceBinding(), ObligationLifecycle(), VerifyAndCommit()], policy=pol,
                      env_type="tool_sandbox", judge_fn=judge, budget={"max_semantic_checks": budget},
                      semantic_allocation=section)
    k.after_action({"type": "tool", "tool": "ImageDescription", "args": {}}, "a 3cm spiculated RUL nodule",
                   None, None, step=0, result_ok=True)
    return k


# ---------------------------------------------------------------------------------------- 1 the allocation decides
def test_an_edited_allocation_changes_which_consumer_is_refused(tmp_path, prompts):
    def sequence(k):
        k.ctx.budgeted_judge("goal_spec")("compile the goal")
        assert _view(k, "GoalAlignment").semantic_check() is not None
        _view(k, "ActiveEvidence").semantic_check()
        _host(k, "AnswerCandidate", "on_answer", lambda ctx: ctx.semantic_check())
        assert k.ctx.semantic_check("final_verification") is not None
        return _refused(k)

    h0 = _kernel(W.H0["tool_sandbox"])
    assert h0.audit()["semantic_budget"]["allocation_source"] == "manifest"
    assert sequence(h0) == ["middleware:AnswerCandidate"]

    edited = S.normalized(None)
    edited["pools"] = {"candidate": {"cap": 1}, "reserve": {"share": 0.5, "min": 1, "max": 5}, "general": {"rest": True}}
    edited["draws"]["middleware:AnswerCandidate"] = ["candidate"]
    k = _kernel(_medcta_with(tmp_path / "ws", edited))
    assert k.audit()["semantic_budget"]["pools"] == {"candidate": {"cap": 1, "spent": 0},
                                                     "reserve": {"cap": 2, "spent": 0}, "general": {"cap": 2, "spent": 0}}
    assert sequence(k) == ["kernel_middleware:ActiveEvidence"]


def test_an_allocation_without_verification_units_leaves_the_final_unverified(prompts):
    assert _verify_kernel(prompts).before_final("RUL nodule", step=1).type == D.ALLOW
    assert len(prompts) == 1
    del prompts[:]
    section = {"pools": {"general": {"rest": True}},
               "draws": {"goal_spec": ["general"], "kernel_middleware": ["general"], "middleware": ["general"]}}
    k = _verify_kernel(prompts, section=section)
    eff = k.before_final("RUL nodule", step=1)
    assert (eff.type, eff.raw.reason_code) == (D.ESCALATE, "unverifiable_commit") and prompts == []
    assert _refused(k) == ["final_verification"]


# ---------------------------------------------------------------------------------------- 2 validation at check
def test_a_pool_sum_above_the_total_or_an_unknown_consumer_is_refused_at_check(tmp_path):
    def budget_problems(path):
        return [p for p in W.Workspace(path).check([]) if "semantic_budget" in p]
    assert budget_problems(W.H0["tool_sandbox"]) == []

    over = dict(S.normalized(None), pools={"reserve": {"cap": 3}, "general": {"cap": 3}})
    problems = budget_problems(_medcta_with(tmp_path / "over", over))
    assert len(problems) == 1 and problems[0].endswith(
        "semantic_budget: the pools sum to 6 units (reserve 3, general 3), above the shell total 5"), problems
    with pytest.raises(S.SemanticBudgetError):
        HarnessContext(Ledger(), None, {}, semantic_budget=5, semantic_allocation=over)

    unknown = S.normalized(None)
    unknown["draws"].update({"planner": ["general"], "kernel_middleware:NotListed": ["general"],
                             "middleware:AnswerCandidate": ["nowhere"]})
    problems = budget_problems(_medcta_with(tmp_path / "unknown", unknown))
    assert len(problems) == 3, problems
    assert any("unknown consumer 'planner'" in p for p in problems)
    assert any("'kernel_middleware:NotListed' names no class the manifest lists under kernel_middleware" in p
               for p in problems)
    assert any("draws middleware:AnswerCandidate: unknown pool 'nowhere'" in p for p in problems)


# ---------------------------------------------------------------------------------------- 3 one meter, one total
def test_the_total_is_enforced_across_all_consumers_through_one_meter(tmp_path, prompts):
    takes = [
        ("goal_spec", lambda k: k.ctx.budgeted_judge("goal_spec")("p")),
        ("kernel_middleware:GoalAlignment", lambda k: _view(k, "GoalAlignment").semantic_check()),
        ("kernel_middleware:RequiredContext", lambda k: _view(k, "RequiredContext").budgeted_judge()("p")),
        ("middleware:AnswerCandidate judge", lambda k: _host(k, "AnswerCandidate", "before_final",
                                                             lambda ctx: ctx.judge("p"))),
        ("middleware:AnswerCandidate check", lambda k: _host(k, "AnswerCandidate", "on_answer",
                                                             lambda ctx: ctx.semantic_check())),
        ("final_verification", lambda k: k.ctx.semantic_check("final_verification")),
    ]
    # default workspace: the pools partition the total; with every consumer allowed to draw every pool: still the total
    everywhere = {"pools": {"reserve": {"cap": 1}, "general": {"rest": True}},
                  "draws": dict((c, ["general", "reserve"]) for c in S.CONSUMERS)}
    for k in (_kernel(W.H0["tool_sandbox"]), _kernel(_medcta_with(tmp_path / "ws", everywhere))):
        meter, granted = k.ctx.semantic_meter, 0
        for _round in range(3):
            for _name, take in takes:
                granted += take(k) is not None
        assert granted == meter.spent_total() == meter.total == k.budget["max_semantic_checks"] == 5
        rec = k.audit()["semantic_budget"]
        assert rec["total"] == 5 and rec["spent"] == 5 and rec["refused"] == 3 * len(takes) - 5
        assert sum(r["units"] for r in rec["consumers"].values()) == 5
        for _name, take in takes:
            assert take(k) is None
    # the run's total is the kernel's budget: a larger one is enforced the same way
    wide = _kernel(W.H0["tool_sandbox"], budget={"max_semantic_checks": 50})
    assert wide.ctx.semantic_meter.caps == {"reserve": 5, "general": 45}


def test_no_shell_module_draws_or_counts_judge_units_outside_the_meter():
    """The meter's two issuers (HarnessContext.semantic_check / budgeted_judge) are the only callers of draw(), and no
    module keeps a refusal counter of its own."""
    offenders = []
    for dirpath, dirnames, filenames in os.walk(_RUNNER):
        dirnames[:] = [d for d in dirnames if d not in ("__pycache__", "tests")]
        for fn in filenames:
            if not fn.endswith(".py") or fn.startswith("test_"):
                continue
            p = os.path.join(dirpath, fn)
            rel = os.path.relpath(p, _RUNNER).replace(os.sep, "/")
            src = io.open(p, encoding="utf-8").read()
            if rel not in ("harness/capability.py", "harness/semantic_budget.py") and \
                    re.search(r"\.(draw|answered|expired)\(", src):
                offenders.append(rel + ": draws")
            if re.search(r"n_semantic_refused\s*\+=|refused_(checks|calls)|expired_calls", src) and \
                    rel != "harness/semantic_budget.py":
                offenders.append(rel + ": counts refusals")
    assert offenders == []


# ---------------------------------------------------------------------------------------- 4 refusals counted once
def test_refusals_are_counted_once_whichever_consumer_is_refused(prompts):
    k = _kernel(W.H0["tool_sandbox"], budget={"max_semantic_checks": 0})
    meter = k.ctx.semantic_meter
    refusals = [
        ("goal_spec", lambda: k.ctx.budgeted_judge("goal_spec")("p")),
        ("kernel_middleware:GoalAlignment", lambda: _view(k, "GoalAlignment").semantic_check()),
        ("kernel_middleware:RequiredContext", lambda: _view(k, "RequiredContext").budgeted_judge()("p")),
        ("middleware:DeliverableMiddleware", lambda: _host(k, "DeliverableMiddleware", "before_final",
                                                           lambda ctx: ctx.judge("p"))),
        ("middleware:AnswerCandidate", lambda: _host(k, "AnswerCandidate", "on_answer",
                                                     lambda ctx: ctx.semantic_check())),
        ("final_verification", lambda: k.ctx.semantic_check("final_verification")),
    ]
    for i, (consumer, refuse) in enumerate(refusals, 1):
        assert refuse() is None
        audit = k.audit()
        assert meter.refused_total() == audit["n_semantic_refused"] == audit["semantic_budget"]["refused"] == i
        assert audit["semantic_budget"]["consumers"][consumer]["refused"] == 1
    assert prompts == []

    # a check judge kept past its hook call: each refused call once, as expired, and nothing else moves
    k = _kernel(W.H0["tool_sandbox"])
    judge = _view(k, "ActiveEvidence").semantic_check()
    k.ctx.hook_serial += 1
    assert judge("a") is None and judge("b") is None
    row = k.audit()["semantic_budget"]["consumers"]["kernel_middleware:ActiveEvidence"]
    assert (row["checks"], row["calls"], row["expired_calls"], row["refused"]) == (1, 0, 2, 2)
    assert k.audit()["n_semantic_refused"] == 2

    # the shell's final verification refused inside verify_commit: once
    vk = _verify_kernel(prompts, budget=0)
    assert vk.before_final("RUL nodule", step=1).type == D.ESCALATE
    assert vk.audit()["n_semantic_refused"] == 1 and _refused(vk) == ["final_verification"]


# ---------------------------------------------------------------------------------------- 5 default allocation
class _BeforeD6(object):
    """Reference model of the default semantic budget allocation: MAX_ACQ_ROUNDS = MAX_REV_ROUNDS = 2, a verification
    reserve of min(2 + 2 + 1, max(1, total // 2)) (0 for a zero total) and a general pool of the rest; the final
    verification draws the reserve, then the general pool; every other consumer the general pool."""

    def __init__(self, total):
        self.reserve = min(2 + 2 + 1, max(1, total // 2)) if total > 0 else 0
        self.general = total - self.reserve

    def draw(self, consumer):
        if consumer == "final_verification" and self.reserve > 0:
            self.reserve -= 1
            return True
        if self.general > 0:
            self.general -= 1
            return True
        return False


@pytest.mark.parametrize("env", sorted(W.H0))
def test_h0_declares_the_allocation_the_shell_had_before_d6(env):
    section = S.manifest_section(W.H0[env])
    assert section == S.normalized(None) == S.DEFAULT_ALLOCATION        # a missing section is the default allocation
    assert section["rounds"] == {"acquire": 2, "revise": 2}
    consumers = ("goal_spec", "final_verification", "kernel_middleware:GoalAlignment", "middleware:AnswerCandidate")
    for total in (0, 1, 2, 3, 4, 5, 9, 10, 11, 50):
        before = _BeforeD6(total)
        meter = S.SemanticMeter(total, section)
        assert meter.caps == {"reserve": before.reserve, "general": before.general}
        for seq in itertools.product(consumers, repeat=5):
            before, meter = _BeforeD6(total), S.SemanticMeter(total, section)
            for c in seq:
                assert meter.draw(c, S.CHECK) == before.draw(c), (total, seq)
