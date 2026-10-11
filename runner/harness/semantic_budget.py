# -*- coding: utf-8 -*-
"""The semantic budget: the shell's total ceiling, the workspace's allocation of it, and the one meter every
harness judge unit goes through.

The shell fixes one number: the total, HarnessKernel budget `max_semantic_checks` (harness/kernel.py _DEFAULT_BUDGET, 5;
recorded in result.json as harness.audit.budget.max_semantic_checks and harness.audit.semantic_budget.total). The meter
refuses every unit past it, whoever asks.

The workspace declares how the total is split, in the manifest section `semantic_budget`:

    semantic_budget:
      pools:                          # named pools; each declares exactly one of
        reserve: {share: 0.5, min: 1, max: 5}   #   share: floor(total * share), held within [min, max], never above
                                                #          the total (min and max optional)
        general: {rest: true}                   #   rest:  what the other pools leave (at most one pool)
                                                #   cap:   a fixed number of units
      draws:                          # consumer -> the pools it draws from, in fallback order
        final_verification: [reserve, general]
        goal_spec: [general]
        kernel_middleware: [general]
        middleware: [general]
      rounds: {acquire: 2, revise: 2} # per-component round bounds workspace middleware reads (ctx.semantic_rounds)

A missing section is DEFAULT_ALLOCATION (the allocation the shipped workspaces/*/h0 manifests declare); a section
missing one of the three keys takes that key from DEFAULT_ALLOCATION.

Consumers. The shell's: goal_spec (goal-spec compilation before step 1, runner/run.py) and final_verification
(verify_commit's audit of a final). The workspace's: kernel_middleware (a kernel middleware class) and middleware (a host
middleware class), each also nameable per class, `kernel_middleware:<Class>` / `middleware:<Class>`, for a class the
manifest lists under that key; a class entry wins over its category. A consumer without a draws entry gets no unit.

Validation (allocation_problems, run by Workspace.check, and again when the kernel is built for a run's total): the pool
caps resolved for the total sum to at most the total; an unknown key, pool field, pool, consumer or class, and a
malformed value, are refused.

Pricing. The meter draws one unit for each metered judge the shell hands out, at one of two prices:
    HarnessContext.budgeted_judge(consumer)   one unit per judge call
    HarnessContext.semantic_check(consumer)   one unit for every call of one decision during the hook call that took it;
                                              a call after that hook call is refused (expired), and so is a call past
                                              MAX_JUDGE_CALLS_PER_UNIT on the one unit (over_ceiling)
Units, answered judge calls and refusals are counted here, once each, per consumer (record(): result.json
harness.audit.semantic_budget; the per-unit ceiling, the most calls one unit answered and the calls refused past the
ceiling per consumer are its `calls_per_unit`).
"""
import copy
import fractions
import io
import os
import re

SHELL_CONSUMERS = ("goal_spec", "final_verification")
COMPONENT_CONSUMERS = ("kernel_middleware", "middleware")     # the manifest keys listing the classes
CONSUMERS = SHELL_CONSUMERS + COMPONENT_CONSUMERS
SECTION_KEYS = ("pools", "draws", "rounds")
CHECK, CALL = "check", "call"

# The most judge calls one semantic_check unit answers (a shell constant). A unit buys every call of one decision in the
# hook call that took it, so without a ceiling one unit could buy any number of calls. The longest path of the MedCTA h0
# workspace's adoption gates (workspaces/medcta/h0/middleware/answer_candidate.py evaluate_candidate: a REVISE core flip)
# makes 12 calls, so the shipped workspaces never reach the ceiling.
MAX_JUDGE_CALLS_PER_UNIT = 12

DEFAULT_ALLOCATION = {
    "pools": {"reserve": {"share": 0.5, "min": 1, "max": 5},
              "general": {"rest": True}},
    "draws": {"final_verification": ["reserve", "general"],
              "goal_spec": ["general"],
              "kernel_middleware": ["general"],
              "middleware": ["general"]},
    "rounds": {"acquire": 2, "revise": 2},
}

_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class SemanticBudgetError(ValueError):
    """An allocation the shell refuses (allocation_problems), raised when a kernel is built with it."""


def shell_total():
    """The shell's total semantic ceiling: the kernel's default budget `max_semantic_checks`."""
    from .kernel import _DEFAULT_BUDGET
    return _DEFAULT_BUDGET["max_semantic_checks"]


def normalized(section):
    """The allocation a manifest section declares, with DEFAULT_ALLOCATION's value for each key it does not declare.
    A section that is not a mapping is returned as is (allocation_problems refuses it)."""
    if section is None:
        return copy.deepcopy(DEFAULT_ALLOCATION)
    if not isinstance(section, dict):
        return section
    out = copy.deepcopy(section)
    for k in SECTION_KEYS:
        if k not in out:
            out[k] = copy.deepcopy(DEFAULT_ALLOCATION[k])
    return out


def _int(v):
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def _pool_problems(name, spec):
    if not isinstance(name, str) or not _NAME.match(name):
        return ["pool name %r is not an identifier" % (name,)]
    if not isinstance(spec, dict):
        return ["pool %s is not a mapping" % name]
    problems = ["pool %s: unknown field %r" % (name, k) for k in spec if k not in ("cap", "share", "min", "max", "rest")]
    kinds = [k for k in ("cap", "share", "rest") if k in spec]
    if len(kinds) != 1:
        return problems + ["pool %s must declare exactly one of cap, share, rest (declares %s)"
                           % (name, ", ".join(kinds) or "none")]
    kind = kinds[0]
    if kind != "share" and ("min" in spec or "max" in spec):
        problems.append("pool %s: min and max go with share" % name)
    if kind == "cap" and not _int(spec["cap"]):
        problems.append("pool %s: cap must be a non-negative integer" % name)
    if kind == "rest" and spec["rest"] is not True:
        problems.append("pool %s: rest must be true" % name)
    if kind == "share":
        s = spec["share"]
        if isinstance(s, bool) or not isinstance(s, (int, float)) or not 0 <= s <= 1:
            problems.append("pool %s: share must be a number in [0, 1]" % name)
        for k in ("min", "max"):
            if k in spec and not _int(spec[k]):
                problems.append("pool %s: %s must be a non-negative integer" % (name, k))
        if _int(spec.get("min")) and _int(spec.get("max")) and spec["min"] > spec["max"]:
            problems.append("pool %s: min is above max" % name)
    return problems


def resolve_caps(allocation, total):
    """{pool: cap} for `total`, in declared order. A rest pool may resolve negative (allocation_problems refuses it)."""
    total = max(0, int(total or 0))
    pools = allocation["pools"]
    caps = {}
    for name, spec in pools.items():
        if "cap" in spec:
            caps[name] = spec["cap"]
        elif "share" in spec:
            v = int(fractions.Fraction(str(spec["share"])) * total)
            if "min" in spec:
                v = max(spec["min"], v)
            if "max" in spec:
                v = min(spec["max"], v)
            caps[name] = min(v, total)
    rest = [n for n, s in pools.items() if s.get("rest") is True]
    if rest:
        caps[rest[0]] = total - sum(caps.values())
    return {n: caps[n] for n in pools}


def consumer_problem(consumer, classes=None):
    """Why `consumer` is not a consumer name, or None. `classes` ({manifest key: class names}) also requires a
    per-class consumer to name a class the manifest lists; None skips that."""
    if not isinstance(consumer, str):
        return "consumer %r is not a name" % (consumer,)
    cat, sep, cls = consumer.partition(":")
    if cat not in CONSUMERS:
        return "unknown consumer %r (consumers: %s, or %s)" % (
            consumer, ", ".join(CONSUMERS), " / ".join("%s:<Class>" % c for c in COMPONENT_CONSUMERS))
    if sep:
        if cat not in COMPONENT_CONSUMERS or not _NAME.match(cls):
            return "unknown consumer %r (only %s name a class)" % (consumer, " and ".join(COMPONENT_CONSUMERS))
        if classes is not None and cls not in (classes.get(cat) or ()):
            return "consumer %r names no class the manifest lists under %s" % (consumer, cat)
    return None


def allocation_problems(section, total, classes=None):
    """-> list of problems of the manifest section `section` (None: the default) for the shell total `total`."""
    alloc = normalized(section)
    if not isinstance(alloc, dict):
        return ["semantic_budget is not a mapping"]
    problems = ["unknown key %r (keys: %s)" % (k, ", ".join(SECTION_KEYS)) for k in alloc if k not in SECTION_KEYS]
    pools, draws, rounds = alloc.get("pools"), alloc.get("draws"), alloc.get("rounds")
    if not isinstance(pools, dict) or not pools:
        return problems + ["pools must be a non-empty mapping"]
    pool_problems = []
    for name, spec in pools.items():
        pool_problems += _pool_problems(name, spec)
    if sum(1 for s in pools.values() if isinstance(s, dict) and s.get("rest") is True) > 1:
        pool_problems.append("at most one pool may be rest")
    problems += pool_problems
    if not isinstance(draws, dict):
        problems.append("draws must be a mapping")
    else:
        for consumer, order in draws.items():
            p = consumer_problem(consumer, classes)
            if p:
                problems.append(p)
            if not isinstance(order, list) or not order:
                problems.append("draws %s must be a non-empty list of pools" % (consumer,))
                continue
            problems += ["draws %s: unknown pool %r" % (consumer, n) for n in order if n not in pools]
            if len(set(map(str, order))) != len(order):
                problems.append("draws %s names a pool twice" % (consumer,))
    if not isinstance(rounds, dict):
        problems.append("rounds must be a mapping")
    else:
        problems += ["rounds %s must be a non-negative integer" % (k,) for k, v in rounds.items()
                     if not (isinstance(k, str) and _NAME.match(k)) or not _int(v)]
    if pool_problems:
        return problems
    t = max(0, int(total or 0))
    caps = resolve_caps(alloc, t)
    fixed = sum(c for n, c in caps.items() if pools[n].get("rest") is not True)
    if fixed > t or sum(caps.values()) > t:
        problems.append("the pools sum to %d units (%s), above the shell total %d"
                        % (max(fixed, sum(caps.values())), ", ".join("%s %d" % (n, c) for n, c in caps.items()), t))
    return problems


def manifest_section(workspace):
    """The `semantic_budget` section of the workspace directory `workspace`'s manifest.yaml (None: no workspace, or a
    manifest without the section)."""
    if workspace is None:
        return None
    import yaml
    with io.open(os.path.join(workspace, "manifest.yaml"), encoding="utf-8") as fh:
        manifest = yaml.safe_load(fh) or {}
    return manifest.get("semantic_budget") if isinstance(manifest, dict) else None


def _row():
    return {"checks": 0, "call_units": 0, "calls": 0, "refused_checks": 0, "refused_calls": 0, "expired_calls": 0}


def _ceiling_row():
    return {"max_calls": 0, "refused_calls": 0}


class SemanticMeter(object):
    """The one meter of a task's harness judge units. draw() grants or refuses a unit for a consumer (the total first,
    then the consumer's pools in order); answered() and expired() count what a granted judge did with a call."""

    def __init__(self, total, section=None):
        self.total = max(0, int(total or 0))
        self.source = "default" if section is None else "manifest"
        problems = allocation_problems(section, self.total)
        if problems:
            raise SemanticBudgetError("semantic_budget: " + "; ".join(problems))
        self.allocation = normalized(section)
        self.caps = resolve_caps(self.allocation, self.total)
        self.spent = dict((p, 0) for p in self.caps)
        self.consumers = {}
        self.per_unit = {}          # consumer -> {max_calls: most calls one unit answered, refused_calls: past the ceiling}

    def _row(self, consumer):
        p = consumer_problem(consumer)
        if p:
            raise SemanticBudgetError(p)
        return self.consumers.setdefault(consumer, _row())

    def pools_for(self, consumer):
        draws = self.allocation["draws"]
        if consumer in draws:
            return list(draws[consumer])
        return list(draws.get(consumer.partition(":")[0]) or [])

    def draw(self, consumer, price):
        """One unit for `consumer` at `price` (CHECK or CALL) -> True, or False: refused, counted once."""
        row = self._row(consumer)
        if self.spent_total() < self.total:
            for pool in self.pools_for(consumer):
                if self.spent[pool] < self.caps[pool]:
                    self.spent[pool] += 1
                    row["checks" if price == CHECK else "call_units"] += 1
                    return True
        row["refused_checks" if price == CHECK else "refused_calls"] += 1
        return False

    def answered(self, consumer, unit_calls=None):
        """A granted judge answered a call; `unit_calls` is the number of calls its unit has answered with this one (a
        semantic_check unit)."""
        self._row(consumer)["calls"] += 1
        if unit_calls is not None:
            rec = self.per_unit.setdefault(consumer, _ceiling_row())
            rec["max_calls"] = max(rec["max_calls"], unit_calls)

    def expired(self, consumer):
        self._row(consumer)["expired_calls"] += 1

    def over_ceiling(self, consumer):
        """A call past MAX_JUDGE_CALLS_PER_UNIT on one unit: refused, counted once."""
        self._row(consumer)
        self.per_unit.setdefault(consumer, _ceiling_row())["refused_calls"] += 1

    def rounds(self, name):
        return self.allocation["rounds"][name]

    def spent_total(self):
        return sum(self.spent.values())

    def refused_total(self):
        return (sum(r["refused_checks"] + r["refused_calls"] + r["expired_calls"] for r in self.consumers.values())
                + sum(r["refused_calls"] for r in self.per_unit.values()))

    def pool_cap(self, pool):
        return self.caps.get(pool)

    def pool_spent(self, pool):
        return self.spent.get(pool)

    def record(self):
        consumers = dict((c, dict(r, units=r["checks"] + r["call_units"],
                                  refused=(r["refused_checks"] + r["refused_calls"] + r["expired_calls"]
                                           + (self.per_unit.get(c) or {}).get("refused_calls", 0))))
                         for c, r in sorted(self.consumers.items()))
        return {"total": self.total, "allocation_source": self.source, "allocation": copy.deepcopy(self.allocation),
                "pools": dict((p, {"cap": self.caps[p], "spent": self.spent[p]}) for p in self.caps),
                "spent": self.spent_total(), "calls": sum(r["calls"] for r in self.consumers.values()),
                "refused": self.refused_total(), "consumers": consumers,
                "calls_per_unit": {"ceiling": MAX_JUDGE_CALLS_PER_UNIT,
                                   "consumers": dict((c, dict(r)) for c, r in sorted(self.per_unit.items()))}}
