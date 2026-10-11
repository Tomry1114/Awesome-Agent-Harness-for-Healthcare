# -*- coding: utf-8 -*-
"""Judge binding: which models score a run, resolved once from configuration, required before any agent step,
and recorded in result.json run_config.

No judge comes from an implicit default or is silently off. Without an explicit binding an unset variable would
change scoring unnoticed, for example:

    LLM_JUDGE_MODEL unset         -> every PhysicianBench content checkpoint raises inside upstream eval_helpers,
                                     comes back FAIL and is recorded as agent_failure
    MH_HARNESS_JUDGE_MODEL unset  -> the kernel runs with no semantic layer at all

Every result records who judged it, so two runs judged differently do not pool unnoticed.

Roles. Each is configuration (model names, run counts, token limits), not a switch: every role a benchmark
needs is always on, and a run whose configuration is missing is refused before its first agent step.

    role                 what it judges                                     configured by
    harness              the kernel's semantic checks (every benchmark)     MH_HARNESS_JUDGE_MODEL,
                                                                            MH_HARNESS_JUDGE_MAX_TOKENS
    pb_content           PhysicianBench content checkpoints, via upstream   LLM_JUDGE_MODEL
                         utils/eval_helpers.call_llm (one call per judged
                         item, temperature 0, max_completion_tokens 4000)
    hab_rubric           HealthAdminBench rubric evals, native LLMJudge     MH_JUDGE_MODEL, MH_RUBRIC_NUM_RUNS
                         majority vote over `runs` calls
    medcta_gacc          MedCTA goal accuracy (upstream goal_accuracy.py)   MH_GACC_MODEL
    medcta_checkpoint    MedCTA verification and governance checkpoints     MH_JUDGE_MODEL

Which fields are compared between two arms (SIGNATURE_FIELDS) is the configurable part: model everywhere, the
rubric's run count, the harness judge's token limit. The other token limits are code constants; they are
recorded, not compared, because they change only with the code.
"""
import os


class JudgeConfigError(RuntimeError):
    """A judge the benchmark needs has no (or invalid) configuration. Raised before any agent step."""


HARNESS_JUDGE_MAX_TOKENS_DEFAULT = 300    # default for MH_HARNESS_JUDGE_MAX_TOKENS
HAB_RUBRIC_RUNS_DEFAULT = 3               # upstream health-admin-bench LLMJudge num_runs default
HAB_RUBRIC_MAX_TOKENS = 300
PB_CONTENT_MAX_TOKENS = 4000              # upstream utils/eval_helpers.call_llm max_completion_tokens
MEDCTA_GACC_MAX_TOKENS = 1024
MEDCTA_CHECKPOINT_MAX_TOKENS = 400

ROLES = {
    "harness": {"model_var": "MH_HARNESS_JUDGE_MODEL", "max_tokens_var": "MH_HARNESS_JUDGE_MAX_TOKENS",
                "runs": 1, "max_tokens": HARNESS_JUDGE_MAX_TOKENS_DEFAULT},
    "pb_content": {"model_var": "LLM_JUDGE_MODEL", "runs": 1, "max_tokens": PB_CONTENT_MAX_TOKENS,
                   "temperature": 0},
    "hab_rubric": {"model_var": "MH_JUDGE_MODEL", "runs_var": "MH_RUBRIC_NUM_RUNS",
                   "runs": HAB_RUBRIC_RUNS_DEFAULT, "max_tokens": HAB_RUBRIC_MAX_TOKENS},
    "medcta_gacc": {"model_var": "MH_GACC_MODEL", "runs": 1, "max_tokens": MEDCTA_GACC_MAX_TOKENS},
    "medcta_checkpoint": {"model_var": "MH_JUDGE_MODEL", "runs": 1, "max_tokens": MEDCTA_CHECKPOINT_MAX_TOKENS},
}

BENCH_ROLES = {
    "PhysicianBench": ("harness", "pb_content"),
    "HealthAdminBench": ("harness", "hab_rubric"),
    "MedCTA": ("harness", "medcta_gacc", "medcta_checkpoint"),
}

SIGNATURE_FIELDS = {
    "harness": ("model", "max_tokens"),
    "pb_content": ("model",),
    "hab_rubric": ("model", "runs"),
    "medcta_gacc": ("model",),
    "medcta_checkpoint": ("model",),
}

JUDGE_VARS = tuple(sorted({v for r in ROLES.values() for k, v in r.items() if k.endswith("_var")}))


def roles_for(bench):
    """The judge roles a benchmark needs. An unknown benchmark still runs the harness, so it needs that judge."""
    return BENCH_ROLES.get(bench, ("harness",))


def _get(env, name):
    v = (env if env is not None else os.environ).get(name)
    v = v.strip() if isinstance(v, str) else v
    return v or None


def _positive_int(env, name, default):
    raw = _get(env, name)
    if raw is None:
        return default, None
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return None, "%s=%r is not an integer" % (name, raw)
    if n < 1:
        return None, "%s=%r must be >= 1" % (name, raw)
    return n, None


def _spec(role, env):
    """-> (spec, problems). spec is None when the role cannot be bound."""
    d = ROLES[role]
    problems = []
    model = _get(env, d["model_var"])
    if not model:
        problems.append("%s (judge role %s) is not set" % (d["model_var"], role))
    runs, err = (_positive_int(env, d["runs_var"], d["runs"]) if "runs_var" in d else (d["runs"], None))
    if err:
        problems.append(err)
    mt, err = (_positive_int(env, d["max_tokens_var"], d["max_tokens"]) if "max_tokens_var" in d
               else (d["max_tokens"], None))
    if err:
        problems.append(err)
    if problems:
        return None, problems
    config = {d["model_var"]: model}
    if "runs_var" in d:
        config[d["runs_var"]] = runs
    if "max_tokens_var" in d:
        config[d["max_tokens_var"]] = mt
    spec = {"role": role, "model": model, "runs": runs, "max_tokens": mt, "config": config}
    if "temperature" in d:
        spec["temperature"] = d["temperature"]
    return spec, []


def spec(role, env=None):
    """The bound spec of one role, or JudgeConfigError."""
    s, problems = _spec(role, env)
    if s is None:
        raise JudgeConfigError("judge configuration missing or invalid: " + "; ".join(problems))
    return s


def require(bench, env=None):
    """Every judge `bench` needs, bound from configuration. Refuses (JudgeConfigError) naming EVERY missing or
    invalid variable at once, so a launcher fixes its configuration in one pass."""
    roles, problems = {}, []
    for role in roles_for(bench):
        s, p = _spec(role, env)
        if s is None:
            problems.extend(p)
        else:
            roles[role] = s
    if problems:
        raise JudgeConfigError("REFUSING to run %s: judge configuration missing or invalid -- %s. A judge is "
                               "never optional: without it a checkpoint cannot be scored and the run would "
                               "still produce numbers." % (bench, "; ".join(problems)))
    return {"bench": bench, "roles": roles}


def signature(binding_or_roles):
    """The comparable part of a binding: {role: {field: value}} over SIGNATURE_FIELDS."""
    roles = binding_or_roles.get("roles", binding_or_roles) if isinstance(binding_or_roles, dict) else {}
    out = {}
    for role, s in sorted((roles or {}).items()):
        if not isinstance(s, dict):
            continue
        out[role] = {f: s.get(f) for f in SIGNATURE_FIELDS.get(role, ("model",))}
    return out


def subprocess_env(binding):
    """The environment variables that hand `binding` to a child process (run_batch). The caller drops every
    JUDGE_VARS entry first, so nothing inherited from the shell can stand in for the explicit binding."""
    out = {}
    for s in (binding.get("roles") or {}).values():
        for k, v in (s.get("config") or {}).items():
            out[k] = str(v)
    return out


def _usage(result):
    """What the result says was actually called, per role: checkpoints that carry a judge block, and the
    kernel's own count of semantic judge calls."""
    use = {}
    for c in (result.get("checkpoints") or []):
        j = c.get("judge") if isinstance(c, dict) else None
        if not isinstance(j, dict) or not j.get("role"):
            continue
        u = use.setdefault(j["role"], {"checkpoints": 0, "calls": 0, "errors": 0, "models": []})
        u["checkpoints"] += 1
        u["calls"] += int(j.get("calls") or 0)
        u["errors"] += 1 if j.get("error") else 0
        if j.get("model") and j["model"] not in u["models"]:
            u["models"].append(j["model"])
    audit = ((result.get("harness") or {}).get("audit") or {}) if isinstance(result.get("harness"), dict) else {}
    if audit:
        use["harness"] = {"calls": audit.get("runtime_judge_calls"), "models": [audit.get("runtime_judge_model")]
                          if audit.get("runtime_judge_model") else []}
    return use


def run_record(binding, result):
    """The block written to result.json run_config.judges: the binding the run was refused-or-allowed on, plus
    what the result itself shows was used."""
    roles = {}
    for role, s in sorted((binding.get("roles") or {}).items()):
        roles[role] = {k: s.get(k) for k in ("model", "runs", "max_tokens", "temperature") if k in s}
        roles[role]["source"] = sorted((s.get("config") or {}).keys())
    return {"bench": binding.get("bench"), "roles": roles, "signature": signature(binding),
            "usage": _usage(result or {})}


# ------------------------------------------------------------------------------------------------------------
# Reading a STORED result: what does it record about its judges? Used for stored results without a run_config block.
# Only what is stored counts. A field the result does not record is `None` and is listed in `unrecorded`; it is
# never filled from today's defaults, because that would make an unrecorded control look compatible.
# ------------------------------------------------------------------------------------------------------------

def record_from_result(result, bench):
    """-> {"source": "run_config"|"stored_fields", "roles": {role: {field: value|None}}, "unrecorded": [...]}"""
    rc = (result.get("run_config") or {}) if isinstance(result, dict) else {}
    jr = rc.get("judges") if isinstance(rc, dict) else None
    if isinstance(jr, dict) and isinstance(jr.get("signature"), dict):
        roles = {r: dict(v) for r, v in jr["signature"].items()}
        source = "run_config"
    else:
        roles = _derive_roles(result or {}, bench)
        source = "stored_fields"
    unrecorded = sorted("%s.%s" % (r, f) for r in roles_for(bench)
                        for f in SIGNATURE_FIELDS.get(r, ("model",))
                        if (roles.get(r) or {}).get(f) is None)
    return {"source": source, "roles": roles, "unrecorded": unrecorded}


def _one(values):
    vals = sorted({v for v in values if v is not None}, key=str)
    if not vals:
        return None
    return vals[0] if len(vals) == 1 else {"mixed": vals}


def _derive_roles(result, bench):
    audit = ((result.get("harness") or {}).get("audit") or {}) if isinstance(result.get("harness"), dict) else {}
    snap = audit.get("run_config") if isinstance(audit.get("run_config"), dict) else {}
    roles = {"harness": {"model": audit.get("runtime_judge_model") or snap.get("harness_judge_model"),
                         "max_tokens": snap.get("harness_judge_max_tokens")}}
    cps = [c for c in (result.get("checkpoints") or []) if isinstance(c, dict)]
    if bench == "PhysicianBench":
        roles["pb_content"] = {"model": _one((c.get("judge") or {}).get("model") for c in cps
                                             if (c.get("judge") or {}).get("role") == "pb_content")}
    elif bench == "HealthAdminBench":
        rub = [c for c in cps if c.get("evaluator_kind") == "gateway_rubric_text" and c.get("judge_backend")]
        roles["hab_rubric"] = {"model": _one(c.get("judge_backend") for c in rub),
                               "runs": _one((c.get("detail") or {}).get("num_runs") for c in rub)}
    elif bench == "MedCTA":
        roles["medcta_gacc"] = {"model": _one(c.get("judge_backend") for c in cps
                                              if c.get("evaluator_kind") == "gacc_judge")}
        roles["medcta_checkpoint"] = {"model": _one(c.get("judge_backend") for c in cps if c.get("evaluator_kind")
                                                    in ("verification_judge", "governance_4rule"))}
    return roles


def arm_record(records):
    """Fold per-task records into one arm-level record: each field's set of recorded values. A field no task
    recorded is unrecorded; a field with two values is mixed. -> {"roles": {role: {field: value|{"mixed": [...]}
    |None}}, "unrecorded": [...], "mixed": [...], "n_tasks": n, "sources": {...}}"""
    fields = {}
    sources = {}
    for rec in records:
        sources[rec.get("source")] = sources.get(rec.get("source"), 0) + 1
        for role, fv in (rec.get("roles") or {}).items():
            for f, v in (fv or {}).items():
                bucket = fields.setdefault((role, f), [])
                if isinstance(v, dict) and "mixed" in v:
                    bucket.extend(v["mixed"])
                else:
                    bucket.append(v)
    roles, unrecorded, mixed = {}, [], []
    for (role, f), vals in sorted(fields.items()):
        v = _one(vals)
        roles.setdefault(role, {})[f] = v
        if v is None:
            unrecorded.append("%s.%s" % (role, f))
        elif isinstance(v, dict):
            mixed.append("%s.%s" % (role, f))
    return {"roles": roles, "unrecorded": unrecorded, "mixed": mixed, "n_tasks": len(records), "sources": sources}


def compatibility(binding, control_arm_record, bench):
    """May a run judged by `binding` (the "candidate") be compared against a control arm whose stored results record
    `control_arm_record`? Compatible only when every signature field the benchmark needs is recorded by the
    control, with exactly one value, equal to the candidate's. -> {"compatible": bool, "problems": [...]}"""
    want = signature(binding)
    have = (control_arm_record or {}).get("roles") or {}
    problems = []
    for role in roles_for(bench):
        for f in SIGNATURE_FIELDS.get(role, ("model",)):
            c = (want.get(role) or {}).get(f)
            k = (have.get(role) or {}).get(f)
            if k is None:
                problems.append("%s.%s: the control does not record it (candidate %r)" % (role, f, c))
            elif isinstance(k, dict):
                problems.append("%s.%s: the control recorded several values %r (candidate %r)"
                                % (role, f, k.get("mixed"), c))
            elif c != k:
                problems.append("%s.%s: control %r != candidate %r" % (role, f, k, c))
    return {"compatible": not problems, "problems": problems, "candidate": want, "control": have}
