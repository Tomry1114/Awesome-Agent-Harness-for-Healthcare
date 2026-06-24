#!/usr/bin/env python3
"""dim_tooling.py - benchmark-AGNOSTIC Tooling dimension evaluator.

Splits the Tooling dimension into a generic ToolingCore and a plugin-resolved
ArgumentSemanticEvaluator hook, superseding the MedCTA-coupled `arg_match` /
`tool_path` deterministic checks in runner/scoring.py (the ``_ev_deterministic``
arg_accuracy_3axis + tool_path branches, ~lines 206-289, plus their helpers
``_localization_status`` / ``_arg_semantic_judge``).

ToolingCore consumes ONLY substrate structures and never names a benchmark, a
tool literal (OCR / fhir_* / click / RegionAttribute), an image, a FHIR resource,
or a DOM selector. Four core sub-metrics, all applicable-only (a sub-metric with
no opportunity returns status=not_applicable and is EXCLUDED from the mean):

  path_completion          best of the acceptable acquire-paths, measured purely
                           by milestones produced on the SemanticTrace.
  tool_authorization       invoked tools  subset of  authorized/available tools
                           (CapabilityManifest.authorized, else available_tools).
  argument_schema_validity generic shape check: every invoked tool that declares
                           a non-trivial signature actually carried (typed) args.
  unnecessary_tool_use     fraction of tool calls that advanced no milestone and
                           no progress token and changed no state (pure waste).

The benchmark-SPECIFIC argument semantics (MedCTA bbox/region/localization
relevance, FHIR arg correctness, DOM ref validity, ...) live behind the plugin
hook ``argument_semantic_score(call, task) -> {score, status, ...}``. Core never
hardcodes them: it resolves the hook FROM the plugin (substrate plugin dict key
``argument_semantic_score`` or this module's hook registry keyed by the plugin's
governance_policy_id). If no hook is registered the axis is not_applicable.

Entry function: ``tooling(sem_trace, dimension_policy, manifest, ...)``.
"""
import substrate

EVALUATOR_VERSION = "tooling-core-1"

# system / env-injected argument keys an agent never supplies itself (generic).
_SYSTEM_ARG_KEYS = {"image", "image_path", "img", "image_url", "session", "session_id",
                    "_meta", "_raw", "context_id"}


# --------------------------------------------------------------------------- helpers
def _sm(score, status="valid", opportunities=None, **kw):
    """applicable-only sub-metric record (mirrors lifecycle_exec._sm)."""
    try:
        from lifecycle_exec import _sm as _le_sm
        return _le_sm(score, status, opportunities, **kw)
    except Exception:
        d = {"score": score, "status": status}
        if opportunities is not None:
            d["opportunities"] = opportunities
        d.update(kw)
        return d


def _aggregate(subs):
    """Average ONLY applicable (status=valid, numeric) sub-metrics. Never vacuous-1.0."""
    try:
        from lifecycle_exec import _aggregate as _le_agg
        return _le_agg(subs)
    except Exception:
        valid = {k: v for k, v in subs.items()
                 if v.get("status") == "valid" and isinstance(v.get("score"), (int, float))}
        vals = [v["score"] for v in valid.values()]
        score = round(sum(vals) / len(vals), 3) if vals else None
        return {"score": score, "submetrics": subs, "applicable_submetrics": sorted(valid),
                "n_applicable": len(valid), "zero_variance": (len(set(vals)) == 1) if vals else None}


def _is_tool_call(sem):
    """A SemanticEvent that originated from a raw tool_call (role acquire/act/verify/commit)."""
    raw = sem.get("raw") or {}
    return raw.get("event_type") == "tool_call"


def _tool_name(sem):
    return (sem.get("raw") or {}).get("tool")


def _norm_args(raw_call):
    """Tool-call args -> dict, tolerating a JSON string (mirrors scoring.parse_args)."""
    a = (raw_call or {}).get("args")
    if isinstance(a, dict):
        return a
    if isinstance(a, str):
        import json
        try:
            j = json.loads(a)
            return j if isinstance(j, dict) else {"_raw": j}
        except Exception:
            return {"_raw": a}
    return {} if a is None else {"_raw": a}


def _authorized_set(manifest, available_tools):
    """Tools the agent was permitted to use: CapabilityManifest.authorized (four-state) UNION the
    task-declared available_tools roster. Returns (auth_set, source) or (None, ...) when unknown."""
    auth = set()
    src = []
    for name, cap in (manifest or {}).items():
        if isinstance(cap, dict) and cap.get("authorized") is not False and cap.get("available") is not False:
            auth.add(name)
    if auth:
        src.append("capability_manifest")
    roster = set()
    for t in (available_tools or []):
        if isinstance(t, dict) and t.get("name"):
            roster.add(t["name"])
        elif isinstance(t, str):
            roster.add(t)
    if roster:
        auth |= roster
        src.append("available_tools")
    return (auth if auth else None), "+".join(src) if src else "none"


def _unauthorized_set(manifest, available_tools):
    """Tools the manifest EXPLICITLY marks not-authorized (authorized is False) -> hard penalty set."""
    bad = set()
    for name, cap in (manifest or {}).items():
        if isinstance(cap, dict) and cap.get("authorized") is False:
            bad.add(name)
    return bad


def _tool_signatures(available_tools):
    """name -> signature string, used only as a GENERIC shape oracle (does it take args at all?)."""
    sig = {}
    for t in (available_tools or []):
        if isinstance(t, dict) and t.get("name"):
            sig[t["name"]] = str(t.get("signature") or "")
    return sig


def _signature_takes_args(sig):
    """Heuristic: signature '(a,b)->c' declares >=1 input param. Empty '()' -> no required args.
    Generic (no tool literal): inspects the parenthesised input list of the declared signature."""
    if not sig:
        return None  # unknown -> cannot judge
    head = sig.split("->", 1)[0]
    if "(" in head and ")" in head:
        inside = head[head.find("(") + 1:head.rfind(")")].strip()
        return bool(inside)
    return None


# --------------------------------------------------------------------------- plugin hook (arg semantics)
# Benchmark-specific argument SEMANTICS are resolved FROM the plugin, never hardcoded in core.
# A plugin may expose a callable under the dict key 'argument_semantic_score', OR a hook may be
# registered here against the plugin's governance_policy_id. Core only ever CALLS the resolved hook.
_ARG_SEMANTIC_HOOKS = {}


def register_argument_semantic_hook(governance_policy_id, fn):
    """Bind an argument_semantic_score(call, task)->dict hook to a governance policy id."""
    _ARG_SEMANTIC_HOOKS[governance_policy_id] = fn


def resolve_argument_semantic_hook(plugin, policy):
    """Return the plugin-supplied argument-semantic hook, or None. Resolution order:
    (1) plugin dict key 'argument_semantic_score'; (2) registry by governance_policy_id."""
    if isinstance(plugin, dict) and callable(plugin.get("argument_semantic_score")):
        return plugin["argument_semantic_score"]
    gid = (policy or {}).get("governance_policy_id") or (isinstance(plugin, dict) and plugin.get("benchmark"))
    return _ARG_SEMANTIC_HOOKS.get(gid)


# --------------------------------------------------------------------------- ToolingCore
def tooling(sem_trace, dimension_policy=None, manifest=None, available_tools=None,
            task=None, plugin=None):
    """Benchmark-agnostic Tooling score.

    sem_trace        : list[SemanticEvent]  (from substrate.map_trace)
    dimension_policy : DimensionPolicy dict  (substrate.dimension_policy) - provides required_milestones
                       and required_tool_groups (acceptable acquire-paths, as milestone bundles).
    manifest         : CapabilityManifest    (substrate.capability_manifest)
    available_tools  : task.available_tools roster (names/signatures) - used for authorization fallback
                       and the GENERIC argument-shape oracle only.
    task / plugin    : passed THROUGH to the resolved plugin argument-semantic hook (opaque to core).
    """
    policy = dimension_policy or {}
    calls = [s for s in sem_trace if _is_tool_call(s)]
    reached = substrate.milestones_reached(sem_trace)
    sub = {}

    # ---- path_completion: best acceptable acquire-path, measured by MILESTONES only --------------
    # An acceptable path is a bundle of required milestones; we credit the best-covered bundle.
    paths = []
    rtg = policy.get("required_tool_groups")
    # required_tool_groups are tool names; map each to the milestones the plugin says they produce,
    # so the metric stays tool-literal-free in its SCORING (the mapping is plugin knowledge).
    tool_sem = (plugin or {}).get("tool_semantics", {}) if isinstance(plugin, dict) else {}
    if rtg:
        for g in rtg:
            ms = set()
            for t in g:
                ms.update((tool_sem.get(t) or {}).get("success_milestones") or [])
            if ms:
                paths.append(ms)
    # fall back to the flat required_milestones bundle as a single acceptable path.
    req_ms = set(policy.get("required_milestones") or [])
    if req_ms:
        paths.append(req_ms)
    if paths:
        best = max(len(p & reached) / len(p) for p in paths)
        sub["path_completion"] = _sm(round(best, 3), opportunities=len(paths))
    else:
        sub["path_completion"] = _sm(None, "not_applicable", 0)

    # ---- tool_authorization: invoked tools subset of authorized/available -----------------------
    invoked = {_tool_name(s) for s in calls if _tool_name(s)}
    auth_set, auth_src = _authorized_set(manifest, available_tools)
    explicit_bad = _unauthorized_set(manifest, available_tools)
    if invoked and (auth_set is not None or explicit_bad):
        unauthorized = (invoked & explicit_bad) | (invoked - auth_set if auth_set is not None else set())
        n = len(invoked)
        sub["tool_authorization"] = _sm(round((n - len(unauthorized)) / n, 3), opportunities=n,
                                        unauthorized=sorted(unauthorized), source=auth_src)
    else:
        sub["tool_authorization"] = _sm(None, "not_applicable", 0,
                                        reason="no_invocations" if not invoked else "no_authorization_signal")

    # ---- argument_schema_validity: generic shape check on invoked tools -------------------------
    sigs = _tool_signatures(available_tools)
    judged = 0
    ok_args = 0
    for s in calls:
        name = _tool_name(s)
        takes = _signature_takes_args(sigs.get(name))
        if takes is None:
            continue  # unknown signature -> no generic basis to judge; skip (not penalised)
        judged += 1
        args = _norm_args(s.get("raw"))
        nonsys = {k: v for k, v in args.items()
                  if k not in _SYSTEM_ARG_KEYS and v not in (None, "", [], {}, ())}
        if takes:
            ok_args += 1 if nonsys else 0          # tool needs args -> at least one typed non-system arg
        else:
            ok_args += 1                            # no-arg tool -> always shape-valid
    if judged:
        sub["argument_schema_validity"] = _sm(round(ok_args / judged, 3), opportunities=judged)
    else:
        sub["argument_schema_validity"] = _sm(None, "not_applicable", 0, reason="no_typed_signatures")

    # ---- unnecessary_tool_use: calls that advanced NOTHING (no milestone/progress/state) --------
    if calls:
        seen_progress = set()
        wasted = 0
        for s in calls:
            added = set(s.get("milestones_added") or [])
            tok = s.get("progress_token")
            new_progress = (tok is not None) and (tok not in seen_progress)
            if tok is not None:
                seen_progress.add(tok)
            advanced = bool(added) or new_progress or bool(s.get("state_changed"))
            if not advanced:
                wasted += 1
        n = len(calls)
        # report as fraction that WAS necessary (higher = better, consistent with other sub-metrics)
        sub["unnecessary_tool_use"] = _sm(round((n - wasted) / n, 3), opportunities=n, wasted_calls=wasted)
    else:
        sub["unnecessary_tool_use"] = _sm(None, "not_applicable", 0)

    out = _aggregate(sub)

    # ---- plugin-resolved argument SEMANTICS (reported, NOT folded into the core mean) -----------
    hook = resolve_argument_semantic_hook(plugin, policy)
    if hook is not None:
        per_call = []
        for s in calls:
            try:
                r = hook(s.get("raw") or {}, task)
            except Exception as e:
                r = {"status": "error", "note": repr(e)}
            if isinstance(r, dict) and r.get("status") not in (None, "not_applicable"):
                per_call.append(r)
        scored = [r for r in per_call if isinstance(r.get("score"), (int, float))]
        if scored:
            out["argument_semantic"] = {
                "status": "valid", "n": len(scored),
                "score": round(sum(r["score"] for r in scored) / len(scored), 3),
                "source": "plugin_hook", "per_call": per_call}
        else:
            out["argument_semantic"] = {"status": "not_applicable", "source": "plugin_hook",
                                        "reason": "hook_returned_no_scorable_calls"}
    else:
        out["argument_semantic"] = {"status": "not_applicable", "reason": "no_plugin_hook",
                                    "note": "benchmark argument semantics not registered for this plugin"}

    out["authorization_source"] = auth_src
    out["reportable"] = out["n_applicable"] > 0
    out["coverage"] = {"n_tool_calls": len(calls),
                       "applicable_submetrics": out["applicable_submetrics"],
                       "n_applicable": out["n_applicable"]}
    out["tier"] = "experimental"
    out["evaluator_version"] = EVALUATOR_VERSION
    return out


# --------------------------------------------------------------------------- plugin hook impl: MedCTA-style
# This argument-SEMANTIC evaluator carries the MedCTA-specific bbox/region/localization knowledge that
# used to be hardcoded in scoring.py (_localization_status). It is NOT part of ToolingCore: it is
# registered as a plugin hook below, so core resolves it indirectly and never references its literals.
class ArgumentSemanticEvaluator:
    """Plugin hook: score the SEMANTIC adequacy of a single tool call's localization arguments.

    A region/localization tool earns 1.0 when the call EITHER resolved a region (explicit
    result.localization.resolved) OR carried a non-empty semantic region argument; it scores 0.0 when
    it silently fell back (resolved is False). Calls that are not region-localization are not_applicable.
    """
    REGION_ARG_KEYS = ("region", "region_query", "bbox")
    LOC_RESULT_KEYS = ("localization",)

    def _localization(self, raw):
        r = raw.get("result")
        if isinstance(r, dict):
            o = r.get("output")
            if isinstance(o, dict) and isinstance(o.get("localization"), dict):
                return o["localization"]
            if isinstance(r.get("localization"), dict):
                return r["localization"]
        return None

    def score(self, call, task=None):
        args = call.get("args")
        if isinstance(args, str):
            import json
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        args = args if isinstance(args, dict) else {}
        region = next((args.get(k) for k in self.REGION_ARG_KEYS if args.get(k)), None)
        loc = self._localization(call)
        if region is None and loc is None:
            return {"status": "not_applicable", "reason": "not_a_localization_call"}
        if isinstance(loc, dict) and "resolved" in loc:
            ok = bool(loc.get("resolved"))
            return {"status": "valid", "score": 1.0 if ok else 0.0, "axis": "localization_resolved",
                    "region": str(region)[:120]}
        # no explicit localization signal -> credit a non-empty semantic region request.
        return {"status": "valid", "score": 1.0 if region else 0.0, "axis": "semantic_region_present",
                "region": str(region)[:120]}


# register the MedCTA-flavoured hook against MedCTA's governance policy id WITHOUT core referencing it.
def _register_default_hooks():
    ev = ArgumentSemanticEvaluator()
    p = substrate.get_plugin("MedCTA")
    gid = (p or {}).get("dimension_policy", {}).get("governance_policy_id") or (p or {}).get("benchmark")
    if gid:
        register_argument_semantic_hook(gid, lambda call, task: ev.score(call, task))


_register_default_hooks()


# --------------------------------------------------------------------------- self-check
def _selfcheck():
    import os, json, glob
    def _load(d):
        traj = [json.loads(l) for l in open(os.path.join(d, "trajectory.jsonl")) if l.strip()]
        task = json.load(open(os.path.join(d, "task.json")))
        res = json.load(open(os.path.join(d, "result.json")))
        prov = res.get("provenance") or {}
        return traj, task, prov

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bundles = [("MedCTA", os.path.join(here, "results_mctaGov/gpt5/MCTA-0"))]
    pb = sorted(glob.glob(os.path.join(here, "results_pb_chk3/gpt5/PB-*")))
    if pb:
        bundles.append(("PhysicianBench", pb[0]))
    hab = os.path.join(here, "results_hab10/gpt5/HAB-denial-easy-1")
    if os.path.isdir(hab):
        bundles.append(("HealthAdminBench", hab))

    for bench, d in bundles:
        traj, task, prov = _load(d)
        plugin = substrate.get_plugin(task.get("source_benchmark") or bench)
        sem = substrate.map_trace(traj, plugin)
        policy = substrate.dimension_policy(task, plugin)
        manifest = substrate.capability_manifest(prov)
        out = tooling(sem, policy, manifest, available_tools=task.get("available_tools"),
                      task=task, plugin=plugin)
        print("==== %s  (%s)" % (bench, os.path.basename(d)))
        print("  score:", out["score"], "| reportable:", out["reportable"],
              "| n_applicable:", out["n_applicable"], "| auth_source:", out["authorization_source"])
        for k, v in out["submetrics"].items():
            print("    %-26s score=%-6s status=%-15s %s" % (
                k, v.get("score"), v.get("status"),
                {kk: vv for kk, vv in v.items() if kk not in ("score", "status")}))
        asem = out.get("argument_semantic", {})
        print("    argument_semantic(plugin hook):", asem.get("status"),
              "score=", asem.get("score"), "n=", asem.get("n"))


if __name__ == "__main__":
    _selfcheck()
