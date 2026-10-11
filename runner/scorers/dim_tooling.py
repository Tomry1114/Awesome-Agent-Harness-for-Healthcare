#!/usr/bin/env python3
"""
dim_tooling.py - benchmark-AGNOSTIC Clinical Resource Competence dimension evaluator.

Splits the Clinical Resource Competence dimension into a generic ToolingCore and a plugin-resolved
ArgumentSemanticEvaluator hook, so that no benchmark-coupled argument or
tool-path check lives in the core.

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
``argument_semantic_score``). If no hook is registered the axis is not_applicable.

Entry function: ``tooling(sem_trace, dimension_policy, manifest, ...)``.
"""
import substrate

EVALUATOR_VERSION = "tooling-core-1"

# system / env-injected argument keys an agent never supplies itself (generic).
_SYSTEM_ARG_KEYS = {"image", "image_path", "img", "image_url", "session", "session_id",
                    "_meta", "_raw", "context_id"}


# --------------------------------------------------------------------------- helpers
try:                                                   # flat sys.path (runner/scorers on the path)
    from dimension_core import submetric as _sm, aggregate_applicable as _aggregate
except ImportError:                                    # pragma: no cover - packaged import
    from .dimension_core import submetric as _sm, aggregate_applicable as _aggregate


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
# Benchmark-specific argument SEMANTICS are resolved FROM the plugin ONLY (no central registry): a
# plugin exposes a callable under the dict key 'argument_semantic_score'. A 4th dataset needs only a
# plugin file (plugins/<x>.py) with that key -- dim_tooling.py never changes.
def resolve_argument_semantic_hook(plugin, policy=None):
    """Return the plugin-supplied argument-semantic hook, or None. Plugin dict key ONLY."""
    if isinstance(plugin, dict) and callable(plugin.get("argument_semantic_score")):
        return plugin["argument_semantic_score"]
    return None


# --------------------------------------------------------------------------- ToolingCore
def tooling(sem_trace, dimension_policy=None, manifest=None, available_tools=None,
            task=None, plugin=None, actor_scope="agent"):
    """Benchmark-agnostic Clinical Resource Competence score.

    sem_trace        : list[SemanticEvent]  (from substrate.map_trace)
    dimension_policy : DimensionPolicy dict  (substrate.dimension_policy) - provides required_milestones
                       and required_tool_groups (acceptable acquire-paths, as milestone bundles).
    manifest         : CapabilityManifest    (substrate.capability_manifest)
    available_tools  : task.available_tools roster (names/signatures) - used for authorization fallback
                       and the GENERIC argument-shape oracle only.
    task / plugin    : passed THROUGH to the resolved plugin argument-semantic hook (opaque to core).
    """
    # ACTOR-AWARE: the agent-process Clinical Resource Competence profile scores the AGENT tool use; recovery tool calls
    # are a separate intervention profile (their authorization/params are audited there, not as agent
    # waste/path-incompletion here). Pass actor_scope!="agent" for end-to-end/recovery profiles.
    if actor_scope == "agent" and sem_trace:
        _f = [s for s in sem_trace if s.get("actor", "agent") != "harness"]
        if _f:
            sem_trace = _f

    policy = dimension_policy or {}
    calls = [s for s in sem_trace if _is_tool_call(s)]
    reached = substrate.milestones_reached(sem_trace)
    sub = {}

    # ---- path_completion: best acceptable acquire-path, measured by MILESTONES only --------------
    # An acceptable path is a bundle of required milestones; we credit the best-covered bundle.
    # When the task contract supplies required_milestone_groups, use THEM as the ONLY acceptable
    # paths (the conjunction the task actually requires). Do NOT also add the per-tool-group single-element
    # paths -- those would let a single write_file satisfy a 1-element path -> path_completion=1.0 while every OTHER
    # required tool was skipped. Fall back to the tool-group mapping ONLY when no contract group exists.
    paths = []
    contract_groups = policy.get("required_milestone_groups")
    if contract_groups:
        paths = [set(g) for g in contract_groups if g]
    else:
        rtg = policy.get("required_tool_groups")
        tool_sem = (plugin or {}).get("tool_semantics", {}) if isinstance(plugin, dict) else {}
        if rtg:
            for g in rtg:
                ms = set()
                for t in g:
                    ms.update((tool_sem.get(t) or {}).get("success_milestones") or [])
                if ms:
                    paths.append(ms)
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

    # ---- plugin-resolved argument SEMANTICS -- folded INTO the Clinical Resource Competence mean. Computed into `sub`
    # BEFORE _aggregate so the strict 'were the tool ARGS correct for the task' axis enters the headline
    # Clinical Resource Competence score rather than being reported beside it. Resolved ONLY from
    # plugin['argument_semantic_score']; no benchmark literal reaches core.
    hook = resolve_argument_semantic_hook(plugin, policy)
    _arg_per_call = []
    if hook is not None:
        for s2 in calls:
            try:
                r = hook(s2.get("raw") or {}, task, policy, s2)
            except TypeError:
                try:
                    r = hook(s2.get("raw") or {}, task)
                except Exception as e:
                    r = {"status": "error", "note": repr(e)}
            except Exception as e:
                r = {"status": "error", "note": repr(e)}
            if isinstance(r, dict) and r.get("status") not in (None, "not_applicable"):
                _arg_per_call.append(r)
    _arg_scored = [r for r in _arg_per_call if isinstance(r.get("score"), (int, float))]
    _arg_errors = [r for r in _arg_per_call if r.get("status") == "error"]
    if _arg_scored:
        sub["argument_semantic"] = _sm(round(sum(r["score"] for r in _arg_scored) / len(_arg_scored), 3),
                                       opportunities=len(_arg_scored), per_call=_arg_per_call)
    elif hook is not None and _arg_errors:
        # Fail-closed: the strict hook IS registered but errored on every call -> surface it loudly as
        # status=error (excluded from the mean AND flagged) instead of silently reverting to generic-only.
        sub["argument_semantic"] = _sm(None, "error", len(_arg_errors),
                                       reason="argument_semantic_hook_errored",
                                       errors=[e.get("note") for e in _arg_errors[:3]])
    else:
        sub["argument_semantic"] = _sm(None, "not_applicable", 0,
                                       reason=("no_plugin_hook" if hook is None else "hook_returned_no_scorable_calls"))

    # ---- tool_invocation_success: scored in Clinical Resource Competence, read from dim_execution's computation -------
    # Reading rather than reimplementing: the attribution rules (agent-owned failures count, environment /
    # external_service / harness are excluded, 'unknown' is evidence-insufficient, a malformed action is not
    # an invocation) live in execution() and must have exactly one definition.
    try:
        from dim_execution import execution as _exec_dim
    except Exception:
        try:
            from runner.scorers.dim_execution import execution as _exec_dim  # type: ignore
        except Exception:
            _exec_dim = None
    _tis = None
    if _exec_dim is not None:
        try:
            _tis = ((_exec_dim(sem_trace, dimension_policy, manifest).get("submetrics") or {})
                    .get("tool_invocation_success"))
        except Exception:
            _tis = None
    sub["tool_invocation_success"] = dict(_tis) if isinstance(_tis, dict) else _sm(
        None, "not_applicable", 0, reason="execution() did not yield tool_invocation_success")

    out = _aggregate(sub)
    out["argument_semantic"] = sub["argument_semantic"]
    out["strict_arg_axis_error"] = (sub["argument_semantic"].get("status") == "error")

    out["authorization_source"] = auth_src
    out["reportable"] = out["n_applicable"] > 0
    out["coverage"] = {"n_tool_calls": len(calls),
                       "applicable_submetrics": out["applicable_submetrics"],
                       "n_applicable": out["n_applicable"]}
    out["tier"] = "experimental"
    out["evaluator_version"] = EVALUATOR_VERSION
    return out


# --------------------------------------------------------------------------- The MedCTA argument-semantic
# evaluator lives in runner/plugins/medcta.py, registered via the plugin dict key
# 'argument_semantic_score'. Core does not hardcode or auto-register any benchmark hook.


# --------------------------------------------------------------------------- self-check
def _selfcheck():
    import os, json, glob
    def _load(d):
        traj = [json.loads(l) for l in open(os.path.join(d, "trajectory.jsonl")) if l.strip()]
        task = json.load(open(os.path.join(d, "task.json")))
        res = json.load(open(os.path.join(d, "result.json")))
        prov = res.get("provenance") or {}
        return traj, task, prov

    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scorers/ -> runner/ -> repo root
    bundles = [("MedCTA", os.path.join(here, "examples/medcta/MCTA-0"))]
    pb = sorted(glob.glob(os.path.join(here, "examples/pb/PB-*")))
    if pb:
        bundles.append(("PhysicianBench", pb[0]))
    hab = os.path.join(here, "examples/hab/HAB-denial-hard-11")
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
