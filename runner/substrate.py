#!/usr/bin/env python3
"""Universal substrate for benchmark-AGNOSTIC dimension scoring.

The 7 dimension evaluators are meant to consume ONLY these structures and NEVER a benchmark name, tool
name, image, FHIR resource, or DOM selector:
  CanonicalTask     - the unified task dict (spec/task.schema.json)
  SemanticTrace     - list[SemanticEvent]  (event_role / milestones / progress / failure_attribution)
  CapabilityManifest- four-state {implemented, available, authorized, healthy} per capability
  DimensionPolicy   - per-dimension declarative policy (required_milestones / required_context_units /
                      lifecycle_policy / governance_policy_id)
  EvidenceView      - list[EvidenceUnit] (id / delivered / fidelity / error_visible / acknowledged)

A BenchmarkPlugin supplies the ONLY benchmark-specific knowledge: how a tool maps to a semantic role and
which milestones it produces, how to extract evidence units, the dimension policy, and the source outcome
evaluator. A 4th dataset registers a plugin; the dimension evaluators never change.

SemanticEvent v2 CONTRACT (every tool_call event carries these; consumers read via .get so adding fields
stays back-compatible):
  capability_id : the invoked tool/capability id. SET ON EVERY tool_call INCLUDING failures (Execution /
                  attribution joins it to the manifest healthy/authorized state). Never dropped.
  obligation_id : the PRIMARY milestone this tool is declared to produce (binds a FAILURE to its RECOVERY:
                  a failed obligation is recovered ONLY by a later event producing the SAME obligation_id,
                  i.e. the same tool retried or an alt tool in the same required-tool group -- not by "any
                  later progress"). None when the tool serves no declared milestone.
  progress_token: a SEMANTIC token of the RESULT, never the tool name
                  (evidence:<source>:<hash8> / state:<k>=<v> / resource:<type>/<id>:created /
                  region:<id>:resolved). Derived from the rendered/source CONTENT so OCR(page1) and
                  OCR(page2) differ; a repeated identical call repeats its token (no progress). None when
                  no new evidence/state.
  status        : success | partial | failure  (partial = ran without error but did NOT meet its semantic
                  goal -> milestone withheld).
  state_changed : True ONLY if this event yielded a NEW milestone OR a NEW progress_token."""
import re, json, hashlib

# ----------------------------------------------------------------------------- SemanticEvent
ROLES = ("acquire", "act", "verify", "commit", "final", "escalate", "other")
FAILURE_ATTRIB = ("agent", "environment", "external_service", "harness", "unknown", None)
STATUSES = ("success", "partial", "failure")

def semantic_event(role, status="success", capability_id=None, obligation_id=None, progress_token=None,
                   milestones_added=None, state_changed=False, terminal=None, failure_attribution=None,
                   raw=None, action_valid=True):
    """v2 SemanticEvent. capability_id/obligation_id default None for non-tool events (final/escalate);
    map_trace sets them on every tool_call. Consumers read every field via .get(), so unset fields are
    simply absent-valued, never schema-breaking.

    action_valid : PURE protocol/schema validity of the action that produced this event — True when the
    agent emitted a WELL-FORMED action (a usable tool_call/final/control/etc.), False ONLY for a
    MALFORMED/unparseable action (run.py's agent_error markers: invalid_action / bad_action_type /
    truncated_tool_call). It is INDEPENDENT of execution outcome: a well-formed tool_call that RAN and
    failed is still action_valid=True (the failure is tool_invocation_success's concern, not validity).
    Defaults True so legacy/hand-built events (no malformed marker) are well-formed unless set otherwise."""
    assert role in ROLES, "unknown role %r" % role
    assert status in STATUSES, "unknown status %r" % status
    return {"event_role": role, "status": status, "capability_id": capability_id,
            "obligation_id": obligation_id, "progress_token": progress_token,
            "milestones_added": list(milestones_added or []), "state_changed": bool(state_changed),
            "terminal": terminal, "failure_attribution": failure_attribution, "raw": raw,
            "action_valid": bool(action_valid)}

# ----------------------------------------------------------------------------- BenchmarkPlugin registry
_PLUGINS = {}
def register_plugin(p): _PLUGINS[p["benchmark"]] = p
def get_plugin(name):
    """FAIL-CLOSED: returns None for an unknown/unregistered benchmark (no silent default plugin). Callers
    that must score MUST go through require_plugin() so a missing plugin flips score_eligible off rather
    than being scored against an empty/default policy."""
    return _PLUGINS.get(name)
def list_plugins(): return sorted(_PLUGINS)

def require_plugin(name):
    """Resolve a plugin or FLAG the gap. Returns (plugin, problem):
       - (plugin, None)                          when registered,
       - (None, 'missing_benchmark_plugin:<n>')  when not -- the caller sets score_eligible=False instead
         of scoring against a vacuous default. Never raises silently-defaulting state."""
    p = _PLUGINS.get(name)
    if p is None:
        return None, "missing_benchmark_plugin:%s" % name
    return p, None

# ----------------------------------------------------------------------------- helpers
def _errored(e):
    try:
        import proxy_verifiers as _pv
        return _pv._errored(e)
    except Exception:
        return str(e.get("status", "")).lower() == "error" or bool(e.get("error") or e.get("error_type"))

def _attr(e):
    try:
        import lifecycle_exec as _le
        return _le.error_attribution(e)
    except Exception:
        return "unknown"

def _hash8(s):
    return hashlib.sha1((s or "").encode("utf-8", "replace")).hexdigest()[:8]

def _result_output(e):
    """The raw tool OUTPUT object. Most MedCTA tools nest it under result.output; PB/HAB put the resource /
    page dict directly at result. Returns whatever the tool produced (dict|str|None)."""
    r = e.get("result")
    if isinstance(r, dict) and "output" in r:
        return r.get("output")
    return r

def _no_payload(e):
    """A degenerate event with NO observable output (no result, no observation, no canonical_observation):
    the resolver has nothing to read, so it cannot prove the semantic goal was UNMET. Real traces always
    carry a payload; only synthetic stubs are empty. In that case the mapper falls back to the static
    CONSERVATIVE partial (milestone withheld) instead of an unverifiable success -- a missing payload
    must never fail-open into a granted milestone."""
    if e.get("result") not in (None, "", {}):
        return False
    ob = e.get("observation")
    if ob not in (None, "", {}):
        return False
    co = e.get("canonical_observation")
    if isinstance(co, dict) and (co.get("modalities") or {}).get("text"):
        return False
    return True

def _primary_obligation(meta):
    """The obligation_id a tool is declared responsible for = its PRIMARY success milestone (explicit
    'obligation' override, else the first declared success_milestone). None when the tool earns none."""
    if not isinstance(meta, dict):
        return None
    if meta.get("obligation"):
        return meta.get("obligation")
    ms = meta.get("success_milestones") or []
    return ms[0] if ms else None

# ----------------------------------------------------------------------------- SemanticEventMapper
# run.py agent_error markers that denote a MALFORMED/unparseable AGENT ACTION (the only ones action_validity
# penalizes). Other agent_error values (max_steps_exceeded) are lifecycle/run-management, not malformed
# actions, and are NOT mapped to an action_valid=False event.
_MALFORMED_ACTION_ERRORS = ("invalid_action", "bad_action_type", "truncated_tool_call")

def map_trace(trace, plugin):
    """SemanticEventMapper: raw canonical trace -> [SemanticEvent] under the v2 contract. A tool's meaning
    comes from the plugin: a RESULT-CONDITIONAL resolver(event, prev_state) if declared (reads the actual
    OUTPUT to decide real state change / which milestones were truly achieved), else the static
    tool_semantics (optimistic).

    Every tool_call event carries capability_id (= the tool name, INCLUDING failures), obligation_id (= the
    tool's primary declared milestone, or None), a SEMANTIC progress_token (content-hashed, never the tool
    name), a v2 status (success|partial|failure), and a NOVELTY-gated state_changed (a success that repeats
    a prior token/milestone is state_changed=False so stagnation can see it). The core names no
    benchmark/tool; the plugin does."""
    tool_sem = (plugin or {}).get("tool_semantics", {})
    resolvers = (plugin or {}).get("resolvers", {})
    default_role = (plugin or {}).get("default_tool_role", "act")
    out = []
    seen_ms, seen_pt = set(), set()
    prev_state = {"milestones": seen_ms, "tokens": seen_pt}
    for e in trace:
        et = e.get("event_type")
        if et == "tool_call":
            tool = e.get("tool")
            meta = tool_sem.get(tool, {"role": default_role, "success_milestones": []})
            cap_id = tool                                   # capability_id ALWAYS set, even on failure
            obligation = _primary_obligation(meta)
            ok = not _errored(e)
            res = resolvers.get(tool)
            if _no_payload(e) and not e.get("semantic_assume_success"):
                # P6: a resolver exists (this tool HAS a semantic completion condition) but there is NO output
                # to prove it was met -> CONSERVATIVE partial, milestone WITHHELD (never optimistic). Test
                # fixtures that legitimately assume success set semantic_assume_success=True.
                role, status, ms, pt, explicit_changed = meta.get("role", default_role), "partial", [], None, False
            else:
                if res and _no_payload(e):
                    res = None      # flagged test stub -> static optimistic semantics
                if res:
                    r = res(e, prev_state) or {}
                    role = r.get("role") or meta.get("role", default_role)
                    status = r.get("status", "success" if ok else "failure")
                    ms = list(r.get("milestones_added") or [])
                    pt = r.get("progress_token")
                    obligation = r.get("obligation_id", obligation)
                    explicit_changed = r.get("state_changed")
                else:
                    role = meta.get("role", default_role)
                    status = "success" if ok else "failure"
                    ms = list(meta.get("success_milestones", [])) if ok else []
                    pt = _default_token(e, tool) if ok else None     # content-hashed evidence, never tool name
                    explicit_changed = None
            if status not in STATUSES:
                status = "success" if ok else "failure"
            attr = None if status != "failure" else _attr(e)
            new_ms = [m for m in ms if m not in seen_ms]
            new_pt = pt is not None and pt not in seen_pt
            _sr = e.get("state_record") if isinstance(e.get("state_record"), dict) else None
            _rec_changed = _sr.get("state_changed") if _sr else None
            if _rec_changed is not None:                         # authoritative real state diff (review 3.4)
                changed = bool(_rec_changed) and status == "success"
            elif explicit_changed is not None:
                changed = bool(explicit_changed) and status == "success"
            else:
                changed = (status == "success") and (bool(new_ms) or new_pt)
            out.append(semantic_event(role, status=status, capability_id=cap_id, obligation_id=obligation,
                       progress_token=pt, milestones_added=ms, state_changed=changed,
                       failure_attribution=attr, raw=e))      # a tool_call that ran is WELL-FORMED -> action_valid default True
            seen_ms.update(ms)
            if pt is not None: seen_pt.add(pt)
        elif et == "agent_error" and e.get("error") in _MALFORMED_ACTION_ERRORS:
            # A MALFORMED / unparseable agent action (invalid_action / bad_action_type / truncated_tool_call):
            # run.py logged it instead of dispatching a tool. Emit a counted action event with
            # action_valid=False so action_validity (schema-validity only) sees the bad action. It is an
            # 'act' that FAILED to even be a well-formed action; attributed to the agent (it is the agent's
            # protocol violation), produced no state change. Other agent_error markers (max_steps_exceeded,
            # repeated_failing_call) are NOT malformed actions and are intentionally not emitted here.
            out.append(semantic_event("act", status="failure", failure_attribution="agent",
                       state_changed=False, action_valid=False, raw=e))
        elif et == "final_answer":
            out.append(semantic_event("final", terminal="final", raw=e))
        elif et and ("escalat" in str(et).lower() or et == "deliverable_budget_warning"):
            out.append(semantic_event("escalate", terminal="escalate", raw=e))
    return out

def _default_token(e, tool):
    """Content-hashed evidence token for a tool with no declared resolver: evidence:<tool>:<hash8 of the
    rendered/source content>. Two different outputs of the same tool get DIFFERENT tokens (new evidence); a
    byte-identical repeat repeats the token (no progress). None when there is no content."""
    content = _rendered_text(e) or _source_text(e)
    content = (content or "").strip()
    if not content:
        return None
    return "evidence:%s:%s" % (tool, _hash8(content))

def milestones_reached(sem_trace):
    m = set()
    for s in sem_trace:
        m.update(s.get("milestones_added") or [])
    return m

# ----------------------------------------------------------------------------- EvidenceView
def _rendered_text(e):
    """What the agent ACTUALLY saw. Priority: runner-recorded agent_visible_text (exact string put into the
    agent context) > raw observation > canonical_observation (audit mirror, compat fallback only)."""
    av = e.get("agent_visible_text")
    if isinstance(av, str) and av:
        return av
    ob = e.get("observation")
    if isinstance(ob, str) and ob:
        return ob
    co = e.get("canonical_observation")
    if isinstance(co, dict):
        t = (co.get("modalities") or {}).get("text")
        if t: return str(t)
    ob = e.get("observation")
    if isinstance(ob, str): return ob
    if isinstance(ob, dict): return json.dumps(ob, ensure_ascii=False)
    return ""

def _source_text(e):
    r = e.get("result")
    out = r.get("output") if isinstance(r, dict) else r
    if isinstance(out, dict): out = out.get("text") or json.dumps(out, ensure_ascii=False)
    return str(out or "")

def _real_delivery(e):
    """Prefer the RECORDED delivery_record (run.py renderer-level instrumentation, review 4.2); fall back to
    deriving from canonical_observation when an older trace carries no record."""
    dr = e.get("delivery_record")
    if isinstance(dr, dict):
        errored = _errored(e)
        # delivered = the agent's NEXT decision actually consumed it (backfilled by the runner). Falls back
        # to rendered_to_agent for older traces without the consumed flag. A circuit-broken last call was
        # rendered but never consumed -> NOT delivered.
        _consumed = dr.get("consumed_by_agent")
        _reach = _consumed if _consumed is not None else dr.get("rendered_to_agent")
        delivered = bool(_reach) and bool(dr.get("produced")) and not errored
        fid = 0.0 if errored else (0.5 if dr.get("truncated") else (1.0 if dr.get("rendered_to_agent") else 0.0))
        return {"delivered": delivered, "fidelity": fid, "error_visible": bool(dr.get("error_state_rendered"))}
    # --- legacy fallback (no delivery_record): derive from canonical_observation ---

    """REAL info-flow from the RECORDED rendering, NOT inferred from tool status. delivered = content was
    actually rendered into the agent context; fidelity = how much of the source survived (truncation/drop);
    error_visible = the failure actually appears in the agent-visible surface."""
    errored = _errored(e)
    rendered = _rendered_text(e); src = _source_text(e)
    if errored:
        ev = any(k in rendered.lower() for k in ("error", "fail", "exception", "operationoutcome"))
        return {"delivered": False, "fidelity": 0.0, "error_visible": ev}
    if not rendered.strip():
        return {"delivered": False, "fidelity": 0.0, "error_visible": False}   # renderer dropped the result
    if src.strip():
        sl = src.strip().lower(); rl = rendered.lower()
        fid = 1.0 if (sl[:200] in rl or len(rl) >= 0.8 * len(sl)) else round(min(1.0, len(rl) / max(1, len(sl))), 3)
    else:
        fid = 1.0
    return {"delivered": True, "fidelity": fid, "error_visible": False}

def evidence_view(trace, plugin):
    """EvidenceUnit = {id, delivered_to_agent, delivery_fidelity, error_visible, payload}. Delivery is read
    from the RECORDED agent-visible rendering (canonical_observation/observation), NOT inferred from status.
    A plugin extractor may refine fidelity with modality-specific signals (e.g. localization)."""
    ext = (plugin or {}).get("evidence_extractor")
    if ext: return ext(trace)
    units = []
    for i, e in enumerate(trace):
        if e.get("event_type") != "tool_call": continue
        d = _real_delivery(e)
        units.append({"id": "%s#%d" % (e.get("tool"), i), "delivered_to_agent": d["delivered"],
                      "delivery_fidelity": d["fidelity"], "error_visible": d["error_visible"],
                      "payload": _source_text(e)[:300]})
    return units

# ----------------------------------------------------------------------------- CapabilityManifest accessor
def capability_manifest(provenance):
    caps = (provenance or {}).get("capabilities") or {}
    return {k: {"implemented": v.get("implemented"), "available": v.get("available"),
                "authorized": v.get("authorized"), "healthy": v.get("healthy")}
            for k, v in caps.items() if isinstance(v, dict)}

# ----------------------------------------------------------------------------- DimensionPolicy accessor
def dimension_policy(task, plugin=None):
    """Merge the task's declared policy with the benchmark plugin defaults. Dimension evaluators read
    required_milestones / required_context_units / lifecycle_policy / governance_policy_id from HERE.

    FAIL-CLOSED: if the task's benchmark has NO registered plugin, the returned policy carries
    score_eligible=False + plugin_problem=missing_benchmark_plugin so the dimension evaluators do NOT
    silently score against a vacuous default policy."""
    name = task.get("source_benchmark")
    if plugin is None:
        plugin, problem = require_plugin(name)
        if plugin is None:
            return {"score_eligible": False, "plugin_problem": problem,
                    "required_milestones": [], "required_context_units": [],
                    "governance_policy_id": None}
    base = dict(plugin.get("dimension_policy") or {})
    ref = task.get("reference") or {}
    if ref.get("required_tool_groups"):
        base.setdefault("required_tool_groups", ref["required_tool_groups"])
        # PER-TASK readiness/completion: map each required tool GROUP to the milestone set its tools emit,
        # so |reached_milestones ∩ group_milestones| / |group_milestones| reproduces the old tool-path
        # completion fraction -- expressed in milestones (the evaluator never sees a tool name). This keeps
        # per-task discrimination (a 4-tool task the agent half-finished scores ~0.5), not a flat 1.0.
        tsem = (plugin or {}).get("tool_semantics", {})
        groups_ms = []
        for grp in ref["required_tool_groups"]:
            ms = set()
            for tool in (grp or []):
                ms.update((tsem.get(tool) or {}).get("success_milestones") or [])
            if ms: groups_ms.append(sorted(ms))
        if groups_ms:
            base["required_milestone_groups"] = groups_ms
            base["required_milestones"] = max(groups_ms, key=len)   # the most complete required path
    lifecycle_policy = task.get("lifecycle_policy") or {}
    base["lifecycle_policy"] = lifecycle_policy                      # P3: SINGLE normalization entry --
    for _f in ("ordering_constraints", "terminal_policy", "escalation_conditions", "stagnation_window",
               "irrecoverable_evidence_gap", "non_recoverable_obligations"):
        if _f in lifecycle_policy:                                  # lift sub-fields to top level so the
            base[_f] = lifecycle_policy[_f]                         # evaluator never guesses where they live
    base.setdefault("governance_policy_id", task.get("source_benchmark"))
    return base

# ============================================================================= PLUGIN REGISTRATION
# The benchmark-specific knowledge (tool->role/milestone semantics, RESULT-CONDITIONAL resolvers, EvidenceView
# extractors, dimension policy) lives in the runner/plugins/ package -- NOT here. Importing that package
# auto-registers all three plugins (each module calls register_plugin at import time). This import is LAST in
# the module body so register_plugin + every shared helper (_errored/_result_output/_hash8/_real_delivery/
# _no_payload/_default_token/_rendered_text/_source_text/...) already exist when the plugins import substrate
# back. The dependency is strictly one-directional (plugins -> substrate); the core above names no benchmark.
# A 4TH DATASET = drop runner/plugins/<name>.py + a spec/registry.json entry. No edit to this file.
import plugins as _plugins  # noqa: E402,F401  (import side effect: registers MedCTA/PhysicianBench/HealthAdminBench)
