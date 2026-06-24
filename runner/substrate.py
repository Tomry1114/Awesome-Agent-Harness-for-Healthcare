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
# A plugin MUST declare these keys; anything else makes scoring vacuous, so registration fails closed.
_REQUIRED_PLUGIN_KEYS = ("benchmark", "tool_semantics")

def _equivalent_plugin(a, b):
    """Two plugin dicts are the SAME plugin re-registering (a module RELOAD), not a hostile duplicate, when
    they declare the same benchmark name and the same tool surface (sorted tool_semantics tool ids). A
    genuinely DIFFERENT plugin claiming the name has a different tool set -> not equivalent -> conflict."""
    if not (isinstance(a, dict) and isinstance(b, dict)):
        return False
    if a.get("benchmark") != b.get("benchmark"):
        return False
    return sorted((a.get("tool_semantics") or {}).keys()) == sorted((b.get("tool_semantics") or {}).keys())

def register_plugin(p):
    """V9 FAIL-CLOSED registration. A plugin that is malformed or collides MUST raise at import time
    (loudly) rather than silently shadowing another benchmark or registering a vacuous policy:
      * not a dict, or missing a REQUIRED key (benchmark / tool_semantics)            -> ValueError
      * a CONFLICTING duplicate benchmark name -- a STRUCTURALLY-DIFFERENT plugin claiming an already
        registered id -- -> ValueError (fail closed: refuse to shadow a different existing plugin).
      * any declared resolver / evidence_extractor / trace predicate that is NOT callable -> TypeError
    A benign RE-registration is explicitly allowed (no raise): the SAME object, OR a structurally-equivalent
    plugin (same benchmark name + same tool_semantics tool set) re-registering. That covers the legitimate
    `import importlib; del sys.modules['plugins.*']; import_module('plugins')` package-RELOAD used by the
    drop-file auto-discovery flow -- a reload runs the module body again with a NEW dict that is the same
    plugin, which must NOT be treated as a hostile duplicate. Validating here (not at score time) means a
    broken 4th-dataset drop-in is caught the moment its module imports."""
    if not isinstance(p, dict):
        raise ValueError("register_plugin: plugin must be a dict, got %r" % type(p).__name__)
    for k in _REQUIRED_PLUGIN_KEYS:
        if not p.get(k):
            raise ValueError("register_plugin: plugin missing required key %r (have %s)"
                             % (k, sorted(p)))
    name = p["benchmark"]
    if not isinstance(p.get("tool_semantics"), dict):
        raise ValueError("register_plugin: %r tool_semantics must be a dict" % name)
    existing = _PLUGINS.get(name)
    if existing is not None and existing is not p and not _equivalent_plugin(existing, p):
        # a DIFFERENT plugin (different tool surface) claiming an already-owned benchmark name -> conflict.
        raise ValueError("register_plugin: duplicate benchmark name %r already registered with a DIFFERENT "
                         "tool surface (fail-closed: refuse to shadow an existing plugin)" % name)
    # every declared resolver / extractor / predicate that a dimension evaluator will CALL must be callable
    for fld in ("resolvers", "trace_predicates"):
        for tn, fn in (p.get(fld) or {}).items():
            if not callable(fn):
                raise TypeError("register_plugin: %r %s[%r] is not callable" % (name, fld, tn))
    ext = p.get("evidence_extractor")
    if ext is not None and not callable(ext):
        raise TypeError("register_plugin: %r evidence_extractor is not callable" % name)
    _PLUGINS[name] = p
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

# ----------------------------------------------------------------------------- Obligation resolution (CONTRACT-E)
def _tool_obligation_map(plugin):
    """{tool: primary_obligation} for every tool the plugin declares -- the bridge that lets a
    required_tool_group of TOOLS become an equivalence class of OBLIGATIONS (no tool literal leaks into the
    evaluators). Same primary-obligation rule the mapper uses (_primary_obligation)."""
    tsem = (plugin or {}).get("tool_semantics") or {}
    out = {}
    for tool, meta in tsem.items():
        ob = _primary_obligation(meta)
        if ob:
            out[tool] = ob
    return out


def resolve_obligations(sem_trace, dimension_policy, manifest=None):
    """CONTRACT-E: the ONE canonical obligation-resolution shared by Lifecycle AND Governance.

    Returns {obligation_id: bool} over EVERY obligation that has at least one FAILED (or partial) event in
    the trace. A failed obligation O is RESOLVED iff a LATER event (strictly after the failure) either
      * re-achieves the SAME obligation_id (or an obligation in O's equivalence class -- two tools in the
        same required_tool_group recover each other) with status=success, OR
      * is a JUSTIFIED escalation terminal (a capability is unimplemented/unavailable/unauthorized/unhealthy,
        or the policy marks the evidence irrecoverable / the obligation non-recoverable).
    An UNRELATED later success does NOT resolve it; an UNJUSTIFIED escalation does NOT resolve it.

    This is NOT a private simplified copy: it DELEGATES to dim_lifecycle's exact helpers
    (_obligation_equivalence / _obligation_resolved_after), so Lifecycle's recovery, Governance's
    concealed-failure check, and this function can never drift apart. Governance consumes THIS (no local
    mirror). dimension_policy may carry _tool_obligations (tool->obligation) for the equivalence lift; if
    absent we derive it here so callers needn't pre-populate it."""
    sem_trace = list(sem_trace or [])
    policy = dict(dimension_policy or {})
    import dim_lifecycle as _L                                  # lazy: dim_lifecycle never imports us at load
    equiv = _L._obligation_equivalence(policy)
    out = {}
    for i, s in enumerate(sem_trace):
        ob = s.get("obligation_id")
        if not ob:
            continue
        if str(s.get("status", "")).lower() not in ("failure", "partial"):
            continue
        ok, _how = _L._obligation_resolved_after(sem_trace, i, ob, equiv, manifest, policy)
        # AND across multiple failures of the same obligation: if ANY occurrence is left unresolved, the
        # obligation is unresolved (a later resolution flips an earlier one, but a still-pending tail does
        # not get masked by an earlier resolved instance).
        out[ob] = bool(ok) and out.get(ob, True)
    return out


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

    # --- tool->obligation bridge (CONTRACT-E): so a required_tool_group of TOOLS becomes an equivalence
    # class of OBLIGATIONS for resolve_obligations / Lifecycle recovery WITHOUT a tool literal reaching the
    # evaluator. Derived from the plugin, never authored by the task.
    base.setdefault("_tool_obligations", _tool_obligation_map(plugin))

    # --- V10 (CONTRACT-B): MERGE the task's verification_policy OVER the plugin default. The plugin supplies
    # the sensible DEFAULT (None present -> dim_verification treats cross-source as NOT_APPLICABLE, never
    # 'require 2 sources for every claim'); a task override (e.g. cross_source_required_for=[claim_type]) wins.
    task_vp = task.get("verification_policy")
    if isinstance(task_vp, dict):
        merged_vp = dict(base.get("verification_policy") or {})
        merged_vp.update(task_vp)                                # task keys win, plugin defaults stay as fallback
        base["verification_policy"] = merged_vp
    # (if the task declares none, base keeps the plugin default -- or nothing, which is correctly NOT_APPLICABLE)

    # --- V10 (CONTRACT-F): MERGE the task's TYPED context_requirements OVER the plugin's required_context_units.
    # Each entry is {id, type}; a task may add/override units (keyed by id) the benchmark default did not list.
    # A bare-string entry degrades to {id:s, type:s} so older authoring still parses.
    task_cr = task.get("context_requirements")
    if task_cr:
        def _norm_unit(u):
            if isinstance(u, dict):
                return {"id": u.get("id") or u.get("type"), "type": u.get("type") or u.get("id")}
            return {"id": str(u), "type": str(u)}
        by_id = {}
        order = []
        for u in (base.get("required_context_units") or []):
            nu = _norm_unit(u)
            if nu["id"] not in by_id:
                order.append(nu["id"])
            by_id[nu["id"]] = nu
        for u in task_cr:                                        # task entries override/append by id
            nu = _norm_unit(u)
            if nu["id"] not in by_id:
                order.append(nu["id"])
            by_id[nu["id"]] = nu
        base["required_context_units"] = [by_id[i] for i in order]

    # --- V10 (CONTRACT-C): expose expected_subject = {type, id} derived from the task. dim_context reports
    # subject_consistency (evidence converges on one subject) AND expected_subject_match (that subject ==
    # THIS). 'Consistently reading the WRONG patient' converges but fails the match. Derivation precedence:
    #   1. an explicit task.expected_subject {type,id}            (author override)
    #   2. context.patient_ref  -> {type:'Patient', id:patient_ref}   (PB)
    #   3. context.subject_ref / case_ref {type,id} | id            (generic subject/case)
    #   4. a plugin default expected_subject                         (benchmark fallback)
    #   5. a reference subject (task.reference.subject / context.case_id)
    # type defaults to 'Patient' for a bare patient_ref; otherwise the declared/derived type, else None.
    base["expected_subject"] = _expected_subject(task, base)

    return base


def _expected_subject(task, base_policy):
    """Derive {type, id} for CONTRACT-C. id None means 'no expected subject declared' (the consumer then
    skips expected_subject_match rather than failing every run). Benchmark-agnostic: it reads only generic
    task fields (context.patient_ref / subject_ref / case_ref) plus an optional plugin/task default."""
    ctx = (task or {}).get("context") or {}
    # 1. explicit task override
    es = (task or {}).get("expected_subject")
    if isinstance(es, dict) and es.get("id"):
        return {"type": es.get("type") or "Patient", "id": es.get("id")}
    # 2. patient_ref (PB)
    pref = ctx.get("patient_ref")
    if pref:
        return {"type": "Patient", "id": str(pref)}
    # 3. generic subject_ref / case_ref (dict {type,id} or bare id)
    for key, dflt_type in (("subject_ref", "Subject"), ("case_ref", "Case")):
        v = ctx.get(key)
        if isinstance(v, dict) and v.get("id"):
            return {"type": v.get("type") or dflt_type, "id": v.get("id")}
        if v:
            return {"type": dflt_type, "id": str(v)}
    # 4. plugin/base default
    pes = (base_policy or {}).get("expected_subject")
    if isinstance(pes, dict) and pes.get("id"):
        return {"type": pes.get("type") or "Subject", "id": pes.get("id")}
    # 5. a reference subject as last resort
    ref = (task or {}).get("reference") or {}
    rsub = ref.get("subject") or ctx.get("case_id")
    if rsub:
        return {"type": "Subject", "id": str(rsub)}
    return {"type": None, "id": None}

# ============================================================================= PLUGIN REGISTRATION
# The benchmark-specific knowledge (tool->role/milestone semantics, RESULT-CONDITIONAL resolvers, EvidenceView
# extractors, dimension policy) lives in the runner/plugins/ package -- NOT here. Importing that package
# auto-registers all three plugins (each module calls register_plugin at import time). This import is LAST in
# the module body so register_plugin + every shared helper (_errored/_result_output/_hash8/_real_delivery/
# _no_payload/_default_token/_rendered_text/_source_text/...) already exist when the plugins import substrate
# back. The dependency is strictly one-directional (plugins -> substrate); the core above names no benchmark.
# A 4TH DATASET = drop runner/plugins/<name>.py + a spec/registry.json entry. No edit to this file.
#
# V13 IMPORT-SAFETY: this module may be imported as either top-level `substrate` (PYTHONPATH=runner) OR as
# `runner.substrate` (PYTHONPATH=repo-root). Those are two DIFFERENT module objects, each with its OWN
# _PLUGINS dict. If we always did a bare `import plugins`, the plugin modules' `import substrate` would
# register into the top-level substrate even when WE are runner.substrate -> a future `import runner.substrate`
# would see list_plugins()==0 (split registry). To keep ONE registry, we import the plugins package RELATIVE
# to however THIS module was imported (substrate.__package__) so the plugin->substrate back-import resolves to
# the SAME module object, and we re-export our registry under both `substrate` and `runner.substrate` so the
# package's `import substrate` finds the already-initialized module rather than re-importing a second copy.
import importlib as _il, sys as _sys
_self = _sys.modules[__name__]
_sys.modules.setdefault("substrate", _self)                 # plugins' `import substrate` -> THIS module
if __package__:                                             # imported as <pkg>.substrate
    _sys.modules.setdefault("%s.substrate" % __package__, _self)
    _plugins = _il.import_module("%s.plugins" % __package__)  # package-relative -> shares this registry
else:
    _plugins = _il.import_module("plugins")
del _il, _sys, _self
