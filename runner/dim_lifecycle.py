#!/usr/bin/env python3
"""Dimension: LIFECYCLE (benchmark-AGNOSTIC). Supersedes lifecycle_exec.lifecycle() heuristics.

Consumes ONLY substrate structures:
  sem_trace        - list[SemanticEvent] under the v2 CONTRACT (event_role / status / capability_id /
                     obligation_id / progress_token / milestones_added / state_changed / terminal /
                     failure_attribution)  via substrate.map_trace
  dimension_policy - dict from substrate.dimension_policy(task, plugin): required_milestones /
                     required_milestone_groups / required_tool_groups / ordering_constraints / ...
  manifest         - dict from substrate.capability_manifest(provenance) (advisory; used ONLY to judge
                     escalation appropriateness, never folded into the agent's management score).

NO benchmark name, NO tool literal (OCR/fhir_/click/RegionAttribute), NO image/DOM/FHIR appears here.
Lifecycle = how the agent MANAGES the run across steps toward an allowed terminal state. Every construct
is expressed over the generic SemanticEvent fields, so adding a 4th benchmark needs no change here.

Each sub-metric measures EXACTLY ONE thing (construct overlaps removed in the v2 rewrite):

  readiness_before_terminal : at the FIRST terminal, fraction of the BEST required_milestone_group
                              satisfied going INTO the terminal. ONLY readiness. N/A if no required
                              milestones declared; 0 if the run never terminated.            [CORE]

  ordering_precedence       : DECLARATIVE precedence. Reads dimension_policy.ordering_constraints, each
                              { predecessor:{milestone|role}, successor:{milestone|role}, weight } (the
                              canonical form). A constraint is ACTIVATED when its SUCCESSOR occurs; it is
                              SATISFIED only when the predecessor also occurs strictly BEFORE the successor.
                              If the successor never occurs the constraint is not exercised. Score =
                              satisfied_weight / activated_weight over ACTIVATED constraints ONLY. N/A when
                              the task declares no non-trivial ordering constraint (NOT an auto 1.0). The
                              trivial 'acquire-before-terminal' score is DELETED.             [CORE]

  stagnation                : a window of N>=3 consecutive non-terminal events that add NO new milestone,
                              NO new progress_token, and have state_changed=False -> stagnant. Judged by
                              real progress (now meaningful under v2), never tool-name novelty.

  recovery                  : OBLIGATION-bound. A failure on obligation O is recovered ONLY by a LATER
                              event producing the SAME obligation_id (same tool retried, or an alt tool in
                              the same required-tool group), OR by a justified escalation. NOT by "any
                              later progress". N/A if no obligation-bound failures. Failures with
                              obligation_id=None are non-recoverable noise, excluded from the denominator.

  termination_quality       : terminal MANAGEMENT only -- correct terminal TYPE chosen + not truncated +
                              no post-goal flailing + no UNRESOLVED agent-attributed failure. Does NOT
                              re-use the readiness factor (the readiness/termination overlap is removed).
                              Escalation judged on appropriateness via the manifest.          [CORE]

Tier: experimental.
"""

# Generic terminal / role vocabulary (substrate constants, not benchmark literals).
_TERMINAL_ROLES = ("final", "commit", "escalate")   # answer / commit / escalate are all run end-states
_ACQUIRE_ROLE = "acquire"


def _sm(score, status="valid", opportunities=None, **kw):
    d = {"score": score, "status": status}
    if opportunities is not None:
        d["opportunities"] = opportunities
    d.update(kw)
    return d


def _aggregate(subs):
    """Mean over ONLY applicable (status=valid, numeric) sub-metrics -> never a vacuous 1.0."""
    valid = {k: v for k, v in subs.items()
             if v.get("status") == "valid" and isinstance(v.get("score"), (int, float))}
    vals = [v["score"] for v in valid.values()]
    score = round(sum(vals) / len(vals), 3) if vals else None
    return {"score": score, "submetrics": subs,
            "applicable_submetrics": sorted(valid), "n_applicable": len(valid),
            "zero_variance": (len(set(vals)) == 1) if vals else None}


def _is_failure(s):
    return str(s.get("status", "")).lower() == "failure"


def _is_terminal(s):
    return bool(s.get("terminal")) or s.get("event_role") in _TERMINAL_ROLES


def _milestones_upto(sem_trace, idx):
    """All milestones added strictly before event index `idx` (inclusive of idx-1)."""
    m = set()
    for s in sem_trace[:idx]:
        m.update(s.get("milestones_added") or [])
    return m


# -------------------------------------------------------------------- ordering constraint resolution
def _ordering_constraints(dimension_policy):
    """Normalize dimension_policy.ordering_constraints. CANONICAL authoring form (the only one new tasks
    should emit): {"predecessor": {milestone|role}, "successor": {milestone|role}, "weight"}. The legacy
    trigger/required_before(_role)/required_after(_role) form is still PARSED for back-compat but each such
    constraint is flagged deprecated_syntax=True and a DeprecationWarning is emitted (it reads backwards:
    required_before=B under trigger=A actually means A-before-B). Normalizes to a list of
       {trigger, kind, target, weight}  where trigger/target each carry one of
       ("milestone", <name>) or ("role", <role>), and kind=="before" means trigger MUST precede target.
       A 'required_after' clause is stored as the symmetric 'before' with endpoints swapped.
       Authoring forms per constraint dict:
         trigger=<milestone> | trigger_role=<role>
         required_before=<milestone> | required_before_role=<role>
         required_after=<milestone>  | required_after_role=<role>
         weight=<float, default 1.0>
    """
    raw = (dimension_policy or {}).get("ordering_constraints") or []
    out = []
    def _ep(d):
        if not isinstance(d, dict): return None
        if d.get("milestone") is not None: return ("milestone", d["milestone"])
        if d.get("role") is not None: return ("role", d["role"])
        return None
    for c in raw:
        if not isinstance(c, dict):
            continue
        # canonical form: {"predecessor": {milestone|role}, "successor": {milestone|role}, "weight"}
        if c.get("predecessor") is not None and c.get("successor") is not None:
            tp, sp = _ep(c["predecessor"]), _ep(c["successor"])
            if tp and sp:
                try: _w = float(c.get("weight", 1.0))
                except (TypeError, ValueError): _w = 1.0
                out.append({"trigger": tp, "kind": "before", "target": sp, "weight": _w})
            continue
        if c.get("trigger") is not None:
            trig = ("milestone", c["trigger"])
        elif c.get("trigger_role") is not None:
            trig = ("role", c["trigger_role"])
        else:
            continue
        try:
            w = float(c.get("weight", 1.0))
        except (TypeError, ValueError):
            w = 1.0
        import warnings
        warnings.warn("ordering_constraints legacy trigger/required_before syntax is deprecated; use "
                      "predecessor/successor", DeprecationWarning)
        if c.get("required_before") is not None:
            out.append({"trigger": trig, "kind": "before", "target": ("milestone", c["required_before"]),
                        "weight": w, "deprecated_syntax": True})
        elif c.get("required_before_role") is not None:
            out.append({"trigger": trig, "kind": "before", "target": ("role", c["required_before_role"]),
                        "weight": w, "deprecated_syntax": True})
        elif c.get("required_after") is not None:
            out.append({"trigger": ("milestone", c["required_after"]), "kind": "before", "target": trig,
                        "weight": w, "deprecated_syntax": True})
        elif c.get("required_after_role") is not None:
            out.append({"trigger": ("role", c["required_after_role"]), "kind": "before", "target": trig,
                        "weight": w, "deprecated_syntax": True})
    return out


def _first_occurrence(sem_trace, endpoint):
    """Earliest event index where `endpoint` ("milestone",name) | ("role",role) first occurs (its
    milestone is added / its role appears on a non-failure event). None if it never occurs."""
    kind, val = endpoint
    for i, s in enumerate(sem_trace):
        if kind == "milestone":
            if val in (s.get("milestones_added") or []):
                return i
        else:
            if s.get("event_role") == val and not _is_failure(s):
                return i
    return None


def _score_ordering(sem_trace, dimension_policy):
    constraints = _ordering_constraints(dimension_policy)
    if not constraints:
        return _sm(None, "not_applicable", 0,
                   note="policy declares no ordering_constraints (trivial acquire-before-terminal removed)")
    activated_w = satisfied_w = 0.0
    activated = 0
    detail = []
    for c in constraints:
        t_idx = _first_occurrence(sem_trace, c["trigger"])   # predecessor
        g_idx = _first_occurrence(sem_trace, c["target"])    # successor
        if g_idx is None:                           # successor never happened -> constraint NOT triggered
            detail.append({"trigger": c["trigger"], "target": c["target"], "activated": False})
            continue
        activated += 1                              # ACTIVATED once the successor occurs
        activated_w += c["weight"]
        ok = (t_idx is not None) and (t_idx < g_idx)  # predecessor present AND strictly before successor
        if ok:
            satisfied_w += c["weight"]
        detail.append({"trigger": c["trigger"], "target": c["target"], "activated": True,
                       "satisfied": ok, "trigger_at": t_idx, "target_at": g_idx, "weight": c["weight"]})
    if activated == 0:
        return _sm(None, "not_applicable", 0,
                   note="no declared ordering_constraint was activated (endpoints absent)",
                   declared=len(constraints))
    return _sm(round(satisfied_w / activated_w, 3), opportunities=activated,
               activated_constraints=activated, declared=len(constraints),
               satisfied_weight=round(satisfied_w, 3), activated_weight=round(activated_w, 3),
               constraints=detail)


# -------------------------------------------------------------------- recovery (obligation-bound)
def _obligation_equivalence(dimension_policy):
    """Obligations interchangeable for recovery: any two obligations produced by tools in the SAME
    required_tool_group recover each other. Returns obligation -> frozenset(equivalent obligations).
    Lifts a required_tool_group of TOOLS to a group of OBLIGATIONS via dimension_policy._tool_obligations
    ({tool: primary_obligation}) so no tool name is named HERE (benchmark-agnostic)."""
    groups = (dimension_policy or {}).get("required_tool_groups") or []
    tool_ob = (dimension_policy or {}).get("_tool_obligations") or {}
    equiv = {}
    for grp in groups:
        obs = {tool_ob[t] for t in (grp or []) if tool_ob.get(t)}
        if len(obs) > 1:
            fz = frozenset(obs)
            for ob in obs:
                equiv[ob] = fz
    return equiv


def _escalation_justified(manifest, dimension_policy, obligation_id=None):
    """An escalation terminal is APPROPRIATE only when the environment blocked progress (a capability is
    unimplemented/unavailable/unauthorized/unhealthy) or the task declares the evidence irrecoverable. A bare
    'I give up' escalation is NOT justified (and must not count as recovery)."""
    manifest = manifest or {}; policy = dimension_policy or {}
    for cap in manifest.values():
        if isinstance(cap, dict) and (cap.get("implemented") is False or cap.get("available") is False
                                      or cap.get("authorized") is False or cap.get("healthy") is False):
            return True
    if policy.get("irrecoverable_evidence_gap") is True:
        return True
    if obligation_id is not None and obligation_id in set(policy.get("non_recoverable_obligations") or []):
        return True
    return False


def _obligation_resolved_after(sem_trace, fail_idx, obligation_id, equiv, manifest=None, dimension_policy=None):
    """THE single definition of 'a failure was resolved' shared by Recovery AND Termination: a LATER success
    re-achieving the SAME obligation (or one in its equivalence class), OR a JUSTIFIED escalation. NEVER 'any
    later unrelated milestone'."""
    targets = equiv.get(obligation_id, frozenset([obligation_id])) if obligation_id else frozenset()
    for s in sem_trace[fail_idx + 1:]:
        if _is_terminal(s) and s.get("event_role") == "escalate":
            if _escalation_justified(manifest, dimension_policy, obligation_id):
                return True, "justified_escalation"
            return False, None                       # an UNjustified escalation does not resolve the obligation
        if _is_failure(s):
            continue
        if obligation_id is not None and s.get("obligation_id") in targets and \
           str(s.get("status", "")).lower() == "success":
            return True, "obligation_reachieved"
    return False, None


def _unresolved_obligations(sem_trace, equiv, manifest, dimension_policy, term_idx):
    """ALL obligation-bound failures still unresolved going INTO the terminal -- NOT filtered by attribution,
    matching Recovery's denominator -- so the obligation-specific escalation justification is consistent
    between Recovery and Termination even for an ENVIRONMENT-attributed failure on a non_recoverable
    obligation. (The agent-only unresolved-failure penalty for final/commit is a SEPARATE concern handled by
    _has_unresolved_agent_failure.)"""
    pre = sem_trace[:term_idx] if term_idx is not None else sem_trace
    unresolved = []
    for i, ev in enumerate(pre):
        ob = ev.get("obligation_id")
        if _is_failure(ev) and ob:
            resolved, _ = _obligation_resolved_after(pre, i, ob, equiv, manifest, dimension_policy)
            if not resolved:
                unresolved.append(ob)
    return sorted(set(unresolved))


def _score_recovery(sem_trace, equiv, manifest=None, dimension_policy=None):
    """A failure on obligation O is recovered ONLY by a LATER SUCCESS producing the SAME obligation_id (or
    an obligation in O's equivalence class), OR by a justified escalation terminal. Failures with
    obligation_id=None are excluded. N/A if no obligation-bound failures."""
    bound = [(i, s.get("obligation_id")) for i, s in enumerate(sem_trace)
             if _is_failure(s) and s.get("obligation_id")]
    if not bound:
        return _sm(None, "not_applicable", 0, note="no obligation-bound failure events to recover from")
    recovered = 0
    detail = []
    for i, ob in bound:
        ok, how = _obligation_resolved_after(sem_trace, i, ob, equiv, manifest, dimension_policy)
        recovered += 1 if ok else 0
        detail.append({"failed_obligation": ob, "recovered": ok, "via": how})
    return _sm(round(recovered / len(bound), 3), opportunities=len(bound),
               recovered=recovered, failures=len(bound), recoveries=detail)


# -------------------------------------------------------------------- truncation / agent-failure helpers
def _trace_truncated(sem_trace, dimension_policy):
    if (dimension_policy or {}).get("truncated"):
        return True
    for s in sem_trace:
        raw = s.get("raw") or {}
        et = str(raw.get("event_type", "")).lower()
        if "budget" in et or "truncat" in et or raw.get("truncated") is True:
            return True
        if str(raw.get("finish_reason", "")).lower() in ("length", "max_tokens", "truncated"):
            return True
    return False


def _has_unresolved_agent_failure(sem_trace, equiv=None, manifest=None, dimension_policy=None):
    """An AGENT-attributed failure never resolved (SAME definition as Recovery via _obligation_resolved_after:
    same/equivalent obligation re-achieved OR justified escalation). An unrelated later milestone does NOT
    resolve it -- Recovery and Termination now agree."""
    equiv = equiv or {}
    for i, s in enumerate(sem_trace):
        if _is_failure(s) and s.get("failure_attribution") == "agent":
            ok, _ = _obligation_resolved_after(sem_trace, i, s.get("obligation_id"), equiv, manifest, dimension_policy)
            if not ok:
                return True
    return False


# -------------------------------------------------------------------- main
def lifecycle(sem_trace, dimension_policy, manifest=None):
    sem_trace = list(sem_trace or [])
    dimension_policy = dimension_policy or {}

    term_idx = next((i for i, s in enumerate(sem_trace) if _is_terminal(s)), None)
    has_terminal = term_idx is not None
    if has_terminal:
        ready_set = _milestones_upto(sem_trace, term_idx)
    else:
        ready_set = set()
        for s in sem_trace:
            ready_set.update(s.get("milestones_added") or [])

    sub = {}

    # 1. readiness_before_terminal [CORE] -- ONLY readiness ------------------------------------------
    required = set(dimension_policy.get("required_milestones") or [])
    groups = dimension_policy.get("required_milestone_groups") or ([sorted(required)] if required else [])
    groups = [g for g in groups if g]
    if not groups:
        sub["readiness_before_terminal"] = _sm(None, "not_applicable", 0,
                                                note="policy declares no required_milestones")
    elif not has_terminal:
        sub["readiness_before_terminal"] = _sm(0.0, opportunities=len(max(groups, key=len)),
                                               note="no terminal event reached")
    else:
        best = max(groups, key=lambda g: len(set(g) & ready_set) / len(g))
        sub["readiness_before_terminal"] = _sm(round(len(set(best) & ready_set) / len(best), 3),
                                               opportunities=len(best), n_paths=len(groups),
                                               satisfied=sorted(set(best) & ready_set),
                                               missing=sorted(set(best) - ready_set))

    # 2. ordering_precedence [CORE] -- DECLARATIVE constraints only ----------------------------------
    sub["ordering_precedence"] = _score_ordering(sem_trace, dimension_policy)

    # 3. stagnation -- NEW milestone / NEW progress_token / state_changed -----------------------------
    work = [s for s in sem_trace if not _is_terminal(s)]
    N = 3
    if len(work) >= N:
        seen_ms, seen_pt = set(), set()
        progress_flags = []
        for s in work:
            new_ms = set(s.get("milestones_added") or []) - seen_ms
            pt = s.get("progress_token")
            new_pt = (pt is not None) and (pt not in seen_pt)
            progress_flags.append(bool(new_ms) or new_pt or bool(s.get("state_changed")))
            seen_ms |= set(s.get("milestones_added") or [])
            if pt is not None:
                seen_pt.add(pt)
        windows = len(progress_flags) - N + 1
        stagnant = sum(1 for j in range(windows) if not any(progress_flags[j:j + N]))
        sub["stagnation"] = _sm(round(1.0 - stagnant / windows, 3), opportunities=windows,
                                stagnant_windows=stagnant, window=N)
    else:
        sub["stagnation"] = _sm(None, "not_applicable", len(work),
                                note="fewer than %d non-terminal events" % N)

    # 4. recovery -- OBLIGATION-bound ----------------------------------------------------------------
    _equiv = _obligation_equivalence(dimension_policy)
    sub["recovery"] = _score_recovery(sem_trace, _equiv, manifest, dimension_policy)

    # 5. termination_quality [CORE] -- terminal MANAGEMENT, NOT readiness ----------------------------
    terminal_role = sem_trace[term_idx].get("event_role") if has_terminal else None
    truncated = _trace_truncated(sem_trace, dimension_policy)
    if not has_terminal:
        sub["termination_quality"] = _sm(0.0, terminal=None, truncated=truncated,
                                         note="no terminal event reached (or truncated before terminal)")
    elif terminal_role == "escalate":
        # SAME escalation policy as Recovery/unresolved-failure (single source of truth): justified when a
        # capability is unimplemented/unavailable/unauthorized/unhealthy or the policy declares the evidence
        # irrecoverable. No longer a narrower healthy/authorized-only check (which disagreed with Recovery).
        _unres = _unresolved_obligations(sem_trace, _equiv, manifest, dimension_policy, term_idx)
        justified = (_escalation_justified(manifest, dimension_policy, obligation_id=None)
                     or any(_escalation_justified(manifest, dimension_policy, obligation_id=ob) for ob in _unres))
        sub["termination_quality"] = _sm(1.0 if justified else 0.5, terminal="escalate",
                                         escalation_justified=justified, unresolved_obligations=_unres,
                                         note="shared escalation policy (global + obligation-specific); readiness NOT applied")
    else:
        post_goal = sum(1 for s in sem_trace[term_idx + 1:] if not _is_terminal(s))
        unresolved_agent = _has_unresolved_agent_failure(sem_trace, _equiv, manifest, dimension_policy)
        score = 1.0                                  # a final/commit terminal is a valid TYPE
        if truncated:
            score *= 0.5
        if post_goal > 0:
            score *= 0.5
        if unresolved_agent:
            score *= 0.5
        sub["termination_quality"] = _sm(round(score, 3), terminal=terminal_role, truncated=truncated,
                                         post_goal_events=post_goal,
                                         unresolved_agent_failure=unresolved_agent)

    out = _aggregate(sub)

    _core = ("readiness_before_terminal", "ordering_precedence", "termination_quality")
    valid_core = [k for k in _core if sub.get(k, {}).get("status") == "valid"]
    out["reportable_score"] = len(valid_core) >= 2
    out["coverage_status"] = "ok" if len(valid_core) >= 2 else "insufficient_construct_coverage"
    out["valid_core_submetrics"] = valid_core
    out["submetric_status"] = {k: v.get("status") for k, v in sub.items()}
    out["opportunity_count"] = {k: v.get("opportunities") for k, v in sub.items()
                                if v.get("opportunities")}
    out["state_path"] = {"has_terminal": has_terminal, "terminal_role": terminal_role,
                         "n_events": len(sem_trace), "milestones_at_terminal": sorted(ready_set)}
    if manifest:
        out["degraded_capability"] = any(isinstance(v, dict) and v.get("healthy") is False
                                         for v in manifest.values())
    out["tier"] = "experimental"
    return out


# ----------------------------------------------------------------------------- self-verification
if __name__ == "__main__":
    import os, sys, json
    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, HERE)
    import substrate as S

    def _load_trace(p):
        return [json.loads(l) for l in open(p) if l.strip()]

    def _load_task(p):
        return json.load(open(p)) if os.path.exists(p) else {}

    def _provenance(p):
        if not os.path.exists(p):
            return {}
        d = json.load(open(p))
        return d.get("provenance") or d.get("result", {}).get("provenance") or {}

    ROOT = os.path.dirname(HERE)
    bundles = [
        ("MedCTA", os.path.join(ROOT, "results_mctaGov/gpt5/MCTA-0")),
        ("PB", os.path.join(ROOT, "results_pb_chk3/gpt5/PB-aberrant_drug_screen")),
        ("HAB", os.path.join(ROOT, "results_hab10/gpt5/HAB-denial-easy-1")),
    ]
    for label, bdir in bundles:
        traj = os.path.join(bdir, "trajectory.jsonl")
        if not os.path.exists(traj):
            print("SKIP", label, "(no trajectory at %s)" % bdir)
            continue
        trace = _load_trace(traj)
        task = _load_task(os.path.join(bdir, "task.json"))
        prov = _provenance(os.path.join(bdir, "result.json"))
        plugin = S.get_plugin(task.get("source_benchmark"))
        sem = S.map_trace(trace, plugin)
        dp = S.dimension_policy(task, plugin)
        man = S.capability_manifest(prov)
        res = lifecycle(sem, dp, man)
        print("\n===== %s  (%s) =====" % (label, task.get("source_benchmark")))
        print("score            :", res["score"], "| reportable:", res["reportable_score"],
              "|", res["coverage_status"])
        print("applicable       :", res["applicable_submetrics"])
        for k, v in res["submetrics"].items():
            print("  %-26s score=%-6s status=%-15s opp=%s" % (
                k, v.get("score"), v.get("status"), v.get("opportunities")))
        print("state_path       :", res["state_path"])

    print("\n========== SYNTHETIC CONSTRUCT-VALIDITY PROOFS ==========")

    def ev(role, status="success", ob=None, ms=None, pt=None, changed=None, terminal=None, attr=None, raw=None):
        e = S.semantic_event(role, status=status, obligation_id=ob, progress_token=pt,
                             milestones_added=ms or [], terminal=terminal, failure_attribution=attr,
                             raw=raw or {})
        if changed is not None:
            e["state_changed"] = bool(changed)
        elif status == "success" and (ms or pt):
            e["state_changed"] = True
        return e

    # (a) ordering: wrong-order < right-order when an ordering_constraint is DECLARED
    pol_ord = {"required_milestones": ["A", "B"], "required_milestone_groups": [["A", "B"]],
               "ordering_constraints": [{"predecessor": {"milestone": "A"}, "successor": {"milestone": "B"}, "weight": 1.0}]}
    right = [ev("acquire", ms=["A"], pt="evidence:x:1"),
             ev("acquire", ms=["B"], pt="evidence:y:2"), ev("final", terminal="final")]
    wrong = [ev("acquire", ms=["B"], pt="evidence:y:2"),
             ev("acquire", ms=["A"], pt="evidence:x:1"), ev("final", terminal="final")]
    r_right = lifecycle(right, pol_ord)["submetrics"]["ordering_precedence"]
    r_wrong = lifecycle(wrong, pol_ord)["submetrics"]["ordering_precedence"]
    print("(a) ordering   right=%s  wrong=%s  -> right>wrong: %s" % (
        r_right["score"], r_wrong["score"], r_right["score"] > r_wrong["score"]))
    na = lifecycle(right, {"required_milestones": ["A", "B"],
                           "required_milestone_groups": [["A", "B"]]})["submetrics"]["ordering_precedence"]
    print("    ordering   no-constraint status=%s score=%s (must be not_applicable/None)" % (
        na["status"], na["score"]))

    # (b) termination != readiness
    pol_rt = {"required_milestones": ["A"], "required_milestone_groups": [["A"]]}
    clean = [ev("acquire", ms=["A"], pt="p1"), ev("final", terminal="final")]
    flail = [ev("acquire", ms=["A"], pt="p1"), ev("final", terminal="final"),
             ev("act", status="partial", pt=None)]
    rc = lifecycle(clean, pol_rt)["submetrics"]
    rf = lifecycle(flail, pol_rt)["submetrics"]
    print("(b) clean   readiness=%s termination=%s" % (
        rc["readiness_before_terminal"]["score"], rc["termination_quality"]["score"]))
    print("    flail   readiness=%s termination=%s  -> term<readiness (decoupled): %s" % (
        rf["readiness_before_terminal"]["score"], rf["termination_quality"]["score"],
        rf["termination_quality"]["score"] < rf["readiness_before_terminal"]["score"]))
    premature = [ev("acquire", status="partial", ob="A"), ev("final", terminal="final")]
    Lp = lifecycle(premature, pol_rt)["submetrics"]
    print("    premature readiness=%s termination=%s  -> NOT equal: %s" % (
        Lp["readiness_before_terminal"]["score"], Lp["termination_quality"]["score"],
        Lp["termination_quality"]["score"] != Lp["readiness_before_terminal"]["score"]))

    # (c) a failure 'recovered' by an UNRELATED tool is NOT counted
    unrelated = [ev("acquire", status="failure", ob="O1", attr="agent"),
                 ev("acquire", status="success", ob="O2", ms=["m2"], pt="p2"),
                 ev("final", terminal="final")]
    real = [ev("acquire", status="failure", ob="O1", attr="agent"),
            ev("acquire", status="success", ob="O1", ms=["m1"], pt="p1"),
            ev("final", terminal="final")]
    rec_unrel = lifecycle(unrelated, {})["submetrics"]["recovery"]
    rec_real = lifecycle(real, {})["submetrics"]["recovery"]
    print("(c) recovery  unrelated-tool=%s  same-obligation=%s  -> unrelated NOT counted: %s" % (
        rec_unrel["score"], rec_real["score"],
        rec_unrel["score"] == 0.0 and rec_real["score"] == 1.0))
    pol_grp = {"required_tool_groups": [["toolO1", "toolO1b"]],
               "_tool_obligations": {"toolO1": "O1", "toolO1b": "O1b"}}
    alt = [ev("acquire", status="failure", ob="O1", attr="agent"),
           ev("acquire", status="success", ob="O1b", ms=["m1b"], pt="p1b"),
           ev("final", terminal="final")]
    rec_alt = lifecycle(alt, pol_grp)["submetrics"]["recovery"]
    print("    recovery  alt-tool-in-group=%s  -> equivalence recovers: %s" % (
        rec_alt["score"], rec_alt["score"] == 1.0))

    print("\nimport OK")
