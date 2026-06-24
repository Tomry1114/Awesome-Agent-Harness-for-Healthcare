#!/usr/bin/env python3
"""Dimension: LIFECYCLE (benchmark-AGNOSTIC). Supersedes lifecycle_exec.lifecycle() heuristics.

Consumes ONLY substrate structures:
  sem_trace        - list[SemanticEvent]  (event_role / status / failure_attribution /
                     milestones_added / progress_token / state_changed / terminal)  via substrate.map_trace
  dimension_policy - dict from substrate.dimension_policy(task, plugin): required_milestones / ...
  manifest         - dict from substrate.capability_manifest(provenance) (optional, advisory only)

NO benchmark name, NO tool literal (OCR/fhir_/click/RegionAttribute), NO image/DOM/FHIR appears here.
Lifecycle = how the agent MANAGES the run across steps toward an allowed terminal state. Every construct
is expressed over the generic SemanticEvent fields, so adding a 4th benchmark needs no change here.

Sub-metrics (applicable-only + coverage gate; reportable_score requires >=2 CORE valid):
  readiness_before_terminal : at the FIRST terminal (final/commit) event, are the policy
                              required_milestones satisfied? Premature terminal -> low.  [CORE]
  ordering_precedence       : do acquire-role events occur before the terminal? (generic role precedence,
                              NOT a tool-name regex)                                      [CORE]
  stagnation                : a window of N>=3 consecutive events that add NO new milestone, NO new
                              progress_token, and state_changed is False -> penalize (cross-tool).
  recovery                  : a 'failure' sem event later followed by a 'success' that adds a new
                              milestone / new progress_token -> opportunity-conditioned, N/A if no failures.
  termination_quality       : terminal reached AND ready AND no unresolved AGENT-attributed failure.  [CORE]

Tier: experimental.
"""

# Generic terminal / role vocabulary (from substrate.ROLES & semantic_event.terminal). These are
# substrate constants, not benchmark literals.
_TERMINAL_ROLES = ("final", "commit", "escalate")   # a run can end by emitting an answer, committing, or escalating
_ACQUIRE_ROLE = "acquire"
_COMMIT_ROLE = "commit"


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


def _required_milestones(dimension_policy):
    return set((dimension_policy or {}).get("required_milestones") or [])


def _milestones_upto(sem_trace, idx):
    """All milestones added strictly before event index `idx` (inclusive of idx-1)."""
    m = set()
    for s in sem_trace[:idx]:
        m.update(s.get("milestones_added") or [])
    return m


def lifecycle(sem_trace, dimension_policy, manifest=None):
    sem_trace = list(sem_trace or [])
    dimension_policy = dimension_policy or {}
    required = _required_milestones(dimension_policy)

    # locate the FIRST terminal event (the run's commitment point). escalate counts as a terminal too.
    term_idx = next((i for i, s in enumerate(sem_trace) if _is_terminal(s)), None)
    has_terminal = term_idx is not None
    # readiness at the commitment point: milestones gathered BEFORE terminal (a commit event may itself
    # add a milestone, but readiness is about what was true going INTO the terminal decision).
    if has_terminal:
        ready_set = _milestones_upto(sem_trace, term_idx)
    else:
        ready_set = set()
        for s in sem_trace:
            ready_set.update(s.get("milestones_added") or [])

    sub = {}

    # 1. readiness_before_terminal [CORE] -------------------------------------------------------------
    # At the terminal, fraction of policy-required milestones already satisfied. No required milestones
    # declared -> not applicable (cannot judge readiness without a target). No terminal at all -> the run
    # never reached a commitment point => readiness 0 (it could not have been ready to terminate).
    if not required:
        sub["readiness_before_terminal"] = _sm(None, "not_applicable", 0,
                                                note="policy declares no required_milestones")
    elif not has_terminal:
        sub["readiness_before_terminal"] = _sm(0.0, opportunities=len(required),
                                               note="no terminal event reached")
    else:
        sat = len(required & ready_set)
        sub["readiness_before_terminal"] = _sm(round(sat / len(required), 3),
                                               opportunities=len(required),
                                               satisfied=sorted(required & ready_set),
                                               missing=sorted(required - ready_set))

    # 2. ordering_precedence [CORE] -------------------------------------------------------------------
    # Generic role precedence: acquire (information-gathering) events must precede the terminal commit.
    # Opportunity = there exists at least one acquire event AND a terminal. Score = fraction of acquire
    # events that occurred BEFORE the terminal index. (Not a tool-name regex; pure role ordering.)
    acquire_idx = [i for i, s in enumerate(sem_trace)
                   if s.get("event_role") == _ACQUIRE_ROLE and not _is_failure(s)]
    if has_terminal and acquire_idx:
        before = sum(1 for i in acquire_idx if i < term_idx)
        sub["ordering_precedence"] = _sm(round(before / len(acquire_idx), 3),
                                         opportunities=len(acquire_idx),
                                         acquire_before_terminal=before,
                                         acquire_after_terminal=len(acquire_idx) - before)
    else:
        sub["ordering_precedence"] = _sm(None, "not_applicable", 0,
                                         note="no acquire events or no terminal to order against")

    # 3. stagnation -----------------------------------------------------------------------------------
    # Cross-tool, cross-benchmark loop replacement: slide a window of N=3 over NON-terminal events; a
    # window stagnates if every event in it added NO new milestone, introduced NO new progress_token,
    # and had state_changed=False. Penalize by the fraction of windows that stagnate.
    work = [s for s in sem_trace if not _is_terminal(s)]
    N = 3
    if len(work) >= N:
        seen_ms = set()
        seen_pt = set()
        progress_flags = []   # True if event i contributed *something new*
        for s in work:
            new_ms = set(s.get("milestones_added") or []) - seen_ms
            pt = s.get("progress_token")
            new_pt = (pt is not None) and (pt not in seen_pt)
            advanced = bool(new_ms) or new_pt or bool(s.get("state_changed"))
            progress_flags.append(advanced)
            seen_ms |= set(s.get("milestones_added") or [])
            if pt is not None:
                seen_pt.add(pt)
        windows = len(progress_flags) - N + 1
        stagnant = sum(1 for j in range(windows)
                       if not any(progress_flags[j:j + N]))
        sub["stagnation"] = _sm(round(1.0 - stagnant / windows, 3), opportunities=windows,
                                stagnant_windows=stagnant, window=N)
    else:
        sub["stagnation"] = _sm(None, "not_applicable", len(work),
                                note="fewer than %d non-terminal events" % N)

    # 4. recovery -------------------------------------------------------------------------------------
    # Opportunity-conditioned: for each failure event, does a LATER success event add a new milestone or
    # a new progress_token (i.e. the run actually recovered the lost ground)? N/A if no failures.
    fail_idx = [i for i, s in enumerate(sem_trace) if _is_failure(s)]
    if fail_idx:
        recovered = 0
        for i in fail_idx:
            ms_before = _milestones_upto(sem_trace, i + 1)
            pt_before = {s.get("progress_token") for s in sem_trace[:i + 1]
                         if s.get("progress_token") is not None}
            ok = False
            for s in sem_trace[i + 1:]:
                if _is_failure(s):
                    continue
                new_ms = set(s.get("milestones_added") or []) - ms_before
                pt = s.get("progress_token")
                new_pt = (pt is not None) and (pt not in pt_before)
                if new_ms or new_pt:
                    ok = True
                    break
            recovered += 1 if ok else 0
        sub["recovery"] = _sm(round(recovered / len(fail_idx), 3), opportunities=len(fail_idx),
                              recovered=recovered, failures=len(fail_idx))
    else:
        sub["recovery"] = _sm(None, "not_applicable", 0, note="no failure events to recover from")

    # 5. termination_quality [CORE] -------------------------------------------------------------------
    # A clean termination = a terminal was reached, it was ready (required milestones satisfied), and
    # there is NO unresolved AGENT-attributed failure outstanding at the end. Escalation is a legitimate
    # terminal only when it is the chosen end-state (already captured as terminal). No terminal -> 0.
    unresolved_agent = False
    for i, s in enumerate(sem_trace):
        if _is_failure(s) and s.get("failure_attribution") == "agent":
            pt = s.get("progress_token")
            # resolved if a later success re-establishes the same goal or adds any new milestone
            later_ok = False
            ms_before = _milestones_upto(sem_trace, i + 1)
            for x in sem_trace[i + 1:]:
                if _is_failure(x):
                    continue
                if (set(x.get("milestones_added") or []) - ms_before) or \
                   (pt is not None and x.get("progress_token") == pt):
                    later_ok = True
                    break
            if not later_ok:
                unresolved_agent = True
    if not has_terminal:
        sub["termination_quality"] = _sm(0.0, note="no terminal event reached")
    else:
        rd = sub["readiness_before_terminal"]
        rd_factor = rd["score"] if (rd.get("status") == "valid"
                                    and isinstance(rd.get("score"), (int, float))) else 1.0
        sub["termination_quality"] = _sm(round(rd_factor * (0.5 if unresolved_agent else 1.0), 3),
                                         unresolved_agent_failure=unresolved_agent)

    out = _aggregate(sub)

    # coverage gate: do NOT emit a confident Lifecycle number off a single sub-metric. Require >=2 of the
    # CORE constructs (readiness / ordering / termination) to be applicable to be reportable.
    _core = ("readiness_before_terminal", "ordering_precedence", "termination_quality")
    valid_core = [k for k in _core if sub.get(k, {}).get("status") == "valid"]
    out["reportable_score"] = len(valid_core) >= 2
    out["coverage_status"] = "ok" if len(valid_core) >= 2 else "insufficient_construct_coverage"
    out["valid_core_submetrics"] = valid_core
    out["submetric_status"] = {k: v.get("status") for k, v in sub.items()}
    out["opportunity_count"] = {k: v.get("opportunities") for k, v in sub.items()
                                if v.get("opportunities")}
    out["state_path"] = {
        "has_terminal": has_terminal,
        "terminal_role": sem_trace[term_idx].get("event_role") if has_terminal else None,
        "n_events": len(sem_trace),
        "milestones_at_terminal": sorted(ready_set),
    }
    # manifest is advisory only (a degraded capability is the environment's fault, never the agent's
    # lifecycle management); surfaced for transparency, NOT folded into the score.
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
        ("PB/HAB", os.path.join(ROOT, "results_pb_chk3/gpt5/PB-aberrant_drug_screen")),
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
    print("\nimport OK")
