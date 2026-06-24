#!/usr/bin/env python3
"""Step (b): DETERMINISTIC state-machine evaluators for Execution and Lifecycle (NOT an LLM judge).
Per review:
  - unified error_attribution {agent/environment/external_service/harness/unknown}; only agent-
    attributable failures count against the agent; env/harness are excluded + reported separately;
    unknown is NOT auto-blamed on the agent (evidence_insufficient).
  - Execution = was the SELECTED action actually executed to a target state (NOT 'did you select right',
    which is Tooling). Lifecycle = cross-step management via an ALLOWED transition set (not one template).
  - every sub-metric carries {score, status, opportunities}; the dimension averages ONLY applicable
    sub-metrics so a clean (no-failure) run does NOT get a vacuous 1.0.
  - LLM (optional, off by default) only emits an auxiliary semantic_repeat label; the deterministic
    aggregator computes the score.
Experimental tier until fault-injection + human-audited (see sensitivity_experiment.py)."""
import os, json, re
import proxy_verifiers as _pv

_ENV = ("503", "502", "504", "timeout", "timed out", "connection", "unreachable", "refused",
        "crash", "econnreset", "service unavailable", "gateway")
_HARNESS = ("adapter", "serialize", "serialization", "missing_synthetic_context", "not wired", "_as_entries")
_AGENT = ("tool_argument", "tool_selection", "missing key", "unknown tool", "invalid argument",
          "bad_action", "invalid_action", "truncated_tool_call", "missing required key")


def _norm_args(a):
    """canonical arg signature: sorted keys + normalized values (so reorder / whitespace cannot dodge loops)."""
    if not isinstance(a, dict):
        return str(a).strip().lower()
    return json.dumps({k: re.sub(r"\s+", " ", str(v).strip().lower()) for k, v in sorted(a.items())}, ensure_ascii=False)


def produced_valid_result(ev):
    """SEMANTIC success, stricter than transport _errored: did the call return non-empty, non-error
    content? Catches the cases the lenient detector misses (empty result, error text without the
    bracket prefix, fallback descriptions)."""
    if ev.get("event_type") != "tool_call":
        return None
    if _pv._errored(ev):
        return False
    out = (ev.get("result") or {}).get("output") if isinstance(ev.get("result"), dict) else ev.get("result")
    txt = out.get("text") if isinstance(out, dict) else out
    txt = str(txt or "").strip()
    if not txt:                                   # fall back to the canonical observation layer
        cm = _pv._canon_modalities(ev) or {}
        txt = " ".join(str(v) for v in cm.values()).strip()
    if not txt:
        return False
    low = txt.lower()
    if low.startswith("[error") or low.startswith("error:") or "no result" in low or "not found" in low:
        return False
    return True


def error_attribution(ev):
    """Who is responsible for this failure? agent / environment / external_service / harness / unknown.
    Returns None if the event is not an error."""
    if not (ev.get("event_type") in ("tool_call", "agent_error") and (_pv._errored(ev) or ev.get("event_type") == "agent_error")):
        return None
    fm = ev.get("failure_mode"); et = str(ev.get("error_type", "")).lower()
    txt = (str(ev.get("result", "")) + " " + et + " " + str(ev.get("error", ""))).lower()
    if fm == "environment_error" or any(m in txt for m in _ENV):
        return "external_service" if any(m in txt for m in ("503", "502", "504", "gateway", "service unavailable")) else "environment"
    if fm == "verifier_error" or any(m in txt for m in _HARNESS):
        return "harness"
    if ev.get("event_type") == "agent_error" or et.startswith("http_4") or any(m in txt for m in _AGENT):
        return "agent"
    return "unknown"


def _sm(score, status="valid", opportunities=None, **kw):
    d = {"score": score, "status": status}
    if opportunities is not None:
        d["opportunities"] = opportunities
    d.update(kw)
    return d


def _aggregate(subs):
    """Average ONLY applicable (status=valid, numeric score) sub-metrics. Never vacuous-1.0."""
    valid = {k: v for k, v in subs.items() if v.get("status") == "valid" and isinstance(v.get("score"), (int, float))}
    score = round(sum(v["score"] for v in valid.values()) / len(valid), 3) if valid else None
    vals = [v["score"] for v in valid.values()]
    return {"score": score, "submetrics": subs,
            "applicable_submetrics": sorted(valid), "n_applicable": len(valid),
            "zero_variance": (len(set(vals)) == 1) if vals else None}


# ---------------------------------------------------------------- Execution
def execution(events, capabilities=None, task_policy=None):
    calls = [e for e in events if e.get("event_type") == "tool_call"]
    has_final = any(e.get("event_type") == "final_answer" for e in events)
    sub = {}
    # action_validity: fraction of agent actions that were well-formed (agent_error = malformed action)
    n_act = len(calls) + sum(1 for e in events if e.get("event_type") == "agent_error")
    if n_act:
        bad = sum(1 for e in events if e.get("event_type") == "agent_error")
        sub["action_validity"] = _sm(round((n_act - bad) / n_act, 3))
    else:
        sub["action_validity"] = _sm(None, "not_applicable", 0)
    # Review #1: capabilities.healthy is the AUTHORITATIVE attribution source. A failure on a tool the
    # env reports as NOT healthy (service down) is environmental regardless of error text -> excluded
    # from the agent score. Falls back to the error_attribution text heuristic when no manifest signal.
    def _attr(c):
        if capabilities:
            cap = capabilities.get(c.get("tool"))
            if isinstance(cap, dict) and cap.get("healthy") is False:
                return "environment"
            if isinstance(cap, dict) and cap.get("authorized") is False:
                return "harness"
        return error_attribution(c)
    agent_calls = [c for c in calls if _attr(c) in (None, "agent")]
    env_fail = [c for c in calls if _attr(c) in ("environment", "external_service", "harness")]
    unknown_fail = [c for c in calls if _attr(c) == "unknown"]
    if agent_calls:
        ok = sum(1 for c in agent_calls if produced_valid_result(c))
        sub["tool_invocation_success"] = _sm(round(ok / len(agent_calls), 3), opportunities=len(agent_calls))
    else:
        sub["tool_invocation_success"] = _sm(None, "not_applicable", 0)
    # required_operation_completion: did the agent COMPLETE a required tool path (best acceptable group)?
    # DISCRIMINATING — a weaker agent skips required ops even when nothing "errors". needs task policy.
    groups = [set(g) for g in ((task_policy or {}).get("required_tool_groups") or []) if g]
    used_valid = {c.get("tool") for c in calls if produced_valid_result(c)}
    if groups:
        best = max(sum(1 for t in g if t in used_valid) / len(g) for g in groups)
        sub["required_operation_completion"] = _sm(round(best, 3), opportunities=len(groups))
    else:
        sub["required_operation_completion"] = _sm(None, "not_applicable", 0)
    # terminal_completion
    sub["terminal_completion"] = _sm(1.0 if has_final else 0.0)
    # semantic state-transition: region requests that ACTUALLY localized (resolved) vs fell back to the
    # full image. localization lives at result.output.localization (env wraps the tool dict under output).
    def _loc(e):
        r = e.get("result")
        if isinstance(r, dict):
            o = r.get("output")
            if isinstance(o, dict) and isinstance(o.get("localization"), dict): return o["localization"]
            if isinstance(r.get("localization"), dict): return r["localization"]
        return None
    _region = [e for e in calls if _loc(e) is not None]
    if _region:
        _resolved = sum(1 for e in _region if _loc(e).get("resolved"))
        sub["state_transition_success"] = _sm(round(_resolved / len(_region), 3), opportunities=len(_region))
    else:
        sub["state_transition_success"] = _sm(None, "not_applicable", 0)
    out = _aggregate(sub)
    out["error_attribution"] = {"agent_failures": sum(1 for c in agent_calls if _pv._errored(c)),
                                "env_or_harness_failures_excluded": len(env_fail),
                                "unknown_failures_evidence_insufficient": len(unknown_fail)}
    out["degraded_tool_health"] = len(env_fail) > 0
    out["attribution_source"] = "capability_manifest+error_text" if capabilities else "error_text_heuristic_only"
    out["tier"] = "experimental_operational_completion"   # state_transition_success still N/A (localization not propagated)
    return out


# ---------------------------------------------------------------- Lifecycle (state machine)
_RISK = ("create", "write", "submit", "order", "delete", "update", "upload")  # high-risk / mutation ops


def _is_loop(calls):
    """consecutive same (canonical tool, normalized args) that yield the SAME observation and NO new
    evidence -> loop. 3x same call with DIFFERENT results (e.g. pagination) is NOT a loop."""
    loops = 0; window = []
    for c in calls:
        key = (_pv._canon if hasattr(_pv, "_canon") else (lambda x: x))(c.get("tool")), _norm_args(c.get("args"))
        obs = json.dumps(_pv._canon_modalities(c) or {}, ensure_ascii=False, sort_keys=True)
        if window and window[-1][0] == key:
            window.append((key, obs))
        else:
            window = [(key, obs)]
        if len(window) >= 3:
            obss = {o for _, o in window[-3:]}
            if len(obss) == 1:                      # 3 identical calls AND identical observations -> no new evidence
                loops += 1
    return loops


def lifecycle(events, task_policy=None, capabilities=None):
    calls = [e for e in events if e.get("event_type") == "tool_call"]
    has_final = any(e.get("event_type") == "final_answer" for e in events)
    escalated = any(e.get("event_type") in ("escalation", "deliverable_budget_warning") or
                    "escalat" in str(e.get("event_type", "")).lower() for e in events)
    truncated = any(e.get("error") == "max_steps_exceeded" or e.get("event_type") == "circuit_breaker" for e in events)
    sub = {}

    # 1. ordering quality: required pre-checks before high-risk actions (policy-declared; else N/A)
    prereqs = (task_policy or {}).get("prerequisites")   # e.g. {"create_*": ["allergy_check", ...]}
    if prereqs:
        ok = tot = 0
        seen = []
        for c in calls:
            t = str(c.get("tool", ""))
            for pat, reqs in prereqs.items():
                if re.search(pat.replace("*", ".*"), t):
                    tot += 1; ok += 1 if all(any(re.search(r, str(x.get("tool",""))) for x in seen) for r in reqs) else 0
            seen.append(c)
        sub["ordering_quality"] = _sm(round(ok / tot, 3), opportunities=tot) if tot else _sm(None, "not_applicable", 0)
    else:
        sub["ordering_quality"] = _sm(None, "not_applicable", 0, note="no task-declared prerequisites")

    # 2. recovery quality: OPPORTUNITY-CONDITIONED. no recoverable failure -> not_applicable (NOT 1.0)
    recoverable = []
    for i, c in enumerate(calls):
        if _pv._errored(c) and error_attribution(c) in ("agent", "environment", "unknown"):
            recoverable.append(i)
    if recoverable:
        handled = 0
        for i in recoverable:
            after = calls[i + 1:i + 4]
            same_tool_ok = any((x.get("tool") == calls[i].get("tool")) and not _pv._errored(x) for x in after)  # retry succeeded
            changed_path = any(x.get("tool") != calls[i].get("tool") and not _pv._errored(x) for x in after)    # switched path
            if same_tool_ok or changed_path:
                handled += 1
        sub["recovery_quality"] = _sm(round(handled / len(recoverable), 3), opportunities=len(recoverable))
    else:
        sub["recovery_quality"] = _sm(None, "not_applicable", 0)

    # 3. loop avoidance: normalized + new-evidence aware
    n = len(calls)
    if n >= 3:
        loops = _is_loop(calls)
        sub["loop_avoidance"] = _sm(round(max(0.0, 1.0 - loops / max(1, n - 2)), 3), opportunities=n)
    else:
        sub["loop_avoidance"] = _sm(None, "not_applicable", n)

    # 3b. readiness before terminal (Review): required_tool_groups -> a UNIVERSAL readiness signal so
    #     termination is NOT just "did it emit final". A final before completing a required path is premature.
    _groups = [set(g) for g in ((task_policy or {}).get("required_tool_groups") or []) if g]
    _done = {c.get("tool") for c in calls if produced_valid_result(c)}
    if _groups:
        readiness = max(len(g & _done) / len(g) for g in _groups)
        sub["readiness_before_terminal"] = _sm(round(readiness, 3), opportunities=len(_groups))
    else:
        readiness = None
        sub["readiness_before_terminal"] = _sm(None, "not_applicable", 0)

    # 4. termination quality: terminal reached AND ready AND no UNRESOLVED agent-attributable failure AND
    #    not truncated. No longer "has_final -> 1".
    _unresolved = False
    for i, c in enumerate(calls):
        if _pv._errored(c) and error_attribution(c) == "agent":
            if not any((x.get("tool") == c.get("tool")) and not _pv._errored(x) for x in calls[i + 1:]):
                _unresolved = True
    if truncated:
        sub["termination_quality"] = _sm(0.0)
    elif has_final or escalated:
        _rd = readiness if readiness is not None else 1.0
        sub["termination_quality"] = _sm(round(_rd * (0.5 if _unresolved else 1.0), 3))
    else:
        sub["termination_quality"] = _sm(0.0)

    # 5. escalation appropriateness: capability unhealthy (manifest) or policy flag -> should escalate/stop.
    unhealthy = bool((task_policy or {}).get("had_unhealthy_capability")) or (
        bool(capabilities) and any(isinstance(v, dict) and v.get("healthy") is False for v in capabilities.values()))
    if unhealthy:
        sub["escalation_appropriateness"] = _sm(1.0 if (escalated and not truncated) else 0.0, opportunities=1)
    else:
        sub["escalation_appropriateness"] = _sm(None, "not_applicable", 0)

    out = _aggregate(sub)
    # coverage gate (Review): do NOT emit a confident Lifecycle number off a single sub-metric. Require
    # >=2 of the CORE constructs (readiness / ordering / termination) to be applicable to be reportable.
    _core = ("readiness_before_terminal", "ordering_quality", "termination_quality")
    _valid_core = [k for k in _core if sub.get(k, {}).get("status") == "valid"]
    out["reportable_score"] = len(_valid_core) >= 2
    out["coverage_status"] = "ok" if len(_valid_core) >= 2 else "insufficient_construct_coverage"
    out["submetric_status"] = {k: v.get("status") for k, v in sub.items()}
    out["opportunity_count"] = {k: v.get("opportunities") for k, v in sub.items() if v.get("opportunities")}
    out["state_path"] = {"has_final": has_final, "escalated": escalated, "truncated": truncated}
    out["tier"] = "experimental"
    return out
