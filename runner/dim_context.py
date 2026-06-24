#!/usr/bin/env python3
"""Context dimension — benchmark-AGNOSTIC CONTEXT MANAGEMENT scorer.

Measures whether the agent *managed its context* (acquired / had sufficient / bound to the right
subject / pulled relevant evidence) — NOT whether the final answer is correct (that is Verification /
Outcome). This module supersedes the MedCTA image-answer `context_grounding` / mm_judge "Context" path
in scoring.py: that path judged the ANSWER against the IMAGE (answer correctness); Context here judges
only the EVIDENCE the agent gathered.

Hard 诚信门 invariants (enforced by construction):
  * Consumes ONLY substrate structures: SemanticTrace (map_trace), EvidenceView (evidence_view),
    DimensionPolicy (dimension_policy), CapabilityManifest. No benchmark name, tool literal
    (OCR/fhir_/click/RegionAttribute), image, FHIR resource, or DOM appears in the scoring logic.
  * NEVER reads the final answer or the gold. `relevance` judges the OBSERVATIONS (evidence payloads)
    against the TASK INSTRUCTION only; the `final`/terminal SemanticEvent payload is explicitly dropped.
  * Applicable-only aggregation: a sub-metric with no opportunity returns status=not_applicable and is
    EXCLUDED from the mean (never a vacuous 1.0). Every sub-metric carries a `reportable` flag.

Sub-metrics (applicable-only):
  acquisition  — were the dimension_policy.required_context_units obtained (delivered acquire-role
                 evidence units + reached required_milestones mapped to the count of required units).
  sufficiency  — enough context units present to PROCEED (boolean floor: delivered units & required
                 milestones met before the terminal commit/final).
  relevance    — delivered evidence pertains to the task (gateway judge over OBSERVATIONS only).
  binding      — evidence bound to a single consistent subject/case (generic identifier-consistency
                 check over delivered evidence payloads; no patient/image/case literal).

tier = experimental.
"""
import os
import re

try:
    import substrate as _sub
except Exception:                       # pragma: no cover - allow `from runner import`
    from runner import substrate as _sub  # type: ignore

try:
    from lifecycle_exec import _sm, _aggregate          # reuse the canonical sub-metric helpers
except Exception:                                       # pragma: no cover
    def _sm(score, status="valid", opportunities=None, **kw):
        d = {"score": score, "status": status}
        if opportunities is not None:
            d["opportunities"] = opportunities
        d.update(kw)
        return d

    def _aggregate(subs):
        valid = {k: v for k, v in subs.items()
                 if v.get("status") == "valid" and isinstance(v.get("score"), (int, float))}
        score = round(sum(v["score"] for v in valid.values()) / len(valid), 3) if valid else None
        vals = [v["score"] for v in valid.values()]
        return {"score": score, "submetrics": subs, "applicable_submetrics": sorted(valid),
                "n_applicable": len(valid), "zero_variance": (len(set(vals)) == 1) if vals else None}

CONTEXT_VERSION = "context-1.0-experimental"

# A generic identifier token: an alnum string carrying a digit (MRNs, DEN-001, asset ids, refs ...).
# Deliberately NOT any benchmark-specific prefix — just "looks like an id that names a subject/case".
_ID_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9]*-?\d[A-Za-z0-9\-]*|\d{3,})\b")
_STOP_ID = {"http", "https", "2023", "2024", "2025", "2026"}   # dates/urls are not subjects


def _delivered(evidence):
    return [u for u in (evidence or []) if u.get("delivered_to_agent")]


def _subject_ids(text):
    """Generic candidate subject identifiers in a string (no domain knowledge)."""
    out = []
    for m in _ID_RE.findall(str(text or "")):
        t = m.strip().strip("-")
        if len(t) >= 3 and t.lower() not in _STOP_ID and any(c.isdigit() for c in t):
            out.append(t)
    return out


# ---------------------------------------------------------------- acquisition
def _acquisition(sem_trace, evidence, policy):
    """Coverage ratio: distinct delivered acquire-role evidence + reached required milestones vs the
    number of declared required_context_units. No opportunity (no required units) -> not_applicable."""
    req_units = list(policy.get("required_context_units") or [])
    req_ms = set(policy.get("required_milestones") or [])
    if not req_units:
        return _sm(None, "not_applicable", 0, reportable=False, reason="policy declares no required_context_units")

    deliv = _delivered(evidence)
    # acquire-role events carry distinct progress tokens (one per *kind* of context obtained)
    acquire_tokens = {s.get("progress_token") for s in sem_trace
                      if s.get("event_role") == "acquire" and s.get("status") == "success"
                      and s.get("progress_token")}
    reached = _sub.milestones_reached(sem_trace)
    ms_hit = req_ms & reached
    # "obtained context kinds" = distinct acquire tokens, capped by what delivered evidence backs.
    obtained = len(acquire_tokens) if acquire_tokens else len(deliv)
    if req_ms:
        # if the policy names required milestones, prefer the milestone-coverage signal
        obtained = max(obtained, len(ms_hit))
    score = round(min(1.0, obtained / max(1, len(req_units))), 3)
    return _sm(score, "valid", len(req_units), reportable=True,
               required_units=req_units, obtained_context_kinds=obtained,
               required_milestones=sorted(req_ms), milestones_reached=sorted(ms_hit))


# ---------------------------------------------------------------- sufficiency
def _sufficiency(sem_trace, evidence, policy):
    """Did the agent have ENOUGH context to proceed: a boolean floor — at least one delivered evidence
    unit per required_context_unit AND every required milestone reached, BEFORE the terminal step.
    Distinct from acquisition (which is a graded coverage ratio)."""
    req_units = list(policy.get("required_context_units") or [])
    req_ms = set(policy.get("required_milestones") or [])
    if not req_units and not req_ms:
        return _sm(None, "not_applicable", 0, reportable=False, reason="no required units/milestones to gate on")

    # evidence obtained strictly before the terminal (final/commit) event
    terminal_seen = False
    pre_units = 0
    for s in sem_trace:
        if s.get("terminal") in ("final", "escalate") or s.get("event_role") == "commit":
            terminal_seen = True
            break
        if s.get("event_role") == "acquire" and s.get("status") == "success":
            pre_units += 1
    n_deliv = len(_delivered(evidence)) if not terminal_seen or pre_units == 0 else pre_units
    n_deliv = max(n_deliv, pre_units)

    units_ok = (not req_units) or (n_deliv >= len(req_units))
    ms_ok = req_ms.issubset(_sub.milestones_reached(sem_trace))
    score = 1.0 if (units_ok and ms_ok) else 0.0
    return _sm(score, "valid", 1, reportable=True,
               required_units=len(req_units), units_obtained_pre_terminal=n_deliv,
               required_milestones=sorted(req_ms), milestones_satisfied=ms_ok)


# ---------------------------------------------------------------- binding
def _binding(evidence):
    """Generic subject-consistency check: identifier-like tokens that recur across delivered evidence
    payloads should point at ONE dominant subject/case (not a scatter of unrelated ids). Applicable
    only when delivered evidence actually carries identifier-like tokens. No patient/image/case literal."""
    deliv = _delivered(evidence)
    # subject ids that appear in MORE THAN ONE delivered unit (a one-off id is not a binding signal)
    per_unit = [set(_subject_ids(u.get("payload"))) for u in deliv]
    from collections import Counter
    cross = Counter()
    for s in per_unit:
        for i in s:
            cross[i] += 1
    recurring = {i: c for i, c in cross.items() if c >= 2}
    if not recurring:
        return _sm(None, "not_applicable", 0, reportable=False,
                   reason="no identifier-like tokens recur across delivered evidence")

    units_with_id = [s for s in per_unit if s]
    dominant, dom_n = max(recurring.items(), key=lambda kv: kv[1])
    # fraction of id-bearing delivered units that reference the dominant subject (single-subject focus)
    bound = sum(1 for s in units_with_id if dominant in s)
    score = round(bound / max(1, len(units_with_id)), 3)
    return _sm(score, "valid", len(units_with_id), reportable=True,
               dominant_subject=dominant, units_referencing_dominant=bound,
               id_bearing_units=len(units_with_id), distinct_recurring_ids=len(recurring))


# ---------------------------------------------------------------- relevance (judge over OBSERVATIONS)
def _relevance(sem_trace, evidence, task_instruction, judge_model, char_budget=7000, per_unit=700):
    """Gateway judge: do the delivered OBSERVATIONS pertain to the task? Reads evidence payloads +
    the task INSTRUCTION only. The terminal/final SemanticEvent is excluded so the answer never leaks;
    gold is never passed. No judge backend / instruction / evidence -> not_applicable (no vacuous 1.0)."""
    deliv = _delivered(evidence)
    instr = str(task_instruction or "").strip()
    if not deliv or not instr:
        return _sm(None, "not_applicable", 0, reportable=False, reason="no delivered evidence or no task instruction")
    if not judge_model or os.environ.get("MH_CONTEXT_JUDGE", "1") == "0":
        return _sm(None, "not_applicable", 0, reportable=False, reason="relevance judge backend unavailable/disabled")

    # build OBSERVATION-ONLY context; defensively skip any unit that came from a terminal/final event
    terminal_ids = {s.get("raw", {}).get("tool") for s in sem_trace if s.get("terminal")}
    parts, used = [], 0
    for u in deliv:
        seg = "- %s" % str(u.get("payload"))[:per_unit]
        if used + len(seg) > char_budget and parts:
            break
        parts.append(seg)
        used += len(seg)
    obs = "\n".join(parts)
    try:
        import gateway
        sysp = ("You judge whether the EVIDENCE an agent gathered is RELEVANT to the stated task. "
                "Judge only the evidence/observations against the task description — do NOT judge whether "
                "any conclusion is correct, and there is no answer key. Reply with exactly RELEVANT or "
                "IRRELEVANT on the first line, then a one-line reason.")
        usr = "TASK:\n%s\n\nEVIDENCE GATHERED:\n%s" % (instr[:1500], obs[:char_budget])
        r = gateway.chat([{"role": "system", "content": sysp}, {"role": "user", "content": usr}],
                         model=judge_model, max_tokens=200, judge=True)
        if not r.get("ok"):
            return _sm(None, "not_applicable", 0, reportable=False,
                       reason="relevance judge error: %s" % r.get("error_type"))
        head = (r.get("content") or "").strip().upper()
        rel = not head.startswith("IRRELEVANT")
        return _sm(1.0 if rel else 0.0, "valid", 1, reportable=True,
                   judge_model=judge_model, judge_tier="gateway_observation_relevance",
                   judge_backend=judge_model, n_evidence_units=len(parts),
                   reason=(r.get("content") or "")[:200])
    except Exception as ex:
        return _sm(None, "not_applicable", 0, reportable=False, reason="relevance judge exception: %s" % ex)


# ---------------------------------------------------------------- entry point
def context(sem_trace, evidence, dimension_policy, task_instruction=None, judge_model=None):
    """Context-management score from substrate structures only.

    sem_trace        : list[SemanticEvent]  (substrate.map_trace(trace, plugin))
    evidence         : list[EvidenceUnit]   (substrate.evidence_view(trace, plugin))
    dimension_policy : dict                 (substrate.dimension_policy(task, plugin))
    task_instruction : str | None  — the task GOAL/instruction text (NOT the gold, NOT the answer).
                       Used by the relevance judge only; safe because it is the problem statement.
    judge_model      : str | None  — gateway model for the relevance judge; None -> relevance skipped.
    """
    policy = dimension_policy or {}
    if judge_model is None:
        judge_model = os.environ.get("MH_JUDGE_MODEL")
    subs = {
        "acquisition": _acquisition(sem_trace, evidence, policy),
        "sufficiency": _sufficiency(sem_trace, evidence, policy),
        "binding": _binding(evidence),
        "relevance": _relevance(sem_trace, evidence, task_instruction, judge_model),
    }
    out = _aggregate(subs)
    out["dimension"] = "Context"
    out["tier"] = "experimental"
    out["evaluator_version"] = CONTEXT_VERSION
    out["measures"] = "context_management"               # explicitly NOT answer correctness
    out["reads_final_or_gold"] = False
    out["reportable"] = bool(out.get("n_applicable"))
    out["governance_policy_id"] = policy.get("governance_policy_id")
    return out


# ---------------------------------------------------------------- self-verification harness
def _selfcheck():
    import json
    import sys
    base = os.path.expanduser("~/Medical_harness")
    sys.path.insert(0, os.path.join(base, "runner"))

    def _load(bundle, sb):
        with open(os.path.join(base, bundle, "trajectory.jsonl")) as f:
            trace = [json.loads(l) for l in f if l.strip()]
        tj = os.path.join(base, bundle, "task.json")
        task = json.load(open(tj)) if os.path.exists(tj) else {"source_benchmark": sb}
        task.setdefault("source_benchmark", sb)
        return trace, task

    cases = [("results_mctaGov/gpt5/MCTA-0", "MedCTA"),
             ("results_pb_chk3/gpt5/PB-aberrant_drug_screen", "PhysicianBench"),
             ("results_hab10/gpt5/HAB-denial-easy-1", "HealthAdminBench")]
    for bundle, sb in cases:
        trace, task = _load(bundle, sb)
        plugin = _sub.get_plugin(sb)
        sem = _sub.map_trace(trace, plugin)
        ev = _sub.evidence_view(trace, plugin)
        pol = _sub.dimension_policy(task, plugin)
        instr = (task.get("context") or {}).get("text") or task.get("goal")
        # judge_model intentionally left None here so the offline self-check is deterministic / no network
        res = context(sem, ev, pol, task_instruction=instr, judge_model=None)
        print("\n==== %s [%s] ====" % (bundle, sb))
        print(" score       :", res["score"], "| applicable:", res["applicable_submetrics"],
              "| reportable:", res["reportable"], "| tier:", res["tier"])
        for k, v in res["submetrics"].items():
            print("   %-12s %-15s score=%s  %s" % (
                k, v.get("status"), v.get("score"),
                {kk: vv for kk, vv in v.items() if kk not in ("score", "status")}))


if __name__ == "__main__":
    _selfcheck()
