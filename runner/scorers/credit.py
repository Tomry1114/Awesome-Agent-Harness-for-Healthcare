# -*- coding: utf-8 -*-
"""Agent vs harness credit: the scoring views, the checkpoint credit record and the agent-credited outcome.

runner/harness/origin.py has the origin model. What the harness completed is not credited to the agent. The native
score stays reportable; the agent-credited outcome is a separate field beside it.

    scoring_views(trajectory, ...)       the agent-credited scoring inputs run.py hands the scorers, and the harness view
    write_trajectory_logs(job_dir, ...)  logs/agent/trajectory.log from agent-origin tool calls only; the harness view in
                                         logs/harness/harness_tool_calls.jsonl
    CheckpointCreditTracker              after every executed action (ActionExecutor.build_event), which state-query
                                         checkpoint holds and which rubric field value is present, and who acted
    task_credit(...)                     result["credit"]: per-origin counts, the final's author, harness-written files
                                         and persisted fields, per-checkpoint credit, native and agent-credited outcome
    agent_credited_outcome(result, bench)  the aggregation hook: the native outcome rule on the result with harness- and
                                         mixed-credited checkpoints not achieved
    report_block(results, bench)         the separate report field (aggregate_report.build "agent_credited_outcome")
    attribution(result, bench)           per task, which scored items are agent- vs harness-credited
"""
import ast
import collections
import copy
import datetime
import hashlib
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.dirname(HERE)
if RUNNER not in sys.path:           # flat sys.path like the dim_* scorers; the origin model is a runner/harness module
    sys.path.insert(0, RUNNER)
from harness import origin as O  # noqa: E402

CREDIT_VERSION = 1
AGENT, HARNESS, MIXED, UNKNOWN = "agent", "harness", "mixed", "unknown"
CREDITS = (AGENT, HARNESS, MIXED, UNKNOWN)
NOT_AGENT_ACHIEVED = (HARNESS, MIXED)        # a passing checkpoint with this credit does not count for the agent
PORTAL_KEYS = ("emr", "payerA", "payerB", "fax")
# eval-root slices a persisted field is read from (aetna_state / anthem_state alias payer_*; signals derive from
# full_state.agentActions)
FIELD_ROOTS = ("full_state", "payer_a_state", "payer_b_state")
MAX_FIELD_RECORDS = 200
_WRITE_TOOLS = ("type", "fill")
_MIN_TEXT_MATCH = 12


# ===================================================================================================== final answers
def final_author(ev, agent_answers=None):
    """Who wrote the text of a final_answer event. A tagged event says it (final_answer_author). An untagged one (a
    trajectory recorded without author tags) is read from what the loop recorded: the withheld stub or its disposition,
    a +claimmerge disposition, else identity with one of the agent's answer_attempt texts (they keep 500 characters)
    when those are given (an Accept whose text is none of them: candidate merge), else the agent."""
    a = ev.get("final_answer_author")
    if a in O.AUTHORS:
        return a
    thought = ev.get("thought") or ""
    disp = str(ev.get("final_disposition") or "")
    if thought.startswith("[WITHHELD]") or disp.startswith("abstained_unresolved"):
        return O.AUTHOR_WITHHELD_STUB
    if "claimmerge" in disp:
        return O.AUTHOR_CANDIDATE_MERGE
    if agent_answers:
        for ans in agent_answers:
            if thought == ans or (len(ans) >= 500 and thought[:500] == ans):
                return O.AUTHOR_AGENT
        if ev.get("intervention_type") is not None:
            return O.AUTHOR_CANDIDATE_MERGE
    return O.AUTHOR_AGENT


def _agent_event(ev, agent_answers):
    et = ev.get("event_type")
    if et == "tool_call":
        return not O.is_harness_event(ev)
    if et == "final_answer":
        return final_author(ev, agent_answers) == O.AUTHOR_AGENT
    return et in ("answer_attempt", "agent_error")


def _sha(text):
    return hashlib.sha1((text or "").encode("utf-8", "replace")).hexdigest()[:12]


# ===================================================================================================== scoring views
def write_target(args):
    """The file name a write_file call writes (its declared file_path, or path)."""
    args = args if isinstance(args, dict) else {}
    p = args.get("file_path") or args.get("path")
    return os.path.basename(str(p)) if p else None


def scoring_views(trajectory, first_answer=None, outdir=None):
    """The scoring inputs run.py builds, split by origin.

    agent-credited (what the scorers read):
        agent_tool_calls   tool_call events of agent origin
        final_texts        thoughts of the final_answer events the agent authored; when there is none (no final, or the
                           delivered final is a candidate merge or the withheld stub) the loop's existing no-final rule:
                           the agent's first answer, else its last valid answer_attempt
        note_texts         write_file contents of agent origin, then every file in the output directory (end state)
        agent_trajectory   agent tool calls, answer_attempt, agent-authored final_answer, agent_error
    harness view (recorded, not scored):
        harness_tool_calls, delivered_final_texts (every final_answer thought), harness_note_texts
    """
    trajectory = trajectory or []
    attempts = [e.get("answer") or "" for e in trajectory if e.get("event_type") == "answer_attempt"]
    agent_calls = [e for e in trajectory if O.is_agent_tool_call(e)]
    harness_calls = [e for e in trajectory if O.is_harness_tool_call(e)]
    finals = [e for e in trajectory if e.get("event_type") == "final_answer"]
    authors = [final_author(e, attempts) for e in finals]
    final_texts = [e.get("thought", "") for e, a in zip(finals, authors) if a == O.AUTHOR_AGENT]
    source = "agent_final" if final_texts else "none"
    if not any((t or "").strip() for t in final_texts):
        # the loop's no-final fallback, as the loop applies it: protect the agent's EARLIEST answer, else its last valid attempt
        valid = [a for a in attempts if a.strip() and "API_BRAIN_ERROR" not in a]
        if (first_answer or "").strip() and "API_BRAIN_ERROR" not in (first_answer or ""):
            fb, src = first_answer, "fallback_first_answer"
        else:
            fb, src = (valid[-1] if valid else ""), "fallback_last_answer_attempt"
        if (fb or "").strip():
            final_texts, source = [fb], src
    agent_notes = [(e.get("args") or {}).get("content", "") for e in agent_calls if e.get("tool") == "write_file"]
    harness_notes = [(e.get("args") or {}).get("content", "") for e in harness_calls if e.get("tool") == "write_file"]
    files = []
    if outdir and os.path.isdir(outdir):
        for fn in os.listdir(outdir):
            try:
                files.append((fn, open(os.path.join(outdir, fn)).read()))
            except Exception:
                pass
    return {"agent_tool_calls": agent_calls, "harness_tool_calls": harness_calls,
            "final_texts": final_texts, "final_texts_source": source,
            "delivered_final_texts": [e.get("thought", "") for e in finals], "final_authors": authors,
            "final_events": finals,
            "note_texts": agent_notes + [t for _fn, t in files], "harness_note_texts": harness_notes,
            "output_files": sorted(fn for fn, _t in files),
            "agent_trajectory": [e for e in trajectory if _agent_event(e, attempts)],
            "trajectory": trajectory}


def _log_line(ev, canon_tool, extra=None):
    meta = {"tool_name": canon_tool(ev["tool"], ev.get("args")), "input": ev.get("args"),
            "output": (ev.get("result") if isinstance(ev.get("result"), str)
                       else json.dumps(ev.get("result"), ensure_ascii=False))}
    if extra:
        meta.update(extra)
    return json.dumps({"timestamp": datetime.datetime.now().isoformat(), "type": "tool_call",
                       "content": ev["tool"], "metadata": meta}) + "\n"


def write_trajectory_logs(job_dir, views, canon_tool):
    """logs/agent/trajectory.log (upstream format, read by the PhysicianBench native tests) from the agent's own tool
    calls; the harness's tool calls, same line format plus origin / requester / step, in
    logs/harness/harness_tool_calls.jsonl (a name the upstream trajectory.log glob never matches)."""
    with open(os.path.join(job_dir, "logs", "agent", "trajectory.log"), "w") as f:
        for ev in views["agent_tool_calls"]:
            f.write(_log_line(ev, canon_tool))
    hdir = os.path.join(job_dir, "logs", "harness")
    os.makedirs(hdir, exist_ok=True)
    with open(os.path.join(hdir, "harness_tool_calls.jsonl"), "w") as f:
        for ev in views["harness_tool_calls"]:
            f.write(_log_line(ev, canon_tool, {"origin": O.HARNESS, "requester": O.requester_of(ev),
                                               "step": ev.get("step")}))


# ===================================================================================================== state credit
def eval_root(snapshot):
    """The native evaluation root (harness.obligation_gate.obligation_view, what GuiEnvReal.eval_state returns) of a state
    snapshot carrying the portal slices (GuiEnvReal.state_summary); None for any other snapshot."""
    if not isinstance(snapshot, dict) or not any(k in snapshot for k in PORTAL_KEYS):
        return None
    from harness.obligation_gate import obligation_view
    return obligation_view({k: snapshot.get(k) for k in PORTAL_KEYS})


def _leaves(obj, prefix="", out=None, depth=0):
    """{dotted path: text} of the non-empty scalar leaves (non-strings as JSON)."""
    if out is None:
        out = {}
    if depth > 8:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            _leaves(v, "%s.%s" % (prefix, k) if prefix else str(k), out, depth + 1)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _leaves(v, "%s.%d" % (prefix, i) if prefix else str(i), out, depth + 1)
    elif obj is not None and not (isinstance(obj, str) and not obj.strip()):
        out[prefix] = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    return out


def _norm_text(s):
    return " ".join(str(s or "").lower().split())


def _text_match(written, leaf):
    w, v = _norm_text(written), _norm_text(leaf)
    if not w or not v:
        return False
    return w == v or (len(w) >= _MIN_TEXT_MATCH and w in v) or (len(v) >= _MIN_TEXT_MATCH and v in w)


def _state_query_holds(cp, root):
    """The native deterministic checkpoint on this root: True / False, None when it errors. The scorer's own evaluator
    (scoring._ev_deterministic), so the query semantics (contains / expected, root handling) are the native ones."""
    import scoring
    r = scoring._ev_deterministic(cp, {"full_state": root}, {"id": cp.get("id")})
    st = (r or {}).get("checkpoint_status")
    return True if st == "passed" else (False if st == "failed" else None)


def _template_exprs(cp):
    tmpl = (cp.get("check") or {}).get("student_answer")
    if not tmpl:
        return []
    import scoring
    return [m.strip() for m in scoring._TEMPLATE_RE.findall(str(tmpl))]


def _search(expr, root):
    """scoring._resolve_student_answer's lookup on a native root."""
    try:
        import jmespath
        return jmespath.search(expr, dict(root))
    except Exception:
        return None


def authors_to_credit(authors):
    """agent + harness -> mixed; harness (with or without unknown) -> harness; agent (with or without unknown) -> agent;
    otherwise unknown."""
    if HARNESS in authors and AGENT in authors:
        return MIXED
    if HARNESS in authors:
        return HARNESS
    if AGENT in authors:
        return AGENT
    return UNKNOWN


class CheckpointCreditTracker(object):
    """Who first made each state checkpoint true, and who authored each rubric field, from the state after every
    executed action.

    Active only for a task with state checkpoints: deterministic jmespath queries, or gateway_rubric_text rubrics whose
    student_answer is a {{jmespath}} template over the persisted state (HealthAdminBench). The state read is the snapshot
    the executor already takes after every environment call (ActionOutcome.snap_after = GuiEnvReal.state_summary,
    refreshed by the environment's own _snap after each tool), turned into the native evaluation root: no environment
    call is added. The last observation is the root native scoring reads (run.py ctx full_state).

    Limits: a portal write that lands after the environment's post-action read shows up at the next observed action and
    is attributed to it; a transition across an observation where the query could not be evaluated (no snapshot, or a
    query error) is unknown."""

    def __init__(self, task, initial_snapshot=None):
        self.cps = []
        for cp in (task or {}).get("checkpoints") or []:
            chk = cp.get("check") or {}
            if cp.get("type") == "deterministic" and chk.get("method") == "jmespath":
                self.cps.append((cp, "state_query", None))
            elif cp.get("type") == "llm_judge" and chk.get("verifier") == "gateway_rubric_text":
                exprs = _template_exprs(cp)
                if exprs:
                    self.cps.append((cp, "state_text", exprs))
        self.active = bool(self.cps)
        self.obs = []           # {"kind": baseline|action|episode_end, "step", "tool", "origin", "requester", "action_id"}
        self.truth = {cp["id"]: [] for cp, kind, _e in self.cps if kind == "state_query"}
        self.leaf_first = {}    # (cp_id, expr, path, text) -> observation index where that value first appeared there
        self.last_leaves = {}   # cp_id -> {(expr, path): text} at the latest observed root
        self.writes = []        # (observation index, origin, requester, text) of executed type / fill tool calls
        self.field_writer = {}  # dotted persisted field -> observation index after which its value last changed
        self.last_fields = {}
        self.elapsed_s = 0.0
        self.unobserved = 0
        self.final_observed = False
        if self.active:
            snap = initial_snapshot() if callable(initial_snapshot) else initial_snapshot
            self._observe(eval_root(snap), {"kind": "baseline"})

    # -------------------------------------------------------------------------------------------- observation
    def observe(self, ev, outcome):
        """ActionExecutor.build_event's observer: one executed tool call and its outcome."""
        if not self.active:
            return
        t0 = time.time()
        origin = O.HARNESS if O.is_harness_event(ev) else O.AGENT
        rec = {"kind": "action", "step": ev.get("step"), "tool": ev.get("tool"), "origin": origin,
               "requester": ev.get("requester") if origin == O.HARNESS else None, "action_id": ev.get("action_id")}
        args = ev.get("args") if isinstance(ev.get("args"), dict) else {}
        if ev.get("tool") in _WRITE_TOOLS and str(args.get("text") or "").strip() and ev.get("status") == "ok":
            self.writes.append((len(self.obs), origin, rec["requester"], str(args.get("text"))))
        self._observe(eval_root(getattr(outcome, "snap_after", None)), rec)
        self.elapsed_s += time.time() - t0

    def _observe(self, root, rec):
        i = len(self.obs)
        self.obs.append(rec)
        if root is None:
            self.unobserved += 1
            for cid in self.truth:
                self.truth[cid].append(None)
            return
        for cp, kind, exprs in self.cps:
            if kind == "state_query":
                self.truth[cp["id"]].append(_state_query_holds(cp, root))
                continue
            now = {}
            for expr in exprs:
                for path, text in _leaves(_search(expr, root)).items():
                    now[(expr, path)] = text
                    self.leaf_first.setdefault((cp["id"], expr, path, text), i)
            self.last_leaves[cp["id"]] = now
        fields = {}
        for top in FIELD_ROOTS:
            fields.update(_leaves(root.get(top), top))
        for path, text in fields.items():
            if self.last_fields.get(path) != text:
                self.field_writer[path] = i
        self.last_fields = fields

    # -------------------------------------------------------------------------------------------- credit
    def who(self, i):
        o = self.obs[i]
        return {"kind": o.get("kind"), "step": o.get("step"), "tool": o.get("tool"), "origin": o.get("origin"),
                "requester": o.get("requester"), "action_id": o.get("action_id")}

    def finalize(self, final_root):
        """Observe the root native scoring read (the episode end); -> {checkpoint id: credit record}."""
        out = {}
        if not self.active:
            return out
        self.final_observed = final_root is not None
        self._observe(final_root, {"kind": "episode_end"})
        for cp, kind, exprs in self.cps:
            out[cp["id"]] = self._query_credit(cp) if kind == "state_query" else self._text_credit(cp)
        return out

    def _query_credit(self, cp):
        seq = self.truth[cp["id"]]
        rec = {"credit_rule": "state_query_became_true_after", "true_at_end": seq[-1] if seq else None}
        if not seq or seq[-1] is not True:
            if seq and seq[-1] is None:
                why = "query_error_at_end" if self.final_observed else "unobserved_at_end"
            else:
                why = "not_true_at_end"
            rec.update(credit=None, credit_reason=why)
            return rec
        k = len(seq) - 1
        while k >= 0 and seq[k] is True:
            k -= 1
        start = k + 1
        rec["became_true_after"] = self.who(start)
        if k >= 0 and seq[k] is None:
            # the observation before the true run: no state snapshot, or the query errored on it
            rec.update(credit=UNKNOWN, credit_reason="transition_not_determinable")
        elif self.obs[start]["kind"] == "baseline":
            rec.update(credit=UNKNOWN, credit_reason="true_before_first_action")
        elif self.obs[start]["kind"] == "episode_end":
            rec.update(credit=UNKNOWN, credit_reason="true_only_in_final_state")
        else:
            o = self.obs[start]["origin"]
            rec.update(credit=o, credit_reason="became_true_after_%s_action" % o)
        return rec

    def _text_credit(self, cp):
        rec = {"credit_rule": "persisted_text_author"}
        if not self.final_observed:
            rec.update(credit=None, credit_reason="unobserved_at_end")
            return rec
        end = self.last_leaves.get(cp["id"]) or {}
        if not end:
            rec.update(credit=None, credit_reason="empty_at_end")
            return rec
        authors, leaves = set(), []
        for (expr, path), text in sorted(end.items()):
            i = self.leaf_first.get((cp["id"], expr, path, text))
            kind = self.obs[i]["kind"] if i is not None else None
            appeared = self.obs[i].get("origin") if kind == "action" else UNKNOWN
            typed = sorted({o for (_j, o, _r, w) in self.writes if _text_match(w, text)})
            la = {appeared} | set(typed)
            authors |= la
            leaves.append({"field": "%s:%s" % (expr, path), "appeared_after": (self.who(i) if i is not None else None),
                           "typed_by": typed, "authors": sorted(la), "text_sha": _sha(text)})
        rec.update(credit=authors_to_credit(authors), credit_reason="authors:%s" % "+".join(sorted(authors)),
                   leaves=leaves)
        return rec

    def harness_persisted_fields(self):
        """Persisted fields whose current value appeared after a harness action (dotted eval-root paths)."""
        out = []
        for path in sorted(self.last_fields):
            i = self.field_writer.get(path)
            if i is None or self.obs[i].get("kind") != "action" or self.obs[i].get("origin") != O.HARNESS:
                continue
            out.append({"field": path, "step": self.obs[i].get("step"), "tool": self.obs[i].get("tool"),
                        "requester": self.obs[i].get("requester"), "action_id": self.obs[i].get("action_id")})
        return out

    def cost(self):
        return {"observations": len(self.obs), "unobserved": self.unobserved, "extra_env_calls": 0,
                "elapsed_s": round(self.elapsed_s, 4), "state_checkpoints": len(self.cps)}


# ===================================================================================================== PhysicianBench
_PB_TRAJECTORY_READERS = {"load_trajectory", "get_tool_calls", "get_tool_outputs",
                          "get_all_fhir_resources_from_trajectory"}
_OUTPUT_NAME = re.compile(r"^[\w.\-]+\.(?:txt|md|json|csv|html)$")
_PB_SRC_CACHE = {}


def pb_checkpoint_inputs(pb_repo, native_test_ref):
    """What a PhysicianBench native test reads, from its source (static analysis of the test function and the module
    functions it calls): the trajectory log, the output directory (and the file names it names), an LLM judge. None
    when the source cannot be read."""
    if not (pb_repo and native_test_ref and "::" in native_test_ref):
        return None
    rel, func = native_test_ref.split("::", 1)
    path = os.path.join(pb_repo, rel)
    if path not in _PB_SRC_CACHE:
        try:
            _PB_SRC_CACHE[path] = ast.parse(open(path, encoding="utf-8").read())
        except Exception:
            _PB_SRC_CACHE[path] = None
    tree = _PB_SRC_CACHE[path]
    if tree is None:
        return None
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    if func not in funcs:
        return None
    seen, todo, names, strings = set(), [func], set(), []
    while todo:
        fn = todo.pop()
        if fn in seen or fn not in funcs:
            continue
        seen.add(fn)
        for n in ast.walk(funcs[fn]):
            if isinstance(n, ast.Name):
                names.add(n.id)
                if n.id in funcs:
                    todo.append(n.id)
            elif isinstance(n, ast.Attribute):
                names.add(n.attr)
            elif isinstance(n, ast.Constant) and isinstance(n.value, str):
                strings.append(n.value)
    return {"reads_trajectory_log": bool(names & _PB_TRAJECTORY_READERS),
            "reads_output_dir": bool(names & {"read_output_file", "OUTPUT_DIR"}),
            "output_files": sorted({s for s in strings if _OUTPUT_NAME.match(s)}),
            "llm_judge": "llm_judge" in names, "analysis": "static_source"}


def written_files(trajectory, output_files):
    """Per output file: which origins wrote it successfully (write_file, status ok), and whether it exists only through
    harness writes (harness_completed)."""
    by = collections.defaultdict(lambda: {"agent": 0, "harness": 0, "requesters": set()})
    for e in trajectory or []:
        if e.get("event_type") != "tool_call" or e.get("tool") != "write_file" or e.get("status") != "ok":
            continue
        name = write_target(e.get("args"))
        if not name:
            continue
        if O.is_harness_event(e):
            by[name]["harness"] += 1
            by[name]["requesters"].add(O.requester_of(e))
        else:
            by[name]["agent"] += 1
    out = {}
    for name in sorted(set(output_files or ()) | set(by)):
        b = by.get(name) or {"agent": 0, "harness": 0, "requesters": set()}
        out[name] = {"exists_at_end": name in set(output_files or ()), "agent_writes": b["agent"],
                     "harness_writes": b["harness"], "harness_requesters": sorted(b["requesters"]),
                     "harness_completed": bool(name in set(output_files or ()) and b["harness"] and not b["agent"])}
    return out


# ===================================================================================================== per-task credit
def origin_counts(trajectory):
    c = {"tool_call": {"agent": 0, "harness": 0}, "tool_call_harness_by_requester": {},
         "state_changing_tool_call": {"agent": 0, "harness": 0}, "answer_attempt": 0, "final_answer_by_author": {},
         "kernel_records_on_harness_actions": 0}
    attempts = [e.get("answer") or "" for e in trajectory or [] if e.get("event_type") == "answer_attempt"]
    for e in trajectory or []:
        et = e.get("event_type")
        if et == "tool_call":
            o = HARNESS if O.is_harness_event(e) else AGENT
            c["tool_call"][o] += 1
            if o == HARNESS:
                r = O.requester_of(e)
                c["tool_call_harness_by_requester"][r] = c["tool_call_harness_by_requester"].get(r, 0) + 1
            if (e.get("state_record") or {}).get("state_changed") is True:
                c["state_changing_tool_call"][o] += 1
        elif et == "answer_attempt":
            c["answer_attempt"] += 1
        elif et == "final_answer":
            a = final_author(e, attempts)
            c["final_answer_by_author"][a] = c["final_answer_by_author"].get(a, 0) + 1
        elif e.get("action_origin"):
            c["kernel_records_on_harness_actions"] += 1
    return c


def _final_record(views):
    finals = views.get("final_events") or []
    if not finals:
        return {"final_answer_author": None, "final_answer_requester": None, "delivered_sha": None,
                "scored_sha": _sha(" ".join(views.get("final_texts") or [])) if views.get("final_texts") else None,
                "final_texts_source": views.get("final_texts_source")}
    ev, author = finals[-1], (views.get("final_authors") or [O.AUTHOR_AGENT])[-1]
    return {"final_answer_author": author, "final_answer_requester": ev.get("requester"),
            "final_disposition": ev.get("final_disposition"), "delivered_sha": _sha(ev.get("thought") or ""),
            "scored_sha": _sha(" ".join(views.get("final_texts") or [])),
            "final_texts_source": views.get("final_texts_source")}


_FINAL_TEXT_SUBDIMS = ("result_verification", "clinical_task_success", "context_grounding", "evidence_verification",
                       "safety_governance")


def checkpoint_credit(task, bench, results, views, state_credit, files, pb_repo=None):
    """{checkpoint id: credit record}: credit in agent | harness | mixed | unknown (None: the checkpoint's thing was not
    there to credit), credit_reason, input_view."""
    final = _final_record(views)
    author = final["final_answer_author"]
    harness_files = sorted(n for n, f in files.items() if f["harness_completed"])
    status = {r.get("id"): r for r in results or []}
    out = {}
    for cp in (task or {}).get("checkpoints") or []:
        cid, chk, typ = cp.get("id"), (cp.get("check") or {}), cp.get("type")
        rec = None
        if cid in state_credit:
            rec = dict(state_credit[cid], input_view="native_end_state")
        elif typ == "deterministic" and chk.get("method") in ("toolset_contains", "arg_match", "tool_path"):
            rec = {"credit": AGENT, "credit_reason": "scored_on_agent_tool_calls", "input_view": "agent_tool_calls"}
        elif typ == "native_pytest":
            inp = pb_checkpoint_inputs(pb_repo, cp.get("native_test_ref"))
            rec = {"input_view": "agent_trajectory_log+fhir_state+output_dir", "test_inputs": inp}
            if inp is None:
                rec.update(credit=UNKNOWN, credit_reason="test_source_unreadable")
            elif inp["reads_output_dir"] and harness_files and (
                    not inp["output_files"] or set(inp["output_files"]) & set(harness_files)):
                rec.update(credit=HARNESS, credit_reason="reads_output_written_only_by_harness:%s"
                           % ",".join(harness_files))
            else:
                rec.update(credit=AGENT, credit_reason="inputs_agent_only")
        elif typ == "policy" and str(chk.get("verifier") or "").endswith("allergy_exists_for_patient"):
            rec = {"credit": UNKNOWN, "credit_reason": "environment_property", "input_view": "fhir_state"}
        elif typ == "policy" and str(chk.get("verifier") or "").endswith("no_allergy_conflicting_medication_documented"):
            rec = ({"credit": HARNESS, "credit_reason": "note_files_written_only_by_harness:%s" % ",".join(harness_files)}
                   if harness_files else {"credit": AGENT, "credit_reason": "inputs_agent_only"})
            rec["input_view"] = "note_texts"
        elif typ == "policy" and str(chk.get("verifier") or "").startswith("augmentation/drug_safety_check"):
            rec = {"credit": AGENT, "credit_reason": "inputs_agent_only",
                   "input_view": "agent_trajectory|created_meds|final_texts"}
        elif typ == "llm_judge" and (bench == "MedCTA" or cp.get("subdimension") in _FINAL_TEXT_SUBDIMS) \
                and not chk.get("student_answer"):
            if author in (None, O.AUTHOR_AGENT):
                rec = {"credit": AGENT, "credit_reason": "delivered_final_authored_by_agent"
                       if author else "no_delivered_final"}
            else:
                rec = {"credit": HARNESS, "credit_reason": "delivered_final_authored_by_%s" % author}
            rec["input_view"] = "agent_final_texts"
        else:
            rec = {"credit": UNKNOWN, "credit_reason": "reads_full_trajectory_not_split" if typ == "policy"
                   else "no_credit_rule", "input_view": "unchanged"}
        r = status.get(cid) or {}
        rec["checkpoint_status"] = r.get("checkpoint_status")
        rec["native_passed"] = (r.get("pass_status") or r.get("checkpoint_status")) == "passed"
        rec["agent_achieved"] = bool(rec["native_passed"] and rec.get("credit") not in NOT_AGENT_ACHIEVED)
        out[cid] = rec
    return out


def task_credit(task, bench, result, views, tracker=None, final_root=None, outdir=None, pb_repo=None):
    """result["credit"] for one run."""
    traj = views.get("trajectory") or []
    state_credit = tracker.finalize(final_root) if (tracker is not None and tracker.active) else {}
    files = written_files(traj, views.get("output_files"))
    final = _final_record(views)
    cps = checkpoint_credit(task, bench, result.get("checkpoints") or [], views, state_credit, files, pb_repo)
    block = {"version": CREDIT_VERSION, "origin_counts": origin_counts(traj),
             "final_answer_author": final["final_answer_author"], "final_answer": final,
             "harness_written": {"files": sorted(n for n, f in files.items() if f["harness_writes"]),
                                 "files_harness_completed": sorted(n for n, f in files.items() if f["harness_completed"]),
                                 "persisted_fields": ((tracker.harness_persisted_fields() if tracker is not None
                                                       and tracker.active else [])[:MAX_FIELD_RECORDS])},
             "output_files": files, "checkpoints": cps,
             "tracker": (tracker.cost() if tracker is not None and tracker.active else None)}
    counts = collections.Counter()
    for rec in cps.values():
        if rec.get("native_passed"):
            counts[rec.get("credit") or "none"] += 1
    block["native_passed_checkpoints_by_credit"] = dict(counts)
    probe = dict(result, credit=block)
    block["native_outcome"] = native_outcome(probe, bench)
    block["agent_credited_outcome"] = agent_credited_outcome(probe, bench)
    return block


# ===================================================================================================== aggregation hook
def native_outcome(result, bench):
    from aggregate_report import native_task_outcome
    return native_task_outcome(result, bench)


def mask_not_agent_achieved(result):
    """A copy of `result` whose passing checkpoints credited harness or mixed read as not achieved (failed, score 0)."""
    cr = (result.get("credit") or {}).get("checkpoints") or {}
    out = copy.deepcopy({k: v for k, v in result.items() if k != "credit"})
    for c in out.get("checkpoints") or []:
        rec = cr.get(c.get("id")) or {}
        if rec.get("credit") in NOT_AGENT_ACHIEVED and (c.get("pass_status") or c.get("checkpoint_status")) == "passed":
            c["checkpoint_status"] = "failed"
            c["pass_status"] = "failed"
            c["score"] = 0.0
            c["credit_masked"] = rec.get("credit")
    return out


def agent_credited_outcome(result, bench):
    """The native outcome rule (aggregate_report.native_task_outcome) applied to the result with harness- and
    mixed-credited checkpoints not achieved. MedCTA: a delivered final not authored by the agent credits its
    final-text checkpoints to the harness, so the task is not agent-credited. None: the result carries no credit
    record, or no native outcome signal."""
    if not isinstance((result or {}).get("credit"), dict) or "checkpoints" not in result["credit"]:
        return None
    return native_outcome(mask_not_agent_achieved(result), bench)


def _passed_ids(result):
    """Checkpoint ids that passed natively, read from the result's checkpoints as they are now."""
    return {c.get("id") for c in (result or {}).get("checkpoints") or []
            if (c.get("pass_status") or c.get("checkpoint_status")) == "passed"}


def passed_by_credit(result):
    cr = ((result or {}).get("credit") or {}).get("checkpoints") or {}
    out = collections.Counter()
    for cid in _passed_ids(result):
        out[(cr.get(cid) or {}).get("credit") or "no_credit_record"] += 1
    return dict(out)


def credit_row(result, bench):
    cr = (result or {}).get("credit")
    if not isinstance(cr, dict) or "checkpoints" not in cr:
        return {"task_id": (result or {}).get("task_id"), "has_credit_record": False,
                "native_outcome": native_outcome(result, bench), "agent_credited_outcome": None}
    return {"task_id": result.get("task_id"), "has_credit_record": True,
            "native_outcome": native_outcome(result, bench),
            "agent_credited_outcome": agent_credited_outcome(result, bench),
            "final_answer_author": cr.get("final_answer_author"),
            "native_passed_checkpoints_by_credit": passed_by_credit(result),
            "tool_calls_by_origin": (cr.get("origin_counts") or {}).get("tool_call") or {},
            "files_harness_completed": (cr.get("harness_written") or {}).get("files_harness_completed") or []}


def report_block(results, bench):
    """The report field aggregate_report.build writes as "agent_credited_outcome", separate from the native outcome."""
    rows = [credit_row(r, bench) for r in results or []]
    withrec = [r for r in rows if r["has_credit_record"]]
    evaluated = [r for r in withrec if r["agent_credited_outcome"] is not None]
    passed = sum(1 for r in evaluated if r["agent_credited_outcome"])
    native_same = sum(1 for r in evaluated if r["native_outcome"])
    by_credit, authors, calls = collections.Counter(), collections.Counter(), collections.Counter()
    for r in withrec:
        by_credit.update(r["native_passed_checkpoints_by_credit"])
        authors[str(r.get("final_answer_author"))] += 1
        calls.update(r["tool_calls_by_origin"])
    return {"agent_credited_task_success_rate": round(passed / len(evaluated), 3) if evaluated else None,
            "agent_credited_passed": passed, "evaluated": len(evaluated),
            "native_task_success_rate_same_tasks": round(native_same / len(evaluated), 3) if evaluated else None,
            "native_success_not_agent_credited": sorted(r["task_id"] for r in evaluated
                                                        if r["native_outcome"] and not r["agent_credited_outcome"]),
            "tasks_without_credit_record": sorted(str(r["task_id"]) for r in rows if not r["has_credit_record"]),
            "native_passed_checkpoints_by_credit": dict(by_credit), "final_answer_author": dict(authors),
            "tool_calls_by_origin": dict(calls),
            "note": ("Separate from outcome / native_metrics (the native score, unchanged). The same native outcome rule "
                     "on each result with harness- and mixed-credited checkpoints not achieved (runner/scorers/credit.py); "
                     "unknown-credited checkpoints still count and are listed in native_passed_checkpoints_by_credit. "
                     "Results written before the credit record have none and are listed, never guessed.")}


def attribution(result, bench=None):
    """Per task, the scored items credited to the agent vs the harness.

    -> {"task_id", "final_answer_author", "native_outcome", "agent_credited_outcome",
        "agent": [...], "harness": [...], "mixed": [...], "unknown": [...], "not_achieved": [...],
        "harness_completed_files": [...], "harness_persisted_fields": [...]}
    Items are checkpoint ids; "not_achieved" lists the ones that did not pass natively."""
    cr = (result or {}).get("credit") or {}
    out = {"task_id": (result or {}).get("task_id"), "final_answer_author": cr.get("final_answer_author"),
           "has_credit_record": "checkpoints" in cr, "agent": [], "harness": [], "mixed": [], "unknown": [],
           "not_achieved": [],
           "harness_completed_files": (cr.get("harness_written") or {}).get("files_harness_completed") or [],
           "harness_persisted_fields": [f.get("field") for f in (cr.get("harness_written") or {}).get("persisted_fields")
                                        or []]}
    passed = _passed_ids(result)
    for cid, rec in sorted((cr.get("checkpoints") or {}).items()):
        if cid not in passed:
            out["not_achieved"].append(cid)
        else:
            out[rec.get("credit") if rec.get("credit") in CREDITS else UNKNOWN].append(cid)
    if bench is not None:
        out["native_outcome"] = native_outcome(result, bench)
        out["agent_credited_outcome"] = agent_credited_outcome(result, bench)
    return out
