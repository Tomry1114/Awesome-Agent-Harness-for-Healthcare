# -*- coding: utf-8 -*-
"""The shell's completion-obligation engine for GUI tasks: one composition of a task's completion conjunction and one
evaluation of it.

Content and engine. Which conjuncts complete a task is task-helping content in the run's workspace: the file its
manifest names under `completion_obligations` (HAB: workspaces/hab/h0/policies/completion_obligations.yaml, which
documents the format). How a conjunction is composed and evaluated is fixed here, and every reader goes through it:

    terminal lock           obligation_snapshot before and after each action, lock_inputs of the two, read by run.py and
                            the recovery action service -> ledger public_obligations_met / observed_submission_realized
                            -> kernel.after_action
    middleware context      gui_obligations -> ctx.obligations() (middleware_host.obligations_of) -> HAB
                            obligation_nudge
    recovery                completion_conjunction -> HabBenchmarkAdapter.resolve_commitments: verify_all, the submission
                            conjuncts and the early decline (evaluate); GenericGuiCompletionWorkflow reads each conjunct
                            back with conjunct_met (workspaces/hab/h0/recovery)
    submit rule             decision_of / decision_landed -> the harness action service's irreversible-commit check
                            (runner/harness/action_service.py)

Sources. A conjunct is composed from exactly two sources, never from a hidden checkpoint and never from a
non-public task field (the task type is not read):
  * the public goal: `when: {goal_mentions: [phrases]}` (a phrase, case-insensitive) or `{goal_matches: [regexes]}`
    (a regex over the lower-cased goal);
  * the agent's own recorded decisions. A decision is an agent-origin tool call -- a trajectory tool_call event with
    origin "agent", whatever its status, or the agent action being executed (`pending`) -- that meets the decision's
    `match` (the manifest action-match keys tool / tool_any / tool_pattern / ref_pattern, harness.semantics._match_rule,
    over the control the action names with the brackets an agent copies from the page dropped). Every decision the
    agent took composes its `effect` conjunct: the effect the agent chose must land. `when: {agent_decision: <id>}`
    makes a requirement depend on that decision.
  The keys of one `when` must all hold.

Composition (completion_conjunction). The effects of the decisions the agent took come first, in the content's decision
order: they are the submission conjuncts (each carries "decision"), which the terminal lock reads as observed
submissions. The `requires` entries whose `when` holds follow in declared order. A conjunct already composed is not
repeated. `{payer}` in a path is the payer the content's `payer` table resolves from the public goal (the first payer
whose phrases the goal contains, else its default); every path of the task uses that one payer. A task whose agent has
decided nothing and whose goal names no requirement composes the empty conjunction; a workspace whose manifest names no
completion_obligations file composes None (no obligation model).

Evaluation (conjunct_met). A conjunct reads its path, then its alt_paths, and stops at the first truthy value; its check
is truthy, nonempty (length >= min_len, default 1) or equals `value`. A conjunction is met when every conjunct holds.

Terminal-lock inputs (lock_inputs). public_obligations_met: the composition is non-empty and every conjunct holds (an
empty composition is not a completed task, so the lock never engages on it). observed_submission_realized: a submission
conjunct that did not hold before the action holds after it. Both None where there is no obligation model.

State view (obligation_view). Every reader evaluates against ONE view built from persisted portal state, so the readers
agree on each conjunct's truth, not only on the conjunct list: GuiEnvReal.eval_state IS obligation_view of the portal
(the lock and ctx.obligations() read it), and recovery evaluates its read-back of the portal through
obligation_view(readback_portals(...)) (the workflow's conjunct checks and the early decline).

Content is read from the workspace on every call (cached until the manifest or the file changes) and validated; content
the engine cannot compose raises CompletionObligationsError, which Workspace.check reports before a run starts. Every
function that reads the content takes the run's workspace as a required argument (None: no workspace, so no obligation
model); none resolves a default.

Safety obligations are not here: they are the kernel's policy modules (runner/policies). Oracle-blind: the goal, the
agent's own trajectory and the portal state are public; no gold or checkpoint value is read.
"""
import os
import re

MANIFEST_KEY = "completion_obligations"
CHECKS = ("truthy", "nonempty", "equals")
PAYER = "{payer}"
WHEN_KEYS = ("goal_mentions", "goal_matches", "agent_decision")
MATCH_KEYS = ("tool", "tool_any", "tool_pattern", "ref_pattern")

_TOP_KEYS = ("name", "payer", "decisions", "conjuncts", "requires")
_CONJUNCT_KEYS = ("path", "check", "alt_paths", "min_len", "value")
_DECISION_KEYS = ("match", "effect")
_CACHE = {}


class CompletionObligationsError(ValueError):
    """A workspace's completion_obligations content that the engine cannot compose."""


class Snapshot(tuple):
    """Per-conjunct truth of a composed conjunction, in conjunct order. The first `submissions` entries are the effects
    of the agent's decisions (the outward submissions the terminal lock observes)."""

    def __new__(cls, truth=(), submissions=0):
        self = tuple.__new__(cls, truth)
        self.submissions = int(submissions)
        return self


# ------------------------------------------------------------------------------------------------------- evaluation
def _dig(state, dotted):
    node = state
    for seg in [p for p in str(dotted or "").split(".") if p]:
        if isinstance(node, dict):
            node = node.get(seg)
        elif isinstance(node, list):
            try:
                node = node[int(seg)]
            except (ValueError, IndexError, TypeError):
                return None
        else:
            return None
    return node


def conjunct_met(conjunct, state):
    """Does one conjunct hold in `state`? The first of path / alt_paths holding a truthy value is read."""
    got = None
    for path in [conjunct.get("path")] + list(conjunct.get("alt_paths") or []):
        got = _dig(state, path)
        if got:
            break
    check = conjunct.get("check")
    if check == "nonempty":
        return bool(got) and (not hasattr(got, "__len__") or len(got) >= int(conjunct.get("min_len", 1)))
    if check == "equals":
        return got == conjunct.get("value")
    return bool(got)


def evaluate(conjuncts, state):
    """Per-conjunct truth of a conjunction over one state, in conjunct order."""
    return tuple(bool(conjunct_met(c, state)) for c in conjuncts)


# ------------------------------------------------------------------------------------------------------ state view
def obligation_view(portals):
    """The one state view a completion conjunction is evaluated against, from persisted portal state: the portals_state
    slices {"emr", "payerA", "payerB", "fax"} (a missing or non-dict slice is empty). It is the native HealthAdminBench
    evaluator root (get_final_state): full_state is the EMR slice with a non-empty fax slice as faxPortal; signals are
    the snake_case flags derived from the EMR's agentActions; payer_{a,b}_state (aliases aetna_state / anthem_state)
    expose a payer slice where the native checkpoints query it. GuiEnvReal.eval_state returns this view; recovery builds
    it from its read-back (readback_portals)."""
    p = portals if isinstance(portals, dict) else {}

    def slice_(key):
        return p.get(key) if isinstance(p.get(key), dict) else {}
    full_state = dict(slice_("emr"))
    fax = slice_("fax")
    if fax:
        full_state["faxPortal"] = fax
    payer_a, payer_b = _payer_view(slice_("payerA")), _payer_view(slice_("payerB"))
    aa = full_state.get("agentActions") if isinstance(full_state.get("agentActions"), dict) else {}
    actions = {"history": [],
               "visited_pages": aa.get("visitedPages") if isinstance(aa.get("visitedPages"), list) else [],
               "viewed_documents": aa.get("viewedDocuments") if isinstance(aa.get("viewedDocuments"), list) else []}
    return {"signals": _signals(full_state), "actions": actions, "actions_history": [],
            "full_state": full_state, "payer_a_state": payer_a, "payer_b_state": payer_b,
            "aetna_state": payer_a, "anthem_state": payer_b}


def readback_portals(readback):
    """The portals_state slices of a recovery read-back (the harness action service's read_recovery_state view:
    full_state = the EMR slice, payer_a_state / payer_b_state, fax; its draft_ui_state is not persisted state)."""
    r = readback if isinstance(readback, dict) else {}
    return {"emr": r.get("full_state"), "payerA": r.get("payer_a_state"), "payerB": r.get("payer_b_state"),
            "fax": r.get("fax")}


def _signals(full_state):
    """Native HAB environment._build_signals: the snake_case signals view derived from emr.agentActions, so signals.*
    checkpoints (clicked_diagnoses_tab / viewed_auth_letter / downloaded_auth_letter / submitted ...) resolve."""
    aa = full_state.get("agentActions", {}) if isinstance(full_state, dict) else {}
    if not isinstance(aa, dict):
        aa = {}
    return {
        "read_clinical_note": bool(aa.get("readClinicalNote")),
        "viewed_auth_letter": bool(aa.get("viewedAuthLetter")),
        "downloaded_auth_letter": bool(aa.get("downloadedAuthLetter")),
        "downloaded_auth_letter_filename": aa.get("downloadedAuthLetterFilename"),
        "downloaded_clinical_note": bool(aa.get("downloadedClinicalNote")),
        "downloaded_clinical_note_filename": aa.get("downloadedClinicalNoteFilename"),
        "clicked_go_to_portal": bool(aa.get("clickedGoToPortal")),
        "clicked_coverages_tab": bool(aa.get("clickedCoveragesTab")),
        "clicked_diagnoses_tab": bool(aa.get("clickedDiagnosesTab")),
        "clicked_services_tab": bool(aa.get("clickedServicesTab")),
        "clicked_referral_tab": bool(aa.get("clickedReferralTab")),
        "submitted": bool(aa.get("submitted") or aa.get("submittedAppeal")),
    }


def _payer_view(raw_state):
    """Native HAB environment._build_payer_state: a payer portal's localStorage slice under the shape the native
    JMESPath checkpoints query (config / full_state.appealActions / differences.{priorAuth,authSearches,
    eligibilityChecks})."""
    state = raw_state if isinstance(raw_state, dict) else {}
    submissions = state.get("submissions") if isinstance(state.get("submissions"), list) else []
    auth_searches = state.get("authSearches") if isinstance(state.get("authSearches"), list) else []
    eligibility_checks = state.get("eligibilityChecks") if isinstance(state.get("eligibilityChecks"), list) else []
    appeal_actions = state.get("appealActions")
    if not isinstance(appeal_actions, dict):
        fb = state.get("agentActions")
        appeal_actions = fb if isinstance(fb, dict) else {}
    added = {"priorAuth": submissions[-1]} if submissions else {}
    return {"config": state.get("initialState", {}),
            "full_state": {"appealActions": appeal_actions, "agentActions": appeal_actions},
            "agentActions": appeal_actions,
            "initialfinaldiff": {"added": added, "updated": {}, "removed": {}},
            "differences": {"priorAuth": {"added": submissions},
                            "authSearches": auth_searches, "eligibilityChecks": eligibility_checks}}


# ---------------------------------------------------------------------------------------------------------- content
def _stamp(path):
    try:
        st = os.stat(path)
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def load_completion_obligations(workspace):
    """The workspace's completion-obligation content, validated and normalized:
    {"payer": {"default", "goal_mentions": [(payer, phrases)]}, "decisions": {id: {"match", "effect"}},
     "conjuncts": {id: conjunct}, "requires": [(id, when or None)]}, `when` = {"goal_mentions": (phrases),
    "goal_matches": (compiled), "agent_decision": id} with only the keys the entry declares.
    None where the run has no workspace (None) or its manifest names no file (no obligation model)."""
    if not workspace:
        return None
    from workspace import Workspace
    ws = os.path.abspath(workspace)
    manifest_stamp = _stamp(os.path.join(ws, "manifest.yaml"))
    hit = _CACHE.get(ws)
    if hit is not None and hit[0] == manifest_stamp and (hit[1] is None or hit[2] == _stamp(hit[1])):
        return hit[3]
    path = Workspace(ws).completion_obligations_file()
    if path is None:
        content, stamp = None, None
    else:
        if not os.path.isfile(path):
            raise CompletionObligationsError("%s: completion_obligations file %s missing" % (ws, path))
        stamp = _stamp(path)
        import yaml
        try:
            doc = yaml.safe_load(open(path, encoding="utf-8"))
        except yaml.YAMLError as e:
            raise CompletionObligationsError("%s: not YAML: %s" % (path, e))
        content = _normalize(doc, path)
    _CACHE[ws] = (manifest_stamp, path, stamp, content)
    return content


def _phrases(ph, where, errors):
    if not isinstance(ph, list) or not ph or not all(isinstance(p, str) and p for p in ph):
        errors.append("%s must be a non-empty list of phrases" % where)
        return None
    return tuple(p.lower() for p in ph)


def _patterns(ps, where, errors):
    if not isinstance(ps, list) or not ps or not all(isinstance(p, str) and p for p in ps):
        errors.append("%s must be a non-empty list of regexes" % where)
        return None
    out = []
    for p in ps:
        try:
            out.append(re.compile(p))
        except re.error as e:
            errors.append("%s: %r is not a regex: %s" % (where, p, e))
            return None
    return tuple(out)


def _when(when, where, decisions, errors):
    if not isinstance(when, dict) or not when or set(when) - set(WHEN_KEYS):
        errors.append("%s: when must map some of %s" % (where, ", ".join(WHEN_KEYS)))
        return None
    out = {}
    if "goal_mentions" in when:
        out["goal_mentions"] = _phrases(when["goal_mentions"], "%s.goal_mentions" % where, errors)
    if "goal_matches" in when:
        out["goal_matches"] = _patterns(when["goal_matches"], "%s.goal_matches" % where, errors)
    if "agent_decision" in when:
        did = when["agent_decision"]
        if not isinstance(did, str) or did not in decisions:
            errors.append("%s: agent_decision must name a decision id, got %r" % (where, did))
        out["agent_decision"] = did
    return out


def _normalize(doc, where):
    errors = []
    if not isinstance(doc, dict):
        raise CompletionObligationsError("%s: not a mapping" % where)
    for k in sorted(set(doc) - set(_TOP_KEYS)):
        errors.append("unknown key %r" % k)

    payer = doc.get("payer") or {}
    default, mentions = None, []
    if not isinstance(payer, dict) or set(payer) - {"default", "goal_mentions"}:
        errors.append("payer must be {default, goal_mentions}")
    else:
        default = payer.get("default")
        if default is not None and not isinstance(default, str):
            errors.append("payer.default must be a string")
        gm = payer.get("goal_mentions") or {}
        if not isinstance(gm, dict):
            errors.append("payer.goal_mentions must map a payer to phrases")
        else:
            for name, ph in gm.items():
                p = _phrases(ph, "payer.goal_mentions.%s" % name, errors)
                if p is not None:
                    mentions.append((str(name), p))

    conjuncts = {}
    raw_conjuncts = doc.get("conjuncts") or {}
    if not isinstance(raw_conjuncts, dict):
        errors.append("conjuncts must map an id to a conjunct")
        raw_conjuncts = {}
    for cid, c in raw_conjuncts.items():
        at = "conjuncts.%s" % cid
        if not isinstance(c, dict):
            errors.append("%s: not a mapping" % at)
            continue
        for k in sorted(set(c) - set(_CONJUNCT_KEYS)):
            errors.append("%s: unknown key %r" % (at, k))
        alt = c.get("alt_paths") or []
        paths = [c.get("path")] + (list(alt) if isinstance(alt, list) else [None])
        if not all(isinstance(p, str) and p for p in paths):
            errors.append("%s: path and alt_paths must be non-empty strings" % at)
            continue
        for p in paths:
            if "{" in p.replace(PAYER, "") or "}" in p.replace(PAYER, ""):
                errors.append("%s: %r: the only placeholder is %s" % (at, p, PAYER))
            elif PAYER in p and default is None:
                errors.append("%s: %r names %s but payer has no default" % (at, p, PAYER))
        if c.get("check") not in CHECKS:
            errors.append("%s: check must be one of %s" % (at, ", ".join(CHECKS)))
        if "min_len" in c and (c.get("check") != "nonempty" or not isinstance(c["min_len"], int)):
            errors.append("%s: min_len is an integer of a nonempty check" % at)
        if (c.get("check") == "equals") != ("value" in c):
            errors.append("%s: value goes with check equals" % at)
        out = {"path": c["path"], "check": c.get("check"), "alt_paths": list(alt)}
        for k in ("min_len", "value"):
            if k in c:
                out[k] = c[k]
        conjuncts[cid] = out

    decisions = {}
    raw_decisions = doc.get("decisions") or {}
    if not isinstance(raw_decisions, dict):
        errors.append("decisions must map an id to a decision")
        raw_decisions = {}
    for did, d in raw_decisions.items():
        at = "decisions.%s" % did
        if not isinstance(d, dict):
            errors.append("%s: not a mapping" % at)
            continue
        for k in sorted(set(d) - set(_DECISION_KEYS)):
            errors.append("%s: unknown key %r" % (at, k))
        m = d.get("match")
        if not isinstance(m, dict) or not m or set(m) - set(MATCH_KEYS) \
                or not (m.get("tool") or m.get("tool_any") or m.get("tool_pattern")):
            errors.append("%s: match must declare a tool (tool / tool_any / tool_pattern) and only %s"
                          % (at, ", ".join(MATCH_KEYS)))
            continue
        if "ref_pattern" in m:
            try:
                re.compile(m["ref_pattern"])
            except (re.error, TypeError) as e:
                errors.append("%s: ref_pattern is not a regex: %s" % (at, e))
                continue
        if not isinstance(d.get("effect"), str) or d.get("effect") not in conjuncts:
            errors.append("%s: effect must name a conjunct id, got %r" % (at, d.get("effect")))
            continue
        decisions[str(did)] = {"match": dict(m), "effect": d["effect"]}

    requires = []
    raw_req = doc.get("requires") or []
    if not isinstance(raw_req, list):
        errors.append("requires must be a list")
        raw_req = []
    for i, r in enumerate(raw_req):
        rat = "requires[%d]" % i
        if isinstance(r, str):
            cid, when = r, None
        elif isinstance(r, dict) and "conjunct" in r and not set(r) - {"conjunct", "when"}:
            cid = r["conjunct"]
            when = _when(r["when"], rat, raw_decisions, errors) if "when" in r else None
        else:
            errors.append("%s: an entry is a conjunct id or {conjunct, when}" % rat)
            continue
        if not isinstance(cid, str) or cid not in conjuncts:
            errors.append("%s: unknown conjunct %r" % (rat, cid))
            continue
        requires.append((cid, when))

    if errors:
        raise CompletionObligationsError("%s: %s" % (where, "; ".join(errors)))
    return {"payer": {"default": default, "goal_mentions": mentions}, "decisions": decisions,
            "conjuncts": conjuncts, "requires": requires}


# ------------------------------------------------------------------------------------------------------ composition
def action_control(action):
    """The control an action names (ref / bid / target / selector / element_id), with surrounding brackets dropped."""
    from .semantics import _action_ref
    return _action_ref(action).strip().strip("[]").strip()


def _meets(match, action):
    from .semantics import _match_rule
    tool = str((action or {}).get("tool") or "")
    return bool(_match_rule(tool, match, {"tool": tool, "args": {"ref": action_control(action)}}))


def agent_actions(actions=None, pending=None):
    """The agent's own tool calls: trajectory tool_call events with origin "agent" (any status), then `pending`, the agent
    action being executed and not yet recorded (a harness read, `_harness_acquire`, is not the agent's)."""
    out = [e for e in (actions or []) if isinstance(e, dict) and e.get("event_type") == "tool_call"
           and e.get("origin") == "agent"]
    if isinstance(pending, dict) and pending.get("tool") and not pending.get("_harness_acquire"):
        out.append(pending)
    return out


def decisions_taken(content, actions=None, pending=None):
    """The ids of the content's decisions the agent took, in the content's decision order."""
    acts = agent_actions(actions, pending)
    return tuple(did for did, d in content["decisions"].items() if any(_meets(d["match"], a) for a in acts))


def decision_of(action, workspace):
    """The id of the first decision whose match an action meets, whoever issues it; None without one (or without an
    obligation model)."""
    content = load_completion_obligations(workspace)
    for did, d in ((content or {}).get("decisions") or {}).items():
        if _meets(d["match"], action):
            return did
    return None


def _goal(task):
    return str(task.get("goal") or "").lower() if isinstance(task, dict) else ""


def _payer(content, goal):
    payer = content["payer"]["default"]
    for name, phrases in content["payer"]["goal_mentions"]:
        if any(p in goal for p in phrases):
            return name
    return payer


def _holds(when, goal, taken):
    if not when:
        return True
    if "goal_mentions" in when and not any(p in goal for p in when["goal_mentions"]):
        return False
    if "goal_matches" in when and not any(r.search(goal) for r in when["goal_matches"]):
        return False
    if "agent_decision" in when and when["agent_decision"] not in taken:
        return False
    return True


def _instance(conjunct, payer, cid):
    out = dict(conjunct)
    fill = (lambda p: p.replace(PAYER, payer)) if payer is not None else (lambda p: p)
    out["path"] = fill(conjunct["path"])
    out["alt_paths"] = [fill(p) for p in conjunct["alt_paths"]]
    out["id"] = cid
    return out


def completion_conjunction(task, workspace, actions=None, pending=None):
    """(decisions, conjuncts): the ids of the decisions the agent took, and the task's completion conjunction as the
    workspace's content composes it from the public goal and those decisions. Each conjunct is a fresh
    {"id", "path", "check", "alt_paths"[, "min_len"][, "value"][, "decision"]} dict; a submission conjunct (the effect
    of a decision) carries "decision". conjuncts is None where the workspace names no completion content."""
    content = load_completion_obligations(workspace)
    if content is None:
        return (), None
    goal = _goal(task)
    payer = _payer(content, goal)
    taken = decisions_taken(content, actions, pending)
    conj, seen = [], set()
    for did in taken:
        cid = content["decisions"][did]["effect"]
        if cid not in seen:
            seen.add(cid)
            c = _instance(content["conjuncts"][cid], payer, cid)
            c["decision"] = did
            conj.append(c)
    for cid, when in content["requires"]:
        if cid not in seen and _holds(when, goal, taken):
            seen.add(cid)
            conj.append(_instance(content["conjuncts"][cid], payer, cid))
    return taken, conj


# ---------------------------------------------------------------------------------------------------------- readers
def _eval_state(env):
    try:
        return env.eval_state() if hasattr(env, "eval_state") else (getattr(env, "full_state", None) or {})
    except Exception:
        return getattr(env, "full_state", None) or {}


def gui_obligations(task, env, workspace, actions=None, pending=None):
    """(met, decisions, conjuncts): the task's completion conjunction evaluated over ONE read of the env's eval_state;
    (True, decisions, None) where nothing is composed (nothing to gate). The shell's obligation model behind the
    middleware context's obligations()."""
    taken, conj = completion_conjunction(task, workspace, actions, pending)
    if not conj:
        return True, taken, None
    truth = evaluate(conj, _eval_state(env))
    return all(truth), taken, truth


def obligation_snapshot(task, env, workspace, actions=None, pending=None):
    """The terminal lock's Snapshot of the task's completion conjunction; None where the workspace names no completion
    content (no obligation model), an empty Snapshot where nothing is composed yet."""
    taken, conj = completion_conjunction(task, workspace, actions, pending)
    if conj is None:
        return None
    if not conj:
        return Snapshot(())
    return Snapshot(evaluate(conj, _eval_state(env)), sum(1 for c in conj if c.get("decision")))


def lock_inputs(before, after):
    """(public_obligations_met, observed_submission_realized) from the snapshots taken before and after one action.
    (None, None) without an obligation model. An empty composition is not met. A submission is observed when a
    submission conjunct that did not hold before holds after (before counts only when it has the same shape)."""
    if after is None:
        return None, None
    subs = int(getattr(after, "submissions", 0) or 0)
    b = before if (before is not None and len(before) == len(after)
                   and int(getattr(before, "submissions", 0) or 0) == subs) else None
    met = bool(len(after)) and all(after)
    realized = any(after[i] and not (b[i] if b is not None else False) for i in range(subs))
    return bool(met), bool(realized)


def decision_landed(task, env, action, workspace):
    """Has the effect of the decision `action` names landed? The decision's effect conjunct, with the goal's payer, over
    the env's obligation view -> True / False; None where the action names no decision (landing undecidable)."""
    content = load_completion_obligations(workspace)
    did = decision_of(action, workspace)
    if content is None or did is None:
        return None
    cid = content["decisions"][did]["effect"]
    c = _instance(content["conjuncts"][cid], _payer(content, _goal(task)), cid)
    return bool(conjunct_met(c, _eval_state(env)))
