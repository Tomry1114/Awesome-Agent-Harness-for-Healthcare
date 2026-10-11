"""Opportunistic read-evidence acquisition for prior-auth (surface-local, ONLINE).

The same mechanism as the remittance acquire in recovery, but fired ONLINE during the agent's run instead of at the
final/force-stop boundary: every turn, if the CURRENT page shows an UNMET public read-only evidence control
(coverages / services / diagnoses tab, clinical note, auth letter -- each one reveals real coverage/clinical data),
the harness dispatches ONE read-only click, then the agent gets the fresh observation back. No cross-page planning,
no churn, read-only, oracle-blind.

POSTCONDITION CLOSURE (not history de-dup): a flag is "done" only when the LIVE state flag actually flipped
True -- an attempted-but-not-landed click is retried up to MAX_ATTEMPTS, then abandoned (never counted as
success). Same standard as the remittance path: not "clicked" but "viewed flag really flipped".

ReadAcquire at the bottom is how the shell (runner/middleware_host.py) calls it: host next_read.
"""
import re

import workspace_api as _W
from workspace_api import portal_agent_actions as _agent_actions
from workspace_api import Read

_hab = _W.load_relative(__file__, "../recovery/hab.py")   # the HAB task classification and public source mapping

MAX_ATTEMPTS = 2


def _snake(name):
    """clickedDiagnosesTab -> clicked_diagnoses_tab"""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", str(name or "")).lower()


def _dig(node, dotted):
    for seg in [p for p in str(dotted or "").split(".") if p]:
        if not isinstance(node, dict):
            return None
        node = node.get(seg)
    return node


def flag_is_set(state_view, flag):
    """Is a read-evidence flag observably TRUE in the live portal state?

    The portal keeps these under `signals` in snake_case (signals.clicked_diagnoses_tab, 301 occurrences
    across the task definitions) while this module names them in camelCase; a lookup of agentActions[camelCase]
    alone would never match and the postcondition would never close. Try every form the real state uses, signals
    first, then agentActions -- the same both-spellings approach the completion obligations content takes
    (policies/completion_obligations.yaml).
    """
    if not isinstance(state_view, dict) or not flag:
        return False
    snake = _snake(flag)
    names = [snake, flag]
    roots = [state_view] + [state_view.get(w) for w in ("_live_full_state", "full_state")]
    for root in roots:
        if not isinstance(root, dict):
            continue
        for container in ("signals", "agentActions"):
            for nm in names:
                if _dig(root, "%s.%s" % (container, nm)):
                    return True
    aa = _agent_actions(state_view) or {}
    return bool(aa.get(flag) or aa.get(snake))


def _norm(s):
    # NOTE: intentionally the SPACED variant (squash whitespace/-/_ to a single space, keep word
    # boundaries). grounding_audit._norm and collection._norm squash to "" with different char classes —
    # do NOT unify them (different behaviour on purpose).
    return re.sub(r"[\s\-_]+", " ", str(s or "")).strip().lower()


def next_read_acquire(sources, controls, state_view, attempts):
    """Return {flag, ref, op, label} for the FIRST unmet read-evidence flag whose control is visible on the
    CURRENT page, or None. `sources` = [{flag, nav_keywords, route}] (raw.acquire_read_sources). `attempts` is a
    caller-owned dict {flag: n} for postcondition-closure retry capping. Read-only; never returns a control that
    mutates business state."""
    if not sources:
        return None
    for src in sources:
        flag = src.get("flag")
        if not flag:
            continue
        if flag_is_set(state_view, flag):                 # POSTCONDITION: flag already flipped True -> done
            continue
        if attempts.get(flag, 0) >= MAX_ATTEMPTS:         # attempted MAX times, never landed -> abandon (no success)
            continue
        kws = [_norm(k) for k in (src.get("nav_keywords") or []) if k]
        for c in (controls or []):
            if str(c.get("role") or "").lower() not in ("button", "link", "a", "tab", "menuitem"):
                continue
            if c.get("visible") is False or c.get("disabled"):
                continue
            lab = _norm(c.get("label"))
            if lab and kws and any(k in lab for k in kws):
                op = "download" if re.search(r"download|export|save.*pdf|get document", str(c.get("label") or ""), re.I) else "click"
                return {"flag": flag, "ref": c.get("ref"), "op": op, "label": c.get("label")}
    return None


def sources_for_task(task):
    """The prior-auth read-evidence acquire sources [{flag, nav_keywords, route}] for a task, or [] for a
    non-prior-auth family. Oracle-blind: public portal naming only (via the adapter's mapping)."""
    if _hab._family(task) != "prior_auth":
        return []
    flags = ["clickedDiagnosesTab", "clickedServicesTab", "clickedCoveragesTab", "readClinicalNote"]
    if _hab._goal_needs_attachment(task):
        flags.append("downloadedAuthLetter")
    return _hab._decision_readiness_sources(flags) or []


# ------------------------------------------------------------------ middleware hook (runner/middleware_host.py)
class ReadAcquire(object):
    """At turn start, if the current page shows an unmet read-evidence control, ask the shell for ONE read-only
    click/download on it in place of the agent's turn (dispatch, budget exemption, provenance and record_execution
    are the shell's). Never while the tool budget is spent, and never once repair has started."""

    def __init__(self, task):
        self.sources = None          # the task's read-evidence sources, resolved on first use
        self.attempts = {}           # {flag: dispatches}: the postcondition-closure retry cap

    def next_read(self, ctx):
        if ctx.budget_spent_at_turn_start or ctx.in_repair:
            return None
        try:
            if self.sources is None:
                self.sources = sources_for_task(ctx.task)
            if self.sources:
                controls = ctx.controls()
                # eval_state(), not state_summary(): the read-evidence flags live under `signals`, which only
                # eval_state assembles. state_summary returns {route, emr, payerA, payerB, fax}, so with it the
                # postcondition would never find a flag and every item would burn both retries.
                state = ctx.persisted_state()
                target = next_read_acquire(self.sources, controls or [], state or {}, self.attempts)
                if target is not None:
                    flag = target["flag"]
                    self.attempts[flag] = self.attempts.get(flag, 0) + 1
                    ctx.emit({"step": ctx.step, "event_type": "read_acquire_dispatch", "evidence_flag": flag,
                              "ref": target["ref"], "op": target["op"], "attempt": self.attempts[flag],
                              "status": "ok"})
                    return Read(target["op"], {"ref": target["ref"], "provenance": "harness-acquired-evidence",
                                               "evidence_flag": flag})
        except Exception as exc:
            ctx.error("read_acquire: %r" % (exc,))
        return None
