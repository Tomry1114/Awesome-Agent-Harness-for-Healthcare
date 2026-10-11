# -*- coding: utf-8 -*-
"""HealthAdminBench H0 middleware: the attachment check before a send (host before_tool, after the pre-commit audit).

Send-readiness. 'Downloaded subset attached' is NOT a universal constraint -- fax-easy-2 explicitly says 'attach the
three required documents and do NOT attach any other documents', so forcing every downloaded file (mis-download /
wrong version / other patient / help PDF) to be attached causes wrong-document failure and over-disclosure. So this
NEVER names a specific document to attach (which one is the agent's / rubric's call). It only flags the one SAFE,
unambiguous process error: a terminal commit on the fax portal with files DOWNLOADED but NOTHING attached. It fires
before the irreversible send, at most MAX_ATTACH times; after that the send goes through. The refusal does not unlock
the repair budget (Nudge). Harness reads are not checked. No gold.
"""
from workspace_api import is_terminal_action
from workspace_api import Nudge

MAX_ATTACH = 2


def terminal_commit(ctx, action):
    """(is the action a terminal commit, the live controls it was classified against: ctx.controls(), the shell's
    read-only control model of the current page)."""
    if action.get("tool") not in ("submit", "click"):
        return False, None
    try:
        controls = ctx.controls()
    except Exception:
        controls = []
    try:
        return bool(is_terminal_action(action.get("tool"), (action.get("args") or {}).get("ref"), controls)), controls
    except Exception:
        return False, controls


def send_readiness_gap(downloaded, attached):
    """True (NOT ready to send) only in the safe case: the agent has downloaded document(s) but attached none.
    Never forces a specific document -> never causes wrong-doc / over-disclosure. Oracle-blind."""
    return bool(downloaded) and not (attached or [])


def feedback():
    """A generic before-send nudge -- never names which document (the agent chooses which to attach)."""
    return ("You are about to SEND, but you have downloaded document(s) and attached NONE. Attach the "
            "document(s) this submission requires before you send -- and only the correct ones.")


class AttachmentBlock(object):
    def __init__(self, task):
        self.rounds = 0

    def before_tool(self, ctx, action):
        if self.rounds >= MAX_ATTACH or action.get("_harness_acquire") or not terminal_commit(ctx, action)[0]:
            return None
        try:
            downloaded = ctx.downloads()                  # the file names the environment downloaded, in order
            state = ctx.persisted_state()                 # the persisted portal state view (None: the env has none)
            state = {} if state is None else state
            fax = ((state.get("full_state") or state).get("faxPortal") or {}) if isinstance(state, dict) else {}
            not_ready = bool(fax) and send_readiness_gap(downloaded, fax.get("attachmentNames") or [])
        except Exception:
            not_ready = False
        if not not_ready:
            return None
        self.rounds += 1
        ctx.emit({"event_type": "attachment_block", "status": "ok", "step": ctx.step, "round": self.rounds})
        return Nudge("attachment", feedback(), "attachment_block", "REVISE")
