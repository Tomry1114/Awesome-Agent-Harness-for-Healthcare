"""Content-preservation guard: a harness action must never destroy the ROOT agent's authored text (triage note /
rationale). The native HAB portal clears the triage note on a surface switch, so a recovery navigation could wipe the
agent's note (failing the documentation checkpoints). Oracle-blind: the agent's OWN prior value is written back.

The guard is the shell's post-action guard for every recovery-origin action (runner/harness/action_service.py): the
restore goes through the executor and is recorded. The run-level behaviour is pinned in
runner/tests/test_harness_action_service.py; these are the detection and restore units."""
import canonical_schema
from harness.action_service import HarnessActionService
from workspace import H0
from harness.executor import ActionExecutor


class _ClearingEnv:
    """Simulates the portal quirk: any surface switch clears triageNotes."""
    page = None

    def __init__(self, clear=True):
        self.clear = clear
        self.emr = {"triageNotes": "Agent reasoning: CO-50 non-covered; route to appeals.",
                    "agentActions": {"selectedDisposition": "Route to Clinical Appeals"}}
        self.payerA = {"appealActions": {"submittedRationale": "medical necessity established"}}
        self.restored_calls = []

    def call_tool(self, name, args):
        if self.clear and name in ("navigate", "click"):
            self.emr["triageNotes"] = ""
        return {"ok": True, "observation": "switched"}

    def reconcile_write(self, name, args, result):
        return {"confirmed": None, "detail": "not_reconcilable"}

    def restore_persisted_content(self, lost):
        self.restored_calls.append(dict(lost))
        for k, v in lost.items():
            if k == "full_state.triageNotes":
                self.emr["triageNotes"] = v
        return len(lost)


def _service(env, trajectory, controls=()):
    charged = []
    ex = ActionExecutor(canonical_schema, "gui", lambda e: None, lambda e: None)
    svc = HarnessActionService(None, ex, env, {}, trajectory, 12, agent_actions=lambda: 0, reserve=lambda: 2, errors=[],
                               workspace=H0["gui"])
    charge = svc._charge

    def counted(requester, kind):              # the channel's own counter, observed
        charged.append(requester)
        charge(requester, kind)
    svc._charge = counted
    svc.read_recovery_state = lambda: {"full_state": env.emr, "payer_a_state": env.payerA, "payer_b_state": {},
                                       "draft_ui_state": {"Triage Note": "a draft is not persisted state"}}
    svc.list_controls = lambda: list(controls)
    return svc, charged


def test_snapshot_captures_content_fields_only():
    svc, _ = _service(_ClearingEnv(), [])
    snap = svc._content_snapshot()["persisted"]
    assert "full_state.triageNotes" in snap
    assert "payer_a_state.appealActions.submittedRationale" in snap
    assert not any("selectedDisposition" in k for k in snap)
    assert not any(k.startswith("draft_ui_state") for k in snap)


def test_navigation_clear_is_restored_and_recorded():
    env, traj = _ClearingEnv(clear=True), []
    svc, charged = _service(env, traj)
    before = svc._content_snapshot()
    env.call_tool("navigate", {"url": "/emr/denied"})          # the harness action the guard follows
    svc._guard(before, "rec-0-1")
    assert env.emr["triageNotes"].startswith("Agent reasoning")
    assert env.restored_calls == [{"full_state.triageNotes": "Agent reasoning: CO-50 non-covered; route to appeals."}]
    assert [(e["event_type"], e.get("tool"), e.get("origin")) for e in traj] == [
        ("tool_call", "restore_agent_content", "harness"), ("agent_content_guard", None, None)]
    assert (traj[0]["requester"], traj[1]["action_requester"]) == ("content_guard", "content_guard")
    assert traj[1]["persisted_fields"] == ["full_state.triageNotes"] and traj[1]["persisted_restored"] == 1
    assert len(charged) == 1


def test_no_clear_no_restore_idempotent():
    env, traj = _ClearingEnv(clear=False), []
    svc, charged = _service(env, traj)
    before = svc._content_snapshot()
    env.call_tool("navigate", {"url": "/emr/denied"})
    svc._guard(before, "rec-0-1")
    assert not env.restored_calls and traj == [] and charged == []


def test_dom_content_capture_unsaved_authored_text():
    """Unsaved authored DOM text (a note textarea the agent typed but did not save) is captured so a
    recovery surface-switch that clears it is detectable/restorable -- not only PERSISTED content."""
    controls = [{"ref": "note-content-input", "role": "textarea", "label": "Progress Note",
                 "value": "Faxed Rx + F2F + H&P to Lincare; confirmation FAX-2025"},
                {"ref": "search-input", "role": "input", "label": "Search", "value": "typed a search"},
                {"ref": "empty-note", "role": "textarea", "label": "Rationale", "value": ""}]
    svc, _ = _service(_ClearingEnv(), [], controls)
    dom = svc._dom_content(svc.list_controls())
    assert dom.get("note-content-input", "").startswith("Faxed Rx")   # content field captured
    assert "empty-note" not in dom                                    # empty ones skipped
    assert "search-input" not in dom                                  # non-content field skipped
