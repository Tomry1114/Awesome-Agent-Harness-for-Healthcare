"""PB-specific deliverable scaffolding (Codex #1).

Extracted verbatim from run.py so the generic runner loop stays benchmark-agnostic.
For non-PB tasks .active is False and every method no-ops.
"""
import os, glob, shutil, json


class DeliverableScaffold:
    # PB-specific deliverable nudging/enforcement, extracted from run_task (Codex #1) so the generic
    # runner loop stays benchmark-agnostic. For non-PB tasks .active is False and every method no-ops.
    # Behavior preserved verbatim from the previous inline version.
    def __init__(self, task):
        import re
        track = os.environ.get("MH_PROMPT_TRACK", "harness")
        scaffold = os.environ.get("MH_DELIV_SCAFFOLD", "0" if track == "native" else "1") != "0"
        m = re.search(r"(?:/?workspace/)?output/[\w.\-]+", task.get("goal", "") or "")
        self.path = (m.group(0) if m else None) if scaffold else None
        self.nudges = 0
        self.budget_nudged = False

    @property
    def active(self):
        return bool(self.path)

    def is_required_write(self, action):
        # True ONLY for the exact required deliverable write (right tool + right filename) so the over-budget
        # exception covers one precise action, not any write_file. Keeps the runner from naming a raw tool.
        if not self.active or (action or {}).get("tool") != "write_file":
            return False
        supplied = os.path.normpath(str(((action.get("args") or {}).get("path") or ""))).lstrip("/")
        required = os.path.normpath(str(self.path or "")).lstrip("/")
        return bool(required) and os.path.basename(supplied) == os.path.basename(required)

    def _want(self, env):
        ws = getattr(env, "workspace", "") or ""
        return os.path.join(ws, os.path.basename(self.path)) if ws else ""

    def _missing(self, env):
        w = self._want(env)
        return not (w and os.path.isfile(w) and os.path.getsize(w) > 0)

    def budget_warning(self, env, used_actions, max_steps, trajectory):
        # one-shot low-budget warning if the deliverable is still missing; keyed on EXECUTED env actions
        # (not loop steps) so harness repair turns don't make PB think the task budget is nearly spent.
        step = used_actions
        if not self.active or self.budget_nudged or (max_steps - step) > 8:
            return None
        if not self._want(env) or not self._missing(env):
            return None
        self.budget_nudged = True
        trajectory.append({"step": step, "event_type": "deliverable_budget_warning",
                           "remaining": max_steps - step, "status": "ok"})
        fb = ("You are RUNNING OUT OF STEPS (only %d left) and have NOT written the "
              "required deliverable. STOP retrieving NOW. Immediately call write_file with EXACTLY "
              "path=\"%s\" and content = your full clinical assessment and management plan from the "
              "data already retrieved." % (max_steps - step, self.path))
        return {"feedback": fb}, "budget_warning"

    def pre_final_nudge(self, env, step, trajectory):
        # before accepting a final answer, force the deliverable to be written first (up to 3x);
        # returns (last_res, last_obs) to inject + signal continue, or None to allow finishing
        if not self.active or self.nudges >= 3 or not self._missing(env):
            return None
        self.nudges += 1
        fb = ("STOP. The task REQUIRES a deliverable file and it is NOT written yet. You MUST "
              "call write_file NOW with EXACTLY path=\"%s\" (this EXACT filename, no other name) "
              "and content = your full clinical assessment and management plan. Do not answer in "
              "chat. Do not use any other filename. After write_file succeeds, then finish." % self.path)
        trajectory.append({"step": step, "event_type": "deliverable_nudge", "path": self.path,
                           "attempt": self.nudges, "status": "ok"})
        return {"feedback": fb}, fb

    def enforce(self, env, agent, task, trajectory, max_steps, harness=None, state_snapshot=None):
        # post-loop: guarantee ONE write attempt if still missing, then normalize a mis-named single file
        if not self.active:
            return
        ws = getattr(env, "workspace", "") or ""
        want = self._want(env)
        if ws and self._missing(env):
            fb = ("You must NOW save the required deliverable. Call write_file with EXACTLY path=\"%s\" and "
                  "content = your full clinical assessment and management plan." % self.path)
            try:
                a = agent.act({"goal": task.get("goal"), "context": task.get("context"),
                               "tools": env.available_tools(), "last_observation": fb,
                               "last_result": {"feedback": fb}})
                if isinstance(a, dict) and a.get("type") == "tool_call" and a.get("tool") == "write_file":
                    # NO side channel: route the forced deliverable through the SAME harness pipeline a normal
                    # tool_call takes -> before_action (RequiredContext / MutationAuthorization see it) and
                    # after_action (read-back + provenance). A BLOCK/ESCALATE from before_action is honored.
                    _snap = (state_snapshot(env) if state_snapshot else None)
                    _blocked = None
                    if harness is not None:
                        try:
                            _hbe = harness.before_action(a, _snap, step=max_steps)
                            if _hbe is not None and getattr(_hbe, "type", None) in ("BLOCK", "ESCALATE"):
                                _blocked = _hbe.type
                        except Exception:
                            pass
                    if _blocked is not None:
                        trajectory.append({"step": max_steps, "event_type": "forced_deliverable_blocked",
                                           "decision": _blocked, "tool": "write_file", "status": "ok"})
                    else:
                        wr = env.call_tool("write_file", a.get("args", {}))
                        _snap2 = (state_snapshot(env) if state_snapshot else None)
                        trajectory.append({"step": max_steps, "event_type": "tool_call", "tool": "write_file",
                                           "args": a.get("args", {}), "result": wr,
                                           "observation": json.dumps(wr)[:200], "ts": str(max_steps),
                                           "status": "ok", "forced_deliverable": True,
                                           "via_pipeline": harness is not None})
                        if harness is not None:
                            try:
                                harness.after_action(a, wr, _snap, _snap2, step=max_steps)
                            except Exception:
                                pass
            except Exception as fe:
                trajectory.append({"step": max_steps, "event_type": "agent_error",
                                   "error": "forced_deliverable_failed", "raw": repr(fe)[:120], "status": "error"})
        if ws and os.path.isdir(ws) and self._missing(env):
            cands = [f for f in glob.glob(os.path.join(ws, "*")) if os.path.isfile(f) and os.path.getsize(f) > 0]
            if cands:
                best = max(cands, key=os.path.getsize)
                shutil.copyfile(best, want)
                trajectory.append({"step": max_steps, "event_type": "deliverable_renamed",
                                   "from": os.path.basename(best), "to": os.path.basename(self.path),
                                   "n_candidates": len(cands), "status": "ok"})
