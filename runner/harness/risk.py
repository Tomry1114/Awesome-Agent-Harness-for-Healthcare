"""Action risk tiers (R0–R3) — drive the control strategy of Module C (Verify-and-Commit).

  R0  read-only          query / view page / OCR / read resource          -> allow + record
  R1  reversible write   fill field / draft material / reversible edit     -> prospective (pre-) check
  R2  commit             order med / submit form / final clinical answer   -> pre-check + post-verify
  R3  unjudgeable        high risk the system cannot reliably adjudicate    -> ESCALATE

Risk is read off the compiled contract's commit_points when the action matches one; otherwise a coarse
keyword default. Per-dataset policy packs may override `commit_actions` / `reversible_actions`.
"""

R0, R1, R2, R3 = "R0", "R1", "R2", "R3"
ORDER = {R0: 0, R1: 1, R2: 2, R3: 3}

# coarse defaults; policy packs refine these per substrate.
_COMMIT_HINTS = ("create", "submit", "send", "order", "sign", "finalize", "appeal", "prescribe")
_REVERSIBLE_HINTS = ("fill", "type", "write", "draft", "upload", "set", "update", "edit")
_READ_HINTS = ("search", "read", "get", "view", "list", "ocr", "describe", "lookup", "navigate", "snapshot")


def classify_risk(action, contract=None, policy=None):
    """Return R0..R3 for a proposed action. `action` is the agent action dict ({type, tool, args} or
    {type:'final'}); a contract commit_point match wins over keyword heuristics."""
    policy = policy or {}
    name = _action_name(action)
    # final answer is always a commit point (R2) unless policy escalates it.
    if action.get("type") == "final":
        return policy.get("final_risk", R2)
    # explicit contract commit point?
    for cp in (getattr(contract, "commit_points", None) or []):
        if cp.get("action") and cp["action"] == name:
            return cp.get("risk", R2)
    # policy-pack explicit lists
    if name in set(policy.get("commit_actions", [])):
        return R2
    if name in set(policy.get("reversible_actions", [])):
        return R1
    if name in set(policy.get("read_actions", [])):
        return R0
    # keyword fallback
    low = (name or "").lower()
    if any(h in low for h in _COMMIT_HINTS):
        return R2
    if any(h in low for h in _REVERSIBLE_HINTS):
        return R1
    return R0  # default: treat unknown as read-only (conservative: never silently auto-commit)


def _action_name(action):
    if not isinstance(action, dict):
        return ""
    if action.get("type") == "final":
        return "final"
    return action.get("tool") or action.get("action") or action.get("type") or ""


def at_least(risk, threshold):
    return ORDER.get(risk, 0) >= ORDER.get(threshold, 0)
