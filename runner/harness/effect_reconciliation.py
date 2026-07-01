"""Effect reconciliation (Phase 4) -- the INCOMPLETE_EFFECT -> COMPLETE recovery, done WITHOUT the harness
making a clinical decision.

An agent may COMMIT to an executable order in its deliverable ("Order pelvic ultrasound") yet never PERFORM
the state mutation that realizes it. The failure is not a wrong decision -- the decision is the agent's and is
correct -- it is an UNREALIZED EFFECT. This module finds committed orders that have NO matching effect in
state, so a downstream authorized mutation can complete EXACTLY what the agent already decided.

Integrity guardrails (why this is a legitimate recovery, not a trick):
  - decision provenance = the AGENT. The order text/modality comes from the agent's deliverable, never a
    checkpoint. `extract_committed_orders` (semantic.py) drops anything conditional/hedged.
  - MUTATION-fail-safe: `is_realized` is CONSERVATIVE toward "already realized" -- on any keyword overlap or
    uncertainty it treats the order as satisfied, so we never place a duplicate/spurious order.
  - completion is still gated downstream by MutationAuthorization + governance preconditions (verify_commit);
    this module only DETECTS the gap. Substrate-agnostic: `state_order_texts` come from the adapter's order
    search; no benchmark names here.
"""
import re
from dataclasses import dataclass, field

_STOP = {"the", "a", "an", "and", "or", "of", "to", "for", "with", "order", "orders", "place", "start",
         "study", "test", "testing", "evaluation", "as", "if", "needed", "first", "line", "approach",
         "obtain", "perform", "consider", "recommend", "patient", "review", "plan", "care", "new", "please"}


def _keywords(text):
    toks = re.findall(r"[a-z0-9]+", str(text or "").lower())
    return {t for t in toks if len(t) > 3 and t not in _STOP}


def is_realized(order_text, state_order_texts):
    """CONSERVATIVE: does an existing state order plausibly satisfy this committed order? True if any state
    order shares a salient keyword with it. Fail-safe toward True (never double-create on uncertainty)."""
    ok = _keywords(order_text)
    if not ok:
        return True                     # nothing salient to match on -> do not risk a spurious create
    for st in (state_order_texts or []):
        if ok & _keywords(st):
            return True
    return False


@dataclass
class UnrealizedCommitment:
    text: str
    category: str = "other"
    keywords: list = field(default_factory=list)


def unrealized_commitments(committed_orders, state_order_texts):
    """committed_orders: [{text,category}] (FIRM only, from extract_committed_orders). state_order_texts:[str]
    orders already present in state (from the adapter's order search). Return the committed orders with NO
    matching state effect -> [UnrealizedCommitment]. Empty input / all realized -> []."""
    out = []
    for o in (committed_orders or []):
        txt = (o or {}).get("text") if isinstance(o, dict) else str(o)
        if not str(txt or "").strip():
            continue
        if is_realized(txt, state_order_texts):
            continue
        out.append(UnrealizedCommitment(text=str(txt), category=str((o or {}).get("category") or "other"),
                                        keywords=sorted(_keywords(txt))))
    return out
