"""Semantic engine — judge-backed checks the deterministic engine cannot do (claim<->evidence support).

The judge is INJECTED as a `judge_fn(prompt:str) -> str|None` (the kernel builds it from the gateway with
an INDEPENDENT judge model, budgeted by max_semantic_checks). This keeps capabilities model-agnostic and
unit-testable (tests inject a fake judge). Fail-SAFE: no judge -> UNKNOWN (never fabricate a pass), and a
low-confidence verdict -> caller ESCALATEs rather than silently allowing.
"""
import json

UNKNOWN = "UNKNOWN"

# Relation tri-state (P0-5): distinguish a real CONTRADICTION (answer conflicts with the evidence) from
# under-coverage (INSUFFICIENT: the evidence neither supports nor refutes). The conservative gate uses
# this to hard-REVISE only on contradiction and to gracefully degrade on under-coverage.
SUPPORTED = "supported"
CONTRADICTED = "contradicted"
INSUFFICIENT = "insufficient"


class SemanticVerdict:
    __slots__ = ("supported", "confidence", "reason", "relation")

    def __init__(self, supported=None, confidence=0.0, reason=None, relation=None):
        self.supported = supported       # True / False / None(unknown)
        self.confidence = confidence     # 0..1
        self.reason = reason
        self.relation = relation         # supported | contradicted | insufficient | None

    def to_dict(self):
        return {"supported": self.supported, "confidence": round(float(self.confidence or 0.0), 3),
                "reason": self.reason, "relation": self.relation}


_PROMPT = (
    "You are an evidence auditor. Using the QUESTION/GOAL and the SELECTED evidence below, decide ONLY "
    "whether the agent's FINAL ANSWER is SUPPORTED by THIS evidence — do NOT judge whether the answer is "
    "correct in general and do NOT use outside knowledge. Classify the relation as exactly one of: "
    "'supported' (evidence directly supports the answer), 'contradicted' (evidence conflicts with/refutes "
    "the answer), or 'insufficient' (evidence neither supports nor contradicts — it under-covers the answer). "
    "Reply with strict JSON: "
    '{{"relation": "supported|contradicted|insufficient", "confidence": 0.0-1.0, "reason": "<short>"}}.\n\n'
    "QUESTION / GOAL:\n{task_goal}\n\nPUBLIC TASK CONTEXT:\n{public_context}\n\n"
    "SELECTED EVIDENCE:\n{evidence}\n\nFINAL ANSWER:\n{answer}\n"
)


def verify_claim_support(task_goal, public_context, answer, evidence, judge_fn=None):
    """Is `answer` supported by the SELECTED `evidence` (already filtered by the postcondition's evidence
    selector — this function does NOT see unrelated evidence). `task_goal` and `public_context` are the
    PUBLIC task info the agent already sees (compiler whitelist, leak-checked) — NEVER gold_answer/reference;
    they only tell the judge WHAT the answer is claiming. Returns a SemanticVerdict. No judge -> UNKNOWN."""
    if not judge_fn:
        return SemanticVerdict(None, 0.0, "semantic_judge_unavailable", relation=None)
    ev_text = _format_evidence(evidence)
    if not ev_text.strip():
        # zero SELECTED evidence = under-coverage (INSUFFICIENT), NOT a high-confidence contradiction.
        return SemanticVerdict(None, 1.0, "no_selected_evidence", relation=INSUFFICIENT)
    prompt = _PROMPT.format(task_goal=str(task_goal or "(not provided)")[:2000],
                            public_context=str(public_context or "(none)")[:3000],
                            evidence=ev_text[:8000], answer=str(answer)[:2000])
    try:
        raw = judge_fn(prompt)
    except Exception as ex:
        return SemanticVerdict(None, 0.0, "semantic_judge_error:%r" % (ex,))
    return _parse(raw)


def _format_evidence(evidence):
    out = []
    for e in (evidence or []):
        if isinstance(e, dict):
            out.append("- [%s] %s" % (e.get("type", "evidence"),
                                      str(e.get("value_full") or e.get("value", ""))[:1800]))
        else:
            out.append("- " + str(e)[:300])
    return "\n".join(out)


def _parse(raw):
    if not raw:
        return SemanticVerdict(None, 0.0, "empty_judge_response", relation=None)
    s = raw if isinstance(raw, str) else str(raw)
    i, j = s.find("{"), s.rfind("}")
    if i >= 0 and j > i:
        try:
            d = json.loads(s[i:j + 1])
            conf = float(d.get("confidence", 0.0) or 0.0)
            reason = d.get("reason")
            rel = d.get("relation")
            rel = rel.strip().lower() if isinstance(rel, str) else None
            if rel not in (SUPPORTED, CONTRADICTED, INSUFFICIENT):
                rel = None
            sup = d.get("supported")
            # STRICT: only an actual JSON boolean counts. A string "false" must NOT become True.
            supported = sup if isinstance(sup, bool) else None
            # Reconcile relation <-> supported. Back-compat: a {"supported": bool} judge carries no
            # relation -> derive it (true->supported, false->contradicted). A {"relation": ...} judge
            # carries no boolean -> derive supported (insufficient -> None, never a contradiction).
            if rel is None and supported is not None:
                rel = SUPPORTED if supported else CONTRADICTED
            elif rel is not None and supported is None:
                supported = True if rel == SUPPORTED else (False if rel == CONTRADICTED else None)
            return SemanticVerdict(supported, conf, reason, relation=rel)
        except Exception:
            pass
    return SemanticVerdict(None, 0.0, "unparseable_judge_response", relation=None)
