"""Semantic engine — judge-backed checks the deterministic engine cannot do (claim<->evidence support).

The judge is INJECTED as a `judge_fn(prompt:str) -> str|None` (the kernel builds it from the gateway with
an INDEPENDENT judge model, budgeted by max_semantic_checks). This keeps capabilities model-agnostic and
unit-testable (tests inject a fake judge). Fail-SAFE: no judge -> UNKNOWN (never fabricate a pass), and a
low-confidence verdict -> caller ESCALATEs rather than silently allowing.
"""
import json

UNKNOWN = "UNKNOWN"


class SemanticVerdict:
    __slots__ = ("supported", "confidence", "reason")

    def __init__(self, supported=None, confidence=0.0, reason=None):
        self.supported = supported       # True / False / None(unknown)
        self.confidence = confidence     # 0..1
        self.reason = reason

    def to_dict(self):
        return {"supported": self.supported, "confidence": round(float(self.confidence or 0.0), 3),
                "reason": self.reason}


_PROMPT = (
    "You are an evidence auditor. Decide ONLY whether the agent's final answer is SUPPORTED by the "
    "SELECTED evidence listed below — do NOT judge whether the answer is correct in general, only whether "
    "THIS evidence supports it. Reply with strict JSON: "
    '{{"supported": true|false, "confidence": 0.0-1.0, "reason": "<short>"}}.\n\n'
    "SELECTED EVIDENCE:\n{evidence}\n\nFINAL ANSWER:\n{answer}\n"
)


def verify_claim_support(answer, evidence, judge_fn=None):
    """Is `answer` supported by the SELECTED `evidence` (already filtered by the postcondition's evidence
    selector — this function does NOT see unrelated evidence). Returns a SemanticVerdict. No judge -> UNKNOWN."""
    if not judge_fn:
        return SemanticVerdict(None, 0.0, "semantic_judge_unavailable")
    ev_text = _format_evidence(evidence)
    if not ev_text.strip():
        return SemanticVerdict(False, 1.0, "no_selected_evidence")
    prompt = _PROMPT.format(evidence=ev_text[:8000], answer=str(answer)[:2000])
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
        return SemanticVerdict(None, 0.0, "empty_judge_response")
    s = raw if isinstance(raw, str) else str(raw)
    i, j = s.find("{"), s.rfind("}")
    if i >= 0 and j > i:
        try:
            d = json.loads(s[i:j + 1])
            sup = d.get("supported")
            # STRICT: only an actual JSON boolean counts. A string "false" must NOT become True.
            supported = sup if isinstance(sup, bool) else None
            return SemanticVerdict(supported, float(d.get("confidence", 0.0) or 0.0), d.get("reason"))
        except Exception:
            pass
    return SemanticVerdict(None, 0.0, "unparseable_judge_response")
