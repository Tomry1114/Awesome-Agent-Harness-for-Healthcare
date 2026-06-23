"""MedCTA Goal-Accuracy (Gacc) judge — faithful 0-1 semantic scoring per upstream goal_accuracy.py.
The two prompts below are copied VERBATIM from benchmark/MedCTA/goal_accuracy.py. Native MedCTA Gacc
uses gpt-5.4 via OpenAI; we use a configurable cheaper strong model on the same gateway (default
deepseek-v3.2) — the judge-model deviation is registered in the paper-align passport.

Config (env): MH_OPENAI_BASE / MH_OPENAI_KEY (~/.xbai_key) / MH_GACC_MODEL (default deepseek-v3.2)
"""
import os, json
import gateway

GOAL_ACCURACY_SYSTEM_PROMPT = """You are a medical answer evaluator.

Compare the predicted FINAL answer against the gold FINAL clinical answer.
Assign a score from 0.0 to 1.0 based on semantic clinical correctness.

CRITICAL RULE (very important):
- If the predicted answer explicitly contains the correct gold answer, assign a score of 1.0.
- Presence of the correct diagnosis/finding overrides extra guesses unless contradictory.

General rules:
- Give partial credit if only partially correct.
- Do NOT give 0.0 unless completely wrong or unrelated.
- Judge by clinical meaning, not wording.
- Synonyms count as correct.

Scoring guide:
- 1.0 = gold answer clearly present OR fully correct
- 0.8-0.95 = correct but minor imprecision
- 0.5-0.75 = partially correct
- 0.2-0.45 = weak overlap
- 0.0-0.1 = wrong/unrelated

Return JSON only:
{
  "score": number
}
"""
GOAL_ACCURACY_USER_PROMPT = """Gold final answer:
{gold_final}

Predicted final answer:
{pred_final}
"""

def _parse_score(txt):
    dec = json.JSONDecoder()
    for i, c in enumerate(txt):
        if c == "{":
            try:
                o, _ = dec.raw_decode(txt[i:])
                if isinstance(o, dict) and "score" in o and isinstance(o["score"], (int, float)):
                    return max(0.0, min(1.0, float(o["score"])))
            except Exception:
                pass
    import re as _re
    m = _re.search(r'score\"?\'?\s*[:=]\s*(1(?:\.0+)?|0?\.\d+|0)', txt)
    if m:
        try: return max(0.0, min(1.0, float(m.group(1))))
        except Exception: pass
    return None

def _flatten_str(x):
    if isinstance(x, str): return [x]
    if isinstance(x, (list, tuple)):
        out = []
        for i in x: out.extend(_flatten_str(i))
        return out
    return [str(x)] if x is not None else []


def score(prediction, gold_answers, model=None):
    """Return {score: 0-1 float | None, raw, model}. gold_answers: list[str] of acceptable gold answers."""
    model = model or os.environ.get("MH_GACC_MODEL", "deepseek-v3.2")
    gold = " | ".join(_flatten_str(gold_answers))
    user = GOAL_ACCURACY_USER_PROMPT.format(gold_final=gold, pred_final=(prediction or "")[:2000])
    res = gateway.chat([{"role": "system", "content": GOAL_ACCURACY_SYSTEM_PROMPT},
                        {"role": "user", "content": user}],
                       model=model, max_tokens=1024, judge=True, timeout=120)
    if res["ok"]:
        content = res["content"]
        return {"score": _parse_score(content), "raw": content[:200], "model": model}
    if res["error_type"] == "billing":
        return {"score": None, "raw": "BILLING/QUOTA " + res["raw"], "model": model}
    return {"score": None, "raw": "gacc_error:" + res["raw"], "model": model}
