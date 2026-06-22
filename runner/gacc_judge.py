"""MedCTA Goal-Accuracy (Gacc) judge — faithful 0-1 semantic scoring per upstream goal_accuracy.py.
The two prompts below are copied VERBATIM from benchmark/MedCTA/goal_accuracy.py. Native MedCTA Gacc
uses gpt-5.4 via OpenAI; we use a configurable cheaper strong model on the same gateway (default
deepseek-v3.2) — the judge-model deviation is registered in the paper-align passport.

Config (env): MH_OPENAI_BASE / MH_OPENAI_KEY (~/.xbai_key) / MH_GACC_MODEL (default deepseek-v3.2)
"""
import os, json, time, urllib.request, urllib.error

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

def _key():
    k = os.environ.get("MH_OPENAI_KEY")
    if k: return k.strip()
    p = os.path.expanduser("~/.xbai_key")
    return open(p).read().strip() if os.path.exists(p) else ""

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
    base = (os.environ.get("MH_JUDGE_BASE") or os.environ.get("MH_OPENAI_BASE", "https://www.micuapi.ai")).rstrip("/")  # judge gateway can differ from agent
    gold = " | ".join(_flatten_str(gold_answers))
    user = GOAL_ACCURACY_USER_PROMPT.format(gold_final=gold, pred_final=(prediction or "")[:2000])
    body = {"model": model, "messages": [{"role": "system", "content": GOAL_ACCURACY_SYSTEM_PROMPT},
                                         {"role": "user", "content": user}], "max_tokens": 1024}
    data = json.dumps(body).encode(); last = ""
    for attempt in range(4):
        try:
            req = urllib.request.Request(base + "/v1/chat/completions", data=data, method="POST", headers={
                "Authorization": "Bearer " + _key(), "Content-Type": "application/json",
                "User-Agent": os.environ.get("MH_OPENAI_UA", "codex_cli_rs/0.20.0"), "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.loads(r.read().decode())
            content = (d.get("choices") or [{}])[0].get("message", {}).get("content")
            if isinstance(content, list):
                content = "".join(x.get("text", "") for x in content if isinstance(x, dict))
            if content and content.strip():
                return {"score": _parse_score(content), "raw": content[:200], "model": model}
            last = "empty"
        except urllib.error.HTTPError as e:
            try: eb = e.read().decode()[:200]
            except Exception: eb = ""
            last = "http_%s:%s" % (e.code, eb)
            if any(k in eb for k in ("额度", "欠费", "预扣费")) or any(k in eb.lower() for k in ("insufficient", "balance", "quota")):
                return {"score": None, "raw": "BILLING/QUOTA " + last, "model": model}
            if e.code in (400, 401): break
        except Exception as e:
            last = "err:%s" % e
        time.sleep(min(12, 2 ** attempt))
    return {"score": None, "raw": "gacc_error:" + last, "model": model}
