# -*- coding: utf-8 -*-
"""pytest plugin loaded into every PhysicianBench checkpoint subprocess (runner/adapters/pb/grader.py).

Upstream utils/eval_helpers.py turns a judge failure into a verdict: llm_judge() catches any exception from
call_llm and returns {"pass": False, "score": "FAIL", "reason": "LLM error: ..."}; llm_extract() returns None or
False. The checkpoint's assertion then fails and our grader reads it as the agent's failure. Upstream code is a
shared vendored tree and is not edited in place, so the boundary is wrapped here instead: this plugin replaces
the module attributes BEFORE the test module is collected (the tests do `from utils.eval_helpers import *`),
records every judge call's outcome to the JSONL file named by MH_PB_JUDGE_LOG, and changes no verdict.

Recorded as a judge error: an exception out of call_llm, an empty reply, and llm_judge's own
"Could not parse LLM response". The grader turns any recorded judge error into checkpoint_status=error.
"""
import json
import os
import time

_LOG = os.environ.get("MH_PB_JUDGE_LOG")


def _rec(row):
    if not _LOG:
        return
    row = dict(row, model=os.environ.get("LLM_JUDGE_MODEL"))
    try:
        with open(_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _install():
    try:
        from utils import eval_helpers as eh
    except Exception as e:                      # cannot certify any judge call -> the grader must not either
        _rec({"ok": False, "stage": "install", "error": "guard_install_failed: %s: %s" % (type(e).__name__, e)})
        return
    if getattr(eh, "_mh_judge_guard", False):
        return
    orig_call, orig_judge = eh.call_llm, eh.llm_judge
    import sys
    _runner = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    if _runner not in sys.path:
        sys.path.insert(0, _runner)
    import responses_api as _RA
    sdk_call = orig_call

    def orig_call(prompt, system=""):
        # upstream's own SDK chat call first; the same request -- system and user message, max_completion_tokens
        # 4000 -- is carried to /v1/responses when the gateway serves this model there only (chat refused with
        # protocol_not_supported, or a 524 on chat)
        model = os.environ.get("LLM_JUDGE_MODEL")
        if not _RA.wants_responses(model):
            try:
                return sdk_call(prompt, system)
            except Exception as e:
                m = str(e)
                code = 524 if "524" in m else (400 if "protocol_not_supported" in m or "不支持 chat" in m else 0)
                if not (code and _RA.note_failure(model, code, m)):
                    raise
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        d = _RA.post(os.environ.get("LLM_JUDGE_BASE_URL") or os.environ.get("MH_JUDGE_BASE") or "",
                     os.environ.get("LLM_JUDGE_API_KEY") or os.environ.get("MH_JUDGE_KEY") or "",
                     os.environ.get("MH_OPENAI_UA", "codex_cli_rs/0.20.0"),
                     {"model": os.environ.get("LLM_JUDGE_MODEL"), "messages": msgs, "temperature": 0,
                      "max_completion_tokens": 4000}, 300, deadline=900)
        return d["choices"][0]["message"].get("content")

    def call_llm(prompt, system=""):
        # A rate-limit / transient refusal (429 "Concurrency limit exceeded", 503 "too frequently" / "temporarily
        # unavailable") passes on a later try; upstream's client does not retry, so without this every such refusal
        # would become a judge_error and void the run. Waited out here with exponential backoff; anything else fails at once.
        delay = 5.0
        for attempt in range(6):
            try:
                out = orig_call(prompt, system)
                break
            except Exception as e:
                msg = "%s: %s" % (type(e).__name__, str(e)[:300])
                low = msg.lower()
                # also transient: a reply the SDK could not parse ("'str' object has no attribute 'choices'" -- the gateway
                # answered with text, not a completion), a gateway 502/504/524 and a dropped connection
                transient = ("ratelimit" in low or " 429" in low or " 503" in low or "concurrency" in low
                             or "too frequently" in low or "temporarily unavailable" in low
                             or "has no attribute 'choices'" in low or " 502" in low or " 504" in low or " 524" in low
                             or "apiconnectionerror" in low or "apitimeouterror" in low or "timed out" in low
                             or "upstream" in low)                  # 403 "BPS upstream response failed"
                if transient and attempt < 5:
                    time.sleep(delay)
                    delay = min(delay * 2, 60.0)
                    continue
                _rec({"ok": False, "stage": "call_llm", "error": msg})
                raise
        if out is None or not str(out).strip():
            _rec({"ok": False, "stage": "call_llm", "error": "empty_response"})
        else:
            _rec({"ok": True, "stage": "call_llm"})
        return out

    def llm_judge(content, rubric, context=""):
        r = orig_judge(content, rubric, context)
        if isinstance(r, dict) and str(r.get("reason") or "") == "Could not parse LLM response":
            _rec({"ok": False, "stage": "llm_judge", "error": "unparseable_response"})
        return r

    eh.call_llm = call_llm
    eh.llm_judge = llm_judge
    eh._mh_judge_guard = True


_install()
