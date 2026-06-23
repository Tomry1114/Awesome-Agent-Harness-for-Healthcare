#!/usr/bin/env python3
"""MedCTA VQA-direct (brain modality #1): the native HEADLINE setting where a MULTIMODAL model sees
the image DIRECTLY and answers (no perception tools) — distinct from our tool-path (text brain forced
through perception tools). Sends image+question to the gateway vision model, scores the answer vs gold
with a GAcc-style judge. Reports report.native_metrics.vqa_direct (the headline, comparable to the
upstream paper). No GPU (gateway vision)."""
import json, os, sys, glob, base64, urllib.request

_BASE = (os.environ.get("MH_OPENAI_BASE", "https://www.micuapi.ai")).rstrip("/")
if _BASE.endswith("/v1"):
    _BASE = _BASE[:-3].rstrip("/")
_MODEL = os.environ.get("MH_VQA_MODEL", "gpt-5.5")
_UA = os.environ.get("MH_OPENAI_UA", "codex_cli_rs/0.20.0")
_IMG_ROOT = os.environ.get("MH_MEDCTA_IMG_ROOT", os.path.join(
    "benchmark", "MedCTA", "opencompass", "data", "medcta_dataset"))


def _key():
    k = os.environ.get("MH_OPENAI_KEY") or os.environ.get("OPENAI_API_KEY")
    if not k:
        kp = os.path.expanduser("~/.xbai_key")
        if os.path.exists(kp):
            k = open(kp).read().strip()
    return k


def _post(messages, max_tokens=1200):
    body = {"model": _MODEL, "max_tokens": max_tokens, "messages": messages}
    req = urllib.request.Request(_BASE + "/v1/chat/completions", data=json.dumps(body).encode(), method="POST",
        headers={"Authorization": "Bearer " + _key(), "Content-Type": "application/json", "User-Agent": _UA})
    d = json.load(urllib.request.urlopen(req, timeout=200))
    c = (d.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    return "".join(x.get("text", "") for x in c if isinstance(x, dict)) if isinstance(c, list) else c


def _img_b64(path):
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    mime = {"png": "image/png", "gif": "image/gif", "webp": "image/webp"}.get(ext, "image/jpeg")
    return mime, base64.b64encode(open(path, "rb").read()).decode()


def vqa_answer(image_path, question):
    mime, b64 = _img_b64(image_path)
    msgs = [{"role": "system", "content": "You are a medical imaging expert. Look at the image and answer the question concisely."},
            {"role": "user", "content": [{"type": "text", "text": question},
                                         {"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (mime, b64)}}]}]
    return _post(msgs)


def gacc(gold, pred):
    out = _post([{"role": "system", "content": "Compare a GOLD answer and a PREDICTED answer to a medical imaging question. Reply with exactly PASS (clinically equivalent) or FAIL on the first line."},
                 {"role": "user", "content": "GOLD: %s\nPREDICTED: %s" % (gold, pred)}], max_tokens=600)
    return 1.0 if out.strip().upper().startswith("PASS") else 0.0


def run(agent_dir):
    scores = []
    for rp in sorted(glob.glob(os.path.join(agent_dir, "*", "result.json"))):
        bdir = os.path.dirname(rp)
        t = json.load(open(os.path.join(bdir, "task.json")))
        imgs = (t.get("context") or {}).get("images") or []
        q = t.get("goal") or t.get("instruction") or ""
        gold = ((t.get("reference") or {}).get("gold_answer")
                or (t.get("reference") or {}).get("answer") or "")
        if not imgs:
            continue
        ip = os.path.join(_IMG_ROOT, imgs[0].get("path") or "")
        if not os.path.exists(ip):
            print("  %-12s (image missing: %s)" % (os.path.basename(bdir), ip)); continue
        try:
            ans = vqa_answer(ip, q)
            sc = gacc(gold, ans)
        except Exception as e:
            print("  %-12s ERR %r" % (os.path.basename(bdir), e)); continue
        scores.append(sc)
        print("  %-12s vqa=%.0f  ans=%s" % (os.path.basename(bdir), sc, ans[:60].replace(chr(10), " ")))
    mean = round(sum(scores) / len(scores), 3) if scores else None
    print("VQA-direct headline accuracy:", mean, "over", len(scores))
    rp = os.path.join(agent_dir, "report.json")
    if os.path.exists(rp) and mean is not None:
        rep = json.load(open(rp))
        rep.setdefault("native_metrics", {})["vqa_direct"] = {"accuracy": mean, "n": len(scores),
            "note": "multimodal-direct headline (image to brain, no tools) — native MedCTA setting #1"}
        json.dump(rep, open(rp, "w"), indent=1, ensure_ascii=False)
        print("-> written to", rp)


if __name__ == "__main__":
    run(sys.argv[1])
