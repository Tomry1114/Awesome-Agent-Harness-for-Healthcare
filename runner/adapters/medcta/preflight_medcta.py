# -*- coding: utf-8 -*-
"""Preflight: exercise ALL FIVE native MedCTA tools for real before spending a run.
Each tool has exactly ONE implementation path, so any failure is visible here rather than being
silently replaced by a non-native substitute mid-run."""
import os, sys, json, ast

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "runner"))
sys.path.insert(0, os.path.join(REPO, "runner", "adapters", "medcta"))

import tools_medcta as T  # noqa: E402

t0 = json.loads(open("benchmark_dataprocess/MedCTA/tasks_unified.jsonl").readline())
ctx = t0["context"]
if isinstance(ctx, str):
    ctx = ast.literal_eval(ctx)
rel = ctx["images"][0]["path"]
img = os.path.join(REPO, "benchmark", "MedCTA", "opencompass", "data", "medcta_dataset", rel)

# a text-bearing image to prove OCR actually recognises characters (a CT slice legitimately has none)
text_img = None
for cand in ("benchmark/MedCTA/opencompass/data/medcta_dataset/image/image_%d.jpg" % i for i in range(2, 40)):
    if os.path.exists(os.path.join(REPO, cand)):
        text_img = os.path.join(REPO, cand)
        break

print("image        = %s (exists=%s)" % (img, os.path.exists(img)))
print("MH_SERPER_KEY set=%s  MH_VLM_API_MODEL=%r" % (bool(os.environ.get("MH_SERPER_KEY")),
                                                     os.environ.get("MH_VLM_API_MODEL")))

checks = [
    ("Calculator",                 lambda: T.run_tool("Calculator", {"expression": "(37*4)+18/2"})),
    ("GoogleSearch",               lambda: T.run_tool("GoogleSearch", {"query": "portal vein thrombosis CT findings"})),
    ("ImageDescription",           lambda: T.run_tool("ImageDescription", {}, image_path=img)),
    ("OCR",                        lambda: T.run_tool("OCR", {}, image_path=img)),
    ("RegionAttributeDescription", lambda: T.run_tool("RegionAttributeDescription",
                                                      {"region_query": "the liver", "attribute": "density"},
                                                      image_path=img)),
]

# markers of a degraded tool output (an error string or a non-native substitute); any of them fails the preflight
DEGRADED = ("[search error]", "[vlm_api_error]", "[EXCEPTION]", "[knowledge retrieval]",
            "[tool error]", "[ocr error]", "no MH_SERPER_KEY")

bad = []
for name, fn in checks:
    try:
        out = str(fn())
    except Exception as e:
        out = "[EXCEPTION] %r" % (e,)
    flat = out.replace("\n", " ")[:170]
    degraded = any(m in out for m in DEGRADED)
    # OCR on an image with no burned-in text legitimately returns "" (upstream agentlego does the same)
    note = ""
    if name == "OCR":
        note = "  <- empty = no text in this image (upstream-faithful)" if out.strip() == "" else ""
    print("  %-28s %s | %s%s" % (name, "DEGRADED" if degraded else "REAL    ", flat, note))
    if degraded:
        bad.append(name)

# prove the OCR engine really recognises characters, not just returns empty
if text_img:
    import re
    hits = 0
    for cand in [text_img] + [os.path.join(REPO, "benchmark/MedCTA/opencompass/data/medcta_dataset/image/image_%d.jpg" % i)
                              for i in range(2, 25)]:
        if not os.path.exists(cand):
            continue
        o = T.ocr(cand)
        if o and not o.startswith("[ocr error]") and re.search(r"\(\d+, \d+, \d+, \d+\)", o):
            print("\nOCR engine proof on %s:\n  %s" % (os.path.basename(cand), o.replace("\n", "\n  ")[:400]))
            hits += 1
            break
    if hits == 0:
        print("\nOCR engine proof: NO text found in the first ~24 images (not necessarily a failure)")

print("\nPREFLIGHT %s%s" % ("FAIL: " if bad else "PASS", ", ".join(bad)))
sys.exit(1 if bad else 0)
