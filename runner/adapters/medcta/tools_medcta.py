"""MedCTA REAL tool implementations (the environment-side "hands").

5 tools matching the MedCTA tool set. **Each tool has exactly ONE implementation path** -- no fallbacks,
no "modes". A fallback that silently substitutes a non-native backend (a model answering from memory in
place of a web search, a captioner paraphrasing in place of an OCR engine) turns a failed tool call into a
plausible-looking observation and corrupts every run that hit it. Failures must stay visible instead.

  ImageDescription / RegionAttributeDescription -> VLM (vlm_backend.get_backend -> ApiVLM);
                       upstream mechanism is also a VLM (agentlego/benchmark.py: llava-7b-v1.5_vqa)
  OCR              -> EasyOCR, the native engine (agentlego tools/ocr/ocr.py), CPU
  Calculator       -> safe AST arithmetic (no eval of arbitrary code)
  GoogleSearch     -> Serper.dev, the native backend (agentlego tools/search/google.py), behind the search guard
                      (runner/search_guard.py): a query targeting the benchmark's own source, or naming the current
                      item's provenance id, is refused, and a result pointing at the benchmark's source or at the
                      current item's direct material provenance is withheld

Each tool returns a STRING (the tool observation fed back to the agent). Image tools take a resolved
absolute image path (the env resolves it from task.context.images).
"""
import os, ast, json, operator, urllib.error, urllib.request, urllib.parse

import search_guard

SEARCH_K = 5   # web search results kept per query
BENCH = "MedCTA"   # this benchmark's section of runner/search_blocklist.yaml

# ---- Calculator: safe arithmetic only ----
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.USub: operator.neg, ast.UAdd: operator.pos,
        ast.FloorDiv: operator.floordiv}
def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)): return node.value
    if isinstance(node, ast.BinOp): return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp): return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")
def calculator(expr):
    try:
        return str(_eval(ast.parse(str(expr), mode="eval").body))
    except Exception as e:
        return "[calculator error] %r" % e

# ---- GoogleSearch: REAL web search via Serper -- the NATIVE backend, and the ONLY path ----
# Upstream MedCTA's GoogleSearch is agentlego/tools/search/google.py = Serper.dev. There is deliberately
# NO fallback: a "knowledge retrieval" fallback (a gateway model answering from parametric memory) and a
# frozen/BM25 local corpus are both NOT the native tool -- substituting either would silently turn "the
# agent searched the web" into "a model recalled something", contaminating every run that used it.
# If Serper is unavailable the tool returns a visible [search error]; it never quietly answers instead.
def google_search(query):
    """The transport: the Serper key lookup and the HTTP exchange. -> the parsed Serper response (a dict), or a visible
    '[search error] ...' string. Needs a Serper key (MH_SERPER_KEY / SERPER_API_KEY / ~/.serper_key). Serper is
    reached over its HTTP API; direct search-engine scraping is not used. The search guard and the formatting run
    above it, in search(), so they see each result's link and apply to whatever answers here (a replay serves its
    recorded, already formatted observation string at this seam)."""
    q = str(query or "").strip()
    key = os.environ.get("MH_SERPER_KEY") or os.environ.get("SERPER_API_KEY")
    if not key:
        try:
            key = open(os.path.expanduser("~/.serper_key")).read().strip()
        except Exception:
            key = ""
    if not key:
        return "[search error] no MH_SERPER_KEY (Serper endpoint is reachable; set the key to enable real web search)"
    try:
        req = urllib.request.Request("https://google.serper.dev/search",
            data=json.dumps({"q": q, "num": SEARCH_K}).encode("utf-8"),
            headers={"X-API-KEY": key, "Content-Type": "application/json"}, method="POST")
        return json.loads(urllib.request.urlopen(req, timeout=25).read().decode("utf-8", "ignore"))
    except urllib.error.HTTPError as e:
        # the body says why: Serper answers 400 {"message":"Not enough credits"} when the key is spent; without the
        # body the observation would read only "HTTPError 400: 'Bad Request'"
        try:
            body = e.read().decode("utf-8", "ignore")[:200]
        except Exception:
            body = ""
        return "[search error] %r %s" % (e, body)
    except Exception as e:
        return "[search error] %r" % e


_RESULTS_HEADER = "[serper web search]\n"
_ENTRY_STARTS = ("- ", "[answer] ", "[kg] ")


def _serper_entries(resp):
    """A Serper response -> its result entries, in the order and at the cut the observation shows them. Each entry is
    {"text": the observation line, "link", "title", "snippet"}; the guard matches all four."""
    entries = []
    ab = resp.get("answerBox") or {}
    if ab.get("answer") or ab.get("snippet"):
        entries.append({"text": "[answer] " + str(ab.get("answer") or ab.get("snippet")), "link": ab.get("link"),
                        "title": ab.get("title"), "snippet": ab.get("snippet")})
    kg = resp.get("knowledgeGraph") or {}
    if kg.get("description"):
        entries.append({"text": "[kg] " + str(kg.get("description")),
                        "link": [kg.get("descriptionLink"), kg.get("website")], "title": kg.get("title")})
    for r in (resp.get("organic") or [])[:SEARCH_K]:
        entries.append({"text": "- %s: %s" % (r.get("title", "")[:80], (r.get("snippet") or "")[:220]),
                        "link": r.get("link"), "title": r.get("title"), "snippet": r.get("snippet")})
    return entries


def _observation_entries(text):
    """An already formatted observation -> its result entries (text only: the format carries no link), or None for a
    line that lists no results (a search error, no results)."""
    if not text.startswith(_RESULTS_HEADER):
        return None
    entries = []
    for line in text[len(_RESULTS_HEADER):].split("\n"):
        if line.startswith(_ENTRY_STARTS) or not entries:
            entries.append({"text": line})
        else:                                   # a line break inside a title or snippet
            entries[-1]["text"] += "\n" + line
    return entries


def _render(entries, q, withheld=0):
    text = (_RESULTS_HEADER + "\n".join(e["text"] for e in entries)) if entries else "[serper: no results] " + q
    return text + ("\n" + search_guard.WITHHELD_NOTE % withheld if withheld else "")


def search(query, task_id=None):
    """The ONE GoogleSearch path. -> (observation string, blocked record or None).

    task_id is the task this search runs for (ToolSandboxEnv.call_tool passes it); the guard applies that item's
    provenance rules as well as the benchmark-level ones, and only the benchmark-level ones without it.
    The query is checked first: one that targets the benchmark's own source, or names the item's provenance id, is
    refused and the backend is never called. Otherwise google_search answers and every result whose link, title or
    snippet points at the benchmark source or at the item's source material is withheld; the agent sees the rest and a
    note saying how many were withheld (the same note whatever the reason). With nothing withheld the observation is
    exactly what the native formatting gives (a formatted string from the seam is returned as is)."""
    refused = search_guard.refuse_query(BENCH, query, task_id)
    if refused is not None:
        return search_guard.REFUSAL, refused
    resp = google_search(query)
    if isinstance(resp, str):
        entries = _observation_entries(resp)
        if entries is None:
            return resp, None
    else:
        try:
            entries = _serper_entries(resp)
        except Exception as e:
            return "[search error] %r" % e, None
    kept, blocked = search_guard.filter_results(BENCH, entries, task_id)
    q = str(query or "").strip()
    if blocked is None:
        return (resp if isinstance(resp, str) else _render(entries, q)), None
    return _render(kept, q, blocked["count"]), blocked

# ---- OCR: EasyOCR -- the NATIVE engine, and the ONLY path ----
# Upstream MedCTA leaves OCR to generic agentlego (tools/ocr/ocr.py) = EasyOCR, CPU-capable (gpu=False).
# A "VLM reads the text off the image" implementation would NOT be the native tool:
# a captioning model paraphrasing text is a different measurement from a CRAFT+CRNN text recogniser.
_OCR_READER = None
_OCR_LOCK = __import__("threading").Lock()

def _ocr_reader():
    global _OCR_READER
    with _OCR_LOCK:                                        # one load, whether the preload or the tool gets here first
        if _OCR_READER is None:
            import easyocr
            _OCR_READER = easyocr.Reader(["en"], gpu=False)   # CPU: CRAFT detector + CRNN recogniser, ~100MB
    return _OCR_READER

def preload_ocr():
    """Load the OCR model in a background thread (tens of seconds on CPU) while the task starts, instead of
    inside the agent's first OCR call. Same model, same reader; only when the load happens changes."""
    t = __import__("threading").Thread(target=lambda: _ocr_reader(), daemon=True)
    t.start()
    return t

def _extract_bbox(char_boxes):
    """Upstream agentlego OCR.extract_bbox: corner points -> (x1, y1, x2, y2)."""
    xs = [int(b[0]) for b in char_boxes]
    ys = [int(b[1]) for b in char_boxes]
    return min(xs), min(ys), max(xs), max(ys)

def ocr(image_path):
    """The ONE OCR implementation. Output format mirrors upstream agentlego OCR.apply():
    one line per detection, '(x1, y1, x2, y2) text'. read_args match upstream defaults for
    line_group_tolerance=-1 -> decoder='beamsearch', paragraph=True."""
    try:
        results = _ocr_reader().readtext(str(image_path), detail=1,
                                         decoder="beamsearch", paragraph=True)
    except Exception as e:
        return "[ocr error] %r" % e
    lines = []
    for item in results:
        lines.append("({}, {}, {}, {}) {}".format(*_extract_bbox(item[0]), item[1]))
    return "\n".join(lines)

# ---- Image tools: the single VLM perception backend (vlm_backend.get_backend -> ApiVLM) ----
def image_description(image_path):
    from adapters.medcta.vlm_backend import get_backend
    return get_backend().image_description(image_path)
def region_attribute_description(image_path, bbox=None, attribute=None, region_query=None):
    from adapters.medcta.vlm_backend import get_backend
    return get_backend().region_attribute_description(image_path, bbox=bbox, attribute=attribute, region_query=region_query)

# ---- dispatch (tool name -> callable on (args, image_path)) ----
def _search_query(a):
    return a.get("query") or a.get("text") or a.get("keyword") or ""


def execute(name, args, image_path=None, task_id=None):
    """One tool call -> {"output": observation string}, plus "blocked" when the search guard refused the query or
    withheld results (runner/search_guard.py). ToolSandboxEnv.call_tool's entry; task_id selects the item's provenance
    rules."""
    if name == "GoogleSearch":
        out, blocked = search(_search_query(args or {}), task_id)
        return {"output": out, "blocked": blocked} if blocked else {"output": out}
    return {"output": run_tool(name, args, image_path)}


def run_tool(name, args, image_path=None, task_id=None):
    a = args or {}
    if name == "Calculator":
        return calculator(a.get("expression") or a.get("expr") or a.get("text") or "")
    if name == "GoogleSearch":
        return search(_search_query(a), task_id)[0]
    if name == "OCR":
        return ocr(image_path)
    if name == "ImageDescription":
        return image_description(image_path)
    if name == "RegionAttributeDescription":
        # a coordinate list / 4-number string is a bbox; a SEMANTIC region name ("porta
        # hepatis") is a region_query -- do NOT dump it into bbox (where it is discarded and the attribute
        # becomes the focus).
        import re as _re
        _rg = a.get("region")
        _bbox = a.get("bbox")
        _rq = a.get("region_query")
        if _rg is not None:
            _coords = ((isinstance(_rg, (list, tuple)) and len(_rg) >= 4)
                       or (isinstance(_rg, str) and len(_re.findall(r"-?\d+\.?\d*", _rg)) >= 4))
            if _coords:
                _bbox = _bbox or _rg
            else:
                _rq = _rq or _rg
        return region_attribute_description(
            image_path, bbox=_bbox, region_query=_rq,
            attribute=a.get("attribute") or a.get("attr"))
    return "[unknown tool] " + str(name)
