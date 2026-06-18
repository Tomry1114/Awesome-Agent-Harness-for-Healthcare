"""MedCTA v1 REAL tool implementations (the environment-side "hands").

5 tools matching the MedCTA tool set:
  ImageDescription / RegionAttributeDescription / OCR -> local VLM backend (vlm_backend.get_backend)
  Calculator       -> safe AST arithmetic (no eval of arbitrary code)
  GoogleSearch     -> frozen offline corpus (reproducible); set MH_SEARCH_MODE=live to hit the web

Each tool returns a STRING (the tool observation fed back to the agent). Image tools take a resolved
absolute image path (the env resolves it from task.context.images).
"""
import os, ast, json, operator, urllib.request, urllib.parse

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

# ---- GoogleSearch: frozen offline corpus by default ----
def _corpus_path():
    return os.environ.get("MH_SEARCH_CORPUS",
        os.path.join(os.path.dirname(__file__), "..", "benchmark_dataprocess", "MedCTA", "search_corpus.json"))
def google_search(query):
    q = str(query or "").strip()
    if os.environ.get("MH_SEARCH_MODE") == "live":
        try:
            req = urllib.request.Request("https://duckduckgo.com/html/?q=" + urllib.parse.quote(q),
                                         headers={"User-Agent": "Mozilla/5.0"})
            html = urllib.request.urlopen(req, timeout=15).read().decode("utf-8", "ignore")
            return "[live search snippet]\n" + html[:800]
        except Exception as e:
            return "[search error] %r" % e
    # frozen corpus: exact-key then substring match
    try:
        corpus = json.load(open(_corpus_path()))
    except Exception:
        corpus = {}
    if q in corpus: return corpus[q]
    for k, v in corpus.items():
        if k.lower() in q.lower() or q.lower() in k.lower(): return v
    return "[no offline result for query] " + q

# ---- Image tools: local VLM ----
def image_description(image_path):
    from vlm_backend import get_backend
    return get_backend().image_description(image_path)
def region_attribute_description(image_path, bbox=None, attribute=None):
    from vlm_backend import get_backend
    return get_backend().region_attribute_description(image_path, bbox=bbox, attribute=attribute)
def ocr(image_path):
    from vlm_backend import get_backend
    return get_backend().ocr(image_path)

# ---- dispatch (tool name -> callable on (args, image_path)) ----
def run_tool(name, args, image_path=None):
    a = args or {}
    if name == "Calculator":
        return calculator(a.get("expression") or a.get("expr") or a.get("text") or "")
    if name == "GoogleSearch":
        return google_search(a.get("query") or a.get("text") or a.get("keyword") or "")
    if name == "OCR":
        return ocr(image_path)
    if name == "ImageDescription":
        return image_description(image_path)
    if name == "RegionAttributeDescription":
        return region_attribute_description(
            image_path,
            bbox=a.get("bbox") or a.get("region"),
            attribute=a.get("attribute") or a.get("attr"))
    return "[unknown tool] " + str(name)
