"""The VLM perception backend for MedCTA's image tools. **Exactly ONE implementation.**

`ApiVLM` = a multimodal model behind the gateway (MH_VLM_API_MODEL, default gpt-5.5). Upstream MedCTA's
own perception tools are ALSO VLM-backed — upstream `agentlego/benchmark.py` defines
ImageDescription and RegionAttributeDescription with `model='llava-7b-v1.5_vqa'` — so this is the native
MECHANISM with a stronger instance, not a substitute mechanism.

There is no local-model backend and no backend switch: a second path would let runs silently differ in
what the tools can see.

OCR is NOT here. It is the native EasyOCR engine in tools_medcta.ocr — a captioning VLM paraphrasing
text is a different measurement from a text recogniser.
"""
import os, functools
import gateway
from PIL import Image

IMAGE_DESC_PROMPT = ("You are a medical image analysis assistant. FIRST identify the image TYPE "
                     "(radiology CT/MRI/X-ray/ultrasound; histology / pathology microscopy; gross specimen "
                     "photograph; chart / graph / figure; clinical photograph). THEN describe it factually "
                     "using the features RELEVANT TO THAT TYPE: radiology -> modality, anatomy, density/signal, "
                     "notable findings; histology -> tissue architecture, cell & nuclear morphology, cytoplasm, "
                     "fibre/gland/crypt arrangement, staining pattern & chromogen colours (e.g. H&E = blue-"
                     "purple nuclei + pink cytoplasm), diagnostic features; gross specimen -> organ/tissue, "
                     "shape, colour, cut surface, lesions; chart -> title, axes, labels, series, values. Be "
                     "concise. Do NOT invent findings that are not visible, and do NOT force radiology "
                     "descriptors onto a non-radiology image.")
def _region_prompt(attribute=None):
    focus = attribute or "the region of interest"
    return ("You are a medical image analysis assistant. The image shown IS the region of interest. FIRST "
            "note the image TYPE (radiology / histology-microscopy / gross specimen / chart / photograph). "
            "THEN describe %s in detail using the features RELEVANT TO THAT TYPE -- histology: cell & nuclear "
            "morphology, cytoplasm, tissue architecture, fibre/gland/crypt arrangement, staining & chromogen "
            "colour; radiology: shape, density/signal, margins, size; gross specimen: colour, texture, cut "
            "surface, structure; chart: labels, axes, values. Report ONLY what is actually visible; do NOT "
            "force radiology descriptors (density/margins/size) onto a non-radiology image." % focus)

def _crop_to_bbox(img, bbox):
    """Crop a PIL image to bbox=[x1,y1,x2,y2]. Accepts pixel coords, normalized (<=1) coords, or a
    free-text/number string. Returns (image, applied_bool). This is REAL pixel grounding — the model
    only sees the cropped region, not a prompt that merely mentions coordinates."""
    try:
        import re as _re
        nums = bbox
        if isinstance(nums, str):
            nums = [float(x) for x in _re.findall(r"-?\d+\.?\d*", nums)]
        if not isinstance(nums, (list, tuple)) or len(nums) < 4:
            return img, False
        x1, y1, x2, y2 = [float(v) for v in nums[:4]]
        W, H = img.size
        if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.0:  # normalized 0..1
            x1, x2, y1, y2 = x1 * W, x2 * W, y1 * H, y2 * H
        x1, x2 = sorted((x1, x2)); y1, y2 = sorted((y1, y2))
        x1 = max(0, min(int(x1), W - 1)); x2 = max(x1 + 1, min(int(round(x2)), W))
        y1 = max(0, min(int(y1), H - 1)); y2 = max(y1 + 1, min(int(round(y2)), H))
        return img.crop((x1, y1, x2, y2)), True
    except Exception:
        return img, False

def _region_attr(gen_fn, img, bbox=None, attribute=None, region_query=None):
    """RegionAttributeDescription supporting bbox (precise pixel crop) OR region_query (semantic region,
    for a blind tool-mediated agent that cannot produce pixel coords). Returns an EXPLICIT localization
    status -- never a silent full-image fallback masquerading as a successful region analysis."""
    cropped, ok = (_crop_to_bbox(img, bbox) if bbox is not None else (img, False))
    # a semantic REGION must be carried into the prompt ALONGSIDE the attribute -- never dropped
    # in favour of the attribute. And a semantic prompt merely being SENT is NOT proof the region was
    # localized -> resolved is UNKNOWN (None), to be decided from the actual text (not auto-True).
    _region = region_query or (bbox if (isinstance(bbox, str) and not ok) else None)
    if ok:
        prompt = _region_prompt(attribute); mode, resolved = "bbox", True   # real pixel crop -> grounded
    elif _region:
        prompt = ("You are a medical image analysis assistant (note the image type first and use the "
                  "features relevant to it -- histology: morphology/staining/architecture; radiology: "
                  "density/margins/size). Focus on this region of the image: %s. Then evaluate "
                  "specifically: %s. Report the features RELEVANT TO THE IMAGE TYPE (histology: cell & nuclear "
                  "morphology, cytoplasm, architecture, staining/chromogen colour; radiology: shape, density, "
                  "margins, size; gross: colour/texture/structure; chart: labels/values); do NOT force "
                  "radiology descriptors onto a non-radiology image. If you cannot confidently localize that "
                  "region, SAY SO explicitly. Do NOT invent findings."
                  % (_region, attribute or "the region of interest"))
        cropped = img; mode, resolved = "semantic", None
    else:
        prompt = _region_prompt(attribute); cropped = img; mode, resolved = "none", None
    text = gen_fn(cropped, prompt)
    return {"text": text, "localization": {"requested": (region_query or bbox), "attribute": attribute,
                                           "mode": mode, "resolved": resolved}}


class ApiVLM:
    """THE VLM perception backend (gateway multimodal model, MH_VLM_API_MODEL, default gpt-5.5).
    No local GPU. Real pixel grounding is preserved: region_attribute_description crops the PIL image
    and sends ONLY the crop (not coordinates in text)."""
    name = "api:gateway"
    def __init__(self, model=None):
        self.model = model or os.environ.get("MH_VLM_API_MODEL", "gpt-5.5")  # the default model must be one the configured gateway serves
        _b = (os.environ.get("MH_VLM_API_BASE") or os.environ.get("MH_OPENAI_BASE") or "https://api.openai.com").rstrip("/")
        if _b.endswith("/v1"): _b = _b[:-3].rstrip("/")  # normalize: callers append /v1 (consistent with the agent, Gacc and multimodal judge clients)
        self.base = _b
        # VLM key is INDEPENDENT of the agent key (MH_VLM_API_KEY first): in a mixed-provider run the agent
        # brain may use a key for another provider while VLM perception still needs a key for the VLM model.
        self.key = (os.environ.get("MH_VLM_API_KEY") or os.environ.get("MH_OPENAI_KEY")
                    or os.environ.get("OPENAI_API_KEY"))
        if not self.key:
            kp = os.path.expanduser("~/.mh_openai_key")
            if os.path.exists(kp): self.key = open(kp).read().strip()
    _MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp"}
    def _encode(self, image):
        import io, base64
        if isinstance(image, str):
            ext = os.path.splitext(image)[1].lower()
            with open(image, "rb") as f: raw = f.read()
            return self._MIME.get(ext, "image/jpeg"), base64.b64encode(raw).decode()
        buf = io.BytesIO(); image.convert("RGB").save(buf, format="PNG")
        return "image/png", base64.b64encode(buf.getvalue()).decode()
    def _call(self, image, prompt, max_tokens=256):
        # Uses the unified gateway HTTP client. gateway makes the last user message
        # multimodal from image_path, so write the (possibly cropped) PIL image to a temp file and pass
        # its path -- this preserves REAL pixel grounding: the model still sees only the crop bytes, not
        # coordinates in text. A str image is used directly. Contract: stripped text on
        # success, "[vlm_api_error] <reason>" on failure.
        import tempfile
        tmp = None
        try:
            if isinstance(image, str):
                img_path = image
            else:
                fd, tmp = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                image.convert("RGB").save(tmp, format="PNG")
                img_path = tmp
            res = gateway.chat([{"role": "user", "content": prompt}], model=self.model,
                               max_tokens=max_tokens, judge=False, timeout=120, image_path=img_path,
                               key=self.key, base=self.base)
        finally:
            if tmp and os.path.exists(tmp):
                try: os.remove(tmp)
                except Exception: pass
        if res.get("ok"):
            return res["content"].strip()
        return "[vlm_api_error] " + (res.get("raw") or res.get("error_type") or "unknown")
    def image_description(self, image): return self._call(image, IMAGE_DESC_PROMPT)
    def region_attribute_description(self, image, bbox=None, attribute=None, region_query=None):
        img = Image.open(image).convert("RGB") if isinstance(image, str) else image
        return _region_attr(lambda im, pr: self._call(im, pr), img, bbox, attribute, region_query)

@functools.lru_cache(maxsize=1)
def get_backend():
    """Return THE VLM backend (singleton). One implementation on purpose -- no backend switch."""
    return ApiVLM()
