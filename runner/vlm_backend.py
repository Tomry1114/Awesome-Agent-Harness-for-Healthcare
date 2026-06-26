"""Pluggable VLM backend for MedCTA v1 image tools.

Default = LOCAL open-weight Qwen3-VL (no API key, runs on a srun-allocated GPU). Same interface
(image_description / region_attribute_description / ocr) can be backed by an API later by swapping
the class — the harness only depends on get_backend(). Model is loaded once (singleton) per process.

Select model via env MH_VLM_PATH (default ~/hf_models/Qwen3-VL-2B-Instruct). Validated: Qwen3-VL-2B
on an A40 ~6s load / ~8s per image.
"""
import os, functools
import gateway
from PIL import Image

IMAGE_DESC_PROMPT = ("You are a medical imaging assistant. Describe this image factually: the imaging "
                     "modality, the anatomy shown, and any notable findings. Be concise. Do NOT invent "
                     "findings that are not visible.")
OCR_PROMPT = ("Transcribe ALL text visible in this image verbatim, preserving reading order. "
              "If there is no text, reply exactly: [no text].")

def _region_prompt(attribute=None):
    focus = attribute or "the region of interest"
    return ("You are a medical imaging assistant. The image shown IS the region of interest. "
            "Describe %s factually: shape, density/intensity, margins, and size if estimable. "
            "Do NOT invent findings that are not visible." % focus)

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
    q = region_query or attribute or (bbox if isinstance(bbox, str) else None)
    cropped, ok = (_crop_to_bbox(img, bbox) if bbox is not None else (img, False))
    if ok:
        prompt = _region_prompt(attribute); mode, resolved = "bbox", True
    elif q:
        prompt = ("You are a medical imaging assistant. Focus ONLY on this region of the image: %s. "
                  "Describe its shape, density/intensity, margins, and size if estimable. Do NOT invent "
                  "findings that are not visible." % q)
        cropped = img; mode, resolved = "semantic", True
    else:
        prompt = _region_prompt(None); cropped = img; mode, resolved = "none", False
    text = gen_fn(cropped, prompt)
    return {"text": text, "localization": {"requested": (region_query or bbox), "mode": mode, "resolved": resolved}}


class LocalQwen3VL:
    name = "local:qwen3vl"
    def __init__(self, model_path=None):
        self.model_path = model_path or os.environ.get(
            "MH_VLM_PATH", os.path.expanduser("~/hf_models/Qwen3-VL-2B-Instruct"))
        self._model = None; self._proc = None
    def _ensure(self):
        if self._model is not None: return
        os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        self._proc = AutoProcessor.from_pretrained(self.model_path)
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path, torch_dtype="auto", device_map="auto")
    def _gen(self, image, prompt, max_new_tokens=256):
        self._ensure()
        if isinstance(image, str): image = Image.open(image).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                  {"type": "text", "text": prompt}]}]
        inputs = self._proc.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                return_dict=True, return_tensors="pt").to(self._model.device)
        out = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self._proc.decode(gen, skip_special_tokens=True).strip()
    def image_description(self, image): return self._gen(image, IMAGE_DESC_PROMPT)
    def region_attribute_description(self, image, bbox=None, attribute=None, region_query=None):
        from PIL import Image as _Image
        img = _Image.open(image).convert("RGB") if isinstance(image, str) else image
        return _region_attr(lambda im, pr: self._gen(im, pr), img, bbox, attribute, region_query)
    def ocr(self, image): return self._gen(image, OCR_PROMPT, max_new_tokens=512)
    def chat(self, messages, max_new_tokens=512):
        """Text-only generation for the agent BRAIN (no image in context). messages = list of
        {role, content}. Returns the raw decoded completion (the agent parses tool_call/answer)."""
        self._ensure()
        # Qwen3-VL processor requires structured content (list of typed parts) even for text-only turns
        norm = []
        for m in messages:
            c = m.get("content")
            if isinstance(c, str):
                c = [{"type": "text", "text": c}]
            norm.append({"role": m["role"], "content": c})
        inputs = self._proc.apply_chat_template(norm, tokenize=True, add_generation_prompt=True,
                                                return_dict=True, return_tensors="pt").to(self._model.device)
        out = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self._proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class ApiVLM:
    """API-backed VLM (gateway multimodal model, default gemini-2.5-flash). Same interface as
    LocalQwen3VL so MedCTA image tools run with NO local GPU. Real pixel grounding is preserved:
    region_attribute_description crops the PIL image and sends ONLY the crop (not coordinates in text)."""
    name = "api:gateway"
    def __init__(self, model=None):
        self.model = model or os.environ.get("MH_VLM_API_MODEL", "gpt-5.5")  # micuapi has gpt-5.x, no gemini
        _b = (os.environ.get("MH_VLM_API_BASE") or os.environ.get("MH_OPENAI_BASE") or "https://www.micuapi.ai").rstrip("/")
        if _b.endswith("/v1"): _b = _b[:-3].rstrip("/")  # normalize: callers append /v1 (consistent with api_agent/gacc/mm_judge)
        self.base = _b
        # VLM key is INDEPENDENT of the agent key (MH_VLM_API_KEY first): in a mixed-provider run the agent
        # brain may use a gemini/deepseek-only key while VLM perception still needs a gpt-5.x key.
        self.key = (os.environ.get("MH_VLM_API_KEY") or os.environ.get("MH_OPENAI_KEY")
                    or os.environ.get("OPENAI_API_KEY"))
        if not self.key:
            kp = os.path.expanduser("~/.xbai_key")
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
        # Migrated to the unified gateway HTTP client (Codex #2). gateway makes the last user message
        # multimodal from image_path, so write the (possibly cropped) PIL image to a temp file and pass
        # its path -- this preserves REAL pixel grounding: the model still sees only the crop bytes, not
        # coordinates in text. A str image is used directly. Contract unchanged: stripped text on
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
                               key=self.key)
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
    def ocr(self, image): return self._call(image, OCR_PROMPT, max_tokens=512)

@functools.lru_cache(maxsize=1)
def get_backend():
    """Return the configured VLM backend (singleton). MH_VLM_BACKEND=local (default)."""
    kind = os.environ.get("MH_VLM_BACKEND", "api")  # default gpt-5.5 vision via gateway (was local Qwen3-VL-2B; gpt is multimodal, no reason to bottleneck perception on a weak 2B)
    if kind == "local":
        return LocalQwen3VL()
    if kind == "api":
        return ApiVLM()
    raise ValueError("unknown MH_VLM_BACKEND: %s (use api=gateway gpt-5.5 [default] or local=Qwen3-VL)" % kind)
