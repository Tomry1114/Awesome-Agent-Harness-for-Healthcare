import sys, os, time
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image

model_path = sys.argv[1]; img_path = sys.argv[2]
prompt = sys.argv[3] if len(sys.argv)>3 else "You are a medical imaging assistant. Describe this image: state the imaging modality, the anatomy shown, and any notable findings. Be concise and factual."
assert os.path.exists(img_path), "image not found: " + img_path
dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print("[device] cuda=%s %s" % (torch.cuda.is_available(), dev), flush=True)
t0=time.time()
proc = AutoProcessor.from_pretrained(model_path)
model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
print("[load] %.1fs" % (time.time()-t0), flush=True)
img = Image.open(img_path).convert("RGB")
messages=[{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":prompt}]}]
inputs = proc.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
t1=time.time()
out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
gen = out[0][inputs["input_ids"].shape[1]:]
print("[gen] %.1fs" % (time.time()-t1))
print("=== ImageDescription output ===")
print(proc.decode(gen, skip_special_tokens=True))
