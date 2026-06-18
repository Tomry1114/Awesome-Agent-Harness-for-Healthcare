import sys, os, time
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
model_path, img_path, question = sys.argv[1], sys.argv[2], sys.argv[3]
proc = AutoProcessor.from_pretrained(model_path)
model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, dtype="auto", device_map="auto")
img = Image.open(img_path).convert("RGB")
msgs=[{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":question}]}]
inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
out = model.generate(**inp, max_new_tokens=200, do_sample=False)
print("=== ANSWER (%s) ===" % os.path.basename(model_path))
print(proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip())
