"""
Sample inference script for Bunny-Phi4.

Usage:
    cd phi4mm
    python sample_inference.py
"""
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import os

model_path = "/datadisk/checkpoints/neel-p0-phimm14b-r-nflx-snglmix-1202-lowlr-bsn6l-with-hfclass/phi4mm"  # Change to your model full path

# Load model and processor
print("Loading model...")
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
).eval()

# Import helpers for image processing
from processing_bunny_phi4 import tokenizer_image_token, process_images, DEFAULT_IMAGE_TOKEN

print(f"Model loaded on {model.device}")

#################################################### text-only ####################################################
print("\n" + "="*60)
print("TEST: Text-only generation")
print("="*60)

messages = [{"role": "user", "content": "What is the answer for 1+1? Explain it."}]
prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print(f">>> Prompt\n{prompt}")
inputs = processor(prompt, images=None, return_tensors="pt").to("cuda:0")
generate_ids = model.generate(
    **inputs,
    max_new_tokens=256,
    eos_token_id=processor.tokenizer.eos_token_id,
    do_sample=False,
)
generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(f'>>> Response\n{response}')

#################################################### single image ####################################################
print("\n" + "="*60)
print("TEST: Single image understanding")
print("="*60)

messages = [{"role": "user", "content": DEFAULT_IMAGE_TOKEN + "\nDescribe this image in detail."}]
prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

image_path = os.path.join(model_path, "330px-Cidade_Maravilhosa.jpg")
print(f">>> Loading image from {image_path}")
image = Image.open(image_path).convert("RGB")
print(f"Image size: {image.size}")

print(f">>> Prompt\n{prompt}")

# Tokenize with image token handling
input_ids = tokenizer_image_token(prompt, processor.tokenizer, return_tensors='pt').unsqueeze(0).to("cuda:0")

# Process image
images = process_images([image], processor.image_processor, model.config)
images = {k: v.to("cuda:0", torch.bfloat16) if v.is_floating_point() else v.to("cuda:0") for k, v in images.items()}

with torch.inference_mode():
    generate_ids = model.generate(
        input_ids=input_ids,
        images=images,
        max_new_tokens=256,
        eos_token_id=processor.tokenizer.eos_token_id,
        do_sample=False,
    )

generate_ids = generate_ids[:, input_ids.shape[1]:]
response = processor.tokenizer.decode(generate_ids[0], skip_special_tokens=True)
print(f'>>> Response\n{response}')

print("\n" + "="*60)
print("All tests completed!")
print("="*60)
