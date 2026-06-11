"""
Side-by-side comparison: reference sample_inference.py path vs our pipeline.

Loads the Phi5 model once, then runs the SAME image+prompt through:
  1. Reference path: original Siglip2ImageProcessorNoUpscale (PIL, pads to 3600)
     + tokenizer_image_token from processing_bunny_phi4.py
  2. Our path: Phi5ImageProcessor.process_batched (GPU, no padding)
     + _tokenizer_image_token from phi5/backbone.py

Compares the generated text token-by-token.

Usage:
    cd /home/gmullins/workspace/rho
    python tests/policies/compare_phi5_generation.py
"""

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

MODEL_PATH = "/data/phi-5-5B/Phi-4-vision-5B-frbxq"
IMAGE_PATH = "/data/phi-5-5B/Phi-4-vision-5B-frbxq/video_first_frame_256.png"

# ── Load model & processors ────────────────────────────────────────────

print("Loading model...")
ref_processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="cuda",
).eval()
print(f"Model loaded on {model.device}")

# Reference helpers — we can't import processing_bunny_phi4 directly
# (it uses relative imports), so we inline the two functions we need.
# These are exact copies of what sample_inference.py uses.
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"


def ref_tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    """Exact copy of tokenizer_image_token from processing_bunny_phi4.py."""
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split(DEFAULT_IMAGE_TOKEN)]

    def insert_separator(x, sep):  # noqa: N803
        return [ele for sublist in zip(x, [sep] * len(x), strict=False) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])
    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])
    if return_tensors == "pt":
        return torch.tensor(input_ids, dtype=torch.long)
    return input_ids


def ref_process_images(images, image_processor, model_cfg=None):
    """Exact copy of process_images from processing_bunny_phi4.py."""
    return image_processor(images, return_tensors="pt")


from rho.policies.rhoalpha.phi5.backbone import (  # noqa: E402
    _tokenizer_image_token as our_tokenizer_image_token,
)
from rho.policies.rhoalpha.phi5.processing_phi5 import Phi5ImageProcessor  # noqa: E402

our_image_processor = Phi5ImageProcessor.from_siglip2_processor(ref_processor.image_processor, device="cuda")

# ── Prepare prompt (same for both paths) ────────────────────────────────

# Use the video frame + "Top slider" prompt from sample.ipynb
object_name = "Top slider"
system_message = (  # noqa: E501
    "A chat between a curious user and an artificial intelligence assistant."
    " The assistant gives helpful, detailed, and polite answers to the user's questions."
)
user_content = (  # noqa: E501
    f"<image>\nLocate the physical object this instruction describes: {object_name}."
    " Output its bbox coordinates using the format of [x1, y1, x2, y2]"
    " using relative coordinates from 0.0000 to 1.0000."
)
messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_content},
]
prompt = ref_processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"\n>>> Prompt:\n{prompt}\n")

# Load the real image (same as sample_inference.py would)
image_pil = Image.open(IMAGE_PATH).convert("RGB")
print(f"Image size: {image_pil.size}")

import torchvision.transforms.functional as TF  # noqa: E402

image_tensor = TF.to_tensor(image_pil).to("cuda")  # (3, H, W) float [0,1]
print(f"Image tensor shape: {image_tensor.shape}")

# ══════════════════════════════════════════════════════════════════════════
# PATH 1: Reference (sample_inference.py style — PIL + padded to 3600)
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PATH 1: Reference (original processor, padded to 3600)")
print("=" * 70)

# Tokenize with IMAGE_TOKEN_INDEX (-200) sentinels
ref_input_ids = (
    ref_tokenizer_image_token(prompt, ref_processor.tokenizer, return_tensors="pt").unsqueeze(0).to("cuda")
)

# Process image through original processor (PIL → padded patches)
ref_images = ref_process_images([image_pil], ref_processor.image_processor, model.config)
ref_images = {
    k: v.to("cuda", torch.float16) if v.is_floating_point() else v.to("cuda") for k, v in ref_images.items()
}

print(f"  input_ids shape:          {ref_input_ids.shape}")
print(f"  pixel_values shape:       {ref_images['pixel_values'].shape}")
print(f"  pixel_attention_mask sum: {ref_images['pixel_attention_mask'].sum().item():.0f} active patches")
print(f"  spatial_shapes:           {ref_images['spatial_shapes'].tolist()}")

with torch.inference_mode():
    ref_generate_ids = model.generate(
        input_ids=ref_input_ids,
        images=ref_images,
        max_new_tokens=1024,
        eos_token_id=ref_processor.tokenizer.eos_token_id,
        do_sample=False,
    )

ref_output_ids = ref_generate_ids[:, ref_input_ids.shape[1] :]
ref_response = ref_processor.tokenizer.decode(ref_output_ids[0], skip_special_tokens=True)
print(f"\n>>> Reference response:\n{ref_response}")

# ══════════════════════════════════════════════════════════════════════════
# PATH 2: Our pipeline (GPU tensor → process_batched, no padding)
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PATH 2: Our pipeline (Phi5ImageProcessor.process_batched, no padding)")
print("=" * 70)

# Tokenize with our tokenizer (same sentinel logic)
our_input_ids = (
    our_tokenizer_image_token(prompt, ref_processor.tokenizer, return_tensors="pt").unsqueeze(0).to("cuda")
)

# Process image through our GPU processor (no padding)
our_images = our_image_processor.process_batched(image_tensor.unsqueeze(0))
our_images = {
    k: v.to("cuda", torch.float16) if v.is_floating_point() else v.to("cuda") for k, v in our_images.items()
}

print(f"  input_ids shape:          {our_input_ids.shape}")
print(f"  pixel_values shape:       {our_images['pixel_values'].shape}")
print(f"  pixel_attention_mask sum: {our_images['pixel_attention_mask'].sum().item():.0f} active patches")
print(f"  spatial_shapes:           {our_images['spatial_shapes'].tolist()}")

with torch.inference_mode():
    our_generate_ids = model.generate(
        input_ids=our_input_ids,
        images=our_images,
        max_new_tokens=1024,
        eos_token_id=ref_processor.tokenizer.eos_token_id,
        do_sample=False,
    )

our_output_ids = our_generate_ids[:, our_input_ids.shape[1] :]
our_response = ref_processor.tokenizer.decode(our_output_ids[0], skip_special_tokens=True)
print(f"\n>>> Our response:\n{our_response}")

# ══════════════════════════════════════════════════════════════════════════
# Comparison
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)

# Token-level comparison
ref_tokens = ref_output_ids[0].tolist()
our_tokens = our_output_ids[0].tolist()
min_len = min(len(ref_tokens), len(our_tokens))
matching = sum(1 for a, b in zip(ref_tokens, our_tokens, strict=False) if a == b)

print(f"Reference tokens: {len(ref_tokens)}")
print(f"Our tokens:       {len(our_tokens)}")
print(f"Matching tokens:  {matching}/{min_len} ({matching / max(min_len, 1) * 100:.1f}%)")
print(f"Responses identical: {ref_response == our_response}")

if ref_response != our_response:
    # Find first divergence
    for i, (a, b) in enumerate(zip(ref_tokens, our_tokens, strict=False)):
        if a != b:
            ref_word = ref_processor.tokenizer.decode([a])
            our_word = ref_processor.tokenizer.decode([b])
            print(f"\nFirst divergence at token {i}:")
            print(f"  Reference: '{ref_word}' (id={a})")
            print(f"  Ours:      '{our_word}' (id={b})")
            break

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
