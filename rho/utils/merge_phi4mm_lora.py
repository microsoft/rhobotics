#!/usr/bin/env python3
"""
Merge Phi4MM vision LoRA into base weights and save checkpoint.

What this script does:
1. Loads Phi4MM
2. Removes audio components (encoder + LoRA) - saves ~600M params
3. Merges vision LoRA into base weights
4. Saves the model

Usage:
    python -m rho.utils.merge_phi4mm_lora --output-dir ./phi4mm-vision-merged
    python -m rho.utils.merge_phi4mm_lora --model-path microsoft/Phi-4-multimodal-instruct \
        --output-dir ./phi4mm-vision-merged --dtype bfloat16
"""

import argparse
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

logger = logging.getLogger(__name__)


def remove_audio_components(model):
    """Remove audio encoder and audio LoRA (following Microsoft's sample_finetune_vision.py)."""
    logger.info("Removing audio components...")

    # Remove audio encoder
    if hasattr(model.model, "embed_tokens_extend") and hasattr(
        model.model.embed_tokens_extend, "audio_embed"
    ):
        del model.model.embed_tokens_extend.audio_embed
        logger.info("  Removed audio encoder")

    # Remove audio/speech LoRA from all layers
    removed_count = 0
    for layer in model.model.layers:
        for proj_name in ["down_proj", "gate_up_proj"]:
            proj = getattr(layer.mlp, proj_name, None)
            if proj and hasattr(proj, "lora_A") and hasattr(proj.lora_A, "speech"):
                del proj.lora_A.speech
                del proj.lora_B.speech
                removed_count += 1
        for proj_name in ["o_proj", "qkv_proj"]:
            proj = getattr(layer.self_attn, proj_name, None)
            if proj and hasattr(proj, "lora_A") and hasattr(proj.lora_A, "speech"):
                del proj.lora_A.speech
                del proj.lora_B.speech
                removed_count += 1

    logger.info(f"  Removed audio LoRA ({removed_count} modules)")


def merge_vision_lora(model):
    """Merge vision LoRA into base weights."""
    logger.info("Merging vision LoRA into base weights...")
    from peft.tuners.lora.layer import LoraLayer

    merged_count = 0
    for _, module in model.named_modules():
        if isinstance(module, LoraLayer) and hasattr(module, "lora_A") and "vision" in module.lora_A:
            module.merge(adapter_names=["vision"])
            merged_count += 1

    logger.info(f"  Merged vision LoRA ({merged_count} layers)")


def merge_and_save_phi4mm(
    model_path: str,
    output_dir: Path,
    dtype: torch.dtype,
):
    """Merge vision LoRA into Phi4MM base weights and save."""
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading Phi4MM from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        _attn_implementation="sdpa",
    )

    logger.info(f"Loading processor from {model_path}...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    # Count parameters before
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters before: {total_params:,}")

    # Remove audio components
    remove_audio_components(model)

    # Merge vision LoRA
    merge_vision_lora(model)

    # Count parameters after
    total_params_after = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters after: {total_params_after:,}")
    logger.info(f"Saved: {(total_params - total_params_after):,} parameters")

    # Save model and processor
    logger.info(f"Saving to {output_dir}...")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    logger.info("Merge completed successfully!")
    logger.info(f"Model saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge Phi4MM vision LoRA into base weights and save checkpoint"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="microsoft/Phi-4-multimodal-instruct",
        help="HuggingFace model ID or local path to Phi4MM (default: microsoft/Phi-4-multimodal-instruct)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save merged checkpoint",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for model weights (default: bfloat16)",
    )

    args = parser.parse_args()

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    merge_and_save_phi4mm(
        model_path=args.model_path,
        output_dir=args.output_dir,
        dtype=dtype_map[args.dtype],
    )


if __name__ == "__main__":
    main()
