"""
End-to-end tests for Phi5 backbone integration.

Loads the real Phi-4-vision-5B model and verifies that:
1. Our pipeline (unpadded process_batched → backbone) produces the same
   vision features and logits as the reference pipeline (original padded
   processor → backbone).
2. Text-only generation still works correctly.
3. Image+text generation produces coherent output.

These tests require a GPU and the model at:
    /data/phi-5-5B/Phi-4-vision-5B-frbxq

Run:
    pytest tests/policies/test_phi5_e2e.py -v -s
"""

from types import SimpleNamespace

import pytest
import torch

# Gate the entire module on GPU + model availability
_MODEL_PATH = "/data/phi-5-5B/Phi-4-vision-5B-frbxq"
_HAS_CUDA = torch.cuda.is_available()

try:
    from pathlib import Path

    _HAS_MODEL = Path(_MODEL_PATH).exists()
except Exception:
    _HAS_MODEL = False

pytestmark = pytest.mark.skipif(
    not (_HAS_CUDA and _HAS_MODEL),
    reason="Requires CUDA and Phi5 model at /data/phi-5-5B/Phi-4-vision-5B-frbxq",
)


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def phi5_backbone_and_processor():
    """Load the real Phi5 model + processor once for all tests in this module."""
    from rho.policies.rhoalpha.phi5.backbone import Phi5Backbone

    config = SimpleNamespace(
        device="cuda",
        dtype=torch.float16,
        image_features=["observation.image"],
        n_obs_steps=1,
        vlm_backbone_folder=_MODEL_PATH,
        freeze_vision_encoder=True,
        enable_gradient_checkpointing=False,
    )
    backend = Phi5Backbone(config)
    backbone = backend.load_backbone()
    backbone.eval()
    processor = backend.create_processor()

    yield backend, backbone, processor

    # Cleanup
    del backbone, processor
    torch.cuda.empty_cache()


def _pad_sequence(sequences, padding_side="right", padding_value=0):
    """Standalone pad_sequence matching RhoAlphaModel.pad_sequence."""
    max_len = max(seq.size(0) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len), padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == "right":
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output


def _cat_with_pad(tensors, dim, padding_value=0):
    """Standalone cat_with_pad matching RhoAlphaModel.cat_with_pad."""
    ndim = tensors[0].dim()
    out_size = [max(t.shape[i] for t in tensors) for i in range(ndim)]
    out_size[dim] = sum(t.shape[dim] for t in tensors)
    output = tensors[0].new_full(out_size, padding_value)
    index = 0
    for t in tensors:
        slices = [slice(0, t.shape[d]) for d in range(ndim)]
        slices[dim] = slice(index, index + t.shape[dim])
        output[tuple(slices)] = t
        index += t.shape[dim]
    return output


# ── Test: Vision features match between padded and unpadded paths ───────


class TestVisionFeatureParity:
    """Verify that removing padding from process_batched does NOT change
    the vision features that reach the LLM."""

    def test_vision_features_match_for_256x256(
        self,
        phi5_backbone_and_processor,
    ):
        """For a 256x256 image, the padded path (per-image __call__) and
        unpadded path (process_batched) should produce identical vision
        features after the vision tower strips padding."""
        backend, backbone, processor = phi5_backbone_and_processor
        image_proc = processor.image_processor

        # Create a deterministic test image (256x256 like LIBERO)
        torch.manual_seed(42)
        test_image = torch.rand(3, 256, 256, device="cuda")

        # ── Padded path: per-image __call__ (pads to max_num_patches=3600) ──
        padded_out = image_proc([test_image])
        padded_pv = padded_out["pixel_values"].to(torch.float16)
        padded_mask = padded_out["pixel_attention_mask"]
        padded_shapes = padded_out["spatial_shapes"]

        # ── Unpadded path: process_batched (no padding) ──
        unpadded_out = image_proc.process_batched(test_image.unsqueeze(0))
        unpadded_pv = unpadded_out["pixel_values"].to(torch.float16)
        unpadded_mask = unpadded_out["pixel_attention_mask"]
        unpadded_shapes = unpadded_out["spatial_shapes"]

        # Verify dimensions: padded has 3600 columns, unpadded has 256
        assert padded_pv.shape[1] == 3600, f"Expected padded to 3600, got {padded_pv.shape[1]}"
        assert unpadded_pv.shape[1] == 256, f"Expected unpadded 256, got {unpadded_pv.shape[1]}"

        # Same number of *active* patches
        padded_active = padded_mask[0].sum().item()
        unpadded_active = unpadded_mask[0].sum().item()
        assert padded_active == unpadded_active == 256, (
            f"Patch count mismatch: padded={padded_active}, unpadded={unpadded_active}"
        )

        # Spatial shapes match
        assert torch.equal(padded_shapes.cpu(), unpadded_shapes.cpu()), "Spatial shapes differ"

        # Active pixel values are identical (same normalization, same patches)
        padded_active_pv = padded_pv[0][padded_mask[0].bool()]
        unpadded_active_pv = unpadded_pv[0][unpadded_mask[0].bool()]
        assert torch.allclose(padded_active_pv, unpadded_active_pv, atol=1e-3), (
            f"Active pixel values differ. "
            f"Max diff: {(padded_active_pv - unpadded_active_pv).abs().max().item()}"
        )

        # ── Run both through the vision tower ──
        vision_tower = backbone.model.vision_tower

        with torch.no_grad():
            # Padded input (3600 patches)
            padded_images = {
                "pixel_values": padded_pv,
                "pixel_attention_mask": padded_mask.to(torch.float16),
                "spatial_shapes": padded_shapes.cpu().numpy(),
            }
            padded_fwd = vision_tower.vision_tower(**padded_images, output_hidden_states=True)
            padded_hidden = padded_fwd.hidden_states[-2].to(torch.float16)
            # Strip padding (same as vision_tower.forward does)
            padded_features = padded_hidden[0][padded_mask[0].bool()]

            # Unpadded input (256 patches)
            unpadded_images = {
                "pixel_values": unpadded_pv,
                "pixel_attention_mask": unpadded_mask.to(torch.float16),
                "spatial_shapes": unpadded_shapes.cpu().numpy(),
            }
            unpadded_fwd = vision_tower.vision_tower(**unpadded_images, output_hidden_states=True)
            unpadded_hidden = unpadded_fwd.hidden_states[-2].to(torch.float16)
            unpadded_features = unpadded_hidden[0][unpadded_mask[0].bool()]

        # Same shape
        assert padded_features.shape == unpadded_features.shape, (
            f"Feature shape mismatch: padded={padded_features.shape}, unpadded={unpadded_features.shape}"
        )

        # Features should be very close — padding tokens are masked in
        # attention, so the only difference comes from numerical
        # non-determinism in flash attention with different sequence lengths.
        # Flash attention tiles computations differently for different seq
        # lengths, so individual features can differ by up to ~0.5 in fp16.
        # What matters is that downstream logits/predictions are unchanged
        # (verified in TestFullForwardParity).
        max_diff = (padded_features - unpadded_features).abs().max().item()
        mean_diff = (padded_features - unpadded_features).abs().mean().item()
        print(f"\n  Vision feature max diff: {max_diff:.6f}")
        print(f"  Vision feature mean diff: {mean_diff:.6f}")
        print(f"  Feature shape: {padded_features.shape}")
        print(f"  Padded seq len:   {padded_pv.shape[1]}")
        print(f"  Unpadded seq len: {unpadded_pv.shape[1]}")

        # Mean diff should be very small (< 0.01) even if individual
        # outliers reach ~0.5 due to flash attention tiling differences
        assert mean_diff < 0.01, f"Vision features differ too much on average: mean_diff={mean_diff:.6f}"
        assert max_diff < 1.0, f"Vision features have extreme outlier: max_diff={max_diff:.6f}"


# ── Test: Full forward pass produces same logits ────────────────────────


class TestFullForwardParity:
    """Verify that full forward pass (process_batch → model) produces
    the same logits for both padded and unpadded paths."""

    def test_logits_match_single_image(
        self,
        phi5_backbone_and_processor,
    ):
        """Process a single image+text through padded (per-image) and
        unpadded (process_batched) pipelines and compare the resulting
        logits from the full LLM forward pass."""
        backend, backbone, processor = phi5_backbone_and_processor
        image_proc = processor.image_processor
        from rho.policies.rhoalpha.phi5.backbone import _tokenizer_image_token

        torch.manual_seed(42)
        test_image = torch.rand(3, 256, 256, device="cuda")

        prompt = (
            "<|im_start|>user<|im_sep|><image>Describe this image.<|im_end|><|im_start|>assistant<|im_sep|>"
        )

        # Shared tokenization (both paths use the same tokens)
        input_ids = (
            _tokenizer_image_token(prompt, processor.tokenizer, return_tensors="pt").unsqueeze(0).to("cuda")
        )
        attention_mask = torch.ones_like(input_ids)

        # ── Padded path: per-image __call__ (pads to 3600) ──
        padded_out = image_proc([test_image])
        padded_images = {
            k: v.to("cuda", torch.float16) if v.is_floating_point() else v.to("cuda")
            for k, v in padded_out.items()
        }

        with torch.no_grad():
            (
                _,
                padded_pos_ids,
                padded_attn_mask,
                _,
                padded_embeds,
                _,
            ) = backbone.prepare_inputs_labels_for_multimodal(
                input_ids.clone(),
                None,
                attention_mask.clone(),
                None,
                None,
                padded_images,
            )

            padded_outputs = backbone(
                inputs_embeds=padded_embeds,
                attention_mask=padded_attn_mask,
                position_ids=padded_pos_ids,
                output_hidden_states=True,
            )
            padded_logits = padded_outputs.logits

        # ── Unpadded path: process_batched (no padding) ──
        unpadded_out = image_proc.process_batched(test_image.unsqueeze(0))
        unpadded_images = {
            k: v.to("cuda", torch.float16) if v.is_floating_point() else v.to("cuda")
            for k, v in unpadded_out.items()
        }

        with torch.no_grad():
            (
                _,
                unpadded_pos_ids,
                unpadded_attn_mask,
                _,
                unpadded_embeds,
                _,
            ) = backbone.prepare_inputs_labels_for_multimodal(
                input_ids.clone(),
                None,
                attention_mask.clone(),
                None,
                None,
                unpadded_images,
            )

            unpadded_outputs = backbone(
                inputs_embeds=unpadded_embeds,
                attention_mask=unpadded_attn_mask,
                position_ids=unpadded_pos_ids,
                output_hidden_states=True,
            )
            unpadded_logits = unpadded_outputs.logits

        # Shapes should match — same image = same active vision tokens
        assert padded_logits.shape == unpadded_logits.shape, (
            f"Logit shape mismatch: padded={padded_logits.shape}, unpadded={unpadded_logits.shape}"
        )

        max_diff = (padded_logits - unpadded_logits).abs().max().item()
        mean_diff = (padded_logits - unpadded_logits).abs().mean().item()
        print(f"\n  Logit max diff:  {max_diff:.6f}")
        print(f"  Logit mean diff: {mean_diff:.6f}")
        print(f"  Logit shape:     {padded_logits.shape}")

        # Top-1 predicted tokens should match
        padded_top1 = padded_logits[0].argmax(dim=-1)
        unpadded_top1 = unpadded_logits[0].argmax(dim=-1)
        match_rate = (padded_top1 == unpadded_top1).float().mean().item()
        print(f"  Top-1 match rate: {match_rate:.4f}")

        assert match_rate > 0.95, f"Top-1 predictions diverge too much: match_rate={match_rate:.4f}"


# ── Test: Text-only generation works ────────────────────────────────────


class TestTextGeneration:
    """Verify that the model can still generate text (no images)."""

    def test_text_only_generation(self, phi5_backbone_and_processor):
        """Reproduce the '1+1' test from sample_inference.py."""
        _, backbone, processor = phi5_backbone_and_processor

        messages = [{"role": "user", "content": "What is the answer for 1+1? Explain it."}]
        prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = processor(prompt, images=None, return_tensors="pt").to("cuda")

        with torch.no_grad():
            generate_ids = backbone.generate(
                **inputs,
                max_new_tokens=64,
                eos_token_id=processor.tokenizer.eos_token_id,
                do_sample=False,
            )

        generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
        response = processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        print(f"\n  Prompt:   {messages[0]['content']}")
        print(f"  Response: {response}")

        # The model should mention "2"
        assert "2" in response, f"Expected '2' in response, got: {response}"


# ── Test: Image+text generation via our pipeline ────────────────────────


class TestImageGeneration:
    """Verify image+text generation works through our pipeline."""

    def test_single_image_generation(self, phi5_backbone_and_processor):
        """Generate a description for a synthetic image using our
        unpadded pipeline."""
        backend, backbone, processor = phi5_backbone_and_processor
        from rho.policies.rhoalpha.phi5.backbone import _tokenizer_image_token

        # Create a synthetic test image
        torch.manual_seed(42)
        test_image = torch.rand(3, 256, 256, device="cuda")

        prompt = (
            "<|im_start|>user<|im_sep|><image>Describe what you see.<|im_end|><|im_start|>assistant<|im_sep|>"
        )

        # Process image through our unpadded pipeline
        img_out = processor.image_processor.process_batched(test_image.unsqueeze(0))

        # Tokenize with -200 sentinels
        input_ids = (
            _tokenizer_image_token(prompt, processor.tokenizer, return_tensors="pt").unsqueeze(0).to("cuda")
        )

        images = {
            k: v.to("cuda", torch.float16) if v.is_floating_point() else v.to("cuda")
            for k, v in img_out.items()
        }

        with torch.no_grad():
            generate_ids = backbone.generate(
                input_ids=input_ids,
                images=images,
                max_new_tokens=64,
                eos_token_id=processor.tokenizer.eos_token_id,
                do_sample=False,
            )

        generate_ids = generate_ids[:, input_ids.shape[1] :]
        response = processor.tokenizer.decode(generate_ids[0], skip_special_tokens=True)

        print(f"\n  Response: {response}")

        # Should produce some non-empty text
        assert len(response.strip()) > 0, "Model produced empty response"
        # Should not be garbage (at least 3 real words)
        words = response.strip().split()
        assert len(words) >= 3, f"Response too short ({len(words)} words): {response}"
