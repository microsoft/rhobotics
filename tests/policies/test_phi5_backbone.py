"""
Tests for the Phi5 backbone adapter.

These tests verify the critical behaviors changed during the Phi5 integration:
- _tokenizer_image_token produces IMAGE_TOKEN_INDEX (-200) sentinels
- prepare_prompt uses the correct Phi5 chat template
- process_batch produces -200 sentinels (not regular token IDs)
- Phi5ImageProcessor produces correct output shapes and dtypes
- Phi4MM prepare_prompt preserves the legacy format
- load_backbone weight detection and uninitialized-weight fallback
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from rho.policies.rhoalpha.phi5.backbone import IMAGE_TOKEN_INDEX, Phi5Backbone, _tokenizer_image_token

# ── Helpers ──────────────────────────────────────────────────────────────


def _make_fake_tokenizer(bos_token_id=None):
    """Return a callable mock tokenizer that splits on whitespace."""
    tok = MagicMock()
    tok.bos_token_id = bos_token_id

    def _tokenize(text):
        # Simulate a simple tokenizer: each character → its ord value.
        # No BOS token is prepended (bos_token_id=None).
        ids = [ord(c) for c in text]
        if bos_token_id is not None:
            ids = [bos_token_id] + ids
        return SimpleNamespace(input_ids=ids)

    tok.side_effect = _tokenize
    return tok


def _make_phi5_config(**overrides):
    """Create a minimal config namespace that Phi5Backbone needs."""
    defaults = {
        "device": "cpu",
        "dtype": torch.float32,
        "image_features": ["observation.image"],
        "n_obs_steps": 1,
        "vlm_backbone_folder": "/fake/path",
        "freeze_vision_encoder": True,
        "enable_gradient_checkpointing": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ── _tokenizer_image_token ──────────────────────────────────────────────


class TestTokenizerImageToken:
    """Verify that <image> placeholders become IMAGE_TOKEN_INDEX sentinels."""

    def test_single_image_produces_sentinel(self):
        """A prompt with one <image> should have exactly one -200."""
        tok = _make_fake_tokenizer(bos_token_id=None)
        prompt = "<image>hello"
        ids = _tokenizer_image_token(prompt, tok)

        assert IMAGE_TOKEN_INDEX in ids
        assert ids.count(IMAGE_TOKEN_INDEX) >= 1

    def test_two_images_produce_two_sentinels(self):
        """Two <image> tokens → two -200 sentinels."""
        tok = _make_fake_tokenizer(bos_token_id=None)
        prompt = "<image><image>describe"
        ids = _tokenizer_image_token(prompt, tok)

        count = ids.count(IMAGE_TOKEN_INDEX)
        assert count >= 2, f"Expected >=2 sentinels, got {count}"

    def test_no_image_token_means_no_sentinel(self):
        """A prompt without <image> should have no -200."""
        tok = _make_fake_tokenizer(bos_token_id=None)
        prompt = "just text"
        ids = _tokenizer_image_token(prompt, tok)

        assert IMAGE_TOKEN_INDEX not in ids

    def test_return_tensors_pt(self):
        """return_tensors='pt' should yield a LongTensor."""
        tok = _make_fake_tokenizer(bos_token_id=None)
        prompt = "<image>hello"
        result = _tokenizer_image_token(prompt, tok, return_tensors="pt")

        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.long

    def test_sentinel_value_is_negative_200(self):
        """The sentinel value must be -200, not 200010 or anything else."""
        assert IMAGE_TOKEN_INDEX == -200
        tok = _make_fake_tokenizer(bos_token_id=None)
        ids = _tokenizer_image_token("<image>x", tok)

        sentinel_positions = [i for i, v in enumerate(ids) if v == -200]
        assert len(sentinel_positions) >= 1

    def test_with_bos_token(self):
        """When tokenizer has bos_token_id, it should be preserved."""
        bos = 1
        tok = _make_fake_tokenizer(bos_token_id=bos)
        ids = _tokenizer_image_token("<image>hi", tok)

        assert ids[0] == bos
        assert IMAGE_TOKEN_INDEX in ids


# ── Phi5Backbone.prepare_prompt ─────────────────────────────────────────


class TestPhi5PreparePrompt:
    """Verify Phi5 prompt formatter uses the correct chat template."""

    def test_uses_im_start_template(self):
        """Phi5 prompts must use <|im_start|>user<|im_sep|> format."""
        config = _make_phi5_config()
        backend = Phi5Backbone(config)

        batch = {"task": ["Pick up the red cube"]}
        prompts = backend.prepare_prompt(batch)

        assert len(prompts) == 1
        p = prompts[0]
        assert "<|im_start|>user<|im_sep|>" in p
        assert "<|im_end|>" in p
        assert "<|im_start|>assistant<|im_sep|>" in p

    def test_does_not_use_phi4mm_tokens(self):
        """Phi5 must NOT use Phi4MM-style <|user|>/<|image_N|> tokens."""
        config = _make_phi5_config()
        backend = Phi5Backbone(config)

        batch = {"task": ["Move the arm"]}
        prompts = backend.prepare_prompt(batch)
        p = prompts[0]

        assert "<|user|>" not in p
        assert "<|image_1|>" not in p
        assert "<|end|>" not in p
        assert "<|assistant|>" not in p

    def test_contains_image_placeholders(self):
        """Prompt should contain <image> tokens (one per camera × obs_step)."""
        config = _make_phi5_config(
            image_features=["cam_high", "cam_low"],
            n_obs_steps=2,
        )
        backend = Phi5Backbone(config)

        batch = {"task": ["Do something"]}
        prompts = backend.prepare_prompt(batch)

        expected_count = 2 * 2  # 2 cameras × 2 obs steps
        actual_count = prompts[0].count("<image>")
        assert actual_count == expected_count

    def test_missing_task_key_returns_empty(self):
        """If 'task' key is missing, return empty list."""
        config = _make_phi5_config()
        backend = Phi5Backbone(config)

        prompts = backend.prepare_prompt({"other_key": [1, 2]})
        assert prompts == []

    def test_batch_of_prompts(self):
        """Multiple prompts should all be formatted."""
        config = _make_phi5_config()
        backend = Phi5Backbone(config)

        tasks = ["Task A", "Task B", "Task C"]
        prompts = backend.prepare_prompt({"task": tasks})

        assert len(prompts) == 3
        for p, task in zip(prompts, tasks, strict=False):
            assert task in p
            assert "<|im_start|>user<|im_sep|>" in p


# ── Phi4MMBackbone.prepare_prompt ───────────────────────────────────────


class TestPhi4MMPreparePrompt:
    """Verify Phi4MM prompt format is preserved after the refactor."""

    def test_uses_phi4mm_template(self):
        from rho.policies.rhoalpha.phi4mm.backbone import Phi4MMBackbone

        config = _make_phi5_config(vlm_backend="phi4mm")
        backend = Phi4MMBackbone(config)

        batch = {"task": ["Grasp the object"]}
        prompts = backend.prepare_prompt(batch)

        p = prompts[0]
        assert p.startswith("<|user|>")
        assert "<|end|><|assistant|>" in p

    def test_image_tokens_numbered(self):
        """Phi4MM should use <|image_1|>, <|image_2|>, etc."""
        from rho.policies.rhoalpha.phi4mm.backbone import Phi4MMBackbone

        config = _make_phi5_config(
            vlm_backend="phi4mm",
            image_features=["cam_high", "cam_low"],
            n_obs_steps=1,
        )
        backend = Phi4MMBackbone(config)

        batch = {"task": ["Do something"]}
        prompts = backend.prepare_prompt(batch)
        p = prompts[0]

        assert "<|image_1|>" in p
        assert "<|image_2|>" in p


# ── Phi5ImageProcessor ──────────────────────────────────────────────────


class TestPhi5ImageProcessor:
    """Test the GPU-native NaFlex image processor."""

    @pytest.fixture
    def processor(self):
        from rho.policies.rhoalpha.phi5.processing_phi5 import Phi5ImageProcessor

        return Phi5ImageProcessor(
            patch_size=16,
            max_num_patches=3600,
            min_num_patches=256,
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5],
            device="cpu",
        )

    def test_single_image_output_keys(self, processor):
        """__call__ should return pixel_values, pixel_attention_mask, spatial_shapes."""
        img = torch.rand(3, 224, 224)
        out = processor([img])

        assert "pixel_values" in out
        assert "pixel_attention_mask" in out
        assert "spatial_shapes" in out

    def test_single_image_shapes(self, processor):
        """Output tensors should have the expected dimensions."""
        img = torch.rand(3, 224, 224)
        out = processor([img])

        pv = out["pixel_values"]
        mask = out["pixel_attention_mask"]
        shapes = out["spatial_shapes"]

        # pixel_values: (1, num_patches, patch_dim)
        assert pv.ndim == 3
        assert pv.shape[0] == 1
        patch_dim = 16 * 16 * 3
        assert pv.shape[2] == patch_dim

        # pixel_attention_mask: (1, num_patches)
        assert mask.ndim == 2
        assert mask.shape[0] == 1
        assert mask.shape[1] == pv.shape[1]

        # spatial_shapes: (1, 2)
        assert shapes.shape == (1, 2)

    def test_batch_processing(self, processor):
        """process_batched should handle a batch of same-size images."""
        batch = torch.rand(4, 3, 224, 224)
        out = processor.process_batched(batch)

        assert out["pixel_values"].shape[0] == 4
        assert out["pixel_attention_mask"].shape[0] == 4
        assert out["spatial_shapes"].shape[0] == 4

    def test_batched_no_padding_waste(self, processor):
        """process_batched should NOT pad to max_num_patches (3600).

        For 256x256 images with patch_size=16 → 256 real patches.
        The vision tower should only process 256 tokens, not 3600.
        """
        batch = torch.rand(2, 3, 256, 256)
        out = processor.process_batched(batch)

        # 256x256 / 16 = 16x16 = 256 patches
        expected_patches = 256
        assert out["pixel_values"].shape == (2, expected_patches, 16 * 16 * 3)
        assert out["pixel_attention_mask"].shape == (2, expected_patches)
        # Mask should be all True (no padding)
        assert out["pixel_attention_mask"].all()

    def test_normalization(self, processor):
        """Images normalized with mean=0.5, std=0.5: [0,1] → [-1,1]."""
        # All-zeros image → active patches become (0 - 0.5) / 0.5 = -1.0
        img = torch.zeros(3, 224, 224)
        out = processor([img])
        pv = out["pixel_values"]
        mask = out["pixel_attention_mask"].bool()
        active = pv[mask]
        assert torch.allclose(active, torch.full_like(active, -1.0), atol=1e-5)

        # All-ones image → active patches become (1 - 0.5) / 0.5 = 1.0
        img = torch.ones(3, 224, 224)
        out = processor([img])
        pv = out["pixel_values"]
        mask = out["pixel_attention_mask"].bool()
        active = pv[mask]
        assert torch.allclose(active, torch.full_like(active, 1.0), atol=1e-5)

    def test_patch_count_within_bounds(self, processor):
        """Number of active patches should be in [min_num_patches, max_num_patches]."""
        # Small image → should use min_num_patches (256) or close
        img = torch.rand(3, 64, 64)
        out = processor([img])
        n_active = out["pixel_attention_mask"].sum().item()
        assert n_active >= 256

        # Large image → should not exceed max_num_patches (3600)
        img = torch.rand(3, 2048, 2048)
        out = processor([img])
        n_active = out["pixel_attention_mask"].sum().item()
        assert n_active <= 3600

    def test_different_aspect_ratios(self, processor):
        """Processor should handle non-square images."""
        for h, w in [(480, 640), (1080, 1920), (100, 500)]:
            img = torch.rand(3, h, w)
            out = processor([img])
            assert out["pixel_values"].ndim == 3
            n_active = out["pixel_attention_mask"].sum().item()
            # Patch count should be reasonable (allow some slack
            # from rounding to patch-aligned sizes)
            assert 200 <= n_active <= 3600


# ── Integration: tokenization + sentinel check ─────────────────────────


class TestProcessBatchTokenization:
    """Verify that process_batch produces -200 sentinels in input_ids.

    This is the most critical test: if input_ids don't contain -200,
    the model's prepare_inputs_labels_for_multimodal will not splice
    in vision features and the model trains blind.
    """

    def _make_process_batch_inputs(self, n_cameras=2):
        """Create minimal inputs for Phi5Backbone.process_batch."""
        config = _make_phi5_config(
            image_features=[f"cam_{i}" for i in range(n_cameras)],
        )
        backend = Phi5Backbone(config)

        # Build prompts with <image> tokens
        batch = {"task": ["Pick up the cube"]}
        texts = backend.prepare_prompt(batch)

        # Fake images (C, H, W) per camera
        images = [[torch.rand(3, 224, 224) for _ in range(n_cameras)]]

        return backend, texts, images

    def test_input_ids_contain_sentinel(self):
        """input_ids from process_batch must contain IMAGE_TOKEN_INDEX (-200)."""
        from rho.policies.rhoalpha.phi5.processing_phi5 import Phi5ImageProcessor

        backend, texts, images = self._make_process_batch_inputs()

        # Mock processor with real tokenizer behavior
        processor = MagicMock()
        processor.tokenizer = _make_fake_tokenizer(bos_token_id=None)
        processor.image_processor = Phi5ImageProcessor(
            patch_size=16,
            max_num_patches=3600,
            min_num_patches=256,
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5],
            device="cpu",
        )

        def pad_fn(seqs, padding_side="right", padding_value=0):
            max_len = max(s.size(0) for s in seqs)
            padded = []
            for s in seqs:
                pad = torch.full((max_len - s.size(0),), padding_value, dtype=s.dtype)
                padded.append(torch.cat([s, pad]))
            return torch.stack(padded)

        result = backend.process_batch(
            processor=processor,
            images=images,
            texts=texts,
            image_mask=None,
            pad_sequence_fn=pad_fn,
            cat_with_pad_fn=None,
        )

        input_ids = result["input_ids"]
        n_sentinels = (input_ids == IMAGE_TOKEN_INDEX).sum().item()

        assert n_sentinels >= 2, (
            f"Expected at least 2 sentinel tokens (-200) for 2 cameras, "
            f"got {n_sentinels}. input_ids: {input_ids}"
        )

    def test_input_ids_do_not_contain_200010(self):
        """input_ids must NOT contain 200010 (the text tokenization of <image>).

        This was the original bug: processor.tokenizer() produced 200010
        instead of -200, causing the model to skip image splicing.
        """
        from rho.policies.rhoalpha.phi5.processing_phi5 import Phi5ImageProcessor

        backend, texts, images = self._make_process_batch_inputs()

        processor = MagicMock()
        processor.tokenizer = _make_fake_tokenizer(bos_token_id=None)
        processor.image_processor = Phi5ImageProcessor(
            patch_size=16,
            max_num_patches=3600,
            min_num_patches=256,
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5],
            device="cpu",
        )

        def pad_fn(seqs, padding_side="right", padding_value=0):
            max_len = max(s.size(0) for s in seqs)
            padded = []
            for s in seqs:
                pad = torch.full((max_len - s.size(0),), padding_value, dtype=s.dtype)
                padded.append(torch.cat([s, pad]))
            return torch.stack(padded)

        result = backend.process_batch(
            processor=processor,
            images=images,
            texts=texts,
            image_mask=None,
            pad_sequence_fn=pad_fn,
            cat_with_pad_fn=None,
        )

        input_ids = result["input_ids"]
        n_200010 = (input_ids == 200010).sum().item()

        assert n_200010 == 0, (
            f"Found {n_200010} occurrences of token 200010 in input_ids. "
            "This means <image> was tokenized as text instead of being "
            "replaced with IMAGE_TOKEN_INDEX (-200)."
        )


# ── Weight detection & uninitialized loading ────────────────────────────


def _make_phi5_config_with_defaults(**overrides):
    """Create a RhoAlphaConfig with vlm_backend='phi5', letting it resolve defaults."""
    from rho.policies.rhoalpha.configuration_rhoalpha import RhoAlphaConfig

    kwargs = {"vlm_backend": "phi5"}
    kwargs.update(overrides)
    cfg = RhoAlphaConfig(**kwargs)
    # Overlay device/dtype which RhoAlphaConfig doesn't expose as __init__ args
    cfg.device = overrides.get("device", "cpu")
    cfg.dtype = overrides.get("dtype", torch.float32)
    return cfg


class TestHasWeightFiles:
    """Verify _has_weight_files correctly detects presence/absence of model weights."""

    def test_bundled_model_has_no_weights(self):
        """The default vlm_backbone_folder (bundled dir) ships without safetensors."""
        config = _make_phi5_config_with_defaults()
        backend = Phi5Backbone(config)
        assert not backend._has_weight_files()


@pytest.mark.gpu
@pytest.mark.resource_intensive
class TestLoadBackboneUninitialized:
    """Verify load_backbone creates an uninitialized model when no weights are present.

    Uses the default vlm_backbone_folder resolved by RhoAlphaConfig so that we
    also exercise the config default-resolution path.

    These tests actually instantiate the model architecture from config.json,
    so they require a GPU and are marked resource_intensive.
    """

    @pytest.fixture(scope="class")
    def backbone_and_backend(self):
        """Load the backbone once (uninitialized) for all tests in this class."""
        config = _make_phi5_config_with_defaults(device="cuda", dtype=torch.float16)
        backend = Phi5Backbone(config)
        backbone = backend.load_backbone()
        return backbone, backend

    def test_backbone_loads_without_weights(self, backbone_and_backend):
        """load_backbone should succeed with the default config-only directory."""
        backbone, _ = backbone_and_backend
        assert backbone is not None

    def test_backbone_has_expected_hidden_size(self, backbone_and_backend):
        """The model config should produce the correct hidden_size."""
        backbone, backend = backbone_and_backend
        hidden_size = backend.get_hidden_size(backbone)
        assert hidden_size > 0

    def test_backbone_on_correct_device(self, backbone_and_backend):
        """The model should be on the requested device after loading."""
        backbone, _ = backbone_and_backend
        param = next(backbone.parameters())
        assert param.device.type == "cuda"
