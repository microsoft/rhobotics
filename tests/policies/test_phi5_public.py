import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn
from transformers import Siglip2VisionConfig, Siglip2VisionModel, dynamic_module_utils

from rho.policies.rho.phi5.backbone import IMAGE_TOKEN_INDEX, Phi5Backbone, _tokenizer_image_token
from rho.policies.rho.phi5.processing_phi5 import Phi5ImageProcessor


def _fake_tokenizer(bos_token_id=None):
    tokenizer = MagicMock()
    tokenizer.bos_token_id = bos_token_id
    tokenizer.eos_token_id = 2
    tokenizer.model_max_length = 8192

    def tokenize(text, **kwargs):
        token_ids = [ord(character) for character in text]
        if bos_token_id is not None:
            token_ids.insert(0, bos_token_id)
        return SimpleNamespace(input_ids=token_ids)

    tokenizer.side_effect = tokenize
    return tokenizer


def test_tokenizer_image_token_inserts_sentinel():
    token_ids = _tokenizer_image_token("<image>move", _fake_tokenizer())

    assert token_ids.count(IMAGE_TOKEN_INDEX) == 1
    assert IMAGE_TOKEN_INDEX == -200


def test_tokenizer_image_token_supports_tensor_output():
    token_ids = _tokenizer_image_token(
        "<image>move",
        _fake_tokenizer(),
        return_tensors="pt",
    )

    assert isinstance(token_ids, torch.Tensor)
    assert token_ids.dtype == torch.long


def test_phi5_backbone_uses_public_configuration_contract():
    config = SimpleNamespace(
        device="cpu",
        dtype=torch.float32,
        image_features=["observation.image"],
        n_obs_steps=1,
        vlm_backbone_folder="/tmp/model",
        freeze_vision_encoder=True,
        enable_gradient_checkpointing=False,
    )

    backbone = Phi5Backbone(config)

    assert backbone.config is config


@pytest.fixture
def phi5_batch():
    backend = Phi5Backbone(SimpleNamespace(device="cpu", dtype=torch.float32))
    processor = SimpleNamespace(
        tokenizer=_fake_tokenizer(),
        image_processor=Phi5ImageProcessor(
            patch_size=2,
            min_num_patches=1,
            max_num_patches=4,
            image_mean=[0.5] * 3,
            image_std=[0.5] * 3,
            device="cpu",
        ),
    )
    images = torch.rand(3, 2, 3, 2, 4)
    return backend, processor, images, ["<image>a<image>b"] * 3


def _process_inputs(backend, processor, images, prompts, image_mask, lm):
    pad_fn = partial(nn.utils.rnn.pad_sequence, batch_first=True)
    if lm:
        return backend._build_lm_inputs(processor, images, prompts, ["answer"] * 3, image_mask, pad_fn)
    return backend.process_batch(processor, images, prompts, image_mask, pad_fn, None)


@pytest.mark.parametrize("lm", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize(
    "mask_values",
    [None, [[1, 1]] * 3, [[1, 0], [0, 0], [0, 1]], [[0, 0]] * 3],
)
def test_phi5_image_validity_masks_preserve_patch_padding_and_slot_order(phi5_batch, lm, legacy, mask_values):
    backend, processor, images, prompts = phi5_batch
    image_mask = None if mask_values is None else torch.tensor(mask_values, dtype=torch.int32)
    reference = processor.image_processor.process_batched(images.flatten(0, 1))
    reference["pixel_values"] = nn.functional.pad(reference["pixel_values"], (0, 0, 0, 1))
    reference["pixel_attention_mask"] = nn.functional.pad(reference["pixel_attention_mask"], (0, 1))
    processor.image_processor.process_batched = MagicMock(return_value=dict(reference))
    expected_mask = reference["pixel_attention_mask"].clone()
    assert (expected_mask == 0).any()
    if image_mask is not None:
        expected_mask &= image_mask.reshape(-1, 1).bool()
    if legacy:
        images = [list(sample) for sample in images]
        image_mask = None if image_mask is None else list(image_mask)

    result = _process_inputs(backend, processor, images, prompts, image_mask, lm)

    assert torch.equal(result["pixel_attention_mask"], expected_mask)
    assert torch.equal(result["pixel_values"], reference["pixel_values"])
    assert torch.equal(result["spatial_shapes"], reference["spatial_shapes"])
    assert torch.equal((result["input_ids"] == IMAGE_TOKEN_INDEX).sum(dim=1), torch.tensor([2, 2, 2]))


@pytest.mark.parametrize("lm", [False, True])
@pytest.mark.parametrize("shape", [(6,), (3, 1)])
def test_phi5_rejects_misaligned_image_masks(phi5_batch, lm, shape):
    backend, processor, images, prompts = phi5_batch
    with pytest.raises(ValueError, match="image_mask must have shape"):
        _process_inputs(backend, processor, images, prompts, torch.ones(shape), lm)


@pytest.fixture
def tiny_phi5_backbone(tmp_path, monkeypatch):
    monkeypatch.setattr(dynamic_module_utils, "HF_MODULES_CACHE", str(tmp_path / "modules"))
    model_dir = Path(__file__).resolve().parents[2] / "rho" / "models" / "Phi-4-vision-5B"
    model_class = dynamic_module_utils.get_class_from_dynamic_module(
        "modeling_bunny_phi4.BunnyPhi4ForCausalLM", str(model_dir), local_files_only=True
    )
    tower_class = sys.modules[model_class.__module__].Siglip2VisionTower
    tower = object.__new__(tower_class)
    nn.Module.__init__(tower)
    tower.vision_tower = Siglip2VisionModel(
        Siglip2VisionConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            patch_size=2,
            num_patches=4,
            vision_use_head=False,
        )
    )
    tower.select_layer = -1
    backbone = object.__new__(model_class)
    nn.Module.__init__(backbone)
    backbone.config = SimpleNamespace(tokenizer_padding_side="right")
    backbone.model = nn.Module()
    backbone.model.embed_tokens = nn.Embedding(256, 8)
    backbone.model.vision_tower = tower
    backbone.model.get_vision_tower = lambda: tower
    backbone.model.mm_projector = nn.Identity()
    return backbone.eval()


@pytest.mark.parametrize("lm", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mask_values", [[[1, 0], [0, 0], [0, 1]], [[0, 0]] * 3])
def test_phi5_padded_images_cannot_affect_lm_inputs_or_gradients(
    phi5_batch, tiny_phi5_backbone, lm, dtype, mask_values
):
    backend, processor, images, prompts = phi5_batch
    backbone = tiny_phi5_backbone.to(dtype)
    image_mask = torch.tensor(mask_values, dtype=torch.bool)
    images.requires_grad_()

    def splice(source_images):
        inputs = _process_inputs(backend, processor, source_images, prompts, image_mask, lm)
        result = backbone.prepare_inputs_labels_for_multimodal(
            input_ids=inputs["input_ids"].clone(),
            position_ids=None,
            attention_mask=inputs["attention_mask"],
            past_key_values=None,
            labels=inputs["labels"],
            images={key: inputs[key] for key in ("pixel_values", "pixel_attention_mask", "spatial_shapes")},
        )
        expected_lengths = inputs["attention_mask"].sum(dim=1) - 2 + 2 * image_mask.sum(dim=1)
        assert torch.equal(result[2].sum(dim=1), expected_lengths)
        if lm:
            assert torch.equal(result[5][result[5] != -100], inputs["labels"][inputs["labels"] != -100])
        return result[4]

    embeddings = splice(images)
    changed_images = images.detach().clone()
    changed_images[~image_mask] = 1
    torch.testing.assert_close(embeddings, splice(changed_images))
    embeddings[..., 0].sum().backward()

    assert torch.isfinite(embeddings).all()
    assert torch.isfinite(images.grad).all()
    assert images.grad[~image_mask].count_nonzero() == 0
    if image_mask.any():
        assert images.grad[image_mask].abs().sum() > 0
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in backbone.parameters()
    )
