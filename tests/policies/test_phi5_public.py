from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from rho.policies.rho.phi5.backbone import IMAGE_TOKEN_INDEX, Phi5Backbone, _tokenizer_image_token


def _fake_tokenizer(bos_token_id=None):
    tokenizer = MagicMock()
    tokenizer.bos_token_id = bos_token_id

    def tokenize(text):
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
