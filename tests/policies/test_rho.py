from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import draccus
import pytest
import torch
from torch import nn

from rho.checkpoints import load_bundle_weights, save_state_dict_bundle
from rho.common.constants import ACTION, OBSERVATION_IMAGE, OBSERVATION_LANG, OBSERVATION_STATE
from rho.common.transforms import ResizeWithPadding
from rho.common.types import FeatureType, PolicyFeature
from rho.policies import POLICY_REGISTRY, make_policy_from_checkpoint
from rho.policies.base import PolicyConfig
from rho.policies.rho import RhoConfig, RhoPolicy
from rho.policies.rho.rho_model import FlowMatchingModel
from rho.policies.rho.rho_policy import convert_rhoalpha_state_dict
from rho.policies.rho.rho_processor import RhoProcessor


def _make_observation_policy(*, resize: tuple[int, int] | None = (4, 4), n_obs_steps: int = 1) -> RhoPolicy:
    feature_dict = {
        f"{OBSERVATION_IMAGE}.0": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 6, 8)),
        f"{OBSERVATION_IMAGE}.1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 6, 8)),
        OBSERVATION_STATE: PolicyFeature(type=FeatureType.STATE, shape=(5,)),
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
    }
    config = RhoConfig(
        device="cpu",
        dtype="float32",
        feature_dict=feature_dict,
        resize_imgs_with_padding=resize,
        max_state_dim=8,
        max_action_dim=8,
        n_obs_steps=n_obs_steps,
    )
    policy = object.__new__(RhoPolicy)
    nn.Module.__init__(policy)
    policy.config = config
    policy.processor = RhoProcessor(config)
    return policy


def _observation_batch(images: list[torch.Tensor]) -> dict:
    return {f"{OBSERVATION_IMAGE}.{index}": image for index, image in enumerate(images)} | {
        OBSERVATION_STATE: torch.zeros(images[0].shape[0], 5),
        OBSERVATION_LANG: ["do the task"] * images[0].shape[0],
    }


def test_rho_is_registered_as_distinct_policy():
    config = draccus.decode(PolicyConfig, {"type": "rho", "feature_dict": {}})

    assert isinstance(config, RhoConfig)
    assert config.name == "rho"
    assert config.vlm_backend == "phi5"
    assert config.vlm_backbone_folder == str(
        Path(__file__).resolve().parents[2] / "rho" / "models" / "Phi-4-vision-5B"
    )
    assert config.resize_imgs_with_padding == (256, 256)
    assert config.embed_dim == 2048
    assert config.num_heads == 16
    assert config.norm == "adaptive"
    assert config.hidden_state_idx == 14
    assert config.num_blocks == 12
    assert config.n_action_steps == 25
    assert config.attention_type == "cross"
    assert config.adaln_mode == "shared"
    assert config.gqa_groups == 4
    assert config.dropout_p == 0.02
    assert config.num_flow_samples == 8
    assert POLICY_REGISTRY["rho"] is RhoPolicy


@pytest.mark.parametrize("dropout_p", [-0.1, 1.01, float("nan")])
def test_rho_rejects_dropout_outside_allowed_range(dropout_p):
    with pytest.raises(ValueError, match="dropout_p must be between 0.0 and 1.0"):
        RhoConfig(feature_dict={}, dropout_p=dropout_p)


@pytest.mark.parametrize("dropout_p", [0.0, 0.01, 0.019, 0.02, 0.1, 1.0])
def test_rho_preserves_valid_dropout(dropout_p):
    assert RhoConfig(feature_dict={}, dropout_p=dropout_p).dropout_p == dropout_p


def test_rho_allows_single_flow_sample_override():
    config = draccus.decode(PolicyConfig, {"type": "rho", "feature_dict": {}, "num_flow_samples": 1})
    assert config.num_flow_samples == 1


@pytest.mark.parametrize(("alpha", "beta"), [(1.0, 1.0), (1.5, 1.0), (1.0, 1.5)])
def test_rho_beta_sampler_matches_distribution_moments(alpha, beta):
    model = object.__new__(FlowMatchingModel)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        samples = model.sample_beta(alpha, beta, 100_000, torch.device("cpu"))

    expected_mean = alpha / (alpha + beta)
    expected_variance = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))
    assert samples.dtype == torch.float32
    assert samples.shape == (100_000,)
    assert ((samples > 0) & (samples < 1)).all()
    assert samples.mean().item() == pytest.approx(expected_mean, abs=0.004)
    assert samples.var().item() == pytest.approx(expected_variance, abs=0.002)


@pytest.mark.parametrize(("strategy", "alpha", "beta"), [("beta", 1.5, 1.0), ("beta_reverse", 1.0, 1.5)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rho_time_sampler_preserves_rng_restore_scaling_and_dtype(strategy, alpha, beta, dtype):
    model = object.__new__(FlowMatchingModel)
    nn.Module.__init__(model)
    model.config = RhoConfig(feature_dict={}, time_sampling_strategy=strategy)
    model.dtype = dtype
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        rng_state = torch.get_rng_state()
        times = model.sample_time(128, torch.device("cpu"))
        torch.set_rng_state(rng_state)
        expected = model.sample_beta(alpha, beta, 128, torch.device("cpu"))
        torch.set_rng_state(rng_state)
        resumed = model.sample_time(128, torch.device("cpu"))

    assert times.dtype == dtype
    assert torch.equal(times, (expected * 0.999 + 0.001).to(dtype))
    assert torch.equal(times, resumed)


@pytest.mark.parametrize("scale", [1.0, 255.0])
def test_rho_batched_image_preparation_preserves_existing_float_outputs(scale):
    policy = _make_observation_policy()
    source = torch.arange(2 * 3 * 6 * 8, dtype=torch.float32).reshape(2, 3, 6, 8)
    source = source.remainder(256).div(255.0).mul(scale)
    batch = _observation_batch([source, source.flip(-1)])
    consolidated = policy.consolidate_images(batch)

    prepared, image_mask = policy.prepare_image(consolidated)

    resize = ResizeWithPadding(4, 4)
    expected = torch.stack(
        [
            torch.stack(
                [
                    resized / 255.0 if resized.max() > 1.0 else resized
                    for image in sample
                    for resized in [resize(image)]
                ]
            )
            for sample in consolidated[OBSERVATION_IMAGE]
        ]
    )
    assert torch.equal(prepared, expected)
    assert torch.equal(image_mask, torch.ones((2, 2), dtype=torch.int32))


def test_rho_accepts_uint8_images_before_resizing():
    policy = _make_observation_policy()
    source = torch.arange(3 * 6 * 8, dtype=torch.uint8).reshape(1, 3, 6, 8)
    consolidated = policy.consolidate_images(_observation_batch([source, source]))

    prepared, _ = policy.prepare_image(consolidated)

    expected = ResizeWithPadding(4, 4)(
        consolidated[OBSERVATION_IMAGE].flatten(0, 1).float().div(255.0)
    ).reshape(1, 2, 3, 4, 4)
    assert torch.equal(prepared, expected)


def test_rho_preserves_mixed_valid_float_ranges_per_camera():
    policy = _make_observation_policy()
    normalized = torch.linspace(0.0, 1.0, 3 * 6 * 8).reshape(1, 3, 6, 8)
    high_range = normalized * 255.0
    consolidated = policy.consolidate_images(_observation_batch([normalized, high_range]))

    prepared, _ = policy.prepare_image(consolidated)

    resize = ResizeWithPadding(4, 4)
    assert torch.equal(prepared[:, 0], resize(normalized))
    assert torch.equal(prepared[:, 1], resize(high_range) / 255.0)


def test_rho_clamps_small_normalized_float_overshoot():
    policy = _make_observation_policy(resize=None)
    source = torch.full((1, 3, 6, 8), 1.0005)
    consolidated = policy.consolidate_images(_observation_batch([source, source]))

    prepared, _ = policy.prepare_image(consolidated)

    assert torch.equal(prepared, torch.ones_like(prepared))


@pytest.mark.parametrize("maximum", [1.5, 255.1])
def test_rho_rejects_invalid_float_image_ranges(maximum):
    policy = _make_observation_policy(resize=None)
    source = torch.zeros((1, 3, 6, 8))
    source[..., 0, 0] = maximum
    consolidated = policy.consolidate_images(_observation_batch([source, source]))

    with pytest.raises(ValueError, match="image"):
        policy.prepare_image(consolidated)


def test_rho_reports_missing_required_camera():
    policy = _make_observation_policy()
    source = torch.zeros((1, 3, 6, 8))
    batch = _observation_batch([source])

    with pytest.raises(KeyError, match=r"observation\.image\.1"):
        policy.consolidate_images(batch)


def test_rho_preserves_per_camera_padding_masks():
    policy = _make_observation_policy(resize=None)
    source = torch.ones((2, 3, 6, 8))
    batch = _observation_batch([source, source])
    batch[f"{OBSERVATION_IMAGE}.1_is_pad"] = torch.tensor([[False], [True]])

    consolidated = policy.consolidate_images(batch)
    _, image_mask = policy.prepare_image(consolidated)

    assert torch.equal(image_mask, torch.tensor([[1, 1], [1, 0]], dtype=torch.int32))


def test_rho_rejects_preconsolidated_image_slot_mismatch():
    policy = _make_observation_policy(resize=None, n_obs_steps=2)
    batch = {
        OBSERVATION_IMAGE: torch.ones((1, 3, 3, 6, 8)),
        OBSERVATION_STATE: torch.zeros((1, 5)),
        OBSERVATION_LANG: ["do the task"],
    }

    with pytest.raises(
        ValueError,
        match=r"has 3 image slots.*expects 4.*2 configured camera\(s\).*n_obs_steps=2",
    ):
        policy.processor.prepare(batch)


def test_rho_accepts_expected_preconsolidated_image_slots_and_pad_mask():
    policy = _make_observation_policy(resize=None, n_obs_steps=2)
    batch = {
        OBSERVATION_IMAGE: torch.ones((1, 4, 3, 6, 8)),
        f"{OBSERVATION_IMAGE}_is_pad": torch.tensor([[False, False, True, False]]),
        OBSERVATION_STATE: torch.zeros((1, 5)),
        OBSERVATION_LANG: ["do the task"],
    }

    prepared = policy.processor.prepare(batch)

    assert prepared.image.shape == (1, 4, 3, 6, 8)
    assert torch.equal(prepared.image_mask, torch.tensor([[1, 1, 0, 1]], dtype=torch.int32))


def test_rho_reports_missing_required_state_before_padding():
    policy = _make_observation_policy()
    source = torch.zeros((1, 3, 6, 8))
    batch = _observation_batch([source, source])
    del batch[OBSERVATION_STATE]

    with pytest.raises(KeyError, match=OBSERVATION_STATE):
        policy._validate_required_observations(batch)


def test_rho_keeps_flow_weights_and_drops_cotraining_heads():
    flow_weight = torch.tensor([1.0])
    state_dict = {
        "model.flow_model.action_head.weight": flow_weight,
        "model.text_output_model.lm_head.weight": torch.tensor([2.0]),
        "model.fast_model.action_head.weight": torch.tensor([3.0]),
    }

    converted = convert_rhoalpha_state_dict(state_dict)

    assert converted == {"model.flow_model.action_head.weight": flow_weight}


def test_rho_maps_knowledge_insulation_flow_weights():
    state_dict = {
        "model.knowledge_insulation_endstate_model.flow_model.action_head.weight": torch.tensor([1.0]),
        "model.knowledge_insulation_endstate_model.vlm_backbone.weight": torch.tensor([2.0]),
        "model.text_output_model.lm_head.weight": torch.tensor([3.0]),
    }

    converted = convert_rhoalpha_state_dict(state_dict)

    assert set(converted) == {"model.flow_model.action_head.weight"}


def test_rho_maps_legacy_direct_flow_weights():
    state_dict = {
        "model.vlm_backbone.weight": torch.tensor([1.0]),
        "model.action_head.weight": torch.tensor([2.0]),
        "model.tactile_projector.weight": torch.tensor([3.0]),
    }

    converted = convert_rhoalpha_state_dict(state_dict)

    assert set(converted) == {
        "model.flow_model.vlm_backbone.weight",
        "model.flow_model.action_head.weight",
    }


def test_rho_remaps_legacy_siglip2_checkpoint_keys_for_transformers_510():
    policy = object.__new__(RhoPolicy)
    nn.Module.__init__(policy)
    policy.model = SimpleNamespace(
        flow_model=SimpleNamespace(
            vlm_backbone=SimpleNamespace(
                model=SimpleNamespace(
                    vision_tower=SimpleNamespace(
                        vision_tower=SimpleNamespace(),
                    )
                )
            )
        )
    )
    tensor = torch.tensor([1.0])
    legacy_key = (
        "model.flow_model.vlm_backbone.model.vision_tower."
        "vision_tower.vision_model.encoder.layers.0.layer_norm1.weight"
    )

    remapped = policy._remap_backwards_compatible_keys({legacy_key: tensor})

    assert list(remapped) == [
        "model.flow_model.vlm_backbone.model.vision_tower.vision_tower.encoder.layers.0.layer_norm1.weight"
    ]
    assert remapped[next(iter(remapped))] is tensor


def test_loading_weights_clears_uninitialized_backbone_guard():
    policy = object.__new__(RhoPolicy)
    nn.Module.__init__(policy)
    policy._has_uninitialized_backbone = True

    policy.load_state_dict({})

    assert not policy._has_uninitialized_backbone


def test_make_policy_from_rho_bundle_uses_portable_metadata(tmp_path):
    class TinyRhoPolicy(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.weight = nn.Parameter(torch.zeros(2))

        def load_from_pretrained(self, checkpoint):
            load_bundle_weights(self, checkpoint)

    checkpoint = save_state_dict_bundle(
        {"weight": torch.tensor([1.0, 2.0])},
        tmp_path / "checkpoint_step_0000040",
        step=40,
        policy_config={
            "type": "rho",
            "name": "rho",
            "empty_cameras": 0,
            "feature_dict": {"action": {"type": "ACTION", "shape": [2]}},
        },
    )

    with patch.dict(POLICY_REGISTRY, {"rho": TinyRhoPolicy}):
        policy = make_policy_from_checkpoint(str(checkpoint))

    assert isinstance(policy.config, RhoConfig)
    assert not hasattr(policy.config, "empty_cameras")
    assert torch.equal(policy.weight, torch.tensor([1.0, 2.0]))


@pytest.mark.gpu
@pytest.mark.resource_intensive
def test_rho_constructs_full_phi5_flow_model():
    config = RhoConfig(
        feature_dict={
            OBSERVATION_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            OBSERVATION_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        }
    )

    policy = RhoPolicy(config)

    assert policy.model.flow_model is not None
    assert policy._has_uninitialized_backbone
