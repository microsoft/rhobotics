"""
Tests for Qwen backbone adapters and backbone adapter contract.

Covers:
- BackboneAdapter contract: all adapters implement required abstract methods
- create_backbone_adapter factory: correct dispatch and error handling
- Qwen25VLBackbone: model ID, hidden size, prompt formatting, no-op methods
- Qwen3VLBackbone: model validation, text_config hidden size, prompt formatting
- Registry aliases: qwen25vl and qwen3vl resolve to RhoAlphaPolicy
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from rho.policies.rhoalpha.backbone import create_backbone_adapter
from rho.policies.rhoalpha.qwen3vl.backbone import Qwen3VLBackbone
from rho.policies.rhoalpha.qwen25vl.backbone import Qwen25VLBackbone

# ── Helpers ──────────────────────────────────────────────────────────────


def _make_qwen_config(**overrides):
    """Create a minimal config namespace for Qwen backbone adapters."""
    defaults = {
        "device": "cpu",
        "dtype": torch.float32,
        "image_features": ["observation.image.cam_high", "observation.image.cam_wrist"],
        "n_obs_steps": 1,
        "vlm_model_name": None,
        "vlm_backend": "qwen25vl",
        "enable_gradient_checkpointing": False,
        "freeze_vision_encoder": True,
        "freeze_language_model": True,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_mock_backbone(hidden_size=3584, has_text_config=False):
    """Create a mock VLM backbone with a config attribute."""
    backbone = MagicMock()
    if has_text_config:
        backbone.config.text_config.hidden_size = hidden_size
    else:
        backbone.config.hidden_size = hidden_size
    backbone.device = torch.device("cpu")

    # Mock visual module for freezing
    visual_param = torch.nn.Parameter(torch.randn(2, 2))
    backbone.visual.parameters.return_value = [visual_param]

    return backbone


# ── BackboneAdapter Contract ────────────────────────────────────────────


class TestBackboneAdapterContract:
    """Verify all concrete adapters implement every abstract method."""

    ABSTRACT_METHODS = [
        "load_backbone",
        "create_processor",
        "get_model_id",
        "get_hidden_size",
        "prepare_prompt",
        "remove_audio_components",
        "set_vision_requires_grad",
        "get_freezing_components",
        "process_batch",
        "get_image_text_hidden_state",
    ]

    @pytest.mark.parametrize(
        "adapter_cls",
        [Qwen25VLBackbone, Qwen3VLBackbone],
        ids=["qwen25vl", "qwen3vl"],
    )
    def test_adapter_implements_all_abstract_methods(self, adapter_cls):
        """Each adapter must implement every abstract method from BackboneAdapter."""
        for method_name in self.ABSTRACT_METHODS:
            method = getattr(adapter_cls, method_name, None)
            assert method is not None, f"{adapter_cls.__name__} missing method: {method_name}"
            assert not getattr(method, "__isabstractmethod__", False), (
                f"{adapter_cls.__name__}.{method_name} is still abstract"
            )

    def test_supports_builtin_lora_defaults_false(self):
        """Non-Phi4MM adapters should return False for supports_builtin_lora."""
        config = _make_qwen_config()
        assert Qwen25VLBackbone(config).supports_builtin_lora() is False
        assert Qwen3VLBackbone(config).supports_builtin_lora() is False


# ── create_backbone_adapter Factory ─────────────────────────────────────


class TestCreateBackboneAdapter:
    """Verify factory dispatches to the correct adapter class."""

    def test_factory_returns_qwen25vl(self):
        config = _make_qwen_config(vlm_backend="qwen25vl")
        adapter = create_backbone_adapter(config)
        assert isinstance(adapter, Qwen25VLBackbone)

    def test_factory_returns_qwen3vl(self):
        config = _make_qwen_config(vlm_backend="qwen3vl")
        adapter = create_backbone_adapter(config)
        assert isinstance(adapter, Qwen3VLBackbone)

    def test_factory_raises_for_unknown_backend(self):
        config = _make_qwen_config(vlm_backend="nonexistent")
        with pytest.raises(ValueError, match="Unknown vlm_backend"):
            create_backbone_adapter(config)

    def test_factory_returns_phi4mm(self):
        config = _make_qwen_config(vlm_backend="phi4mm")
        from rho.policies.rhoalpha.phi4mm.backbone import Phi4MMBackbone

        adapter = create_backbone_adapter(config)
        assert isinstance(adapter, Phi4MMBackbone)

    def test_factory_returns_phi5(self):
        config = _make_qwen_config(vlm_backend="phi5", vlm_backbone_folder="/fake/path")
        from rho.policies.rhoalpha.phi5.backbone import Phi5Backbone

        adapter = create_backbone_adapter(config)
        assert isinstance(adapter, Phi5Backbone)


# ── Qwen25VLBackbone ───────────────────────────────────────────────────


class TestQwen25VLBackbone:
    """Tests for the Qwen2.5-VL backbone adapter."""

    def test_get_model_id_default(self):
        config = _make_qwen_config(vlm_model_name=None)
        adapter = Qwen25VLBackbone(config)
        assert adapter.get_model_id() == "Qwen/Qwen2.5-VL-7B-Instruct"

    def test_get_model_id_custom(self):
        config = _make_qwen_config(vlm_model_name="custom/model")
        adapter = Qwen25VLBackbone(config)
        assert adapter.get_model_id() == "custom/model"

    def test_get_hidden_size(self):
        config = _make_qwen_config()
        adapter = Qwen25VLBackbone(config)
        backbone = _make_mock_backbone(hidden_size=3584)
        assert adapter.get_hidden_size(backbone) == 3584

    def test_remove_audio_components_is_noop(self):
        config = _make_qwen_config()
        adapter = Qwen25VLBackbone(config)
        backbone = _make_mock_backbone()
        # Should not raise
        adapter.remove_audio_components(backbone)

    def test_set_vision_requires_grad_freezes_visual(self):
        config = _make_qwen_config()
        adapter = Qwen25VLBackbone(config)

        # Create a real parameter to test freezing
        param = torch.nn.Parameter(torch.randn(2, 2))
        assert param.requires_grad is True

        backbone = MagicMock()
        backbone.visual.parameters.return_value = [param]

        adapter.set_vision_requires_grad(backbone)
        assert param.requires_grad is False

    def test_get_freezing_components_returns_dict(self):
        config = _make_qwen_config()
        adapter = Qwen25VLBackbone(config)
        backbone = _make_mock_backbone()
        components = adapter.get_freezing_components(backbone)
        assert "VLM Backbone" in components
        assert "  Vision Encoder" in components

    def test_prepare_prompt_formatting(self):
        config = _make_qwen_config(
            image_features=["observation.image.cam1", "observation.image.cam2"],
            n_obs_steps=1,
        )
        adapter = Qwen25VLBackbone(config)
        batch = {"task": ["Pick up the cup"]}

        prompts = adapter.prepare_prompt(batch)

        assert len(prompts) == 1
        prompt = prompts[0]
        # Should contain 2 image tokens (one per camera)
        assert prompt.count("<|vision_start|>") == 2
        assert prompt.count("<|image_pad|>") == 2
        assert prompt.count("<|vision_end|>") == 2
        # Should contain the task description
        assert "Pick up the cup" in prompt
        # Should have system, user, assistant structure
        assert "<|im_start|>system" in prompt
        assert "<|im_start|>user" in prompt
        assert "<|im_start|>assistant" in prompt

    def test_prepare_prompt_empty_batch(self):
        config = _make_qwen_config()
        adapter = Qwen25VLBackbone(config)
        prompts = adapter.prepare_prompt({"other_key": ["test"]})
        assert prompts == []

    def test_prepare_prompt_multi_obs_steps(self):
        config = _make_qwen_config(
            image_features=["observation.image.cam1"],
            n_obs_steps=3,
        )
        adapter = Qwen25VLBackbone(config)
        batch = {"task": ["task"]}
        prompts = adapter.prepare_prompt(batch)
        # 1 camera * 3 obs steps = 3 image tokens
        assert prompts[0].count("<|vision_start|>") == 3


# ── Qwen3VLBackbone ────────────────────────────────────────────────────


class TestQwen3VLBackbone:
    """Tests for the Qwen3-VL backbone adapter."""

    def test_get_model_id_default(self):
        config = _make_qwen_config(vlm_model_name=None)
        adapter = Qwen3VLBackbone(config)
        assert adapter.get_model_id() == "Qwen/Qwen3-VL-8B-Instruct"

    def test_get_model_id_custom(self):
        config = _make_qwen_config(vlm_model_name="Qwen/Qwen3-VL-4B-Instruct")
        adapter = Qwen3VLBackbone(config)
        assert adapter.get_model_id() == "Qwen/Qwen3-VL-4B-Instruct"

    def test_get_hidden_size_uses_text_config(self):
        """Qwen3VL hidden size is under config.text_config.hidden_size."""
        config = _make_qwen_config()
        adapter = Qwen3VLBackbone(config)
        backbone = _make_mock_backbone(hidden_size=3584, has_text_config=True)
        assert adapter.get_hidden_size(backbone) == 3584

    def test_remove_audio_components_is_noop(self):
        config = _make_qwen_config()
        adapter = Qwen3VLBackbone(config)
        backbone = _make_mock_backbone()
        adapter.remove_audio_components(backbone)

    def test_prepare_prompt_same_as_qwen25vl(self):
        """Qwen3VL uses the same chat template as Qwen2.5VL."""
        config = _make_qwen_config(
            image_features=["observation.image.cam1"],
            n_obs_steps=1,
        )
        batch = {"task": ["Pick up the block"]}

        qwen25_prompts = Qwen25VLBackbone(config).prepare_prompt(batch)
        qwen3_prompts = Qwen3VLBackbone(config).prepare_prompt(batch)

        assert qwen25_prompts == qwen3_prompts

    def test_set_vision_requires_grad_freezes_visual(self):
        config = _make_qwen_config()
        adapter = Qwen3VLBackbone(config)

        param = torch.nn.Parameter(torch.randn(2, 2))
        backbone = MagicMock()
        backbone.visual.parameters.return_value = [param]

        adapter.set_vision_requires_grad(backbone)
        assert param.requires_grad is False


# ── Registry Aliases ────────────────────────────────────────────────────


class TestRegistryAliases:
    """Verify qwen25vl/qwen3vl resolve to RhoAlphaPolicy in the registry."""

    def test_qwen25vl_alias(self):
        from rho.policies import POLICY_REGISTRY
        from rho.policies.rhoalpha.rhoalpha_policy import RhoAlphaPolicy

        assert "qwen25vl" in POLICY_REGISTRY
        assert POLICY_REGISTRY["qwen25vl"] is RhoAlphaPolicy

    def test_qwen3vl_alias(self):
        from rho.policies import POLICY_REGISTRY
        from rho.policies.rhoalpha.rhoalpha_policy import RhoAlphaPolicy

        assert "qwen3vl" in POLICY_REGISTRY
        assert POLICY_REGISTRY["qwen3vl"] is RhoAlphaPolicy

    def test_all_expected_policies_registered(self):
        from rho.policies import POLICY_REGISTRY

        expected = {
            "behavioral_cloning",
            "diffusion",
            "rhoalpha",
            "rhoalpha_tactile",
            "phi4mm",
            "phi4mm_tactile",
            "pi0",
            "pi0fast",
            "qwen25vl",
            "qwen3vl",
        }
        assert expected.issubset(set(POLICY_REGISTRY.keys()))


# ── RhoAlphaConfig validation ──────────────────────────────────────────


class TestRhoAlphaConfigValidation:
    """Verify config validation accepts Qwen backends."""

    def test_qwen25vl_backend_accepted(self):
        from rho.policies.rhoalpha.configuration_rhoalpha import RhoAlphaConfig

        # Should not raise — qwen25vl is a valid backend
        config = RhoAlphaConfig(vlm_backend="qwen25vl")
        assert config.vlm_backend == "qwen25vl"

    def test_qwen3vl_backend_accepted(self):
        from rho.policies.rhoalpha.configuration_rhoalpha import RhoAlphaConfig

        config = RhoAlphaConfig(vlm_backend="qwen3vl")
        assert config.vlm_backend == "qwen3vl"

    def test_invalid_backend_rejected(self):
        from rho.policies.rhoalpha.configuration_rhoalpha import RhoAlphaConfig

        with pytest.raises(ValueError, match="vlm_backend must be one of"):
            RhoAlphaConfig(vlm_backend="invalid_backend")
