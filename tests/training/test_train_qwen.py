"""
Integration tests for Qwen2.5-VL and Qwen3-VL backbone training.

These tests verify the full training pipeline works end-to-end with
Qwen backends routed through RhoAlphaPolicy + BackboneAdapter:
- RhoAlphaConfig with vlm_backend="qwen25vl" / "qwen3vl"
- Dataset creation, batching, and normalization
- Forward pass, loss computation, backward pass
- Checkpoint saving and loading

Marked @pytest.mark.resource_intensive — requires GPU + model downloads.
Run with: pytest --all tests/training/test_train_qwen.py
"""

import gc
import shutil
from pathlib import Path

import pytest
import torch

from rho.common.wandb_logging import WandBConfig
from rho.policies.rhoalpha.configuration_rhoalpha import RhoAlphaConfig
from rho.training.train import TrainConfig, train

# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def cleanup_gpu():
    """Free GPU memory before and after each test."""
    torch.cuda.empty_cache()
    gc.collect()
    yield
    torch.cuda.empty_cache()
    gc.collect()


def _make_qwen_rhoalpha_config(vlm_backend, sample_features, **overrides):
    """Create a RhoAlphaConfig configured for a Qwen backend."""
    defaults = {
        "name": "rhoalpha",
        "vlm_backend": vlm_backend,
        "feature_dict": sample_features,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "n_obs_steps": 1,
        "n_action_steps": 1,
        "max_state_dim": 5,
        "max_action_dim": 2,
        "enable_gradient_checkpointing": True,
        "chunk_size": 10,
        "hidden_state_idx": -1,
        "train_expert_only": True,
    }
    defaults.update(overrides)
    return RhoAlphaConfig(**defaults)


@pytest.fixture
def qwen25vl_config(sample_features):
    return _make_qwen_rhoalpha_config("qwen25vl", sample_features)


@pytest.fixture
def qwen3vl_config(sample_features):
    return _make_qwen_rhoalpha_config(
        "qwen3vl",
        sample_features,
        vlm_model_name="Qwen/Qwen3-VL-2B-Instruct",
    )


def _make_train_config(dataset_config_fixture, policy_config, output_name):
    return TrainConfig(
        wandb=WandBConfig(enabled=False),
        dataset=dataset_config_fixture,
        validation_dataset=None,
        policy=policy_config,
        pretrained_checkpoint=None,
        resume=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=2,
        learning_rate=1e-4,
        steps=5,
        output_dir=f"test_outputs/{output_name}",
        save_checkpoint_every=5,
        logging_interval=2,
        validation_interval=5,
        eval_interval=5,
        eval_batch_size=2,
        eval_num_episodes=1,
        record_videos=False,
    )


@pytest.fixture
def qwen25vl_train_config(dataset_config_fixture, qwen25vl_config):
    return _make_train_config(dataset_config_fixture, qwen25vl_config, "qwen25vl_train")


@pytest.fixture
def qwen3vl_train_config(dataset_config_fixture, qwen3vl_config):
    return _make_train_config(dataset_config_fixture, qwen3vl_config, "qwen3vl_train")


# ── Qwen2.5-VL Training ────────────────────────────────────────────────


class TestTrainQwen25VL:
    """End-to-end training test for Qwen2.5-VL backend."""

    @pytest.mark.resource_intensive
    def test_train_qwen25vl(self, qwen25vl_train_config):
        out_dir = Path(qwen25vl_train_config.output_dir)
        if out_dir.exists():
            shutil.rmtree(out_dir)

        qwen25vl_train_config.checkpoint_folder.parent.mkdir(parents=True, exist_ok=True)
        train(qwen25vl_train_config)

        assert qwen25vl_train_config.checkpoint_folder.exists()
        ckpt_files = list(qwen25vl_train_config.checkpoint_folder.glob("checkpoint_step_*.pt"))
        assert len(ckpt_files) > 0

        config_file = qwen25vl_train_config.checkpoint_folder.parent / "train_config.json"
        assert config_file.exists()

        torch.cuda.empty_cache()

        checkpoint = torch.load(ckpt_files[0], weights_only=False, map_location="cpu")
        assert "policy_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "metrics" in checkpoint
        assert "step" in checkpoint


# ── Qwen3-VL Training ──────────────────────────────────────────────────


class TestTrainQwen3VL:
    """End-to-end training test for Qwen3-VL backend."""

    @pytest.mark.resource_intensive
    def test_train_qwen3vl(self, qwen3vl_train_config):
        out_dir = Path(qwen3vl_train_config.output_dir)
        if out_dir.exists():
            shutil.rmtree(out_dir)

        qwen3vl_train_config.checkpoint_folder.parent.mkdir(parents=True, exist_ok=True)
        train(qwen3vl_train_config)

        assert qwen3vl_train_config.checkpoint_folder.exists()
        ckpt_files = list(qwen3vl_train_config.checkpoint_folder.glob("checkpoint_step_*.pt"))
        assert len(ckpt_files) > 0

        config_file = qwen3vl_train_config.checkpoint_folder.parent / "train_config.json"
        assert config_file.exists()

        torch.cuda.empty_cache()

        checkpoint = torch.load(ckpt_files[0], weights_only=False, map_location="cpu")
        assert "policy_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "metrics" in checkpoint
        assert "step" in checkpoint
