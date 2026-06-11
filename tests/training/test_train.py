import tempfile
from pathlib import Path

import pytest
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from rho.common.types import FeatureType, PolicyFeature
from rho.policies.diffusion import DiffusionConfig, DiffusionPolicy
from rho.training.train import TrainConfig, make_everything, train_policy_step
from rho.training.train_utils import (
    load_training_state,
    make_optimizer_and_scheduler,
    save_checkpoint,
    serialize_train_config,
)


@pytest.fixture(scope="class")
def sample_features_for_training():
    """Create sample features for training tests - class scoped"""
    return {
        "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
    }


@pytest.fixture(scope="class")
def diffusion_config_for_training(sample_features_for_training):
    """Create a DiffusionConfig optimized for training tests with presets"""
    config = DiffusionConfig(
        name="diffusion",
        feature_dict=sample_features_for_training,
        device="cuda" if torch.cuda.is_available() else "cpu",
        n_obs_steps=2,
        horizon=16,
        n_action_steps=8,
        vision_backbone="resnet18",
        crop_shape=(96, 96),  # Match sample_features image size
        down_dims=(512, 1024, 2048),
        num_train_timesteps=100,
        # Training presets for testing
        optimizer_lr=1e-4,
        optimizer_betas=(0.95, 0.999),
        optimizer_eps=1e-8,
        optimizer_weight_decay=1e-6,
        scheduler_name="cosine",
        scheduler_warmup_steps=0,  # No warmup for tests
    )
    return config


@pytest.fixture(scope="class")
def diffusion_policy_for_training(diffusion_config_for_training):
    """Create a DiffusionPolicy for training tests"""
    policy = DiffusionPolicy(config=diffusion_config_for_training)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    return policy


@pytest.fixture(scope="class")
def mock_train_config():
    """Mock training config for testing"""

    class MockTrainConfig:
        def __init__(self):
            self.learning_rate = 1e-4
            self.steps = 1000

    return MockTrainConfig()


@pytest.fixture(scope="class")
def diffusion_training_batch(diffusion_config_for_training):
    """Create a batch compatible with diffusion policy training"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4

    # Match the feature dict from sample_features but adjust for diffusion requirements
    batch = {
        "observation.state": torch.randn(
            batch_size, diffusion_config_for_training.n_obs_steps, 2, device=device
        ),
        "observation.image": torch.randn(
            batch_size, diffusion_config_for_training.n_obs_steps, 3, 96, 96, device=device
        ),
        "action": torch.randn(batch_size, diffusion_config_for_training.horizon, 2, device=device),
        "action_is_pad": torch.zeros(
            batch_size, diffusion_config_for_training.horizon, dtype=torch.bool, device=device
        ),
    }
    return batch


@pytest.fixture(scope="class")
def trained_policy_checkpoint(diffusion_policy_for_training, diffusion_training_batch, mock_train_config):
    """Create a trained policy and save it as a checkpoint for reuse in tests"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create optimizer and scheduler
    optimizer, lr_scheduler = make_optimizer_and_scheduler(mock_train_config, diffusion_policy_for_training)

    # Run training steps using train_policy_step function
    diffusion_policy_for_training.train()
    training_metrics_list = []

    for step in range(1, 6):  # Run 5 training steps
        training_metrics = train_policy_step(
            policy=diffusion_policy_for_training,
            batch=diffusion_training_batch,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            step=step,
            device=device,
            use_amp=False,
        )
        training_metrics_list.append(training_metrics)

    # Create temporary directory and save checkpoint
    temp_dir = tempfile.mkdtemp()
    checkpoint_dir = Path(temp_dir)

    final_step = 5
    final_metrics = training_metrics_list[-1]
    save_checkpoint(diffusion_policy_for_training, optimizer, final_step, final_metrics, str(checkpoint_dir))

    return {
        "checkpoint_path": checkpoint_dir / f"checkpoint_step_{final_step:07d}.pt",
        "checkpoint_dir": checkpoint_dir,
        "final_step": final_step,
        "final_metrics": final_metrics,
        "optimizer_state": optimizer.state_dict(),
        "lr_scheduler_state": lr_scheduler.state_dict() if lr_scheduler else None,
        "training_metrics": training_metrics_list,
    }


@pytest.mark.resource_intensive
class TestTrainingFunctions:
    """GPU-intensive tests for training utility functions"""

    def test_make_optimizer_and_scheduler_with_presets(
        self, diffusion_policy_for_training, mock_train_config
    ):
        """Test make_optimizer_and_scheduler with policy presets"""
        optimizer, lr_scheduler = make_optimizer_and_scheduler(
            mock_train_config, diffusion_policy_for_training
        )

        # Test optimizer - when policy has presets, it uses policy config, not training config
        assert isinstance(optimizer, torch.optim.Adam)
        # The policy config takes precedence over training config when presets are available
        assert optimizer.param_groups[0]["lr"] == diffusion_policy_for_training.config.optimizer_lr
        assert optimizer.param_groups[0]["betas"] == diffusion_policy_for_training.config.optimizer_betas
        assert optimizer.param_groups[0]["eps"] == diffusion_policy_for_training.config.optimizer_eps
        assert (
            optimizer.param_groups[0]["weight_decay"]
            == diffusion_policy_for_training.config.optimizer_weight_decay
        )

        # Test scheduler - using policy preset
        assert isinstance(lr_scheduler, torch.optim.lr_scheduler.LambdaLR)
        assert len(optimizer.param_groups) > 0

        # Test that parameters are correctly assigned
        policy_params = set(diffusion_policy_for_training.parameters())
        optimizer_params = set()
        for group in optimizer.param_groups:
            optimizer_params.update(group["params"])
        assert policy_params == optimizer_params

    def test_make_optimizer_and_scheduler_without_presets(self, mock_train_config):
        """Test make_optimizer_and_scheduler with default optimizer/scheduler"""

        # Create a simple policy without presets
        class SimplePolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 1)
                self.config = None  # No preset methods

        policy = SimplePolicy()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy.to(device)

        optimizer, lr_scheduler = make_optimizer_and_scheduler(mock_train_config, policy)

        # Test default optimizer (Adam)
        assert isinstance(optimizer, Adam)
        assert optimizer.param_groups[0]["lr"] == mock_train_config.learning_rate

        # Test default scheduler (StepLR)
        assert isinstance(lr_scheduler, StepLR)

    def test_serialize_train_config(self):
        """Test serialize_train_config function for various data types"""
        from dataclasses import dataclass
        from pathlib import Path

        import numpy as np

        @dataclass
        class TestConfig:
            name: str
            value: int
            path: Path
            array: np.ndarray
            nested_dict: dict
            nested_list: list

        # Create test data with various types
        test_array = np.array([1, 2, 3])
        test_path = Path("/test/path")
        test_config = TestConfig(
            name="test",
            value=42,
            path=test_path,
            array=test_array,
            nested_dict={"key": "value", "number": 123},
            nested_list=[1, "string", {"nested": True}],
        )

        # Test serialization
        result = serialize_train_config(test_config)

        # Verify dataclass was serialized correctly
        assert isinstance(result, dict)
        assert result["name"] == "test"
        assert result["value"] == 42
        assert result["path"] == "/test/path"  # Path converted to string
        assert result["array"] == [1, 2, 3]  # numpy array converted to list
        assert result["nested_dict"]["key"] == "value"
        assert result["nested_dict"]["number"] == 123
        assert result["nested_list"][0] == 1
        assert result["nested_list"][1] == "string"
        assert result["nested_list"][2]["nested"]

        # Test direct dict serialization
        test_dict = {"path": Path("/another/path"), "array": np.array([4, 5, 6])}
        result_dict = serialize_train_config(test_dict)
        assert result_dict["path"] == "/another/path"
        assert result_dict["array"] == [4, 5, 6]

        # Test list/tuple serialization
        test_list = [Path("/list/path"), np.array([7, 8, 9])]
        result_list = serialize_train_config(test_list)
        assert result_list[0] == "/list/path"
        assert result_list[1] == [7, 8, 9]

        test_tuple = (Path("/tuple/path"), np.array([10, 11, 12]))
        result_tuple = serialize_train_config(test_tuple)
        assert result_tuple[0] == "/tuple/path"
        assert result_tuple[1] == [10, 11, 12]

        # Test primitive types pass through unchanged
        assert serialize_train_config("string") == "string"
        assert serialize_train_config(42) == 42
        assert serialize_train_config(3.14) == 3.14
        assert serialize_train_config(True)
        assert serialize_train_config(None) is None

    def test_save_load_training_state(self, trained_policy_checkpoint, diffusion_config_for_training):
        """Test load_training_state with trained checkpoint"""
        checkpoint_info = trained_policy_checkpoint
        checkpoint_path = checkpoint_info["checkpoint_path"]
        expected_step = checkpoint_info["final_step"]

        # Create fresh optimizer and scheduler using a valid config
        from rho.policies.diffusion.diffusion import DiffusionPolicy

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create a fresh policy for testing with the same config structure
        fresh_policy = DiffusionPolicy(config=diffusion_config_for_training)
        fresh_policy.to(device)

        new_optimizer = Adam(fresh_policy.parameters(), lr=1e-4)
        new_lr_scheduler = StepLR(new_optimizer, step_size=100, gamma=0.9)

        # Load training state
        step, loaded_optimizer, loaded_lr_scheduler, _, _ = load_training_state(
            checkpoint_path, new_optimizer, new_lr_scheduler
        )

        # Verify loaded state
        assert step == expected_step
        assert loaded_optimizer is new_optimizer
        assert loaded_lr_scheduler is new_lr_scheduler

        # Verify optimizer state was restored
        assert loaded_optimizer.state_dict()["param_groups"] is not None

    def test_load_from_pretrained_training_checkpoint(
        self, trained_policy_checkpoint, diffusion_config_for_training
    ):
        """Test load_from_pretrained with training checkpoint format"""
        checkpoint_info = trained_policy_checkpoint
        checkpoint_path = checkpoint_info["checkpoint_path"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create a fresh policy
        fresh_policy = DiffusionPolicy(config=diffusion_config_for_training)
        fresh_policy.to(device)

        # Get original weights from checkpoint for comparison
        original_checkpoint = torch.load(checkpoint_path, weights_only=False)
        original_state_dict = original_checkpoint["policy_state_dict"]

        # Verify they have different weights initially
        fresh_state_dict = fresh_policy.state_dict()
        states_different = False
        for key in original_state_dict:
            if not torch.equal(original_state_dict[key], fresh_state_dict[key]):
                states_different = True
                break
        assert states_different, "Fresh policy should have different weights"

        # Load pretrained weights
        fresh_policy.load_from_pretrained(checkpoint_path)

        # Verify weights match
        loaded_state_dict = fresh_policy.state_dict()
        for key in original_state_dict:
            assert torch.allclose(original_state_dict[key], loaded_state_dict[key], atol=1e-6), (
                f"Mismatch in parameter: {key}"
            )

    def test_training_progression_with_train_policy_step(
        self, diffusion_policy_for_training, diffusion_training_batch, mock_train_config
    ):
        """Test that train_policy_step produces meaningful training progression"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create optimizer and scheduler
        optimizer, lr_scheduler = make_optimizer_and_scheduler(
            mock_train_config, diffusion_policy_for_training
        )

        # Record initial learning rate
        initial_lr = optimizer.param_groups[0]["lr"]
        assert initial_lr == diffusion_policy_for_training.config.optimizer_lr

        # Run several training steps and record metrics
        diffusion_policy_for_training.train()
        training_metrics = []

        for step in range(1, 11):  # Run 10 training steps
            metrics = train_policy_step(
                policy=diffusion_policy_for_training,
                batch=diffusion_training_batch,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                step=step,
                device=device,
                use_amp=False,
            )
            training_metrics.append(metrics)

            # Validate metrics structure
            assert "loss" in metrics
            assert "learning_rate" in metrics
            assert isinstance(metrics["loss"], float)
            assert isinstance(metrics["learning_rate"], float)
            assert metrics["loss"] > 0  # Loss should be positive

        # Verify learning rate changed (depends on scheduler type)
        final_lr = training_metrics[-1]["learning_rate"]
        if lr_scheduler is not None:
            # For cosine scheduler with warmup, LR should change
            assert (
                final_lr != initial_lr
                or len(training_metrics) < diffusion_policy_for_training.config.scheduler_warmup_steps
            )

        # Verify loss progression (should generally decrease or at least be bounded)
        losses = [m["loss"] for m in training_metrics]
        assert all(loss > 0 for loss in losses), "All losses should be positive"
        assert all(loss < 1000 for loss in losses), "Losses should be reasonable (< 1000)"

    def test_checkpoint_consistency_across_training_steps(
        self, diffusion_policy_for_training, diffusion_training_batch, mock_train_config
    ):
        """Test that checkpoints saved at different steps maintain consistency"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        optimizer, lr_scheduler = make_optimizer_and_scheduler(
            mock_train_config, diffusion_policy_for_training
        )
        diffusion_policy_for_training.train()

        checkpoints = []

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir)

            # Save checkpoints at different training steps
            for step in [1, 3, 5]:
                # Run training step
                metrics = train_policy_step(
                    policy=diffusion_policy_for_training,
                    batch=diffusion_training_batch,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    step=step,
                    device=device,
                )

                # Save checkpoint
                save_checkpoint(diffusion_policy_for_training, optimizer, step, metrics, str(checkpoint_dir))
                checkpoint_path = checkpoint_dir / f"checkpoint_step_{step:07d}.pt"

                # Load and verify checkpoint
                checkpoint = torch.load(checkpoint_path, weights_only=False)
                assert checkpoint["step"] == step
                assert "policy_state_dict" in checkpoint
                assert "optimizer_state_dict" in checkpoint
                assert "metrics" in checkpoint
                assert checkpoint["metrics"]["loss"] == metrics["loss"]

                checkpoints.append({"step": step, "checkpoint_path": checkpoint_path, "metrics": metrics})

        # Verify step progression
        for i, checkpoint in enumerate(checkpoints):
            assert checkpoint["step"] == [1, 3, 5][i]
            assert checkpoint["metrics"]["loss"] > 0

    def test_make_everything_creates_all_components(
        self, diffusion_config_for_training, sample_features_for_training
    ):
        """Test that make_everything properly creates all training components"""
        from unittest.mock import MagicMock, patch

        from rho.common.wandb_logging import WandBConfig
        from rho.datasets.lerobot_dataset import LeRobotDatasetConfig

        # Create a complete TrainConfig for testing
        dataset_config = LeRobotDatasetConfig(
            repo_id="lerobot/pusht",
            batch_size=4,
            features=sample_features_for_training,
            stats={
                k: {"mean": torch.zeros(f.shape), "std": torch.ones(f.shape)}
                for k, f in sample_features_for_training.items()
            },
        )

        train_config = TrainConfig(
            wandb=WandBConfig(enabled=False),  # Disable wandb for testing
            dataset=dataset_config,
            validation_dataset=None,
            policy=diffusion_config_for_training,
            pretrained_checkpoint=None,
            resume=False,
            device="cuda" if torch.cuda.is_available() else "cpu",
            batch_size=4,
            learning_rate=1e-4,
            steps=1000,
        )

        # Mock the dataloader creation since we don't have actual datasets
        mock_dataloader = MagicMock()
        mock_dataloader.__iter__ = MagicMock(return_value=iter([]))
        mock_dataloader.__len__ = MagicMock(return_value=0)

        with (
            patch("rho.training.train.make_dataloader", return_value=(mock_dataloader, None)),
            patch(
                "rho.training.train.make_policy",
                return_value=DiffusionPolicy(config=diffusion_config_for_training),
            ),
        ):
            (
                training_dataloader,
                validation_dataloader,
                policy,
                optimizer,
                lr_scheduler,
                train_transforms,
                step,
                training_sampler,
                training_metrics_recorder,
            ) = make_everything(train_config)

        # Verify all components are created
        assert training_dataloader is not None
        assert validation_dataloader is None  # Since validation_dataset is None
        assert isinstance(policy, DiffusionPolicy)
        assert optimizer is not None
        assert isinstance(optimizer, torch.optim.Adam)
        assert lr_scheduler is not None
        assert train_transforms is not None  # Should be created even if empty
        assert step == 0  # Should start at 0 when not resuming

        # Verify policy is in training mode
        assert policy.training

        # Verify optimizer has correct learning rate
        assert optimizer.param_groups[0]["lr"] == diffusion_config_for_training.optimizer_lr

    def test_make_everything_with_resume_from_checkpoint(
        self, trained_policy_checkpoint, diffusion_config_for_training
    ):
        """Test that make_everything properly resumes from checkpoint"""
        from unittest.mock import MagicMock, patch

        from rho.common.wandb_logging import WandBConfig
        from rho.datasets.lerobot_dataset import LeRobotDatasetConfig

        checkpoint_info = trained_policy_checkpoint
        checkpoint_path = checkpoint_info["checkpoint_path"]
        expected_step = checkpoint_info["final_step"]
        sample_features = diffusion_config_for_training.feature_dict

        # Create a complete TrainConfig for testing with resume=True
        dataset_config = LeRobotDatasetConfig(
            repo_id="lerobot/pusht",
            batch_size=4,
            features=sample_features,
            stats={
                k: {"mean": torch.zeros(f.shape), "std": torch.ones(f.shape)}
                for k, f in sample_features.items()
            },
        )

        train_config = TrainConfig(
            wandb=WandBConfig(enabled=False),
            dataset=dataset_config,
            validation_dataset=None,
            policy=diffusion_config_for_training,
            pretrained_checkpoint=str(checkpoint_path),
            resume=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
            batch_size=4,
            learning_rate=1e-4,
            steps=1000,
        )

        # Mock the dataloader creation
        mock_dataloader = MagicMock()
        mock_dataloader.__iter__ = MagicMock(return_value=iter([]))
        mock_dataloader.__len__ = MagicMock(return_value=0)

        with (
            patch("rho.training.train.make_dataloader", return_value=(mock_dataloader, None)),
            patch(
                "rho.training.train.make_policy",
                return_value=DiffusionPolicy(config=diffusion_config_for_training),
            ),
        ):
            (
                training_dataloader,
                validation_dataloader,
                policy,
                optimizer,
                lr_scheduler,
                train_transforms,
                step,
                training_sampler,
                training_metrics_recorder,
            ) = make_everything(train_config)

        # Verify components are created and state is resumed
        assert training_dataloader is not None
        assert validation_dataloader is None
        assert isinstance(policy, DiffusionPolicy)
        assert optimizer is not None
        assert lr_scheduler is not None
        assert train_transforms is not None  # Should be created even if empty
        assert step == expected_step  # Should resume from checkpoint step

        # Verify policy is in training mode
        assert policy.training
