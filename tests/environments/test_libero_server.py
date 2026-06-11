"""Tests for the LiberoServer environment."""

import numpy as np
import pytest
import torch

from rho.common.types import ActionType
from rho.environment import ENV_REGISTRY, make_environment

# =============================================================================
# LiberoServer registration tests
# =============================================================================


class TestLiberoServerRegistration:
    """Tests for LiberoServer environment registration."""

    def test_registered_in_env_registry(self):
        """Test that LiberoServer is registered in the environment registry."""
        from environments.libero.libero_server import LiberoServer  # noqa: F401  — triggers registration

        assert "libero_server" in ENV_REGISTRY

    def test_make_environment(self):
        """Test that make_environment can create a LiberoServer."""
        from environments.libero.libero_server import LiberoServer, LiberoServerConfig

        config = LiberoServerConfig(device="cpu")
        env = make_environment(config)
        assert isinstance(env, LiberoServer)
        assert env.device == "cpu"


# =============================================================================
# LiberoServer initialization tests
# =============================================================================


class TestLiberoServerInit:
    """Tests for LiberoServer initialization."""

    def test_init_sets_image_keys(self):
        """Test that initialization sets expected image keys."""
        from environments.libero.libero_server import LiberoServer, LiberoServerConfig

        config = LiberoServerConfig(device="cpu")
        server = LiberoServer(config)

        assert server.image_keys == ["agentview", "wrist"]

    def test_init_sets_policy_action_type(self):
        """Test that initialization sets policy action type from config."""
        from environments.libero.libero_server import LiberoServer, LiberoServerConfig

        config = LiberoServerConfig(
            device="cpu",
            policy_action_type=ActionType.POSITION,
        )
        server = LiberoServer(config)

        assert server.policy_action_type == ActionType.POSITION


# =============================================================================
# LiberoServer.process_input tests
# =============================================================================


class TestProcessInput:
    """Tests for LiberoServer.process_input."""

    @pytest.fixture
    def server(self):
        """Create a LiberoServer with CPU device for testing."""
        from environments.libero.libero_server import LiberoServer, LiberoServerConfig

        config = LiberoServerConfig(
            device="cpu",
            policy_action_type=ActionType.POSITION,
        )
        return LiberoServer(config)

    def test_image_rotation_and_normalization(self, server):
        """Test that images are rotated and normalized to [0, 1]."""
        # Create an RGB image (H, W, 3) as numpy array
        rgb_image = np.zeros((128, 128, 3), dtype=np.uint8)
        rgb_image[:, :, 0] = 255  # Red channel
        rgb_image[:, :, 1] = 128  # Green channel
        rgb_image[:, :, 2] = 64  # Blue channel

        input_data = {
            "agentview": rgb_image,
            "state": np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]),
        }

        obs = server.process_input(input_data)

        img = obs["agentview"]
        assert img.shape == (1, 3, 128, 128), f"Expected (1, 3, 128, 128), got {img.shape}"
        # Check normalization: Red channel should be 255/255=1.0
        assert torch.allclose(img[0, 0, 0, 0], torch.tensor(255.0 / 255.0), atol=1e-3)
        # Green channel should be 128/255
        assert torch.allclose(img[0, 1, 0, 0], torch.tensor(128.0 / 255.0), atol=1e-3)
        # Blue channel should be 64/255
        assert torch.allclose(img[0, 2, 0, 0], torch.tensor(64.0 / 255.0), atol=1e-3)

    def test_image_dtype_and_range(self, server):
        """Test that images are converted to float32 and normalized to [0, 1] range."""
        input_data = {
            "agentview": np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8),
            "wrist": np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8),
            "state": np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]),
        }

        obs = server.process_input(input_data)

        for key in ["agentview", "wrist"]:
            assert obs[key].dtype == torch.float32
            assert obs[key].min() >= 0.0
            assert obs[key].max() <= 1.0
            assert obs[key].shape == (1, 3, 128, 128)

    def test_state_tensor_2d_to_3d(self, server):
        """Test that 2D state tensors (1, dim) are unsqueezed to (1, 1, dim)."""
        input_data = {
            "state": np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]),
        }

        obs = server.process_input(input_data)

        state = obs["state"]
        assert state.shape == (1, 1, 8), f"Expected (1, 1, 8), got {state.shape}"
        assert state.dtype == torch.float32

    def test_state_tensor_1d_to_3d(self, server):
        """Test that 1D state tensors are unsqueezed to (1, 1, dim)."""
        input_data = {
            "state": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
        }

        obs = server.process_input(input_data)

        state = obs["state"]
        assert state.shape == (1, 1, 8), f"Expected (1, 1, 8), got {state.shape}"
        assert state.dtype == torch.float32

    def test_state_tensor_2d_sequence(self, server):
        """Test that 2D state tensors with sequence length > 1 are unsqueezed to (1, seq, dim)."""
        # Simulate history of states: (seq_len, dim)
        input_data = {
            "state": np.array(
                [
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                    [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                ]
            ),
        }

        obs = server.process_input(input_data)

        state = obs["state"]
        assert state.shape == (1, 3, 8), f"Expected (1, 3, 8), got {state.shape}"

    def test_task_passthrough(self, server):
        """Test that task strings are passed through unchanged."""
        input_data = {
            "task": ["pick up the red cube", "place it on the blue plate"],
            "state": np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]),
        }

        obs = server.process_input(input_data)

        assert "task" in obs
        assert obs["task"] == ["pick up the red cube", "place it on the blue plate"]

    def test_multiple_images_and_state(self, server):
        """Test processing multiple images and state together."""
        input_data = {
            "agentview": np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8),
            "wrist": np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8),
            "state": np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]),
            "task": ["test task"],
        }

        obs = server.process_input(input_data)

        # Check images
        for key in ["agentview", "wrist"]:
            assert obs[key].shape == (1, 3, 128, 128)
            assert obs[key].dtype == torch.float32

        # Check state
        assert obs["state"].shape == (1, 1, 8)
        assert obs["state"].dtype == torch.float32

        # Check task
        assert obs["task"] == ["test task"]


# =============================================================================
# LiberoServer.process_output tests
# =============================================================================


class TestProcessOutput:
    """Tests for LiberoServer.process_output."""

    @pytest.fixture
    def server(self):
        """Create a LiberoServer for testing."""
        from environments.libero.libero_server import LiberoServer, LiberoServerConfig

        config = LiberoServerConfig(
            device="cpu",
            policy_action_type=ActionType.POSITION,
        )
        return LiberoServer(config)

    def test_squeeze_batch_dimension(self, server):
        """Test that batch dimension is squeezed from (1, chunk, dim) to (chunk, dim)."""
        actions = torch.randn(1, 10, 7)
        result = server.process_output(actions)

        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 7)

    def test_no_squeeze_without_batch(self, server):
        """Test that actions without batch dim are passed through."""
        actions = torch.randn(10, 7)
        result = server.process_output(actions)

        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 7)

    def test_output_is_float32_numpy(self, server):
        """Test that output is float32 numpy array."""
        actions = torch.randn(1, 10, 7, dtype=torch.float64)
        result = server.process_output(actions)

        assert result.dtype == np.float32

    def test_action_shape_with_different_chunk_sizes(self, server):
        """Test action tensor handling with different chunk sizes."""
        # Test with chunk_size=1
        actions_1 = torch.randn(1, 1, 7)
        result_1 = server.process_output(actions_1)
        assert result_1.shape == (1, 7)

        # Test with chunk_size=5
        actions_5 = torch.randn(1, 5, 7)
        result_5 = server.process_output(actions_5)
        assert result_5.shape == (5, 7)

        # Test with chunk_size=20
        actions_20 = torch.randn(1, 20, 7)
        result_20 = server.process_output(actions_20)
        assert result_20.shape == (20, 7)

    def test_action_values_preserved(self, server):
        """Test that action values are preserved during conversion."""
        actions = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]])
        result = server.process_output(actions)

        np.testing.assert_allclose(
            result, np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]], dtype=np.float32), rtol=1e-5
        )


# =============================================================================
# LiberoServer dict_to_torch (inherited from Server) tests
# =============================================================================


class TestDictToTorch:
    """Tests for the inherited dict_to_torch method."""

    @pytest.fixture
    def server(self):
        from environments.libero.libero_server import LiberoServer, LiberoServerConfig

        config = LiberoServerConfig(device="cpu")
        return LiberoServer(config)

    def test_numpy_array_converted(self, server):
        """Test that numpy arrays are converted to torch tensors."""
        obs = {"state": np.array([1.0, 2.0, 3.0], dtype=np.float32)}
        result = server.dict_to_torch(obs, "cpu")

        assert isinstance(result["state"], torch.Tensor)
        assert torch.allclose(result["state"], torch.tensor([1.0, 2.0, 3.0]))

    def test_torch_tensor_moved(self, server):
        """Test that torch tensors are moved to the correct device."""
        obs = {"state": torch.tensor([1.0, 2.0, 3.0])}
        result = server.dict_to_torch(obs, "cpu")

        assert isinstance(result["state"], torch.Tensor)
        assert result["state"].device.type == "cpu"

    def test_nested_dict(self, server):
        """Test that nested dicts are processed recursively."""
        obs = {"nested": {"inner": np.array([1.0, 2.0])}}
        result = server.dict_to_torch(obs, "cpu")

        assert isinstance(result["nested"]["inner"], torch.Tensor)

    def test_non_tensor_values_preserved(self, server):
        """Test that non-tensor/non-array values are preserved."""
        obs = {"label": "test_string", "count": 42}
        result = server.dict_to_torch(obs, "cpu")

        assert result["label"] == "test_string"
        assert result["count"] == 42
