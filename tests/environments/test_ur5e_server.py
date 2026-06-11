"""Tests for the UR5eServer environment."""

from unittest.mock import patch

import numpy as np
import pytest
import torch

from rho.common.types import ActionType
from rho.environment import ENV_REGISTRY, make_environment

# =============================================================================
# UR5eServer registration tests
# =============================================================================


class TestUR5eServerRegistration:
    """Tests for UR5eServer environment registration."""

    def test_registered_in_env_registry(self):
        """Test that UR5eServer is registered in the environment registry."""
        from environments.ur5e.ur5 import UR5eServer  # noqa: F401  — triggers registration

        assert "ur5e_server" in ENV_REGISTRY

    def test_make_environment(self):
        """Test that make_environment can create a UR5eServer."""
        from environments.ur5e.ur5 import UR5eServer, UR5eServerConfig

        config = UR5eServerConfig(device="cpu")
        env = make_environment(config)
        assert isinstance(env, UR5eServer)
        assert env.device == "cpu"


# =============================================================================
# UR5eServer initialization tests
# =============================================================================


class TestUR5eServerInit:
    """Tests for UR5eServer initialization."""

    def test_init_sets_image_keys(self):
        """Test that initialization sets expected image keys."""
        from environments.ur5e.ur5 import UR5eServer, UR5eServerConfig

        config = UR5eServerConfig(device="cpu")
        server = UR5eServer(config)

        assert server.image_keys == ["zed_scene_bgr", "zed_wrist_left_bgr", "zed_wrist_right_bgr"]

    def test_init_sets_action_types(self):
        """Test that initialization sets action types from config."""
        from environments.ur5e.ur5 import UR5eServer, UR5eServerConfig

        config = UR5eServerConfig(
            device="cpu",
            input_action_type=ActionType.POSITION,
            output_action_type=ActionType.EE_QUAT_POS_XYZW,
            policy_action_type=ActionType.POSITION,
        )
        server = UR5eServer(config)

        assert server.input_action_type == ActionType.POSITION
        assert server.output_action_type == ActionType.EE_QUAT_POS_XYZW
        assert server.policy_action_type == ActionType.POSITION


# =============================================================================
# UR5eServer.process_input tests
# =============================================================================


class TestProcessInput:
    """Tests for UR5eServer.process_input."""

    @pytest.fixture
    def server(self):
        """Create a UR5eServer with CPU device for testing."""
        from environments.ur5e.ur5 import UR5eServer, UR5eServerConfig

        config = UR5eServerConfig(
            device="cpu",
            input_action_type=ActionType.POSITION,
            policy_action_type=ActionType.POSITION,
        )
        return UR5eServer(config)

    def test_image_bgr_to_rgb_and_normalized(self, server):
        """Test that images are converted from BGR to RGB and normalized to [0, 1]."""
        # Create a BGR image (H, W, 3) as numpy array
        bgr_image = np.zeros((180, 320, 3), dtype=np.uint8)
        bgr_image[:, :, 0] = 255  # Blue channel
        bgr_image[:, :, 2] = 128  # Red channel

        input_data = {
            "zed_scene_bgr": bgr_image,
            "joint_positions": np.array(
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0]
            ),
        }

        obs = server.process_input(input_data)

        img = obs["zed_scene_bgr"]
        assert img.shape == (1, 3, 180, 320), f"Expected (1, 3, 180, 320), got {img.shape}"
        # After BGR->RGB: channel 0 should be Red (128/255), channel 2 should be Blue (255/255)
        assert torch.allclose(img[0, 0, 0, 0], torch.tensor(128.0 / 255.0), atol=1e-3)
        assert torch.allclose(img[0, 2, 0, 0], torch.tensor(255.0 / 255.0), atol=1e-3)

    def test_state_tensor_1d_unsqueeze(self, server):
        """Test that 1D state tensors are unsqueezed to (1, 1, dim)."""
        input_data = {
            "joint_positions": np.array(
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0]
            ),
        }

        obs = server.process_input(input_data)

        jp = obs["joint_positions"]
        assert jp.shape == (1, 1, 14), f"Expected (1, 1, 14), got {jp.shape}"
        assert jp.dtype == torch.float32

    def test_state_tensor_2d_unsqueeze(self, server):
        """Test that 2D state tensors are unsqueezed to (1, seq, dim)."""
        # Simulate history of joint positions: (seq_len, dim)
        input_data = {
            "joint_positions": np.array(
                [
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0],
                    [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.0],
                ]
            ),
        }

        obs = server.process_input(input_data)

        jp = obs["joint_positions"]
        assert jp.shape == (1, 2, 14), f"Expected (1, 2, 14), got {jp.shape}"

    def test_tactile_history_renamed(self, server):
        """Test that 'tactile_history' is renamed to 'tactile'."""
        input_data = {
            "tactile_history": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "joint_positions": np.array(
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0]
            ),
        }

        obs = server.process_input(input_data)

        assert "tactile" in obs
        assert "tactile_history" not in obs

    def test_tcp_forces_history_renamed(self, server):
        """Test that 'tcp_forces_history' is renamed to 'actual_tcp_forces'."""
        input_data = {
            "tcp_forces_history": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "joint_positions": np.array(
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0]
            ),
        }

        obs = server.process_input(input_data)

        assert "actual_tcp_forces" in obs
        assert "tcp_forces_history" not in obs

    def test_multiple_images_processed(self, server):
        """Test that all image keys are processed correctly."""
        input_data = {
            "zed_scene_bgr": np.random.randint(0, 256, (180, 320, 3), dtype=np.uint8),
            "zed_wrist_left_bgr": np.random.randint(0, 256, (180, 320, 3), dtype=np.uint8),
            "zed_wrist_right_bgr": np.random.randint(0, 256, (180, 320, 3), dtype=np.uint8),
            "joint_positions": np.array(
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0]
            ),
        }

        obs = server.process_input(input_data)

        for key in ["zed_scene_bgr", "zed_wrist_left_bgr", "zed_wrist_right_bgr"]:
            assert obs[key].shape == (1, 3, 180, 320), f"{key} shape mismatch: {obs[key].shape}"
            assert obs[key].dtype == torch.float32
            assert obs[key].min() >= 0.0
            assert obs[key].max() <= 1.0


# =============================================================================
# UR5eServer.process_output tests
# =============================================================================


class TestProcessOutput:
    """Tests for UR5eServer.process_output."""

    @pytest.fixture
    def server(self):
        """Create a UR5eServer with matching output and policy action types."""
        from environments.ur5e.ur5 import UR5eServer, UR5eServerConfig

        config = UR5eServerConfig(
            device="cpu",
            output_action_type=ActionType.POSITION,
            policy_action_type=ActionType.POSITION,
        )
        return UR5eServer(config)

    def test_squeeze_batch_dimension(self, server):
        """Test that batch dimension is squeezed from (1, chunk, dim) to (chunk, dim)."""
        actions = torch.randn(1, 5, 14)
        result = server.process_output(actions)

        assert isinstance(result, np.ndarray)
        assert result.shape == (5, 14)

    def test_no_squeeze_without_batch(self, server):
        """Test that actions without batch dim are passed through."""
        actions = torch.randn(5, 14)
        result = server.process_output(actions)

        assert isinstance(result, np.ndarray)
        assert result.shape == (5, 14)

    def test_output_is_float32_numpy(self, server):
        """Test that output is float32 numpy array."""
        actions = torch.randn(1, 5, 14, dtype=torch.float64)
        result = server.process_output(actions)

        assert result.dtype == np.float32

    def test_output_action_type_mismatch_raises(self):
        """Test that mismatched output and policy action types raises assertion."""
        from environments.ur5e.ur5 import UR5eServer, UR5eServerConfig

        config = UR5eServerConfig(
            device="cpu",
            output_action_type=ActionType.EE_EULER_POS,
            policy_action_type=ActionType.POSITION,
        )
        server = UR5eServer(config)
        actions = torch.randn(1, 5, 14)

        with pytest.raises(AssertionError, match="same as"):
            server.process_output(actions)


# =============================================================================
# UR5eServer.convert_observation_state_type_to_policy_type tests
# =============================================================================


class TestConvertObservationStateToPolicyType:
    """Tests for UR5eServer.convert_observation_state_type_to_policy_type."""

    def test_same_input_and_policy_type_noop(self):
        """Test that same input/policy type is a no-op."""
        from environments.ur5e.ur5 import UR5eServer, UR5eServerConfig

        config = UR5eServerConfig(
            device="cpu",
            input_action_type=ActionType.POSITION,
            policy_action_type=ActionType.POSITION,
        )
        server = UR5eServer(config)

        original_data = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0])
        obs = {"joint_positions": original_data.clone()}
        server.convert_observation_state_type_to_policy_type(obs)

        assert torch.allclose(obs["joint_positions"], original_data)

    @patch("environments.ur5e.ur5.compute_eef_poses_for_dual_arm")
    def test_position_to_ee_euler_calls_fk(self, mock_fk):
        """Test that POSITION -> EE_EULER_POS calls FK and stores result in ee_state."""
        from environments.ur5e.ur5 import UR5eServer, UR5eServerConfig

        mock_fk.return_value = np.array(
            [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0]]
        )
        config = UR5eServerConfig(
            device="cpu",
            input_action_type=ActionType.POSITION,
            policy_action_type=ActionType.EE_EULER_POS,
        )
        server = UR5eServer(config)

        original_joints = torch.tensor(
            [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0]]
        )
        obs = {"joint_positions": original_joints.clone()}
        server.convert_observation_state_type_to_policy_type(obs)

        mock_fk.assert_called_once()
        # FK result stored in ee_state, raw joints preserved
        assert "ee_state" in obs
        assert isinstance(obs["ee_state"], torch.Tensor)
        assert torch.allclose(obs["joint_positions"], original_joints)

    def test_non_position_input_raises(self):
        """Test that non-POSITION input raises assertion."""
        from environments.ur5e.ur5 import UR5eServer, UR5eServerConfig

        config = UR5eServerConfig(
            device="cpu",
            input_action_type=ActionType.EE_EULER_POS,
            policy_action_type=ActionType.POSITION,
        )
        server = UR5eServer(config)
        obs = {"joint_positions": torch.tensor([0.1, 0.2, 0.3])}

        with pytest.raises(AssertionError, match="only supports joint_position"):
            server.convert_observation_state_type_to_policy_type(obs)

    @patch("environments.ur5e.ur5.compute_eef_quat_poses")
    def test_position_to_ee_quat_calls_direct_fk_quat(self, mock_fk_quat):
        """Test that POSITION -> EE_QUAT_POS_XYZW uses direct FK-to-quat (no euler intermediate)."""
        from environments.ur5e.ur5 import UR5eServer, UR5eServerConfig

        # Direct FK returns (1, 16) quaternion poses
        mock_fk_quat.return_value = np.array(
            [[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0, 0.0, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0, 0.0]]
        )

        config = UR5eServerConfig(
            device="cpu",
            input_action_type=ActionType.POSITION,
            policy_action_type=ActionType.EE_QUAT_POS_XYZW,
        )
        server = UR5eServer(config)

        original_joints = torch.tensor(
            [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0]]
        )
        obs = {"joint_positions": original_joints.clone()}
        server.convert_observation_state_type_to_policy_type(obs)

        mock_fk_quat.assert_called_once()
        # Quat result stored in ee_state, raw joints preserved
        assert "ee_state" in obs
        assert isinstance(obs["ee_state"], torch.Tensor)
        assert obs["ee_state"].shape == (1, 16)
        assert torch.allclose(obs["joint_positions"], original_joints)

    def test_unsupported_conversion_raises(self):
        """Test that unsupported input->policy conversion raises NotImplementedError."""
        from environments.ur5e.ur5 import UR5eServer, UR5eServerConfig

        config = UR5eServerConfig(
            device="cpu",
            input_action_type=ActionType.POSITION,
            policy_action_type=ActionType.EE_6D_POS,
        )
        server = UR5eServer(config)
        obs = {
            "joint_positions": torch.tensor(
                [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0]]
            )
        }

        with pytest.raises(NotImplementedError, match="not implemented"):
            server.convert_observation_state_type_to_policy_type(obs)


# =============================================================================
# UR5eServer.convert_policy_action_type_to_client_type tests
# =============================================================================


class TestConvertPolicyActionToClientType:
    """Tests for UR5eServer.convert_policy_action_type_to_client_type."""

    def test_matching_types_passthrough(self):
        """Test that matching output/policy types pass actions through."""
        from environments.ur5e.ur5 import UR5eServer, UR5eServerConfig

        config = UR5eServerConfig(
            device="cpu",
            output_action_type=ActionType.EE_EULER_POS,
            policy_action_type=ActionType.EE_EULER_POS,
        )
        server = UR5eServer(config)

        actions = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
        result = server.convert_policy_action_type_to_client_type(actions)

        np.testing.assert_array_equal(result, actions)

    def test_mismatched_types_raises(self):
        """Test that mismatched types raise assertion."""
        from environments.ur5e.ur5 import UR5eServer, UR5eServerConfig

        config = UR5eServerConfig(
            device="cpu",
            output_action_type=ActionType.EE_EULER_POS,
            policy_action_type=ActionType.POSITION,
        )
        server = UR5eServer(config)

        actions = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])

        with pytest.raises(AssertionError, match="same as"):
            server.convert_policy_action_type_to_client_type(actions)


# =============================================================================
# UR5eServer dict_to_torch (inherited from Server) tests
# =============================================================================


class TestDictToTorch:
    """Tests for the inherited dict_to_torch method."""

    @pytest.fixture
    def server(self):
        from environments.ur5e.ur5 import UR5eServer, UR5eServerConfig

        config = UR5eServerConfig(device="cpu")
        return UR5eServer(config)

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
