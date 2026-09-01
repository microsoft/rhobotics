from unittest.mock import MagicMock, patch

import pytest
import torch

from rho.common.constants import ACTION, OBSERVATION_STATE
from rho.common.transforms import (
    AbsoluteActions,
    ColorJitter,
    CombineKeys,
    DeltaActions,
    RandomFlipLeftRight,
    RandomFlipUpDown,
    RandomResizedCrop,
    RandomRot90,
    TaskDescriptionSelector,
    build_key_padding_transform,
    get_target_sequence_lengths,
)
from rho.common.types import ActionType, FeatureType, PolicyFeature
from rho.datasets.data_config import TransformWrapper
from rho.datasets.lerobot_dataset import LeRobotDatasetConfig


class TestTransformWrapper:
    """Test the TransformWrapper class."""

    def test_transform_wrapper_applies_transform(self):
        """Test that TransformWrapper applies transform to the correct key."""
        mock_transform = MagicMock(return_value=torch.tensor([1, 2, 3]))
        wrapper = TransformWrapper("test_key", mock_transform)

        sample = {"test_key": torch.tensor([4, 5, 6]), "other_key": torch.tensor([7, 8, 9])}

        result = wrapper(sample)

        # Check that mock_transform was called with the correct tensor
        mock_transform.assert_called_once()
        call_args = mock_transform.call_args[0][0]
        assert torch.equal(call_args, torch.tensor([4, 5, 6]))

        assert torch.equal(result["test_key"], torch.tensor([1, 2, 3]))
        assert torch.equal(result["other_key"], torch.tensor([7, 8, 9]))

    def test_transform_wrapper_missing_key(self):
        """Test that TransformWrapper handles missing keys gracefully."""
        mock_transform = MagicMock()
        wrapper = TransformWrapper("missing_key", mock_transform)

        sample = {"other_key": torch.tensor([7, 8, 9])}

        result = wrapper(sample)

        mock_transform.assert_not_called()
        assert result == sample


class TestKeyPaddingTransform:
    def test_derives_model_shape_from_temporal_lookup_lengths(self):
        lengths = get_target_sequence_lengths(
            {
                ACTION: [0, 4, 8, 12, 16],
                OBSERVATION_STATE: [-2, -1, 0],
                "observation.image.0": [-2, -1, 0],
            }
        )

        assert lengths == {
            FeatureType.ACTION: 5,
            FeatureType.STATE: 3,
            FeatureType.VISUAL: 3,
        }

    def test_pads_short_action_sequence_and_marks_tail(self):
        transform = build_key_padding_transform(
            features={ACTION: PolicyFeature(FeatureType.ACTION, (4,))},
            target_sequence_lengths={FeatureType.ACTION: 5},
        )
        sample = {
            ACTION: torch.tensor(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                    [5.0, 6.0],
                ]
            ),
            f"{ACTION}_is_pad": torch.tensor([False, True, False]),
        }

        result = transform(sample)

        assert result[ACTION].shape == (5, 4)
        assert torch.equal(result[ACTION][:3, :2], torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        assert torch.equal(result[ACTION][3:], torch.zeros(2, 4))
        assert torch.equal(
            result[f"{ACTION}_is_pad"],
            torch.tensor([False, True, False, True, True]),
        )
        assert torch.equal(
            result[f"{ACTION}_dim_is_pad"],
            torch.tensor([False, False, True, True]),
        )

    def test_emits_all_false_dim_mask_for_full_width_action(self):
        transform = build_key_padding_transform(
            features={ACTION: PolicyFeature(FeatureType.ACTION, (4,))},
            target_sequence_lengths={FeatureType.ACTION: 2},
        )
        sample = {ACTION: torch.ones(2, 4)}

        result = transform(sample)

        assert result[ACTION].shape == (2, 4)
        assert torch.equal(result[f"{ACTION}_dim_is_pad"], torch.zeros(4, dtype=torch.bool))

    def test_expands_existing_native_width_dim_mask(self):
        transform = build_key_padding_transform(
            features={ACTION: PolicyFeature(FeatureType.ACTION, (4,))},
            target_sequence_lengths={FeatureType.ACTION: 2},
        )
        sample = {
            ACTION: torch.ones(2, 2),
            f"{ACTION}_dim_is_pad": torch.tensor([False, False]),
        }

        result = transform(sample)

        assert result[ACTION].shape == (2, 4)
        assert torch.equal(
            result[f"{ACTION}_dim_is_pad"],
            torch.tensor([False, False, True, True]),
        )

    def test_transformed_wider_action_dims_remain_real_through_nested_padding(self):
        leaf_padding = build_key_padding_transform(
            features={ACTION: PolicyFeature(FeatureType.ACTION, (16,))},
            target_sequence_lengths={FeatureType.ACTION: 2},
        )
        parent_padding = build_key_padding_transform(
            features={ACTION: PolicyFeature(FeatureType.ACTION, (20,))},
            target_sequence_lengths={FeatureType.ACTION: 2},
        )
        sample = {ACTION: torch.ones(2, 20)}

        result = parent_padding(leaf_padding(sample))

        assert result[ACTION].shape == (2, 20)
        assert torch.equal(result[f"{ACTION}_dim_is_pad"], torch.zeros(20, dtype=torch.bool))


class TestTaskDescriptionSelector:
    """Test task/subtask language selection."""

    def test_selects_subtask_when_probability_one(self):
        transform = TaskDescriptionSelector(subtask_probability=1.0)
        sample = {"task": "assemble the kit", "subtask": "pick up the screw"}

        result = transform(sample)

        assert result["task"] == "pick up the screw"
        assert "subtask" not in result

    def test_keeps_task_when_probability_zero(self):
        transform = TaskDescriptionSelector(subtask_probability=0.0)
        sample = {"task": "assemble the kit", "subtask": "pick up the screw"}

        result = transform(sample)

        assert result["task"] == "assemble the kit"
        assert "subtask" not in result

    def test_falls_back_to_task_when_subtask_missing_or_empty(self):
        transform = TaskDescriptionSelector(subtask_probability=1.0)

        assert transform({"task": "assemble the kit"})["task"] == "assemble the kit"
        assert transform({"task": "assemble the kit", "subtask": "  "})["task"] == "assemble the kit"

    def test_uses_subtask_when_task_is_missing(self):
        transform = TaskDescriptionSelector(subtask_probability=0.0)

        result = transform({"subtask": "pick up the screw"})

        assert result["task"] == "pick up the screw"
        assert "subtask" not in result

    def test_selector_runs_inside_dataset_transform_pipeline(self):
        config = LeRobotDatasetConfig(
            observation_whitelist=["task", "subtask"],
            transform_mapping={
                "task": [
                    TaskDescriptionSelector(subtask_probability=1.0),
                ]
            },
            features=None,
            normalization_mapping=None,
        )
        transform = config.get_transforms()

        result = transform({"task": "assemble the kit", "subtask": "pick up the screw", "ignored": "x"})

        assert result == {"task": "pick up the screw"}


class TestCombineKeys:
    """Test the CombineKeys transform."""

    def test_combine_keys_basic(self):
        """Test that CombineKeys concatenates multiple keys along the last dimension."""
        # Create test batch with multiple keys
        batch = {
            "observation.state": torch.tensor([[1.0, 2.0, 3.0]]),
            "observation.velocity": torch.tensor([[4.0, 5.0]]),
            "observation.position": torch.tensor([[6.0, 7.0, 8.0, 9.0]]),
        }

        # Create CombineKeys transform
        combine_transform = CombineKeys(
            input_list=["observation.state", "observation.velocity", "observation.position"],
            output_key="observation.combined",
        )

        # Apply transform
        result = combine_transform(batch)

        # Verify the original keys are still present
        assert "observation.state" in result
        assert "observation.velocity" in result
        assert "observation.position" in result

        # Verify the combined key is created
        assert "observation.combined" in result

        # Verify the combined tensor has the correct shape and values
        expected = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]])
        assert torch.equal(result["observation.combined"], expected)

    def test_combine_keys_2d(self):
        """Test CombineKeys with 2D tensors (batch dimension)."""
        batch = {
            "key1": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "key2": torch.tensor([[5.0], [6.0]]),
            "key3": torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]),
        }

        combine_transform = CombineKeys(
            input_list=["key1", "key2", "key3"],
            output_key="combined",
        )

        result = combine_transform(batch)

        # Verify concatenation along last dimension
        expected = torch.tensor([[1.0, 2.0, 5.0, 7.0, 8.0, 9.0], [3.0, 4.0, 6.0, 10.0, 11.0, 12.0]])
        assert torch.equal(result["combined"], expected)

    def test_combine_keys_single_key(self):
        """Test CombineKeys with a single key."""
        batch = {
            "observation.state": torch.tensor([[1.0, 2.0, 3.0]]),
        }

        combine_transform = CombineKeys(
            input_list=["observation.state"],
            output_key="output",
        )

        result = combine_transform(batch)

        # Should just copy the tensor
        assert "output" in result
        assert torch.equal(result["output"], torch.tensor([[1.0, 2.0, 3.0]]))

    def test_combine_keys_no_input_list(self):
        """Test that CombineKeys raises error when input_list is None."""
        batch = {"key": torch.tensor([1.0])}

        combine_transform = CombineKeys(input_list=None, output_key="output")

        with pytest.raises(ValueError, match="input_list must be provided"):
            combine_transform(batch)

    def test_combine_keys_with_dataset_config(self):
        """Test using CombineKeys transform in LeRobotDatasetConfig.get_transforms()."""
        mock_feature_config = MagicMock()
        mock_feature_config.get_input_transform.return_value = None

        # Create a CombineKeys transform
        combine_transform = CombineKeys(
            input_list=["observation.state", "observation.velocity"],
            output_key="observation.combined_state",
        )

        # Note: CombineKeys has input_type="Dict" so it should be added directly to transform_list
        # not wrapped in TransformWrapper
        # transform_mapping expects a list of transforms per key
        transform_mapping = {"dummy_key": [combine_transform]}

        with patch("rho.datasets.lerobot_dataset.rho_features_from_lerobot_dataset") as mock_features:
            mock_features.return_value = mock_feature_config

            config = LeRobotDatasetConfig(features=mock_feature_config, transform_mapping=transform_mapping)

            composed_transform = config.get_transforms()

            # Test that the transform works correctly
            test_batch = {
                "observation.state": torch.tensor([[1.0, 2.0]]),
                "observation.velocity": torch.tensor([[3.0, 4.0]]),
            }

            result = composed_transform(test_batch)

            # Verify the combined key was created
            assert "observation.combined_state" in result
            expected = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
            assert torch.equal(result["observation.combined_state"], expected)


class TestDeltaActions:
    """Test the DeltaActions transform."""

    def test_delta_actions_basic(self):
        """Test that DeltaActions converts absolute actions to deltas correctly."""
        # Create test batch with state and actions
        batch = {
            "observation.state": torch.tensor([[1.0, 2.0, 3.0]]),  # Initial state
            "action": torch.tensor(
                [
                    [
                        [4.0, 5.0, 6.0],  # First action (absolute)
                        [7.0, 8.0, 9.0],  # Second action (absolute)
                        [10.0, 11.0, 12.0],  # Third action (absolute)
                    ]
                ]
            ),
        }

        # Create DeltaActions transform
        delta_transform = DeltaActions(
            state_key="observation.state",
            action_key="action",
        )

        # Apply transform
        result = delta_transform(batch)

        # Verify delta actions
        # With relative_to_state=False:
        # Position 0: actions[0] - state = [4, 5, 6] - [1, 2, 3] = [3, 3, 3]
        # Position 1: actions[1] - actions[0] = [7, 8, 9] - [4, 5, 6] = [3, 3, 3]
        # Position 2: actions[2] - actions[1] = [10, 11, 12] - [7, 8, 9] = [3, 3, 3]
        expected = torch.tensor(
            [
                [
                    [3.0, 3.0, 3.0],
                    [3.0, 3.0, 3.0],
                    [3.0, 3.0, 3.0],
                ]
            ]
        )
        assert torch.equal(result["action"], expected), f"Expected {expected}, got {result['action']}"

    def test_delta_actions_single_timestep(self):
        """Test DeltaActions with a single action timestep."""
        batch = {
            "observation.state": torch.tensor([[10.0, 20.0]]),
            "action": torch.tensor([[[15.0, 25.0]]]),  # Single action
        }

        delta_transform = DeltaActions()

        result = delta_transform(batch)

        # Single action with relative_to_state=False: action[0] - state = [15, 25] - [10, 20] = [5, 5]
        expected = torch.tensor([[[5.0, 5.0]]])
        assert torch.equal(result["action"], expected)

    def test_delta_actions_batched(self):
        """Test DeltaActions with batched data."""
        batch = {
            "observation.state": torch.tensor(
                [
                    [1.0, 2.0],
                    [10.0, 20.0],
                ]
            ),
            "action": torch.tensor(
                [
                    [[3.0, 4.0], [5.0, 6.0]],  # Batch 1: two timesteps
                    [[30.0, 40.0], [50.0, 60.0]],  # Batch 2: two timesteps
                ]
            ),
        }

        delta_transform = DeltaActions()

        result = delta_transform(batch)

        # With relative_to_state=False:
        # Batch 1: [3,4] - [1,2] = [2,2], [5,6] - [3,4] = [2,2]
        # Batch 2: [30,40] - [10,20] = [20,20], [50,60] - [30,40] = [20,20]
        expected = torch.tensor(
            [
                [[2.0, 2.0], [2.0, 2.0]],
                [[20.0, 20.0], [20.0, 20.0]],
            ]
        )
        assert torch.equal(result["action"], expected)

    def test_delta_actions_preserves_shape(self):
        """Test that DeltaActions returns tensor with same shape as input."""
        batch = {
            "observation.state": torch.randn(4, 1, 10),  # Batch of 4, 1 timestep, 10 dims
            "action": torch.randn(4, 5, 10),  # Batch of 4, 5 timesteps, 10 dims
        }

        delta_transform = DeltaActions()

        result = delta_transform(batch)

        # Shape should be preserved
        assert result["action"].shape == batch["action"].shape

    def test_delta_actions_relative_to_state_true(self):
        """Test DeltaActions with relative_to_state=True."""
        batch = {
            "observation.state": torch.tensor([[1.0, 2.0]]),
            "action": torch.tensor(
                [
                    [
                        [4.0, 5.0],  # First action (absolute)
                        [7.0, 8.0],  # Second action (absolute)
                        [10.0, 11.0],  # Third action (absolute)
                    ]
                ]
            ),
        }

        delta_transform = DeltaActions(relative_to_state=True)
        result = delta_transform(batch)

        # With relative_to_state=True, all actions become relative to state
        # First action: [4, 5] - [1, 2] = [3, 3]
        # Second action: [7, 8] - [1, 2] = [6, 6]
        # Third action: [10, 11] - [1, 2] = [9, 9]
        expected = torch.tensor(
            [
                [
                    [3.0, 3.0],  # All actions relative to state
                    [6.0, 6.0],
                    [9.0, 9.0],
                ]
            ]
        )
        assert torch.equal(result["action"], expected)

    def test_delta_actions_relative_to_state_false(self):
        """Test DeltaActions with relative_to_state=False (default behavior)."""
        batch = {
            "observation.state": torch.tensor([[1.0, 2.0]]),
            "action": torch.tensor(
                [
                    [
                        [4.0, 5.0],  # First action (absolute)
                        [7.0, 8.0],  # Second action (absolute)
                        [10.0, 11.0],  # Third action (absolute)
                    ]
                ]
            ),
        }

        delta_transform = DeltaActions(relative_to_state=False)
        result = delta_transform(batch)

        # With relative_to_state=False:
        # Position 0: actions[0] - state = [4, 5] - [1, 2] = [3, 3]
        # Position 1: actions[1] - actions[0] = [7, 8] - [4, 5] = [3, 3]
        # Position 2: actions[2] - actions[1] = [10, 11] - [7, 8] = [3, 3]
        expected = torch.tensor(
            [
                [
                    [3.0, 3.0],
                    [3.0, 3.0],
                    [3.0, 3.0],
                ]
            ]
        )
        assert torch.equal(result["action"], expected)

    def test_delta_actions_relative_to_state_comparison(self):
        """Test that relative_to_state flag produces different results."""
        batch = {
            "observation.state": torch.tensor([[10.0, 20.0]]),
            "action": torch.tensor(
                [
                    [
                        [15.0, 25.0],
                        [20.0, 30.0],
                        [25.0, 35.0],
                    ]
                ]
            ),
        }

        # Test with relative_to_state=True
        delta_transform_true = DeltaActions(relative_to_state=True)
        result_true = delta_transform_true(batch.copy())

        # Test with relative_to_state=False
        delta_transform_false = DeltaActions(relative_to_state=False)
        result_false = delta_transform_false(batch.copy())

        # Results should be different
        assert not torch.equal(result_true["action"], result_false["action"])

        # With relative_to_state=True: all actions become relative to state via broadcasting
        expected_true = torch.tensor(
            [
                [
                    [5.0, 5.0],  # [15, 25] - [10, 20]
                    [10.0, 10.0],  # [20, 30] - [10, 20]
                    [15.0, 15.0],  # [25, 35] - [10, 20]
                ]
            ]
        )
        assert torch.equal(result_true["action"], expected_true)

        # With relative_to_state=False:
        # Position 0: [15, 25] - [10, 20] = [5, 5]
        # Position 1: [20, 30] - [15, 25] = [5, 5]
        # Position 2: [25, 35] - [20, 30] = [5, 5]
        expected_false = torch.tensor(
            [
                [
                    [5.0, 5.0],
                    [5.0, 5.0],
                    [5.0, 5.0],
                ]
            ]
        )
        assert torch.equal(result_false["action"], expected_false)

    def test_ee_6d_delta_actions_can_keep_gripper_absolute(self):
        """EE pose dims are state-relative while gripper dims stay absolute."""
        identity_6d = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        state = torch.tensor(
            [
                [
                    1.0,
                    2.0,
                    3.0,
                    *identity_6d,
                    0.10,
                    10.0,
                    20.0,
                    30.0,
                    *identity_6d,
                    0.20,
                ]
            ]
        )
        actions = torch.tensor(
            [
                [
                    [1.5, 2.5, 3.5, *identity_6d, 0.70, 11.0, 21.0, 31.0, *identity_6d, 0.80],
                    [2.0, 3.0, 4.0, *identity_6d, 0.75, 12.0, 22.0, 32.0, *identity_6d, 0.85],
                ]
            ]
        )
        batch = {"observation.state": state, "action": actions.clone()}

        delta_transform = DeltaActions(
            action_type=ActionType.EE_6D_POS,
            relative_to_state=True,
            use_absolute_grippers=True,
        )
        result = delta_transform(batch)

        assert torch.allclose(result["action"][..., 0:3], actions[..., 0:3] - state[:, None, 0:3])
        assert torch.allclose(result["action"][..., 10:13], actions[..., 10:13] - state[:, None, 10:13])
        assert torch.equal(result["action"][..., 9], actions[..., 9])
        assert torch.equal(result["action"][..., 19], actions[..., 19])

        absolute_transform = AbsoluteActions(
            action_type=ActionType.EE_6D_POS,
            relative_to_state=True,
            use_absolute_grippers=True,
        )
        recovered = absolute_transform({"observation.state": state, "action": result["action"].clone()})

        assert torch.allclose(recovered["action"], actions)


class TestAbsoluteActions:
    """Test the AbsoluteActions transform."""

    def test_absolute_actions_basic(self):
        """Test that AbsoluteActions converts delta actions to absolute correctly."""
        # Create test batch with state and delta actions
        batch = {
            "observation.state": torch.tensor([[1.0, 2.0, 3.0]]),  # Initial state
            "action": torch.tensor(
                [
                    [
                        [3.0, 3.0, 3.0],  # First delta
                        [3.0, 3.0, 3.0],  # Second delta
                        [3.0, 3.0, 3.0],  # Third delta
                    ]
                ]
            ),
        }

        # Create AbsoluteActions transform
        absolute_transform = AbsoluteActions(
            state_key="observation.state",
            action_key="action",
        )

        # Apply transform
        result = absolute_transform(batch)

        # Verify absolute actions
        # First: [1, 2, 3] + cumsum([[3, 3, 3]]) = [1, 2, 3] + [3, 3, 3] = [4, 5, 6]
        # Second: [1, 2, 3] + cumsum([[3, 3, 3], [3, 3, 3]]) = [1, 2, 3] + [6, 6, 6] = [7, 8, 9]
        # Third: [1, 2, 3] + cumsum([[3, 3, 3], [3, 3, 3], [3, 3, 3]]) = [1, 2, 3] + [9, 9, 9] = [10, 11, 12]
        expected = torch.tensor(
            [
                [
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0],
                ]
            ]
        )
        assert torch.equal(result["action"], expected), f"Expected {expected}, got {result['action']}"

    def test_absolute_actions_single_timestep(self):
        """Test AbsoluteActions with a single delta action."""
        batch = {
            "observation.state": torch.tensor([[10.0, 20.0]]),
            "action": torch.tensor([[[5.0, 5.0]]]),  # Single delta
        }

        absolute_transform = AbsoluteActions()

        result = absolute_transform(batch)

        # Should be: [10, 20] + [5, 5] = [15, 25]
        expected = torch.tensor([[[15.0, 25.0]]])
        assert torch.equal(result["action"], expected)

    def test_absolute_actions_batched(self):
        """Test AbsoluteActions with batched data."""
        batch = {
            "observation.state": torch.tensor(
                [
                    [1.0, 2.0],
                    [10.0, 20.0],
                ]
            ),
            "action": torch.tensor(
                [
                    [[2.0, 2.0], [2.0, 2.0]],  # Batch 1: two deltas
                    [[20.0, 20.0], [20.0, 20.0]],  # Batch 2: two deltas
                ]
            ),
        }

        absolute_transform = AbsoluteActions()

        result = absolute_transform(batch)

        # Batch 1: [1, 2] + cumsum([[2, 2]]) = [3, 4], then [1, 2]
        # + cumsum([[2, 2], [2, 2]]) = [5, 6]
        # Batch 2: [10, 20] + cumsum([[20, 20]]) = [30, 40], then [10, 20]
        # + cumsum([[20, 20], [20, 20]]) = [50, 60]
        expected = torch.tensor(
            [
                [[3.0, 4.0], [5.0, 6.0]],
                [[30.0, 40.0], [50.0, 60.0]],
            ]
        )
        assert torch.equal(result["action"], expected)

    def test_delta_and_absolute_roundtrip(self):
        """Test that applying DeltaActions then AbsoluteActions behaves as expected."""
        # Original batch
        original_batch = {
            "observation.state": torch.tensor([[1.0, 2.0, 3.0]]),
            "action": torch.tensor(
                [
                    [
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0],
                        [10.0, 11.0, 12.0],
                    ]
                ]
            ),
        }

        # Apply DeltaActions
        delta_transform = DeltaActions()
        delta_batch = delta_transform(original_batch.copy())

        # Apply AbsoluteActions (use copy to avoid modifying delta_batch)
        absolute_transform = AbsoluteActions()
        recovered_batch = absolute_transform(delta_batch.copy())  # <- Added .copy() here

        # The roundtrip should recover the original actions since DeltaActions computes:
        # Position 0: action[0] - state
        # Position 1+: action[t] - action[t-1]
        # And AbsoluteActions reverses this via cumsum
        expected_recovery = torch.tensor(
            [
                [
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0],
                ]
            ]
        )
        assert torch.equal(recovered_batch["action"], expected_recovery)

    def test_absolute_actions_preserves_shape(self):
        """Test that AbsoluteActions returns tensor with same shape as input."""
        batch = {
            "observation.state": torch.randn(4, 1, 10),  # Batch of 4, 1 timestep, 10 dims
            "action": torch.randn(4, 5, 10),  # Batch of 4, 5 timesteps, 10 dims (deltas)
        }

        absolute_transform = AbsoluteActions()

        result = absolute_transform(batch)

        # Shape should be preserved
        assert result["action"].shape == batch["action"].shape

    def test_absolute_actions_relative_to_state_true(self):
        """Test AbsoluteActions with relative_to_state=True."""
        batch = {
            "observation.state": torch.tensor([[10.0, 20.0]]),
            "action": torch.tensor(
                [
                    [
                        [5.0, 5.0],  # First delta
                        [3.0, 3.0],  # Second delta
                        [2.0, 2.0],  # Third delta
                    ]
                ]
            ),
        }

        absolute_transform = AbsoluteActions(relative_to_state=True)
        result = absolute_transform(batch)

        # With relative_to_state=True, each action is simply state + action (no cumsum)
        # First: [10, 20] + [5, 5] = [15, 25]
        # Second: [10, 20] + [3, 3] = [13, 23]
        # Third: [10, 20] + [2, 2] = [12, 22]
        expected = torch.tensor(
            [
                [
                    [15.0, 25.0],
                    [13.0, 23.0],
                    [12.0, 22.0],
                ]
            ]
        )
        assert torch.equal(result["action"], expected)

    def test_absolute_actions_relative_to_state_false(self):
        """Test AbsoluteActions with relative_to_state=False (default behavior)."""
        batch = {
            "observation.state": torch.tensor([[10.0, 20.0]]),
            "action": torch.tensor(
                [
                    [
                        [5.0, 5.0],  # First delta
                        [3.0, 3.0],  # Second delta
                        [2.0, 2.0],  # Third delta
                    ]
                ]
            ),
        }

        absolute_transform = AbsoluteActions(relative_to_state=False)
        result = absolute_transform(batch)

        # With relative_to_state=False, use cumulative sum
        # First: [10, 20] + cumsum([5, 5]) = [10, 20] + [5, 5] = [15, 25]
        # Second: [10, 20] + cumsum([[5, 5], [3, 3]]) = [10, 20] + [8, 8] = [18, 28]
        # Third: [10, 20] + cumsum([[5, 5], [3, 3], [2, 2]]) = [10, 20] + [10, 10] = [20, 30]
        expected = torch.tensor(
            [
                [
                    [15.0, 25.0],
                    [18.0, 28.0],
                    [20.0, 30.0],
                ]
            ]
        )
        assert torch.equal(result["action"], expected)

    def test_absolute_actions_relative_to_state_comparison(self):
        """Test that relative_to_state flag produces different results."""
        batch = {
            "observation.state": torch.tensor([[1.0, 1.0]]),
            "action": torch.tensor(
                [
                    [
                        [2.0, 2.0],
                        [3.0, 3.0],
                    ]
                ]
            ),
        }

        # Test with relative_to_state=True
        absolute_transform_true = AbsoluteActions(relative_to_state=True)
        result_true = absolute_transform_true(batch.copy())

        # Test with relative_to_state=False
        absolute_transform_false = AbsoluteActions(relative_to_state=False)
        result_false = absolute_transform_false(batch.copy())

        # Results should be different
        assert not torch.equal(result_true["action"], result_false["action"])

        # With relative_to_state=True: state + action
        expected_true = torch.tensor(
            [
                [
                    [3.0, 3.0],  # [1, 1] + [2, 2]
                    [4.0, 4.0],  # [1, 1] + [3, 3]
                ]
            ]
        )
        assert torch.equal(result_true["action"], expected_true)

        # With relative_to_state=False: state + cumsum(action)
        expected_false = torch.tensor(
            [
                [
                    [3.0, 3.0],  # [1, 1] + [2, 2]
                    [6.0, 6.0],  # [1, 1] + [2+3, 2+3]
                ]
            ]
        )
        assert torch.equal(result_false["action"], expected_false)

    def test_delta_and_absolute_roundtrip_with_relative_to_state(self):
        """Test roundtrip behavior with relative_to_state=True."""
        # Original batch
        original_batch = {
            "observation.state": torch.tensor([[5.0, 10.0]]),
            "action": torch.tensor(
                [
                    [
                        [15.0, 20.0],
                        [18.0, 23.0],
                        [21.0, 26.0],
                    ]
                ]
            ),
        }

        # Apply DeltaActions with relative_to_state=True
        delta_transform = DeltaActions(relative_to_state=True)
        delta_batch = delta_transform(original_batch.copy())

        # Apply AbsoluteActions with relative_to_state=True (use copy to avoid modifying delta_batch)
        absolute_transform = AbsoluteActions(relative_to_state=True)
        recovered_batch = absolute_transform(delta_batch.copy())  # <- Added .copy() here

        # With relative_to_state=True, all actions become relative to state via broadcasting
        # Expected deltas: [[10, 10], [13, 13], [16, 16]]
        # Expected recovery: [[15, 20], [18, 23], [21, 26]] (should match original perfectly)
        expected_deltas = torch.tensor(
            [
                [
                    [10.0, 10.0],  # [15, 20] - [5, 10]
                    [13.0, 13.0],  # [18, 23] - [5, 10]
                    [16.0, 16.0],  # [21, 26] - [5, 10]
                ]
            ]
        )
        assert torch.equal(delta_batch["action"], expected_deltas)

        # Should perfectly recover original actions with relative_to_state=True for both transforms
        assert torch.equal(recovered_batch["action"], original_batch["action"])


class TestDeterministicTransforms:
    """Guards for the eval-time `deterministic()` contract.

    Random augmentations must be neutralized when a policy is served, otherwise
    inference silently jitters/crops its own inputs.
    """

    def test_base_transform_defaults_to_self(self):
        """Transforms with no randomness are passed through unchanged."""
        t = DeltaActions(relative_to_state=True)
        assert t.deterministic() is t

    def test_color_jitter_is_dropped(self):
        assert ColorJitter(brightness=0.3, contrast=0.3).deterministic() is None

    @pytest.mark.parametrize("cls", [RandomFlipLeftRight, RandomFlipUpDown, RandomRot90])
    def test_random_flips_and_rotations_are_dropped(self, cls):
        assert cls().deterministic() is None

    def test_task_description_selector_prefers_primary_task(self):
        t = TaskDescriptionSelector(subtask_probability=0.5)
        det = t.deterministic()
        assert det is not None
        assert det.subtask_probability == 0.0

    def test_random_resized_crop_becomes_deterministic_and_keeps_output_size(self):
        rrc = RandomResizedCrop(height=224, width=224, scale=(0.9, 1.0), ratio=(0.98, 1.02))
        det = rrc.deterministic()
        img = torch.rand(3, 448, 448)

        first, second = det(img), det(img)
        assert first.shape == (3, 224, 224)
        assert torch.equal(first, second), "eval crop must not vary between calls"

    def test_random_resized_crop_preserves_training_field_of_view(self):
        """The crop must honour `scale`; a full-frame resize would not."""
        rrc = RandomResizedCrop(height=224, width=224, scale=(0.5, 0.5), ratio=(1.0, 1.0))
        crop_h, crop_w = rrc.deterministic()._crop_size(448, 448)
        area_fraction = (crop_h * crop_w) / (448 * 448)
        assert area_fraction == pytest.approx(0.5, abs=1e-3)

    def test_random_resized_crop_matches_torchvision_fallback(self):
        """Non-square inputs use torchvision's central-crop fallback geometry."""
        rrc = RandomResizedCrop(height=256, width=256, scale=(0.9, 1.0), ratio=(0.98, 1.02))
        # 400x300 cannot satisfy ratio>=0.98, so torchvision clamps: w=300, h=round(300/0.98)
        assert rrc.deterministic()._crop_size(400, 300) == (306, 300)

    def test_eval_transforms_are_stable_end_to_end(self):
        """Building transforms with training=False must yield repeatable output."""
        config = LeRobotDatasetConfig(
            root_dir="/tmp/does-not-exist",
            transform_mapping={
                "image.0": [
                    RandomResizedCrop(height=224, width=224, scale=(0.9, 1.0)),
                    ColorJitter(brightness=0.5, contrast=0.5),
                ]
            },
        )
        transforms = config.get_transforms(remap=False, training=False)

        batch = {"image.0": torch.rand(3, 448, 448)}
        first = transforms(dict(batch))["image.0"]
        second = transforms(dict(batch))["image.0"]

        assert first.shape == (3, 224, 224)
        assert torch.equal(first, second), "serving must not randomly augment inputs"
