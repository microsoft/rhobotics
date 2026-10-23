import json

import numpy as np
import pytest
import torch

from rho.common.constants import ACTION, OBSERVATION_STATE
from rho.common.normalize import Normalize, Unnormalize
from rho.common.types import FeatureType, NormalizationMode, PolicyFeature
from rho.utils.normalize import RunningStatsChunked


def _chunk_quantile_features():
    return {ACTION: PolicyFeature(FeatureType.ACTION, (1,))}


def _chunk_quantile_norm_map():
    return {FeatureType.ACTION: NormalizationMode.ACTIONCHUNK_QUANTILE}


def test_actionchunk_quantile_accepts_q01_q99_stats():
    stats = {
        ACTION: {
            "mean": torch.zeros(1),
            "q01_chunk50": torch.tensor([[0.0], [1.0]]),
            "q99_chunk50": torch.tensor([[10.0], [11.0]]),
        }
    }
    normalizer = Normalize(
        _chunk_quantile_features(),
        _chunk_quantile_norm_map(),
        stats=stats,
        chunk_size=2,
    )

    batch = {ACTION: torch.tensor([[[0.0], [11.0]]])}

    normalized = normalizer(batch)

    assert torch.allclose(normalized[ACTION], torch.tensor([[[-1.0], [1.0]]]))


def test_actionchunk_quantile_slices_chunk50_stats_to_actual_action_length():
    stats = {
        ACTION: {
            "mean": torch.zeros(1),
            "q01_chunk50": torch.arange(50, dtype=torch.float32).reshape(50, 1),
            "q99_chunk50": torch.arange(10, 60, dtype=torch.float32).reshape(50, 1),
        }
    }
    normalizer = Normalize(
        _chunk_quantile_features(),
        _chunk_quantile_norm_map(),
        stats=stats,
        chunk_size=50,
    )
    unnormalizer = Unnormalize(
        _chunk_quantile_features(),
        _chunk_quantile_norm_map(),
        stats=stats,
        chunk_size=50,
    )

    batch = {ACTION: torch.tensor([[[0.0], [11.0]]])}

    normalized = normalizer(batch)
    restored = unnormalizer(normalized)

    assert normalized[ACTION].shape == batch[ACTION].shape
    assert torch.allclose(restored[ACTION], batch[ACTION])


def test_actionchunk_quantile_rejects_q02_q98_only_stats():
    stats = {
        ACTION: {
            "mean": torch.zeros(1),
            "q02_chunk50": torch.tensor([[0.0], [1.0]]),
            "q98_chunk50": torch.tensor([[10.0], [11.0]]),
        }
    }
    with pytest.raises(ValueError, match="q01_chunk50/q99_chunk50"):
        Normalize(
            _chunk_quantile_features(),
            _chunk_quantile_norm_map(),
            stats=stats,
            chunk_size=2,
        )


def test_running_stats_chunked_uses_configured_quantile_keys():
    stats = RunningStatsChunked(quantile_low=0.01, quantile_high=0.99)
    values = torch.arange(200, dtype=torch.float32).reshape(100, 2, 1).numpy()

    stats.update(values)

    result = stats.get_statistics()

    assert "q01" in result
    assert "q99" in result
    assert "q02" not in result
    assert "q98" not in result


def test_running_stats_chunked_can_emit_multiple_quantile_pairs():
    stats = RunningStatsChunked(quantile_pairs=[(0.01, 0.99), (0.02, 0.98)])
    values = torch.arange(200, dtype=torch.float32).reshape(100, 2, 1).numpy()

    stats.update(values)

    result = stats.get_statistics()

    assert result["q01"].shape == (2, 1)
    assert result["q99"].shape == (2, 1)
    assert result["q02"].shape == (2, 1)
    assert result["q98"].shape == (2, 1)


@pytest.fixture
def vector_stats():
    return {
        "mean": [10.0, 20.0],
        "std": [1.0, 2.0],
        "min": [9.0, 18.0],
        "max": [11.0, 22.0],
        "q01": [9.0, 18.0],
        "q99": [11.0, 22.0],
    }


@pytest.mark.parametrize(
    "mode", [NormalizationMode.MEAN_STD, NormalizationMode.MIN_MAX, NormalizationMode.QUANTILE]
)
@pytest.mark.parametrize("stats_format", ["list", "numpy", "tensor"])
@pytest.mark.parametrize("width", [1, 2])
@pytest.mark.parametrize(
    "key,feature_type", [(ACTION, FeatureType.ACTION), (OBSERVATION_STATE, FeatureType.STATE)]
)
def test_vector_stats_preserve_dimensions_and_physical_values(
    vector_stats, mode, stats_format, width, key, feature_type
):
    stats = {name: values[:width] for name, values in vector_stats.items()}
    if stats_format == "numpy":
        stats = {name: np.asarray(values, dtype=np.float32) for name, values in stats.items()}
    elif stats_format == "tensor":
        stats = {name: torch.tensor(values, dtype=torch.float32) for name, values in stats.items()}

    features = {key: PolicyFeature(feature_type, (width,))}
    norm_map = {feature_type: mode}
    normalizer = Normalize(features, norm_map, stats={key: stats})
    unnormalizer = Unnormalize(features, norm_map, stats={key: stats})
    physical = torch.tensor([[[9.0, 18.0], [10.0, 20.0], [11.0, 22.0]]])[..., :width]
    normalized = torch.tensor([[[-1.0], [0.0], [1.0]]]).expand(1, 3, width)

    # Check both directions against known values, not just a round trip through
    # potentially matching normalization and denormalization errors.
    torch.testing.assert_close(normalizer({key: physical})[key], normalized, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(unnormalizer({key: normalized})[key], physical, rtol=1e-5, atol=1e-6)
    for transform in (normalizer, unnormalizer):
        buffer = getattr(transform, "buffer_" + key.replace(".", "_"))
        for stat in buffer.values():
            assert stat.shape == (width,)
            assert stat.dtype == torch.float32
            assert not stat.requires_grad


@pytest.mark.parametrize(
    "mode", [NormalizationMode.MEAN_STD, NormalizationMode.MIN_MAX, NormalizationMode.QUANTILE]
)
def test_json_file_stats_preserve_physical_action_values(tmp_path, vector_stats, mode):
    from rho.datasets.data_config import RobotDataConfig

    path = tmp_path / "stats.json"
    path.write_text(json.dumps({ACTION: vector_stats}))
    config = RobotDataConfig(
        features={ACTION: PolicyFeature(FeatureType.ACTION, (2,))},
        stats=str(path),
        normalization_mapping={FeatureType.ACTION: mode},
    )
    assert isinstance(config.stats[ACTION]["mean"], list)
    normalizer = config.get_input_transform()
    unnormalizer = config.get_action_denormalization()
    physical = torch.tensor([[[9.0, 18.0], [10.0, 20.0], [11.0, 22.0]]])
    normalized = torch.tensor([[[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]])

    torch.testing.assert_close(normalizer({ACTION: physical})[ACTION], normalized, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(unnormalizer({ACTION: normalized})[ACTION], physical, rtol=1e-5, atol=1e-6)
