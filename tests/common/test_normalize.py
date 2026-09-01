import pytest
import torch

from rho.common.constants import ACTION
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
