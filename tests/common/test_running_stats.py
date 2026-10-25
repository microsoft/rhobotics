"""Numerical regression tests for normalization-statistic accumulators."""

import numpy as np
import pytest
import torch

from rho.common.normalize import Normalize
from rho.common.types import FeatureType, NormalizationMode, PolicyFeature
from rho.utils.normalize import RunningStats, RunningStatsChunked


@pytest.fixture(params=[RunningStats, RunningStatsChunked], ids=["vectors", "chunks"])
def stats_class(request):
    return request.param


def _samples(values, stats_class):
    return values[:, None, :] if stats_class is RunningStatsChunked else values


def _statistics(accumulator):
    result = accumulator.get_statistics()
    return result if isinstance(result, dict) else vars(result)


@pytest.mark.parametrize("mode", ["no_quantiles", "exact", "histogram"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_small_variance_is_preserved(stats_class, mode, batch_size):
    values = _samples(np.array([[0.9999, 2.0002], [1.0001, 1.9998]], dtype=np.float32), stats_class)
    kwargs = {"no_quantile": mode == "no_quantiles"}
    if mode == "histogram":
        kwargs.update(known_min=values.min(axis=0), known_max=values.max(axis=0), num_quantile_bins=32)
    accumulator = stats_class(**kwargs)
    for start in range(0, len(values), batch_size):
        accumulator.update(values[start : start + batch_size])

    result = _statistics(accumulator)
    reference = values.astype(np.float64)
    np.testing.assert_allclose(result["mean"], reference.mean(axis=0), rtol=1e-14, atol=0)
    np.testing.assert_allclose(result["std"], reference.std(axis=0), rtol=1e-10, atol=1e-14)
    assert np.all(result["std"] > 0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
@pytest.mark.parametrize("batch_sizes", [(8,), (1,) * 8, (3, 1, 4), (4, 4)])
def test_population_moments_match_reference_across_batch_partitions(stats_class, dtype, batch_sizes):
    # The float64 case also fails when raw second moments are merely upcast.
    # The integer case requires promotion before squaring, not afterwards.
    baseline = 50_000 if dtype == np.int32 else (1e9 if dtype == np.float64 else 1e4)
    offsets = np.arange(-4, 4) * (1.0 if dtype == np.int32 else 0.125)
    values = np.column_stack([baseline + offsets, -baseline + 2 * offsets, np.full(8, 7)])
    values = _samples(values.astype(dtype), stats_class)
    original = values.copy()
    accumulator = stats_class(no_quantile=True)
    start = 0
    for size in batch_sizes:
        accumulator.update(values[start : start + size])
        start += size

    result = _statistics(accumulator)
    reference = values.astype(np.float64)
    assert result["mean"].shape == values.shape[1:]
    assert result["std"].shape == values.shape[1:]
    np.testing.assert_allclose(result["mean"], reference.mean(axis=0), rtol=1e-14, atol=1e-12)
    np.testing.assert_allclose(result["std"], reference.std(axis=0, ddof=0), rtol=1e-6, atol=1e-12)
    np.testing.assert_array_equal(result["min"], values.min(axis=0))
    np.testing.assert_array_equal(result["max"], values.max(axis=0))
    np.testing.assert_array_equal(values, original)


def test_merge_includes_variance_between_constant_batches(stats_class):
    first = _samples(np.tile([10.0, 20.0], (3, 1)), stats_class)
    second = _samples(np.tile([14.0, 30.0], (5, 1)), stats_class)
    accumulator = stats_class(no_quantile=True)
    accumulator.update(first)
    np.testing.assert_array_equal(_statistics(accumulator)["std"], np.zeros(first.shape[1:]))
    accumulator.update(second)

    reference = np.concatenate([first, second])
    result = _statistics(accumulator)
    np.testing.assert_allclose(result["mean"], reference.mean(axis=0), rtol=1e-14)
    np.testing.assert_allclose(result["std"], reference.std(axis=0), rtol=1e-14)


def test_constant_features_keep_zero_variance(stats_class):
    values = _samples(np.tile(np.array([0.1, 1e8], dtype=np.float32), (11, 1)), stats_class)
    accumulator = stats_class(no_quantile=True)
    for part in np.split(values, [3, 5]):
        accumulator.update(part)
    result = _statistics(accumulator)
    np.testing.assert_array_equal(result["mean"], values[0])
    np.testing.assert_array_equal(result["std"], np.zeros(values.shape[1:]))


def test_each_time_and_feature_has_independent_moments(stats_class):
    baseline = np.array([[1.0, 10.0, 100.0], [-3.0, 1e6, 7.0]])
    scale = np.array([[1e-4, 0.01, 0.1], [0.01, 0.125, 0.0]])
    values = baseline + np.arange(-2, 3)[:, None, None] * scale
    if stats_class is RunningStats:
        values = values.reshape(5, -1)
    accumulator = stats_class(no_quantile=True)
    for part in np.split(values, [2, 3]):
        accumulator.update(part)
    result = _statistics(accumulator)
    np.testing.assert_allclose(result["mean"], values.mean(axis=0), rtol=1e-14)
    np.testing.assert_allclose(result["std"], values.std(axis=0), rtol=1e-6, atol=1e-12)
    assert result["std"].shape == values.shape[1:]


def test_extrema_and_exact_quantiles_are_preserved(stats_class):
    values = _samples(np.arange(42, dtype=np.float32).reshape(21, 2) / 8, stats_class)
    kwargs = {"quantile_pairs": [(0.01, 0.99), (0.02, 0.98)]} if stats_class is RunningStatsChunked else {}
    accumulator = stats_class(**kwargs)
    for part in np.split(values, [4, 15]):
        accumulator.update(part)
    result = _statistics(accumulator)
    np.testing.assert_array_equal(result["min"], values.min(axis=0))
    np.testing.assert_array_equal(result["max"], values.max(axis=0))
    for percentile in [1, 2, 98, 99] if stats_class is RunningStatsChunked else [1, 99]:
        np.testing.assert_allclose(result[f"q{percentile:02d}"], np.percentile(values, percentile, axis=0))


@pytest.mark.parametrize("count", [0, 1])
def test_statistics_still_require_two_samples(stats_class, count):
    accumulator = stats_class(no_quantile=True)
    if count:
        accumulator.update(_samples(np.ones((count, 2)), stats_class))
    with pytest.raises(ValueError, match="less than 2"):
        accumulator.get_statistics()


def test_vector_accumulator_preserves_one_dimensional_input():
    values = np.array([0.9999, 1.0001], dtype=np.float32)
    accumulator = RunningStats(no_quantile=True)
    accumulator.update(values)
    result = accumulator.get_statistics()
    assert result.std.shape == (1,)
    np.testing.assert_allclose(result.std, [values.astype(np.float64).std()], rtol=1e-10)


def test_generated_statistics_keep_normalized_values_near_unit_scale(stats_class):
    values = np.array([[0.9999], [1.0001]], dtype=np.float32)
    accumulator = stats_class(no_quantile=True)
    accumulator.update(_samples(values, stats_class))
    result = _statistics(accumulator)
    key = "observation.state"
    normalize = Normalize(
        {key: PolicyFeature(FeatureType.STATE, (1,))},
        {FeatureType.STATE: NormalizationMode.MEAN_STD},
        stats={key: {name: result[name].reshape(1) for name in ("mean", "std")}},
    )
    normalized = normalize({key: torch.from_numpy(values)})[key]
    torch.testing.assert_close(normalized, torch.tensor([[-1.0], [1.0]]), rtol=2e-4, atol=0)
