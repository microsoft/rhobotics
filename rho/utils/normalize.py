# Adapted from openpi/shared/normalize.py

import pathlib
from dataclasses import dataclass

import numpy as np


@dataclass
class NormStats:
    min: np.ndarray
    max: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    q01: np.ndarray | None = None  # 1st percentile
    q99: np.ndarray | None = None  # 99th percentile


class RunningStats:
    """Compute running statistics of a batch of vectors.

    If bounds are provided (known_min and known_max), uses histogram-based quantile
    estimation with fixed bin edges (memory efficient).
    If no_quantile is True, skips quantile computation entirely (useful for first pass).
    Otherwise, stores data to compute exact percentiles.
    """

    def __init__(
        self,
        known_min: np.ndarray | None = None,
        known_max: np.ndarray | None = None,
        num_quantile_bins: int = 5000,
        no_quantile: bool = False,
    ):
        """
        Args:
            known_min: If provided with known_max, min/max are fixed and histogram bins
                       are used for quantile estimation (no data accumulation).
                       Shape: (dims,)
            known_max: If provided with known_min, min/max are fixed and histogram bins
                       are used for quantile estimation (no data accumulation).
                       Shape: (dims,)
            num_quantile_bins: Number of bins for histogram-based quantile estimation.
            no_quantile: If True, skip quantile computation entirely. Useful for first pass
                         of two-pass mode where only min/max/mean/std are needed.
        """
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = known_min
        self._max = known_max
        self._num_quantile_bins = num_quantile_bins
        self._fixed_bounds = known_min is not None and known_max is not None
        self._no_quantile = no_quantile

        if self._no_quantile:
            # Skip quantile computation - no data accumulation or histograms needed
            self._accumulated_data = None
            self._histograms = None
            self._bin_edges = None
        elif self._fixed_bounds:
            # Use histogram-based approach with fixed bins - no data accumulation needed
            self._accumulated_data = None
            vector_length = known_min.shape[0]
            self._histograms = [np.zeros(num_quantile_bins) for _ in range(vector_length)]
            self._bin_edges = [
                np.linspace(known_min[i] - 1e-10, known_max[i] + 1e-10, num_quantile_bins + 1)
                for i in range(vector_length)
            ]
        else:
            # Store all data for exact percentile computation
            self._accumulated_data = []
            self._histograms = None
            self._bin_edges = None

    def update(self, batch: np.ndarray) -> None:
        """
        Update the running statistics with a batch of vectors.

        Args:
            batch (np.ndarray): A 2D array where each row is a new vector.
        """
        if batch.ndim == 1:
            batch = batch.reshape(-1, 1)
        num_elements, vector_length = batch.shape

        if self._count == 0:
            self._mean = np.mean(batch, axis=0)
            self._mean_of_squares = np.mean(batch**2, axis=0)

            if not self._fixed_bounds:
                self._min = np.min(batch, axis=0)
                self._max = np.max(batch, axis=0)
        else:
            if vector_length != self._mean.size:
                raise ValueError("The length of new vectors does not match the initialized vector length.")

            if not self._fixed_bounds:
                # Only update min/max if bounds are not fixed
                self._min = np.minimum(self._min, np.min(batch, axis=0))
                self._max = np.maximum(self._max, np.max(batch, axis=0))

        if self._no_quantile:
            # Skip quantile computation - nothing to do here
            pass
        elif self._fixed_bounds:
            # Update histograms
            self._update_histograms(batch)
        else:
            # Store data for percentile computation
            self._accumulated_data.append(batch.copy())

        self._count += num_elements

        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)

        # Update running mean and mean of squares.
        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (
            num_elements / self._count
        )

    def get_statistics(self) -> NormStats:
        """
        Compute and return the statistics of the vectors processed so far.

        Returns:
            NormStats: A dataclass containing the computed statistics.
        """
        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")

        variance = self._mean_of_squares - self._mean**2
        stddev = np.sqrt(np.maximum(0, variance))

        if self._no_quantile:
            # Skip quantile computation entirely
            return NormStats(mean=self._mean, std=stddev, q01=None, q99=None, min=self._min, max=self._max)

        if self._fixed_bounds:
            # Compute quantiles from histograms
            q01, q99 = self._compute_quantiles_from_histograms([0.01, 0.99])
        else:
            # Compute exact quantiles from accumulated data
            all_data = np.concatenate(self._accumulated_data, axis=0)  # (total_samples, dims)
            q01 = np.percentile(all_data, 1, axis=0)
            q99 = np.percentile(all_data, 99, axis=0)

        return NormStats(mean=self._mean, std=stddev, q01=q01, q99=q99, min=self._min, max=self._max)

    def _compute_quantiles_from_histograms(self, quantiles):
        """Compute quantiles based on histograms."""
        results = []
        for q in quantiles:
            target_count = q * self._count
            q_values = []
            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                cumsum = np.cumsum(hist)
                idx = np.searchsorted(cumsum, target_count)
                val = edges[idx]
                # Clamp tiny epsilon values to 0 (artifact of bin edge padding)
                if abs(val) < 1e-9:
                    val = 0.0
                q_values.append(val)
            results.append(np.array(q_values))
        return results

    def _update_histograms(self, batch: np.ndarray) -> None:
        """Update histograms with new vectors."""
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist


class RunningStatsChunked:
    """Compute running statistics for chunked sequences.

    For inputs of shape (batch, time, dims), computes statistics per (time, dims) position.
    If bounds are provided, uses histogram-based quantile estimation (memory efficient).
    If no_quantile is True, skips quantile computation entirely (useful for first pass).
    Otherwise, stores data to compute exact percentiles.
    """

    def __init__(
        self,
        known_min: np.ndarray | None = None,
        known_max: np.ndarray | None = None,
        num_quantile_bins: int = 5000,
        no_quantile: bool = False,
    ):
        """
        Args:
            known_min: If provided with known_max, min/max are fixed and histogram bins
                       are used for quantile estimation (no data accumulation).
                       Shape: (chunk_len, dims)
            known_max: If provided with known_min, min/max are fixed and histogram bins
                       are used for quantile estimation (no data accumulation).
                       Shape: (chunk_len, dims)
            num_quantile_bins: Number of bins for histogram-based quantile estimation.
            no_quantile: If True, skip quantile computation entirely. Useful for first pass
                         of two-pass mode where only min/max/mean/std are needed.
        """
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = known_min
        self._max = known_max
        self._num_quantile_bins = num_quantile_bins
        self._fixed_bounds = known_min is not None and known_max is not None
        self._no_quantile = no_quantile

        if self._no_quantile:
            # Skip quantile computation - no data accumulation or histograms needed
            self._accumulated_data = None
            self._histograms = None
            self._bin_edges = None
        elif self._fixed_bounds:
            # Use histogram-based approach - no data accumulation needed
            self._accumulated_data = None
            chunk_len, dims = known_min.shape
            # Initialize histograms for each (time, dim) position
            self._histograms = np.zeros((chunk_len, dims, num_quantile_bins))
            # Initialize bin edges for each (time, dim) position
            self._bin_edges = np.zeros((chunk_len, dims, num_quantile_bins + 1))
            for t in range(chunk_len):
                for d in range(dims):
                    self._bin_edges[t, d] = np.linspace(
                        known_min[t, d] - 1e-10, known_max[t, d] + 1e-10, num_quantile_bins + 1
                    )
        else:
            # Store all data for exact percentile computation
            self._accumulated_data = []
            self._histograms = None
            self._bin_edges = None

    def update(self, batch: np.ndarray) -> None:
        """Update running statistics with a batch of sequences.

        Args:
            batch: Array of shape (batch_size, chunk_len, dims)
        """
        if batch.ndim != 3:
            raise ValueError(f"Expected 3D array (batch, time, dims), got shape {batch.shape}")

        batch_size, chunk_len, dims = batch.shape

        if self._count == 0:
            # Initialize with first batch
            self._mean = np.mean(batch, axis=0)  # Shape: (chunk_len, dims)
            self._mean_of_squares = np.mean(batch**2, axis=0)

            if not self._fixed_bounds:
                self._min = np.min(batch, axis=0)
                self._max = np.max(batch, axis=0)
        else:
            if not self._fixed_bounds:
                # Only update min/max if bounds are not fixed
                self._min = np.minimum(self._min, np.min(batch, axis=0))
                self._max = np.maximum(self._max, np.max(batch, axis=0))

        if self._no_quantile:
            # Skip quantile computation - nothing to do here
            pass
        elif self._fixed_bounds:
            # Update histograms for each (time, dim) position
            for t in range(chunk_len):
                for d in range(dims):
                    hist, _ = np.histogram(batch[:, t, d], bins=self._bin_edges[t, d])
                    self._histograms[t, d] += hist
        else:
            # Store data for percentile computation
            self._accumulated_data.append(batch)

        self._count += batch_size

        # Update running mean and mean of squares
        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)

        self._mean += (batch_mean - self._mean) * (batch_size / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (batch_size / self._count)

    def get_statistics(self) -> dict[str, np.ndarray]:
        """Compute and return statistics.

        Returns:
            Dict with keys 'mean', 'std', 'min', 'max', 'q02', 'q98'
            Each with arrays of shape (chunk_len, dims)
        """
        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 samples.")

        variance = self._mean_of_squares - self._mean**2
        stddev = np.sqrt(np.maximum(0, variance))

        chunk_len, dims = self._mean.shape

        if self._no_quantile:
            # Skip quantile computation entirely
            return {
                "mean": self._mean,
                "std": stddev,
                "min": self._min,
                "max": self._max,
                "q02": None,
                "q98": None,
            }

        q02 = np.zeros((chunk_len, dims))
        q98 = np.zeros((chunk_len, dims))

        if self._fixed_bounds:
            # Compute quantiles from histograms
            for t in range(chunk_len):
                for d in range(dims):
                    hist = self._histograms[t, d]
                    edges = self._bin_edges[t, d]
                    cumsum = np.cumsum(hist)

                    # 2nd percentile
                    target_count_02 = 0.02 * self._count
                    idx_02 = np.searchsorted(cumsum, target_count_02)
                    q02[t, d] = edges[idx_02]

                    # 98th percentile
                    target_count_98 = 0.98 * self._count
                    idx_98 = np.searchsorted(cumsum, target_count_98)
                    q98[t, d] = edges[idx_98]
        else:
            # Concatenate all accumulated data
            all_data = np.concatenate(self._accumulated_data, axis=0)  # (total_samples, chunk_len, dims)

            # Compute percentiles per (time, dim) position
            for t in range(chunk_len):
                for d in range(dims):
                    values = all_data[:, t, d]
                    q02[t, d] = np.percentile(values, 2)
                    q98[t, d] = np.percentile(values, 98)

        return {
            "mean": self._mean,
            "std": stddev,
            "min": self._min,
            "max": self._max,
            "q02": q02,
            "q98": q98,
        }


def serialize_json(norm_stats: dict[str, NormStats]) -> str:
    """Serialize the running statistics to a JSON string."""
    raise NotImplementedError("Serialization to JSON is not implemented yet.")


def deserialize_json(data: str) -> dict[str, NormStats]:
    """Deserialize the running statistics from a JSON string."""
    raise NotImplementedError("Serialization to JSON is not implemented yet.")


def save(directory: pathlib.Path | str, norm_stats: dict[str, NormStats]) -> None:
    """Save the normalization stats to a directory."""
    path = pathlib.Path(directory) / "norm_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_json(norm_stats))


def load(directory: pathlib.Path | str) -> dict[str, NormStats]:
    """Load the normalization stats from a directory."""
    path = pathlib.Path(directory) / "norm_stats.json"
    if not path.exists():
        raise FileNotFoundError(f"Norm stats file not found at: {path}")
    return deserialize_json(path.read_text())
