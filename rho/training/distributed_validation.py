"""Sample-weighted validation over Accelerate data-parallel workers."""

from itertools import islice
from numbers import Real

import torch
from accelerate.data_loader import BatchSamplerShard
from accelerate.utils import gather_object
from torch.utils.data import BatchSampler


def _valid_batch_size(dataloader, accelerator, batch_size: int) -> int:
    """Exclude only the padding Accelerate appends to an incomplete final batch."""
    if accelerator.num_processes == 1 or not getattr(dataloader, "end_of_dataloader", False):
        return batch_size

    batch_sampler = getattr(dataloader, "batch_sampler", None)
    if isinstance(batch_sampler, BatchSamplerShard):
        if not batch_sampler.even_batches or batch_sampler.drop_last:
            return batch_size
        original = batch_sampler.batch_sampler
        if type(original) is BatchSampler:
            # Dataset length is wrong for subset/replacement samplers. This is
            # the number of indices actually supplied to Accelerate's sharder.
            remainder = len(original.sampler) % dataloader.total_batch_size
        else:
            remainder = dataloader.remainder
    else:
        # DataLoaderDispatcher records the actual final global batch size here.
        remainder = getattr(dataloader, "remainder", 0)

    if remainder < 0:
        raise ValueError("Distributed validation needs a known final-batch length to exclude padding")
    if remainder == 0:
        return batch_size
    # Final batches are rank-ordered, with genuine samples before padding.
    return max(0, min(batch_size, remainder - accelerator.process_index * batch_size))


def _trim_batch(batch: dict, size: int, original_size: int) -> dict:
    """Slice sample-major tensors and text lists without mutating the loader batch."""
    return {
        key: value[:size]
        if (isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == original_size)
        or (isinstance(value, (list, tuple)) and len(value) == original_size)
        else value
        for key, value in batch.items()
    }


@torch.no_grad()
def validate_policy_distributed(policy, validation_dataloader, accelerator, *, max_batches=None) -> dict:
    """Return the same sample-weighted losses on every data-parallel worker.

    All workers must call this function with the same batch limit. Losses must
    be batch means, as returned by RhoPolicy.compute_loss. Trim duplicated tail
    samples before computing a mean; repeating a batch mean and then truncating
    it cannot recover the loss of its genuine samples.

    Numeric auxiliary values are weighted averages of batch summaries, with a
    separate denominator per key. Nonlinear diagnostics (e.g. standard deviation)
    are not recomputed population statistics. Nonnumeric diagnostics retain the
    latest value from the lowest-numbered worker that reports each key.
    """
    policy.eval()
    model = accelerator.unwrap_model(policy)
    totals, counts, diagnostics = {}, {}, {}
    loss_total, sample_count = 0.0, 0
    iterator = iter(validation_dataloader)
    try:
        for batch in islice(iterator, max_batches):
            original_size = batch["action"].shape[0]
            size = _valid_batch_size(validation_dataloader, accelerator, original_size)
            if size == 0:
                continue
            if size != original_size:
                batch = _trim_batch(batch, size, original_size)
            batch = {
                key: value.to(accelerator.device, non_blocking=True)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in batch.items()
            }
            loss, metrics = model.compute_loss(batch)
            loss_total += loss.item() * size
            sample_count += size
            for key, value in (metrics or {}).items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    value = value.item()
                if isinstance(value, Real):
                    totals[key] = totals.get(key, 0.0) + float(value) * size
                    counts[key] = counts.get(key, 0) + size
                else:
                    diagnostics[key] = value
    finally:
        if hasattr(iterator, "close"):
            iterator.close()
        # A batch limit can leave Accelerate's validation loader active. Restore
        # the enclosing training loader instead of leaking its gradient state.
        if accelerator.gradient_state.active_dataloader is validation_dataloader:
            validation_dataloader.end()

    # Gather only small CPU summaries, not model outputs. No per-batch collective
    # is needed, so uneven unpadded shards and workers with no samples are safe.
    summaries = gather_object([(loss_total, sample_count, totals, counts, diagnostics)])
    global_loss, global_samples = 0.0, 0
    global_totals, global_counts, result = {}, {}, {}
    for loss_sum, count, metric_sums, metric_counts, last_values in summaries:
        global_loss += loss_sum
        global_samples += count
        for key, value in metric_sums.items():
            global_totals[key] = global_totals.get(key, 0.0) + value
            global_counts[key] = global_counts.get(key, 0) + metric_counts[key]
        for key, value in last_values.items():
            result.setdefault(key, value)
    result.update({key: value / global_counts[key] for key, value in global_totals.items()})
    result["validation_loss"] = global_loss / global_samples if global_samples else 0.0
    return result
