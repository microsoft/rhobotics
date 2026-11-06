"""Keep main-process diagnostics separate from distributed loader collectives."""

from collections import deque
from copy import deepcopy

from accelerate.utils import broadcast_object_list

from rho.datasets import make_dataloader


def make_local_monitoring_loader(config, policy_config, device):
    """Build an independent, unprepared loader for main-process diagnostics.

    Rebuilding from a private config keeps sampler positions and shuffle RNGs
    independent of training. Zero workers avoid an extra persistent worker pool;
    the deterministic loader supplies its own generator even in ordinary runs.
    Do not pass this loader to Accelerator.prepare().
    """
    config = deepcopy(config)
    config.set_attribute("num_workers", 0)
    loader, _ = make_dataloader(config, deepcopy(policy_config), device=device, deterministic=True)
    return loader


class LookaheadBatchIterator:
    """Replay collectively fetched future batches, retaining their loader state.

    Every worker must call peek() at the same training step. Only rank zero's
    sample count controls how many batches to fetch; other workers retain the
    same number of their own batches for subsequent training.
    """

    def __init__(self, iterator, dataloader):
        self.iterator = iterator
        self.dataloader = dataloader
        self.pending = deque()
        self._live_state = self._state()

    def _state(self):
        return {
            name: getattr(self.dataloader, name)
            for name in ("end_of_dataloader", "remainder")
            if hasattr(self.dataloader, name)
        }

    def _restore(self, state):
        for name, value in state.items():
            setattr(self.dataloader, name, value)

    def _fetch(self):
        # The underlying iterator may be ahead of the batch being trained.
        self._restore(self._live_state)
        batch = next(self.iterator)
        self._live_state = self._state()
        return batch, self._live_state

    def __iter__(self):
        return self

    def __next__(self):
        batch, state = self.pending.popleft() if self.pending else self._fetch()
        # Accelerate's accumulation boundary must describe the batch actually
        # being trained, not the last batch fetched during monitoring.
        self._restore(state)
        return batch

    def peek(self, num_samples, accelerator):
        """Collect enough rank-zero samples without consuming training batches."""
        if num_samples < 0:
            raise ValueError("num_samples must be nonnegative")
        current_state = self._state()
        batches, sample_count = [], 0
        try:
            while True:
                need_more = [sample_count < num_samples if accelerator.is_main_process else None]
                broadcast_object_list(need_more, from_process=0)
                if not need_more[0]:
                    break
                index = len(batches)
                if index == len(self.pending):
                    self.pending.append(self._fetch())
                batch, _ = self.pending[index]
                batches.append(batch)
                sample_count += batch["action"].shape[0]
        finally:
            self._restore(current_state)
        return batches
