"""Distributed loss aggregation using real CPU Accelerate workers."""

import json
import os
import socket
import subprocess
import sys
from collections import Counter
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import InitProcessGroupKwargs
from torch import nn
from torch.utils.data import DataLoader, Dataset

from rho.training.distributed_validation import _trim_batch, validate_policy_distributed

VALUES = [1.0, 1.0, 3.0, 3.0, 7.0, 2.0, 4.0, 9.0, 5.0, 6.0, 8.0]
CASES = {
    "different_worker_means": {"length": 4},
    "one_real_tail_sample": {"length": 5},
    "two_real_tail_samples": {"length": 6},
    "partially_padded_worker": {"length": 7},
    "multiple_batches": {"length": 11},
    "larger_batch": {"length": 11, "global_batch": 8},
    "split_batches": {"length": 7, "split": True},
    "split_tiny_dataset": {"length": 1, "split": True},
    "tiny_dataset": {"length": 1},
    "empty_dataset": {"length": 0},
    "uneven_shards": {"length": 5, "even": False},
    "uneven_partial_worker": {"length": 7, "even": False},
    "empty_worker": {"length": 1, "even": False},
    "uneven_split": {"length": 5, "even": False, "split": True},
    "drop_last": {"length": 7, "drop": True},
    "drop_tiny": {"length": 1, "drop": True},
    "zero_limit": {"length": 7, "limit": 0},
    "early_limit": {"length": 11, "limit": 1},
    "limit_at_padded_end": {"length": 7, "limit": 2},
    "limit_past_end": {"length": 5, "limit": 9},
    "subset_sampler": {"length": 11, "indices": [4, 6, 8, 9, 10]},
    "repeated_sampler_indices": {"length": 11, "indices": [4, 4, 8, 9, 10]},
    "reversed_order": {"length": 7, "indices": [6, 5, 4, 3, 2, 1, 0]},
    "shuffled": {"length": 11, "shuffle": True},
    "no_auxiliary_metrics": {"length": 7, "aux": False},
    "dispatched_tail": {"length": 7, "dispatch": True},
    "dispatched_short_tail": {"length": 5, "dispatch": True},
    "dispatched_split": {"length": 7, "dispatch": True, "split": True},
}


class _Rows(Dataset):
    def __init__(self, length, text=True):
        self.length = length
        self.text = text

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        row = {"action": torch.full((2, 2), VALUES[index]), "index": index}
        if self.text:
            row["task"] = f"sample-{index}"
        return row


class _LossPolicy(nn.Module):
    def __init__(self, rank, auxiliary=True):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))
        self.rank = rank
        self.auxiliary = auxiliary
        self.seen = []

    def forward(self, batch):
        return self.weight.square(), {}

    def compute_loss(self, batch):
        assert not self.training
        assert not torch.is_grad_enabled()
        ids = batch["index"].tolist()
        if "task" in batch:
            assert batch["task"] == [f"sample-{index}" for index in ids]
        self.seen.extend(ids)
        loss = batch["action"].square().mean() + self.weight * 0
        if not self.auxiliary:
            return loss, None
        return loss, {
            "l2_loss": loss.item(),
            "twice_loss": 2 * loss,
            f"worker_{self.rank}_loss": loss.item(),
            "diagnostic": f"worker-{self.rank}",
        }


def _run_training_loop(accelerator, rank, output_dir):
    from rho.policies import PolicyConfig
    from rho.training import train_accelerate
    from rho.training.train import TrainConfig
    from rho.training.train_utils import TrainLogger

    accelerator.dataloader_config.split_batches = False
    accelerator.dataloader_config.even_batches = True
    accelerator.dataloader_config.dispatch_batches = False
    policy = _LossPolicy(rank)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    transforms = MagicMock(side_effect=lambda batch: batch)
    logger = MagicMock()
    batch_size = 4 // accelerator.num_processes
    cfg = TrainConfig(
        dataset=SimpleNamespace(chunk_size=None, batch_size=batch_size, num_workers=0),
        policy=PolicyConfig(feature_dict={}),
        output_dir=str(Path(output_dir) / "training"),
        run_name="validation",
        steps=1,
        device="cpu",
        mixed_precision="no",
        validation_interval=1,
        action_monitoring=False,
        validation_probe=False,
    )
    training = DataLoader(_Rows(8), batch_size=batch_size)
    validation = DataLoader(_Rows(7), batch_size=batch_size)
    recorder = TrainLogger(dataset_length=8)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(train_accelerate, "Accelerator", lambda **kwargs: accelerator)
        patch.setattr(train_accelerate, "resolve_training_checkpoint", lambda _: None)
        patch.setattr(train_accelerate, "serialize_train_config", lambda _: {})
        patch.setattr(
            train_accelerate,
            "make_everything",
            lambda *args: (training, validation, policy, optimizer, None, transforms, 0, None, recorder),
        )
        patch.setattr(train_accelerate, "make_policy_interface", lambda **kwargs: None)
        patch.setattr(train_accelerate, "WandBLogger", lambda _: logger)
        patch.setattr(train_accelerate, "tqdm", lambda **kwargs: MagicMock())
        with pytest.raises(SystemExit) as exited:
            train_accelerate.train.__wrapped__(cfg)
        assert exited.value.code == 0
    return [call.args[0] for call in logger.log.call_args_list if call.kwargs.get("prefix") == "validation"]


def _run_worker(rank, world_size, port, output_dir):
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_WORLD_SIZE=str(world_size),
        ACCELERATE_USE_CPU="true",
    )
    if sys.platform == "darwin":
        os.environ["GLOO_SOCKET_IFNAME"] = "lo0"
    torch.set_num_threads(1)
    accelerator = Accelerator(
        cpu=True,
        mixed_precision="no",
        dataloader_config=DataLoaderConfiguration(),
        kwargs_handlers=[InitProcessGroupKwargs(backend="gloo", timeout=timedelta(seconds=30))],
    )
    results = {}
    try:
        for name, case in CASES.items():
            config = accelerator.dataloader_config
            config.split_batches = case.get("split", False)
            config.even_batches = case.get("even", True)
            config.dispatch_batches = case.get("dispatch", False)
            global_batch = case.get("global_batch", 4)
            local_batch = global_batch if config.split_batches else global_batch // world_size
            loader = DataLoader(
                _Rows(case["length"], text=not config.dispatch_batches),
                batch_size=local_batch,
                sampler=case.get("indices"),
                shuffle=case.get("shuffle", False),
                generator=torch.Generator().manual_seed(123),
                drop_last=case.get("drop", False),
            )
            raw_policy = _LossPolicy(rank, case.get("aux", True))
            policy, loader = accelerator.prepare(raw_policy, loader)
            # Validation interrupted by a limit must restore the outer loader.
            training_loader = accelerator.prepare(DataLoader(torch.ones(32, 1), batch_size=local_batch))
            training_iterator = iter(training_loader)
            next(training_iterator)
            metrics = validate_policy_distributed(policy, loader, accelerator, max_batches=case.get("limit"))
            restored_training = accelerator.gradient_state.active_dataloader is training_loader
            training_iterator.close()
            training_loader.end()
            results[name] = {
                "metrics": metrics,
                "seen": raw_policy.seen,
                "restored_training": restored_training,
            }
        results["training_log"] = _run_training_loop(accelerator, rank, output_dir)
        Path(output_dir, f"worker-{rank}.json").write_text(json.dumps(results))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.fixture(scope="module", params=[1, 2], ids=["single-process", "two-process"])
def distributed_results(request, tmp_path_factory):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("CPU distributed tests require Gloo")
    output = tmp_path_factory.mktemp(f"distributed-validation-{request.param}")
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(key, None)
    command = [sys.executable, str(Path(__file__).resolve()), str(request.param), str(output)]
    run = subprocess.run(command, capture_output=True, text=True, env=env, timeout=150)
    assert run.returncode == 0, run.stdout + run.stderr
    return [json.loads((output / f"worker-{rank}.json").read_text()) for rank in range(request.param)]


def _expected_indices(case):
    indices = list(case.get("indices", range(case["length"])))
    size = case.get("global_batch", 4)
    if case.get("drop", False):
        indices = indices[: len(indices) // size * size]
    if case.get("limit") is not None:
        indices = indices[: case["limit"] * size]
    return indices


@pytest.mark.parametrize("case_name", CASES)
def test_validation_matches_global_sample_mean(distributed_results, case_name):
    case = CASES[case_name]
    indices = _expected_indices(case)
    expected = sum(VALUES[index] ** 2 for index in indices) / len(indices) if indices else 0.0
    rows = [result[case_name] for result in distributed_results]
    assert Counter(index for row in rows for index in row["seen"]) == Counter(indices)
    for row in rows:
        assert row["metrics"] == rows[0]["metrics"]
        assert row["metrics"]["validation_loss"] == pytest.approx(expected)
        assert row["restored_training"]
        if case.get("aux", True) and indices:
            assert row["metrics"]["l2_loss"] == pytest.approx(expected)
            assert row["metrics"]["twice_loss"] == pytest.approx(2 * expected)
            assert row["metrics"]["diagnostic"] == "worker-0"
            for rank, source in enumerate(rows):
                if source["seen"]:
                    local_expected = sum(VALUES[i] ** 2 for i in source["seen"]) / len(source["seen"])
                    assert row["metrics"][f"worker_{rank}_loss"] == pytest.approx(local_expected)
        else:
            assert row["metrics"] == {"validation_loss": pytest.approx(expected)}


def test_training_entry_point_logs_the_global_validation_loss(distributed_results):
    main_log = distributed_results[0]["training_log"]
    assert len(main_log) == 1
    expected = sum(value**2 for value in VALUES[:7]) / 7
    assert main_log[0]["validation_loss"] == pytest.approx(expected)
    assert main_log[0]["l2_loss"] == pytest.approx(expected)
    for other_worker in distributed_results[1:]:
        assert other_worker["training_log"] == []


def test_trimming_preserves_non_sample_metadata_and_original_batch():
    batch = {
        "action": torch.arange(6).reshape(3, 2),
        "task": ["a", "b", "c"],
        "tuple": ("a", "b", "c"),
        "scalar": torch.tensor(4),
        "constant": "fixed",
    }
    trimmed = _trim_batch(batch, 1, 3)
    assert trimmed["action"].tolist() == [[0, 1]]
    assert trimmed["task"] == ["a"] and trimmed["tuple"] == ("a",)
    assert trimmed["scalar"] is batch["scalar"] and trimmed["constant"] == "fixed"
    assert batch["action"].shape == (3, 2) and len(batch["task"]) == 3


if __name__ == "__main__":
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        free_port = listener.getsockname()[1]
    mp.spawn(_run_worker, args=(int(sys.argv[1]), free_port, sys.argv[2]), nprocs=int(sys.argv[1]), join=True)
