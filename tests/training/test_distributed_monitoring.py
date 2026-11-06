"""Monitoring must not strand data-parallel workers or consume training batches."""

import json
import os
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import InitProcessGroupKwargs
from torch import nn
from torch.utils.data import Dataset

from rho.datasets import make_dataloader
from rho.policies import PolicyConfig
from rho.training import train_accelerate
from rho.training.train import TrainConfig
from rho.training.train_utils import TrainLogger


class _Rows(Dataset):
    def __init__(self):
        self.meta = SimpleNamespace(episodes=[{"dataset_from_index": 0, "dataset_to_index": 8}])

    def __len__(self):
        return 8

    def __getitem__(self, index):
        return {
            "action": torch.full((2, 2), float(index) / 10),
            "observation.state": torch.full((1, 2), float(index)),
            "task": torch.tensor([1, 0], dtype=torch.uint8),
            "index": index,
        }


@dataclass
class _DataConfig:
    chunk_size: int | None = None
    batch_size: int = 2
    num_workers: int = 0
    prefetch_factor: int = 2
    seed: int = 123

    def make_dataset(self, policy_cfg=None):
        return _Rows()

    def make_sampler(self, dataset, policy_cfg=None):
        return None

    def get_length(self):
        return 8

    def get_contributions(self):
        return {"name": "local", "length": 8, "weighted_length": 8, "child_contributions": []}

    def get_action_denormalization(self):
        return None

    def set_attribute(self, key, value):
        setattr(self, key, value)


class _Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(n_action_steps=2)
        self.n_action_steps = 2
        self.seen = []
        self.inferences = 0

    def forward(self, batch):
        self.seen.append(batch["index"].tolist())
        return (self.weight * batch["action"]).mean(), {}

    def compute_loss(self, batch):
        return (self.weight * batch["action"]).mean(), {}

    def reset(self):
        pass

    def select_action(self, batch):
        self.inferences += 1
        return torch.zeros(batch["observation.state"].shape[0], 2)

    def sample_actions(self, batch):
        self.inferences += 1
        return {"actions": torch.zeros(batch["observation.state"].shape[0], 2, 2)}


def _train_once(accelerator, output, source, enabled, dispatch=False, accumulation=1):
    torch.manual_seed(42)
    accelerator.dataloader_config.dispatch_batches = dispatch
    accelerator.gradient_accumulation_steps = accumulation
    accelerator.step = 0
    has_validation = source != "validation_fallback"
    probe_enabled = enabled and source in {"probe", "combined"}
    action_enabled = enabled and source != "probe"
    selected_source = "lookahead" if source in {"probe", "combined"} else source
    if selected_source == "validation_fallback":
        selected_source = "validation"
    cfg = TrainConfig(
        dataset=_DataConfig(),
        validation_dataset=_DataConfig(seed=321) if has_validation else None,
        policy=PolicyConfig(feature_dict={}),
        output_dir=str(output),
        run_name="monitor-test",
        device="cpu",
        batch_size=2,
        num_workers=0,
        steps=4,
        mixed_precision="no",
        gradient_accumulation_steps=accumulation,
        validation_interval=1000,
        save_checkpoint_every=1000,
        logging_interval=1,
        action_monitoring=action_enabled,
        action_monitor_source=selected_source,
        action_monitor_interval=1,
        action_monitor_samples=5,
        validation_probe=probe_enabled,
        validation_probe_interval=1,
        validation_probe_batches=2,
    )
    training, sampler = make_dataloader(cfg.dataset, device="cpu")
    validation = make_dataloader(cfg.validation_dataset, device="cpu")[0] if has_validation else None
    policy = _Policy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    logger, plots = MagicMock(), []
    transforms = MagicMock(side_effect=lambda batch: batch)
    recorder = TrainLogger(dataset_length=8)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(train_accelerate, "Accelerator", lambda **kwargs: accelerator)
        patch.setattr(train_accelerate, "resolve_training_checkpoint", lambda _: None)
        patch.setattr(train_accelerate, "serialize_train_config", lambda _: {})
        patch.setattr(
            train_accelerate,
            "make_everything",
            lambda *args: (training, validation, policy, optimizer, None, transforms, 0, sampler, recorder),
        )
        patch.setattr(train_accelerate, "make_policy_interface", lambda **kwargs: None)
        patch.setattr(train_accelerate, "WandBLogger", lambda _: logger)
        patch.setattr(train_accelerate, "tqdm", lambda **kwargs: MagicMock())
        patch.setattr("rho.training.validation_probe.plot_action_means", lambda *a, **kw: np.zeros((4, 4, 3)))
        patch.setattr("rho.training.validation_probe.wandb.Image", lambda *a, **kw: plots.append(1) or "plot")
        # Keep the worker group alive for the paired no-monitoring run.
        patch.setattr(train_accelerate.dist, "destroy_process_group", lambda: None)
        with pytest.raises(SystemExit) as exited:
            train_accelerate.train.__wrapped__(cfg)
        assert exited.value.code == 0
    result = {
        "seen": policy.seen,
        "weight": policy.weight.item(),
        "inferences": policy.inferences,
        "plots": len(plots),
        "logs": [call.kwargs.get("prefix") for call in logger.log.call_args_list],
        "transform_steps": transforms.step.call_count,
    }
    accelerator.free_memory()
    return result


def _worker(rank, world_size, port, output, source, dispatch, accumulation):
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
        kwargs_handlers=[InitProcessGroupKwargs(backend="gloo", timeout=timedelta(seconds=12))],
    )
    try:
        results = {}
        cases = CASES if source == "all" else [(source, dispatch, accumulation)]
        for source, dispatch, accumulation in cases:
            key = f"{source}-{dispatch}-{accumulation}"
            control = _train_once(
                accelerator, Path(output) / key / "control", source, False, dispatch, accumulation
            )
            monitored = _train_once(
                accelerator, Path(output) / key / "monitored", source, True, dispatch, accumulation
            )
            results[key] = {"control": control, "monitored": monitored}
        Path(output, f"worker-{rank}.json").write_text(json.dumps(results))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


CASES = [
    ("lookahead", False, 1),
    ("training_fresh", False, 1),
    ("validation", False, 1),
    ("validation_fallback", False, 1),
    ("probe", False, 1),
    ("combined", False, 1),
    ("lookahead", True, 1),
    ("training_fresh", True, 1),
    ("validation", True, 1),
    ("probe", True, 1),
    ("lookahead", False, 3),
    ("lookahead", True, 3),
]


@pytest.fixture(scope="module", params=[1, 2], ids=["single-process", "two-process"])
def training_results(request, tmp_path_factory):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("CPU distributed monitoring tests require Gloo")
    output = tmp_path_factory.mktemp(f"monitoring-{request.param}")
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(key, None)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        str(request.param),
        str(output),
        "all",
        "0",
        "1",
    ]
    run = subprocess.run(command, capture_output=True, text=True, env=env, timeout=180)
    assert run.returncode == 0, run.stdout + run.stderr
    return [json.loads((output / f"worker-{rank}.json").read_text()) for rank in range(request.param)]


@pytest.mark.parametrize("source,dispatch,accumulation", CASES)
def test_training_monitoring_completes_without_changing_sample_order(
    training_results, source, dispatch, accumulation
):
    for rank, results in enumerate(training_results):
        result = results[f"{source}-{dispatch}-{accumulation}"]
        control, monitored = result["control"], result["monitored"]
        assert monitored["seen"] == control["seen"]
        assert monitored["weight"] == pytest.approx(control["weight"])
        assert monitored["transform_steps"] == control["transform_steps"] == 4
        if rank == 0:
            assert monitored["inferences"] > 0
            if source in {"probe", "combined"}:
                assert monitored["plots"] == 8
                assert monitored["logs"].count("validation") == 4
        else:
            assert monitored["inferences"] == monitored["plots"] == 0
            assert monitored["logs"] == []


if __name__ == "__main__":
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(
        _worker,
        args=(int(sys.argv[1]), port, sys.argv[2], sys.argv[3], bool(int(sys.argv[4])), int(sys.argv[5])),
        nprocs=int(sys.argv[1]),
        join=True,
    )
