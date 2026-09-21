import os
import random
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from rho.common.determinism import capture_rng_state, configure_training_determinism, restore_rng_state
from rho.datasets import make_dataloader


def _sample_rngs():
    return random.random(), np.random.random(), torch.rand(4)


def test_configure_training_determinism_seeds_all_rngs(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    debug_mode = torch.get_deterministic_debug_mode()
    cudnn_benchmark = torch.backends.cudnn.benchmark
    cuda_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    float32_matmul_precision = torch.get_float32_matmul_precision()

    try:
        configure_training_determinism(123, deterministic=True)
        first = _sample_rngs()

        configure_training_determinism(123, deterministic=True)
        second = _sample_rngs()

        assert first[0] == second[0]
        assert first[1] == second[1]
        assert torch.equal(first[2], second[2])
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.backends.cudnn.benchmark
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32
        assert torch.get_float32_matmul_precision() == "highest"
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    finally:
        torch.set_deterministic_debug_mode(debug_mode)
        torch.set_float32_matmul_precision(float32_matmul_precision)
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.backends.cuda.matmul.allow_tf32 = cuda_matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


def test_capture_and_restore_rng_state():
    configure_training_determinism(7, deterministic=False)
    state = capture_rng_state()
    expected = _sample_rngs()

    _sample_rngs()
    restore_rng_state(state)
    actual = _sample_rngs()

    assert expected[0] == actual[0]
    assert expected[1] == actual[1]
    assert torch.equal(expected[2], actual[2])


@pytest.fixture
def loader_config():
    config = Mock(seed=55, num_workers=0, batch_size=2, prefetch_factor=2)
    config.make_dataset.return_value = torch.arange(8)
    config.make_sampler.return_value = None
    config.get_contributions.return_value = {}
    return config


@pytest.mark.parametrize("deterministic", [False, True])
def test_dataloader_rng_isolation(loader_config, deterministic):
    state = torch.get_rng_state()
    loader, _ = make_dataloader(loader_config, device="cpu", deterministic=deterministic)
    first = torch.cat(list(loader))
    assert torch.equal(torch.get_rng_state(), state) == deterministic
    if deterministic:
        second, _ = make_dataloader(loader_config, device="cpu", deterministic=True)
        assert torch.equal(first, torch.cat(list(second)))


def test_strict_dataloader_rejects_workers_before_loading_data(loader_config):
    loader_config.num_workers = 2
    with pytest.raises(ValueError, match="requires num_workers=0"):
        make_dataloader(loader_config, deterministic=True)
    loader_config.make_dataset.assert_not_called()
    assert loader_config.num_workers == 2


@pytest.mark.parametrize("in_order", ["0", "1"])
def test_normal_dataloader_keeps_worker_settings(loader_config, monkeypatch, in_order):
    monkeypatch.setenv("RHO_DATALOADER_IN_ORDER", in_order)
    loader_config.num_workers = 2
    loader, _ = make_dataloader(loader_config, device="cpu")
    assert loader.num_workers == loader.prefetch_factor == 2
    assert loader.in_order == (in_order == "1")
    assert loader.generator is loader.worker_init_fn is None
