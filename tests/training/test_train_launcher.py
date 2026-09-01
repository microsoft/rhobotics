import sys
from types import SimpleNamespace

from rho import train as train_launcher


def _clear_launch_environment(monkeypatch):
    for name in (
        "ACCELERATE_USE_CPU",
        "ACCELERATE_MIXED_PRECISION",
        "ACCELERATE_NUM_PROCESSES",
        "WORLD_SIZE",
        "RANK",
    ):
        monkeypatch.delenv(name, raising=False)


def test_standard_launch_uses_single_process_trainer(monkeypatch):
    _clear_launch_environment(monkeypatch)
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "rho.training.train",
        SimpleNamespace(train=lambda: calls.append("single")),
    )

    train_launcher.main()

    assert calls == ["single"]


def test_accelerate_launch_uses_distributed_trainer(monkeypatch):
    _clear_launch_environment(monkeypatch)
    monkeypatch.setenv("ACCELERATE_NUM_PROCESSES", "2")
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "rho.training.train_accelerate",
        SimpleNamespace(train=lambda: calls.append("accelerate")),
    )

    train_launcher.main()

    assert calls == ["accelerate"]
