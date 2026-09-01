import draccus
import pytest

from rho.models.schedule import LRSchedulerConfig, migrate_legacy_scheduler_config


@pytest.mark.parametrize(
    ("scheduler_type", "fields"),
    [
        ("constant", {}),
        ("diffuser", {"schedule_name": "cosine", "num_warmup_steps": 10}),
        (
            "cosine_decay_with_warmup",
            {
                "num_warmup_steps": 10,
                "num_decay_steps": 100,
                "peak_lr": 1e-3,
                "decay_lr": 1e-5,
            },
        ),
        (
            "warmup_stable_decay",
            {
                "num_warmup_steps": 10,
                "num_decay_steps": 100,
                "peak_lr": 1e-3,
                "decay_lr": 1e-5,
            },
        ),
    ],
)
def test_scheduler_type_round_trip(scheduler_type, fields):
    config = draccus.decode(LRSchedulerConfig, {"type": scheduler_type, **fields})

    encoded = draccus.encode(config)

    assert config.type == scheduler_type
    assert encoded["type"] == scheduler_type
    assert "name" not in encoded


@pytest.mark.parametrize(
    ("legacy", "expected"),
    [
        ({"name": "constant"}, {"type": "constant"}),
        (
            {"name": "cosine", "num_warmup_steps": 10},
            {"type": "diffuser", "schedule_name": "cosine", "num_warmup_steps": 10},
        ),
        (
            {"type": "diffuser", "name": "linear", "num_warmup_steps": 10},
            {"type": "diffuser", "schedule_name": "linear", "num_warmup_steps": 10},
        ),
    ],
)
def test_legacy_scheduler_name_migration(legacy, expected):
    assert migrate_legacy_scheduler_config(legacy) == expected
    assert "name" in legacy
