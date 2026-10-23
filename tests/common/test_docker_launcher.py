"""Contract tests for the training container's published serving port."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("port", [None, "8765"])
def test_training_launcher_limits_host_port_and_binds_inside_container(tmp_path, port):
    source = Path(__file__).resolve().parents[2] / "docker/training/interactive.sh"
    launcher = tmp_path / "repo/docker/training/interactive.sh"
    launcher.parent.mkdir(parents=True)
    shutil.copyfile(source, launcher)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker_stub = bin_dir / "docker"
    docker_stub.write_text(
        '#!/bin/sh\nif [ "$1" = "run" ]; then\n    printf "%s\\n" "$@" > "$RHO_TEST_DOCKER_ARGS"\nfi\n'
    )
    docker_stub.chmod(0o755)

    args_log = tmp_path / "docker-args.txt"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}{os.pathsep}{env.get('PATH', '')}",
            "HF_HOME": str(tmp_path / "hf-cache"),
            "RHO_TEST_DOCKER_ARGS": str(args_log),
            "RHO_CONTAINER_NAME": "rho-training-test",
            "RHO_SERVER_HOST": "127.0.0.1",  # The container must override this.
        }
    )
    env.pop("RHO_DATA_DIR", None)
    if port is None:
        env.pop("RHO_SERVER_PORT", None)
    else:
        env["RHO_SERVER_PORT"] = port

    subprocess.run(["bash", str(launcher)], check=True, env=env, capture_output=True, text=True, timeout=15)

    args = args_log.read_text().splitlines()
    expected_port = port or "9999"
    assert args[0] == "run"
    assert args[args.index("--publish") + 1] == f"127.0.0.1:{expected_port}:{expected_port}"
    assert ("--env", "RHO_SERVER_HOST=0.0.0.0") in zip(args, args[1:], strict=False)
