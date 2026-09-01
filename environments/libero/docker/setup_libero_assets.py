"""Pre-download libero assets and generate config.yaml during Docker build."""

import os
import subprocess
import sys

import yaml

# Create config.yaml BEFORE importing libero to prevent the interactive
# prompt in libero's __init__.py that asks about custom dataset paths.
config_dir = os.environ.get("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero"))
config_file = os.path.join(config_dir, "config.yaml")

if not os.path.exists(config_file):
    # Find the libero package location without importing it
    result = subprocess.run(
        [sys.executable, "-c", "import libero.libero; print(libero.libero.__file__)"],
        capture_output=True,
        text=True,
        input="N\n",  # answer "No" to the dataset path prompt
    )
    pkg_init = result.stdout.strip()
    pkg_dir = os.path.dirname(os.path.abspath(pkg_init))

    os.makedirs(config_dir, exist_ok=True)
    config = {
        "benchmark_root": pkg_dir,
        "bddl_files": os.path.join(pkg_dir, "./bddl_files"),
        "init_states": os.path.join(pkg_dir, "./init_files"),
        "datasets": os.path.join(pkg_dir, "../datasets"),
        "assets": os.path.join(pkg_dir, "./assets"),
    }
    with open(config_file, "w") as f:
        yaml.dump(config, f)
    print(f"Initial config written to {config_file}")

import libero.libero  # noqa: E402
from libero.libero import get_default_path_dict  # noqa: E402
from libero.libero.utils.download_utils import download_assets_from_huggingface  # noqa: E402

# Download assets into the pip-installed package's assets directory
pkg_dir = os.path.dirname(os.path.abspath(libero.libero.__file__))
assets_dir = os.path.join(pkg_dir, "assets")
print(f"Downloading assets to {assets_dir}...")
download_assets_from_huggingface(download_dir=assets_dir)

# Regenerate config.yaml with correct paths matching the actual install location
config = get_default_path_dict()
with open(config_file, "w") as f:
    yaml.dump(config, f)

print(f"Config written to {config_file}")
for k, v in config.items():
    print(f"  {k}: {v}")

# Verify assets are findable without network access
from libero.libero import get_assets_path  # noqa: E402

resolved = get_assets_path()
print(f"\nget_assets_path() resolves to: {resolved}")
print(f"Assets directory exists: {os.path.exists(resolved)}")
if os.path.exists(resolved):
    subdirs = os.listdir(resolved)
    print(f"Asset subdirectories: {subdirs}")
