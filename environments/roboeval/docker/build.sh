#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

echo "Building Rho RoboEval container..."
docker build \
  -t rho-roboeval:latest \
  -f "$ROOT_DIR/environments/roboeval/docker/Dockerfile" \
  "$ROOT_DIR"
