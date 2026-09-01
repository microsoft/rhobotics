#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONTAINER_NAME="${RHO_ROBOEVAL_CONTAINER_NAME:-rho-roboeval-interactive}"
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"

if [[ -z "${RHO_DATA_DIR:-}" ]]; then
  echo "RHO_DATA_DIR is not set. Point it to the converted RoboEval datasets." >&2
  exit 1
fi
if [[ ! -d "$RHO_DATA_DIR" ]]; then
  echo "RHO_DATA_DIR does not exist: $RHO_DATA_DIR" >&2
  exit 1
fi

mkdir -p "$HF_CACHE_DIR" "$ROOT_DIR/outputs"

docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --rm \
  --volume "$ROOT_DIR:/workspace" \
  --volume "$RHO_DATA_DIR:/data" \
  --volume "$HF_CACHE_DIR:/hf_home" \
  --env "HF_TOKEN=${HF_TOKEN:-}" \
  --env "WANDB_API_KEY=${WANDB_API_KEY:-}" \
  --env "WANDB_BASE_URL=${WANDB_BASE_URL:-}" \
  --env "ROBOEVAL_DATA_ROOT=/data" \
  --name "$CONTAINER_NAME" \
  --detach \
  rho-roboeval:latest \
  tail -f /dev/null

docker exec -it "$CONTAINER_NAME" /bin/bash
