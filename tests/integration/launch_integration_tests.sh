#!/bin/bash
# ==============================================================================
# Launch the Alku integration test suite inside an rho-libero:latest container.
#
# Usage:
#   ./tests/integration/launch_integration_tests.sh [--interactive]
#
# Options:
#   --interactive   Drop into a shell inside the container instead of running
#                   the tests automatically. Useful for debugging failures.
#
# Required environment variables (will skip respective tests if unset):
#   PHI4ROBOTICS_AGIBOT_DATA_ROOT  – path to agibot data on host
#   PHI4ROBOTICS_OXE_DATA_ROOT     – path to OXE data on host
#   TABLETOP_DATA_ROOT             – path to tabletopsim data on host
#   LIBERO_DATA_ROOT              – path to Libero data on host
#   ROBOEVAL_DATA_ROOT             – path to RoboEval data on host
#
# Optional:
#   LIBERO_EVAL_CHECKPOINT         – path (inside container) to a Libero ckpt
#   TABLETOPSIM_EVAL_CHECKPOINT    – path (inside container) to a TabletopSim ckpt
#   HF_TOKEN, WANDB_API_KEY, WANDB_BASE_URL – forwarded if set
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONTAINER_NAME="rho-integration-tests-$(date +%s)"
#IMAGE="msrxworkspace1acr.azurecr.io/phi4robotics/rho-libero:20260203"
IMAGE="rho-libero:latest"
INTERACTIVE=false

for arg in "$@"; do
    case "$arg" in
        --interactive) INTERACTIVE=true ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          Alku Integration Test Launcher                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Project root : ${PROJECT_ROOT}"
echo "Image        : ${IMAGE}"
echo "Container    : ${CONTAINER_NAME}"
echo "Interactive  : ${INTERACTIVE}"
echo ""

# ── Build env-var flags ────────────────────────────────────────────────────────

ENV_FLAGS=()

# Data roots — pass through so tests can find data
for var in PHI4ROBOTICS_AGIBOT_DATA_ROOT PHI4ROBOTICS_OXE_DATA_ROOT \
           TABLETOP_DATA_ROOT LIBERO_DATA_ROOT ROBOEVAL_DATA_ROOT \
           LIBERO_EVAL_CHECKPOINT TABLETOPSIM_EVAL_CHECKPOINT \
           HF_TOKEN WANDB_API_KEY WANDB_BASE_URL; do
    if [ -n "${!var:-}" ]; then
        ENV_FLAGS+=(-e "${var}=${!var}")
    fi
done

# ── Build volume mounts ───────────────────────────────────────────────────────

VOLUME_FLAGS=(
    # Mount source code so we test the latest version
    -v "${PROJECT_ROOT}/rho:/workspace/rho"
    -v "${PROJECT_ROOT}/config:/workspace/config"
    -v "${PROJECT_ROOT}/environments:/workspace/environments"
    -v "${PROJECT_ROOT}/tests:/workspace/tests"
    -v "${PROJECT_ROOT}/scratch:/workspace/scratch"

    # Output goes to host for post-mortem inspection
    -v "${PROJECT_ROOT}/test_outputs:/workspace/test_outputs"

    # HuggingFace cache (for downloading pusht dataset etc.)
    -v "${HOME}/.cache/huggingface:/hf_home"
)

# Mount data directories if they exist on the host
if [ -d "/data" ]; then
    VOLUME_FLAGS+=(-v "/data:/data")
fi
if [ -d "/mnt/shared_data" ]; then
    VOLUME_FLAGS+=(-v "/mnt/shared_data:/mnt/shared_data")
fi
if [ -d "/azuredata" ]; then
    VOLUME_FLAGS+=(-v "/azuredata:/azuredata")
fi

# ── Launch ─────────────────────────────────────────────────────────────────────

if $INTERACTIVE; then
    echo "Launching interactive shell — run the tests manually with:"
    echo "  bash /workspace/tests/integration/run_integration_tests.sh"
    echo ""

    docker run --gpus all --ipc=host \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        --rm -it \
        "${VOLUME_FLAGS[@]}" \
        "${ENV_FLAGS[@]}" \
        --name "${CONTAINER_NAME}" \
        "${IMAGE}" \
        /bin/bash
else
    echo "Running integration tests..."
    echo ""

    docker run --gpus all --ipc=host \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        --rm \
        "${VOLUME_FLAGS[@]}" \
        "${ENV_FLAGS[@]}" \
        --name "${CONTAINER_NAME}" \
        "${IMAGE}" \
        bash /workspace/tests/integration/run_integration_tests.sh

    EXIT_CODE=$?

    echo ""
    if [ ${EXIT_CODE} -eq 0 ]; then
        echo "✅ All integration tests passed."
    else
        echo "❌ Some integration tests failed. Check logs in test_outputs/."
    fi

    exit ${EXIT_CODE}
fi
