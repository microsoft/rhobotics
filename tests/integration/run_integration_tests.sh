#!/bin/bash
# ==============================================================================
# Integration Test Suite for Alku
# Runs inside the rho-libero:latest Docker container
#
# Each test is a separate function that runs a training or eval command and
# checks the exit code. A summary is printed at the end.
#
# Exit code: 0 if all tests pass, 1 if any test fails.
# ==============================================================================

set -o pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"

# Logs persist on the host (mounted volume)
LOG_DIR="${WORKSPACE_DIR}/test_outputs/integration_$(date +%Y%m%d_%H%M%S)/logs"

# Training outputs (checkpoints, videos) are ephemeral — deleted with container
WORK_DIR="/tmp/integration_test_workdir"

mkdir -p "${LOG_DIR}" "${WORK_DIR}"

# Counters
TOTAL=0
PASSED=0
FAILED=0
SKIPPED=0
FAILED_TESTS=()

# ── Helpers ────────────────────────────────────────────────────────────────────

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Run a single test. Usage: run_test "Test Name" command [args...]
# Captures stdout+stderr to a log file and checks exit code.
run_test() {
    local name="$1"
    shift
    local logfile="${LOG_DIR}/$(echo "${name}" | tr ' ' '_' | tr -cd '[:alnum:]_').log"

    TOTAL=$((TOTAL + 1))
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "TEST ${TOTAL}: ${name}"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "Command: $*"
    log "Log: ${logfile}"

    # Run command, tee to logfile so we get live output too
    if eval "$@" 2>&1 | tee "${logfile}"; then
        PASSED=$((PASSED + 1))
        log "✅ PASSED: ${name}"
    else
        FAILED=$((FAILED + 1))
        FAILED_TESTS+=("${name}")
        log "❌ FAILED: ${name}  (see ${logfile})"
    fi
    echo ""
}

skip_test() {
    local name="$1"
    local reason="$2"
    TOTAL=$((TOTAL + 1))
    SKIPPED=$((SKIPPED + 1))
    log "⏭️  SKIPPED: ${name} — ${reason}"
}

print_summary() {
    echo ""
    log "════════════════════════════════════════════════════════════════"
    log "                   INTEGRATION TEST SUMMARY"
    log "════════════════════════════════════════════════════════════════"
    log "Total:   ${TOTAL}"
    log "Passed:  ${PASSED}"
    log "Failed:  ${FAILED}"
    log "Skipped: ${SKIPPED}"
    if [ ${FAILED} -gt 0 ]; then
        log ""
        log "Failed tests:"
        for t in "${FAILED_TESTS[@]}"; do
            log "  ✗ ${t}"
        done
    fi
    log "════════════════════════════════════════════════════════════════"
    log "Logs saved to: ${LOG_DIR}"
}

# ── Pre-flight checks ─────────────────────────────────────────────────────────

preflight() {
    log "Running pre-flight checks..."

    # Verify we can import rho
    python -c "import rho; print(f'rho imported from {rho.__file__}')" || {
        log "FATAL: Cannot import rho. Is the package installed?"
        exit 2
    }

    # Verify GPU is available
    python -c "import torch; assert torch.cuda.is_available(), 'No GPU'; print(f'GPU: {torch.cuda.get_device_name(0)}')" || {
        log "FATAL: No CUDA GPU available."
        exit 2
    }

    log "Pre-flight checks passed."
    echo ""
}

# ── Test 0: Resource-intensive unit tests (GPU required) ───────────────────────

test_unit_tests_gpu() {
    pip install pytest-cov --quiet
    run_test "Resource-intensive unit tests (GPU)" \
        pytest --all --ignore=tests/integration -x
}

# ── Test 1: Agibot multi-dataset training (16 steps) ──────────────────────────

test_agibot_training() {
    # The agibot config already references *_test.yaml datasets via
    # datasets/agibot_alpha_plus_oxe_test.yaml, so no yaml modification needed
    # as long as the data env vars are set.
    if [ -z "${PHI4ROBOTICS_AGIBOT_DATA_ROOT}" ] || [ -z "${PHI4ROBOTICS_OXE_DATA_ROOT}" ]; then
        skip_test "Agibot multi-dataset training (16 steps)" \
            "PHI4ROBOTICS_AGIBOT_DATA_ROOT or PHI4ROBOTICS_OXE_DATA_ROOT not set"
        return
    fi

    run_test "Agibot multi-dataset training (16 steps)" \
        python -m rho.training.train \
            --config_path=config/tests/train_agibot_multidataset.yaml \
            --set_static_graph=true \
            --policy.train_expert_only=true \
            --policy.use_hd_transform=false \
            --batch_size=8 \
            --wandb.enable=false \
            --log_level=INFO \
            --output_dir="${WORK_DIR}/agibot_training" \
            --steps=16
}

# ── Test 2: Libero training (50 steps with eval) ──────────────────────────────

test_libero_training() {
    run_test "Libero training (50 steps with eval)" \
        python environments/libero/train.py \
            --config_path=environments/libero/configs/train_libero_phi4mm.yaml \
            --policy.train_expert_only=true \
            --batch_size=3 \
            --set_static_graph=true \
            --wandb.enable=false \
            --eval_interval=25 \
            --eval_num_episodes=2 \
            --log_level=INFO \
            --policy.use_hd_transform=false \
            --environment.max_episode_steps=10 \
            --save_checkpoint_every=50 \
            --output_dir="${WORK_DIR}/libero_training" \
            --steps=50
}

# ── Test 3: Libero eval (2 episodes on a checkpoint) ──────────────────────────

test_libero_eval() {
    # Use checkpoint produced by test 2 if available, otherwise look for a
    # pre-existing one at the well-known path.
    local ckpt=""

    # Try to find the checkpoint from the training run we just did
    if ls "${WORK_DIR}/libero_training"/*/checkpoints/checkpoint_step_*.pt 1>/dev/null 2>&1; then
        ckpt=$(ls -t "${WORK_DIR}/libero_training"/*/checkpoints/checkpoint_step_*.pt | head -1)
        log "Using checkpoint from training run: ${ckpt}"
    elif [ -n "${LIBERO_EVAL_CHECKPOINT}" ]; then
        ckpt="${LIBERO_EVAL_CHECKPOINT}"
        log "Using checkpoint from LIBERO_EVAL_CHECKPOINT env var: ${ckpt}"
    else
        skip_test "Libero eval (2 episodes)" \
            "No checkpoint found (run test_libero_training first or set LIBERO_EVAL_CHECKPOINT)"
        return
    fi

    run_test "Libero eval (2 episodes)" \
        python environments/libero/eval.py \
            --config_path=environments/libero/configs/eval_libero_phi4mm.yaml \
            --pretrained_checkpoint="${ckpt}" \
            --eval_num_episodes=2 \
            --environment.n_envs=2 \
            --environment.max_episode_steps=10 \
            --record_videos=false \
            --output_dir="${WORK_DIR}/libero_eval"
}

# ── Test 4: PushT Phi4MM training (100 steps with eval at 100) ─────────────

test_pusht_training() {
    # Install gym_pusht and pin numpy<2.0 if not already done
    log "Ensuring gym_pusht is installed and numpy<2.0..."
    pip install gym_pusht 2>&1 | tail -3
    pip install "numpy<2.0" 2>&1 | tail -3

    run_test "PushT Phi4MM training (100 steps, eval@100)" \
        python -m rho.training.train \
            --config_path=config/train_pusht_phi4mm.yaml \
            --policy.train_expert_only=true \
            --batch_size=4 \
            --set_static_graph=true \
            --wandb.enable=false \
            --eval_interval=100 \
            --log_level=INFO \
            --policy.use_hd_transform=false \
            --environment.max_episode_steps=10 \
            --output_dir="${WORK_DIR}/pusht_training" \
            --steps=100
}

# ── Test 4b: PushT Qwen2.5-VL training (100 steps with eval at 100) ───────

test_pusht_qwen25vl_training() {
    log "Ensuring gym_pusht is installed..."
    pip install gym_pusht 2>&1 | tail -3

    run_test "PushT Qwen2.5-VL training (100 steps, eval@100)" \
        python -m rho.training.train \
            --config_path=config/train_pusht_qwen25vl.yaml \
            --policy.train_expert_only=true \
            --batch_size=4 \
            --wandb.enable=false \
            --eval_interval=100 \
            --log_level=INFO \
            --environment.max_episode_steps=10 \
            --output_dir="${WORK_DIR}/pusht_qwen25vl_training" \
            --steps=100
}

# ── Test 4c: PushT Qwen3-VL training (100 steps with eval at 100) ─────────

test_pusht_qwen3vl_training() {
    log "Ensuring gym_pusht is installed..."
    pip install gym_pusht 2>&1 | tail -3

    run_test "PushT Qwen3-VL training (100 steps, eval@100)" \
        python -m rho.training.train \
            --config_path=config/train_pusht_qwen3vl.yaml \
            --policy.train_expert_only=true \
            --batch_size=4 \
            --wandb.enable=false \
            --eval_interval=100 \
            --log_level=INFO \
            --environment.max_episode_steps=10 \
            --output_dir="${WORK_DIR}/pusht_qwen3vl_training" \
            --steps=100
}

# ── Test 5: TabletopSim install + training + eval ──────────────────────────────

install_tabletopsim() {
    log "Installing Tabletop-Sim environment..."
    if python -c "import tabletop" 2>/dev/null; then
        log "Tabletop-Sim already installed."
        return 0
    fi

    (
        cd /tmp
        rm -rf Tabletop-Sim
        git clone https://github.com/jellyho/Tabletop-Sim.git --recursive
        cd Tabletop-Sim
        pip install -r requirements.txt
        pip install -e .
    ) 2>&1 | tail -20

    python -c "import tabletopsim" 2>/dev/null || {
        # Fallback: try a simpler import check
        log "Warning: tabletopsim import check failed, but installation may still work via env.py"
    }
}

test_tabletopsim_training_ee_6d_pos() {
    if [ -z "${TABLETOP_DATA_ROOT}" ]; then
        skip_test "TabletopSim training (50 steps with eval)" \
            "TABLETOP_DATA_ROOT not set"
        return
    fi

    install_tabletopsim

    run_test "TabletopSim training (50 steps with eval)" \
        python environments/tabletopsim/train.py \
            --config_path=environments/tabletopsim/configs/train_ee_6d_pos.yaml \
            --wandb.enable=false \
            --eval_interval=25 \
            --eval_num_episodes=2 \
            --steps=50 \
            --policy.train_expert_only=true \
            --batch_size=2 \
            --mixed_precision=bf16 \
            --environment.max_episode_steps=5 \
            --save_checkpoint_every=50 \
            --output_dir="${WORK_DIR}/tabletopsim_training_ee_6d_pos"
}

test_tabletopsim_eval_ee_6d_pos() {
    # Use checkpoint from training run if available
    local ckpt=""

    if ls "${WORK_DIR}/tabletopsim_training_ee_6d_pos"/*/checkpoints/checkpoint_step_*.pt 1>/dev/null 2>&1; then
        ckpt=$(ls -t "${WORK_DIR}/tabletopsim_training_ee_6d_pos"/*/checkpoints/checkpoint_step_*.pt | head -1)
        log "Using checkpoint from training run: ${ckpt}"
    elif [ -n "${TABLETOPSIM_EVAL_CHECKPOINT}" ]; then
        ckpt="${TABLETOPSIM_EVAL_CHECKPOINT}"
        log "Using checkpoint from TABLETOPSIM_EVAL_CHECKPOINT env var: ${ckpt}"
    else
        skip_test "TabletopSim eval (2 episodes)" \
            "No checkpoint found (run test_tabletopsim_training_ee_6d_pos first or set TABLETOPSIM_EVAL_CHECKPOINT)"
        return
    fi

    local dataset_root="${TABLETOP_DATA_ROOT}/aloha_handover_box_v6"
    if [ ! -d "${dataset_root}" ]; then
        skip_test "TabletopSim eval (2 episodes)" \
            "Dataset dir not found: ${dataset_root}"
        return
    fi

    run_test "TabletopSim eval (2 episodes)" \
        python environments/tabletopsim/eval.py \
            --config_path=environments/tabletopsim/configs/eval_ee_6d_pos.yaml \
            --pretrained_checkpoint="${ckpt}" \
            --dataset_root_dir="${dataset_root}" \
            --environment.task_name=aloha_handover_box \
            --eval_num_episodes=2 \
            --environment.max_episode_steps=5 \
            --record_videos=false \
            --output_dir="${WORK_DIR}/tabletopsim_eval_ee_6d_pos"
}

test_tabletopsim_training_ee_quat_pos() {
    if [ -z "${TABLETOP_DATA_ROOT}" ]; then
        skip_test "TabletopSim training (50 steps with eval)" \
            "TABLETOP_DATA_ROOT not set"
        return
    fi

    run_test "TabletopSim training (50 steps with eval)" \
        python environments/tabletopsim/train.py \
            --config_path=environments/tabletopsim/configs/train_ee_quat_pos.yaml \
            --wandb.enable=false \
            --eval_interval=25 \
            --eval_num_episodes=2 \
            --steps=50 \
            --policy.train_expert_only=true \
            --batch_size=2 \
            --mixed_precision=bf16 \
            --environment.max_episode_steps=5 \
            --save_checkpoint_every=50 \
            --output_dir="${WORK_DIR}/tabletopsim_training_ee_quat_pos"
}

test_tabletopsim_eval_ee_quat_pos() {
    # Use checkpoint from training run if available
    local ckpt=""

    if ls "${WORK_DIR}/tabletopsim_training_ee_quat_pos"/*/checkpoints/checkpoint_step_*.pt 1>/dev/null 2>&1; then
        ckpt=$(ls -t "${WORK_DIR}/tabletopsim_training_ee_quat_pos"/*/checkpoints/checkpoint_step_*.pt | head -1)
        log "Using checkpoint from training run: ${ckpt}"
    elif [ -n "${TABLETOPSIM_EVAL_CHECKPOINT}" ]; then
        ckpt="${TABLETOPSIM_EVAL_CHECKPOINT}"
        log "Using checkpoint from TABLETOPSIM_EVAL_CHECKPOINT env var: ${ckpt}"
    else
        skip_test "TabletopSim eval (2 episodes)" \
            "No checkpoint found (run test_tabletopsim_training_ee_quat_pos first or set TABLETOPSIM_EVAL_CHECKPOINT)"
        return
    fi

    local dataset_root="${TABLETOP_DATA_ROOT}/aloha_handover_box_v6"
    if [ ! -d "${dataset_root}" ]; then
        skip_test "TabletopSim eval (2 episodes)" \
            "Dataset dir not found: ${dataset_root}"
        return
    fi

    run_test "TabletopSim eval (2 episodes)" \
        python environments/tabletopsim/eval.py \
            --config_path=environments/tabletopsim/configs/eval_ee_quat_pos.yaml \
            --pretrained_checkpoint="${ckpt}" \
            --dataset_root_dir="${dataset_root}" \
            --environment.task_name=aloha_handover_box \
            --eval_num_episodes=2 \
            --environment.max_episode_steps=5 \
            --record_videos=false \
            --output_dir="${WORK_DIR}/tabletopsim_eval_ee_quat_pos"
}

test_tabletopsim_training_joint_pos() {
    if [ -z "${TABLETOP_DATA_ROOT}" ]; then
        skip_test "TabletopSim training (50 steps with eval)" \
            "TABLETOP_DATA_ROOT not set"
        return
    fi

    run_test "TabletopSim training (50 steps with eval)" \
        python environments/tabletopsim/train.py \
            --config_path=environments/tabletopsim/configs/train_joint_pos.yaml \
            --wandb.enable=false \
            --eval_interval=25 \
            --eval_num_episodes=2 \
            --steps=50 \
            --policy.train_expert_only=true \
            --batch_size=2 \
            --mixed_precision=bf16 \
            --environment.max_episode_steps=5 \
            --save_checkpoint_every=50 \
            --output_dir="${WORK_DIR}/tabletopsim_training_joint_pos"
}

test_tabletopsim_eval_joint_pos() {
    # Use checkpoint from training run if available
    local ckpt=""

    if ls "${WORK_DIR}/tabletopsim_training_joint_pos"/*/checkpoints/checkpoint_step_*.pt 1>/dev/null 2>&1; then
        ckpt=$(ls -t "${WORK_DIR}/tabletopsim_training_joint_pos"/*/checkpoints/checkpoint_step_*.pt | head -1)
        log "Using checkpoint from training run: ${ckpt}"
    elif [ -n "${TABLETOPSIM_EVAL_CHECKPOINT}" ]; then
        ckpt="${TABLETOPSIM_EVAL_CHECKPOINT}"
        log "Using checkpoint from TABLETOPSIM_EVAL_CHECKPOINT env var: ${ckpt}"
    else
        skip_test "TabletopSim eval (2 episodes)" \
            "No checkpoint found (run test_tabletopsim_training_joint_pos first or set TABLETOPSIM_EVAL_CHECKPOINT)"
        return
    fi

    local dataset_root="${TABLETOP_DATA_ROOT}/aloha_handover_box_v6"
    if [ ! -d "${dataset_root}" ]; then
        skip_test "TabletopSim eval (2 episodes)" \
            "Dataset dir not found: ${dataset_root}"
        return
    fi

    run_test "TabletopSim eval (2 episodes)" \
        python environments/tabletopsim/eval.py \
            --config_path=environments/tabletopsim/configs/eval_joint_pos.yaml \
            --pretrained_checkpoint="${ckpt}" \
            --dataset_root_dir="${dataset_root}" \
            --environment.task_name=aloha_handover_box \
            --eval_num_episodes=2 \
            --environment.max_episode_steps=5 \
            --record_videos=false \
            --output_dir="${WORK_DIR}/tabletopsim_eval_joint_pos"
}

# ── Test 6: Roboeval install + training + eval ──────────────────────────────

install_roboeval() {
    log "Installing RoboEval environment..."
    if python -c "import roboeval" 2>/dev/null; then
        log "RoboEval already installed."
        return 0
    fi

    (
        cd /tmp
        rm -rf RoboEval
        git clone https://github.com/Robo-Eval/RoboEval.git
        cd RoboEval/thirdparty
        git clone https://github.com/helen9975/mujoco_menagerie.git
        cd ..
        pip install -e .
        pip install -e ".[examples]"
        cd ..
    ) 2>&1 | tail -20

    python -c "import roboeval" 2>/dev/null || {
        # Fallback: try a simpler import check
        log "Warning: roboeval import check failed, but installation may still work via env.py"
    }
}

test_roboeval_training_ee_6d_pos() {
    if [ -z "${ROBOEVAL_DATA_ROOT}" ]; then
        skip_test "roboeval training (50 steps with eval)" \
            "ROBOEVAL_DATA_ROOT not set"
        return
    fi

    install_roboeval

    run_test "roboeval training (50 steps with eval)" \
        python environments/roboeval/train.py \
            --config_path=environments/roboeval/configs/train_ee_6d_pos.yaml \
            --wandb.enable=false \
            --eval_interval=25 \
            --eval_num_episodes=2 \
            --steps=50 \
            --policy.train_expert_only=true \
            --batch_size=2 \
            --mixed_precision=bf16 \
            --environment.max_episode_steps=5 \
            --save_checkpoint_every=50 \
            --eval_dataset_root_dir="${ROBOEVAL_DATA_ROOT}/lift_pot" \
            --output_dir="${WORK_DIR}/roboeval_training_ee_6d_pos"
}

test_roboeval_eval_ee_6d_pos() {
    # Use checkpoint from training run if available
    local ckpt=""

    if ls "${WORK_DIR}/roboeval_training_ee_6d_pos"/*/checkpoints/checkpoint_step_*.pt 1>/dev/null 2>&1; then
        ckpt=$(ls -t "${WORK_DIR}/roboeval_training_ee_6d_pos"/*/checkpoints/checkpoint_step_*.pt | head -1)
        log "Using checkpoint from training run: ${ckpt}"
    elif [ -n "${roboeval_EVAL_CHECKPOINT}" ]; then
        ckpt="${roboeval_EVAL_CHECKPOINT}"
        log "Using checkpoint from roboeval_EVAL_CHECKPOINT env var: ${ckpt}"
    else
        skip_test "roboeval eval (2 episodes)" \
            "No checkpoint found (run test_roboeval_training_ee_6d_pos first or set roboeval_EVAL_CHECKPOINT)"
        return
    fi

    local dataset_root="${ROBOEVAL_DATA_ROOT}/lift_pot"
    if [ ! -d "${dataset_root}" ]; then
        skip_test "roboeval eval (2 episodes)" \
            "Dataset dir not found: ${dataset_root}"
        return
    fi

    run_test "roboeval eval (2 episodes)" \
        python environments/roboeval/eval.py \
            --config_path=environments/roboeval/configs/eval_ee_6d_pos.yaml \
            --pretrained_checkpoint="${ckpt}" \
            --dataset_root_dir="${dataset_root}" \
            --environment.task_name=lift_pot \
            --eval_num_episodes=2 \
            --environment.max_episode_steps=5 \
            --record_videos=false \
            --output_dir="${WORK_DIR}/roboeval_eval_ee_6d_pos"
}

test_roboeval_training_ee_quat_pos() {
    if [ -z "${ROBOEVAL_DATA_ROOT}" ]; then
        skip_test "roboeval training (50 steps with eval)" \
            "ROBOEVAL_DATA_ROOT not set"
        return
    fi

    install_roboeval

    run_test "roboeval training (50 steps with eval)" \
        python environments/roboeval/train.py \
            --config_path=environments/roboeval/configs/train_ee_quat_pos.yaml \
            --wandb.enable=false \
            --eval_interval=25 \
            --eval_num_episodes=2 \
            --steps=50 \
            --policy.train_expert_only=true \
            --batch_size=2 \
            --mixed_precision=bf16 \
            --environment.max_episode_steps=5 \
            --save_checkpoint_every=50 \
            --eval_dataset_root_dir="${ROBOEVAL_DATA_ROOT}/lift_pot" \
            --output_dir="${WORK_DIR}/roboeval_training_ee_quat_pos"
}

test_roboeval_eval_ee_quat_pos() {
    # Use checkpoint from training run if available
    local ckpt=""

    if ls "${WORK_DIR}/roboeval_training_ee_quat_pos"/*/checkpoints/checkpoint_step_*.pt 1>/dev/null 2>&1; then
        ckpt=$(ls -t "${WORK_DIR}/roboeval_training_ee_quat_pos"/*/checkpoints/checkpoint_step_*.pt | head -1)
        log "Using checkpoint from training run: ${ckpt}"
    elif [ -n "${roboeval_EVAL_CHECKPOINT}" ]; then
        ckpt="${roboeval_EVAL_CHECKPOINT}"
        log "Using checkpoint from roboeval_EVAL_CHECKPOINT env var: ${ckpt}"
    else
        skip_test "roboeval eval (2 episodes)" \
            "No checkpoint found (run test_roboeval_training_ee_quat_pos first or set roboeval_EVAL_CHECKPOINT)"
        return
    fi

    local dataset_root="${ROBOEVAL_DATA_ROOT}/lift_pot"
    if [ ! -d "${dataset_root}" ]; then
        skip_test "roboeval eval (2 episodes)" \
            "Dataset dir not found: ${dataset_root}"
        return
    fi

    run_test "roboeval eval (2 episodes)" \
        python environments/roboeval/eval.py \
            --config_path=environments/roboeval/configs/eval_ee_quat_pos.yaml \
            --pretrained_checkpoint="${ckpt}" \
            --dataset_root_dir="${dataset_root}" \
            --environment.task_name=lift_pot \
            --eval_num_episodes=2 \
            --environment.max_episode_steps=5 \
            --record_videos=false \
            --output_dir="${WORK_DIR}/roboeval_eval_ee_quat_pos"
}

test_roboeval_training_ee_rpy_pos() {
    if [ -z "${ROBOEVAL_DATA_ROOT}" ]; then
        skip_test "roboeval training (50 steps with eval)" \
            "ROBOEVAL_DATA_ROOT not set"
        return
    fi

    install_roboeval

    run_test "roboeval training (50 steps with eval)" \
        python environments/roboeval/train.py \
            --config_path=environments/roboeval/configs/train_ee_rpy_pos.yaml \
            --wandb.enable=false \
            --eval_interval=25 \
            --eval_num_episodes=2 \
            --steps=50 \
            --policy.train_expert_only=true \
            --batch_size=2 \
            --mixed_precision=bf16 \
            --environment.max_episode_steps=5 \
            --save_checkpoint_every=50 \
            --eval_dataset_root_dir="${ROBOEVAL_DATA_ROOT}/lift_pot" \
            --output_dir="${WORK_DIR}/roboeval_training_ee_rpy_pos"
}

test_roboeval_eval_ee_rpy_pos() {
    # Use checkpoint from training run if available
    local ckpt=""

    if ls "${WORK_DIR}/roboeval_training_ee_rpy_pos"/*/checkpoints/checkpoint_step_*.pt 1>/dev/null 2>&1; then
        ckpt=$(ls -t "${WORK_DIR}/roboeval_training_ee_rpy_pos"/*/checkpoints/checkpoint_step_*.pt | head -1)
        log "Using checkpoint from training run: ${ckpt}"
    elif [ -n "${roboeval_EVAL_CHECKPOINT}" ]; then
        ckpt="${roboeval_EVAL_CHECKPOINT}"
        log "Using checkpoint from roboeval_EVAL_CHECKPOINT env var: ${ckpt}"
    else
        skip_test "roboeval eval (2 episodes)" \
            "No checkpoint found (run test_roboeval_training_ee_rpy_pos first or set roboeval_EVAL_CHECKPOINT)"
        return
    fi

    local dataset_root="${ROBOEVAL_DATA_ROOT}/lift_pot"
    if [ ! -d "${dataset_root}" ]; then
        skip_test "roboeval eval (2 episodes)" \
            "Dataset dir not found: ${dataset_root}"
        return
    fi

    run_test "roboeval eval (2 episodes)" \
        python environments/roboeval/eval.py \
            --config_path=environments/roboeval/configs/eval_ee_rpy_pos.yaml \
            --pretrained_checkpoint="${ckpt}" \
            --dataset_root_dir="${dataset_root}" \
            --environment.task_name=lift_pot \
            --eval_num_episodes=2 \
            --environment.max_episode_steps=5 \
            --record_videos=false \
            --output_dir="${WORK_DIR}/roboeval_eval_ee_rpy_pos"
}


test_roboeval_training_joint_pos() {
    if [ -z "${ROBOEVAL_DATA_ROOT}" ]; then
        skip_test "roboeval training (50 steps with eval)" \
            "ROBOEVAL_DATA_ROOT not set"
        return
    fi

    install_roboeval

    run_test "roboeval training (50 steps with eval)" \
        python environments/roboeval/train.py \
            --config_path=environments/roboeval/configs/train_joint_pos.yaml \
            --wandb.enable=false \
            --eval_interval=25 \
            --eval_num_episodes=2 \
            --steps=50 \
            --policy.train_expert_only=true \
            --batch_size=2 \
            --mixed_precision=bf16 \
            --environment.max_episode_steps=5 \
            --save_checkpoint_every=50 \
            --eval_dataset_root_dir="${ROBOEVAL_DATA_ROOT}/lift_pot" \
            --output_dir="${WORK_DIR}/roboeval_training_joint_pos"
}

test_roboeval_eval_joint_pos() {
    # Use checkpoint from training run if available
    local ckpt=""

    if ls "${WORK_DIR}/roboeval_training_joint_pos"/*/checkpoints/checkpoint_step_*.pt 1>/dev/null 2>&1; then
        ckpt=$(ls -t "${WORK_DIR}/roboeval_training_joint_pos"/*/checkpoints/checkpoint_step_*.pt | head -1)
        log "Using checkpoint from training run: ${ckpt}"
    elif [ -n "${roboeval_EVAL_CHECKPOINT}" ]; then
        ckpt="${roboeval_EVAL_CHECKPOINT}"
        log "Using checkpoint from roboeval_EVAL_CHECKPOINT env var: ${ckpt}"
    else
        skip_test "roboeval eval (2 episodes)" \
            "No checkpoint found (run test_roboeval_training_joint_pos first or set roboeval_EVAL_CHECKPOINT)"
        return
    fi

    local dataset_root="${ROBOEVAL_DATA_ROOT}/lift_pot"
    if [ ! -d "${dataset_root}" ]; then
        skip_test "roboeval eval (2 episodes)" \
            "Dataset dir not found: ${dataset_root}"
        return
    fi

    run_test "roboeval eval (2 episodes)" \
        python environments/roboeval/eval.py \
            --config_path=environments/roboeval/configs/eval_joint_pos.yaml \
            --pretrained_checkpoint="${ckpt}" \
            --dataset_root_dir="${dataset_root}" \
            --environment.task_name=lift_pot \
            --eval_num_episodes=2 \
            --environment.max_episode_steps=5 \
            --record_videos=false \
            --output_dir="${WORK_DIR}/roboeval_eval_joint_pos"
}

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

main() {
    log "Starting Alku Integration Test Suite"
    log "Log directory: ${LOG_DIR}"
    log "Work directory: ${WORK_DIR} (ephemeral — deleted with container)"
    echo ""

    cd "${WORKSPACE_DIR}"
    preflight

    # Run tests in order — later tests may depend on checkpoints from earlier ones
    test_unit_tests_gpu
    test_agibot_training
    test_libero_training
    test_libero_eval
    test_pusht_training
    test_pusht_qwen25vl_training
    test_pusht_qwen3vl_training
    test_tabletopsim_training_ee_6d_pos
    test_tabletopsim_eval_ee_6d_pos
    test_tabletopsim_training_ee_quat_pos
    test_tabletopsim_eval_ee_quat_pos
    test_tabletopsim_training_joint_pos
    test_tabletopsim_eval_joint_pos
    test_roboeval_training_ee_6d_pos
    test_roboeval_eval_ee_6d_pos
    test_roboeval_training_ee_quat_pos
    test_roboeval_eval_ee_quat_pos
    test_roboeval_training_ee_rpy_pos
    test_roboeval_eval_ee_rpy_pos
    test_roboeval_training_joint_pos
    test_roboeval_eval_joint_pos
    print_summary

    if [ ${FAILED} -gt 0 ]; then
        exit 1
    fi
    exit 0
}

main "$@"
