# RoboEval finetuning

This directory contains Rho training and evaluation configurations for
[RoboEval](https://github.com/Robo-Eval/RoboEval). The public examples support
dual-arm joint-position actions and end-effector position with 6D rotation
actions.

## Prepare the datasets

The configurations expect converted LeRobot datasets arranged by task:

```text
/path/to/roboeval_datasets/
├── cube_handover/
├── lift_pot/
├── lift_tray/
├── pack_box/
├── pick_single_book_from_table/
├── rotate_valve/
├── stack_single_book_shelf/
└── stack_two_blocks/
```

Set `RHO_DATA_DIR` on the host to this directory. The container launcher mounts
it at `/data` and sets `ROBOEVAL_DATA_ROOT=/data`.

```bash
export RHO_DATA_DIR=/path/to/roboeval_datasets
```

## Build and launch the container

Build the base Rho image, then the RoboEval image:

```bash
./docker/training/build.sh
./environments/roboeval/docker/build.sh
```

Launch an interactive GPU container:

```bash
./environments/roboeval/docker/run_interactive.sh
```

The remaining commands run inside that container from `/workspace`.

## Finetune

Run a short single-GPU smoke test:

```bash
python environments/roboeval/train.py \
  --config_path=environments/roboeval/configs/train_ee_6d_pos.yaml \
  --steps=10 \
  --batch_size=1 \
  --num_workers=0 \
  --wandb.enabled=false
```

Start a full end-effector training run:

```bash
python environments/roboeval/train.py \
  --config_path=environments/roboeval/configs/train_ee_6d_pos.yaml
```

For joint-position actions, use:

```bash
python environments/roboeval/train.py \
  --config_path=environments/roboeval/configs/train_joint_pos.yaml
```

The configurations use Rho's hosted pretrained checkpoint by default. Set
`--pretrained_checkpoint=<repository-or-path>` only to use another pretrained
source. Set `--resume=true` only when restoring trusted optimizer and scheduler
state from a full training checkpoint.

For multi-GPU training, launch the same environment-specific entry point with
Accelerate:

```bash
accelerate launch --multi-gpu \
  --num_processes=4 \
  environments/roboeval/train.py \
  --config_path=environments/roboeval/configs/train_ee_6d_pos.yaml
```

The effective batch size is:

```text
batch_size × number of processes × gradient_accumulation_steps
```

Increase `gradient_accumulation_steps` when using fewer GPUs if you need to
preserve a target effective batch size.

## Evaluate

The public end-effector checkpoint is available at
[`microsoft/rho-roboeval`](https://huggingface.co/microsoft/rho-roboeval).
Evaluation must explicitly select this hosted checkpoint or a finetuned local
checkpoint.

### Standard benchmark (preferred)

Use the multieval configuration for the standard end-effector benchmark:

```bash
python environments/roboeval/eval.py \
  --config_path=environments/roboeval/configs/multieval_roboeval_ee_6d_pos.yaml \
  --pretrained_checkpoint=microsoft/rho-roboeval \
  --output_dir=outputs/eval_roboeval/multieval
```

This evaluates all eight RoboEval tasks with 100 episodes and a 250-step limit
per task. The policy is loaded once and reused across the task environments,
making this the preferred method for full benchmark runs. Results are written
to one subdirectory per task, along with a `multieval_summary_*.json` file in
the output directory.

The dataset directories must be available under `ROBOEVAL_DATA_ROOT` as
described in [Prepare the datasets](#prepare-the-datasets).

### Single-task evaluation

Use the single-task configuration for smoke tests, debugging, or targeted
evaluation. For the default `lift_pot` task with end-effector actions:

```bash
python environments/roboeval/eval.py \
  --config_path=environments/roboeval/configs/eval_ee_6d_pos.yaml \
  --pretrained_checkpoint=microsoft/rho-roboeval \
  --dataset_root_dir=/data/lift_pot
```

For a joint-position checkpoint:

```bash
python environments/roboeval/eval.py \
  --config_path=environments/roboeval/configs/eval_joint_pos.yaml \
  --pretrained_checkpoint=/path/to/checkpoint_step_0010000 \
  --dataset_root_dir=/data/lift_pot
```

To evaluate another task, override both the environment task and the matching
dataset root:

```bash
python environments/roboeval/eval.py \
  --config_path=environments/roboeval/configs/eval_ee_6d_pos.yaml \
  --pretrained_checkpoint=/path/to/checkpoint_step_0010000 \
  --environment.task_name=stack_two_blocks \
  --dataset_root_dir=/data/stack_two_blocks
```

The dataset root selects the matching preprocessing and normalization
statistics stored in the finetuned checkpoint; the evaluation YAML does not
replace those statistics.
