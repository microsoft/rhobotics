# Project

> This repo has been populated by an initial template to help get you started. Please
> make sure to update the content to build a great experience for community-building.

TO FILL IN


<!--=========================README TEMPLATE INSTRUCTIONS=============================
======================================================================================

- THIS README TEMPLATE LARGELY CONSISTS OF COMMENTED OUT TEXT. THIS UNRENDERED TEXT IS MEANT TO BE LEFT IN AS A GUIDE
  THROUGHOUT THE REPOSITORY'S LIFE WHILE END USERS ONLY SEE THE RENDERED PAGE CONTENT.
- Any italicized text rendered in the initial template is intended to be replaced IMMEDIATELY upon repository creation.

- This template is default but not mandatory. It was designed to compensate for typical gaps in Microsoft READMEs
  that slow the pace of work. You may delete it if you have a fully populated README to replace it with.

- Most README sections below are commented out as they are not known early in a repository's life. Others are commented
  out as they do not apply to every repository. If a section will be appropriate later but not known now, consider
  leaving it in commented out and adding an issue as a reminder.
- There are additional optional README sections in the external instruction link below. These include; "citation",
  "built with", "acknowledgments", "folder structure", etc.
- You can easily find the places to add content that will be rendered to the end user by searching
within the file for "TODO".



- ADDITIONAL EXTERNAL TEMPLATE INSTRUCTIONS:
  -  https://aka.ms/StartRight/README-Template/Instructions

======================================================================================
====================================================================================-->


<!---------------------[  Description  ]------------------<recommended> section below------------------>

# Rho

A repository for experimenting with using rho for robotic applications.

## Prerequisites

We use **WANDB** extensively for testing and training. You will need the following environment variables set:

- `WANDB_API_KEY`
- `WANDB_BASE_URL`
- `HF_TOKEN` (Hugging Face token for model downloads)

If you want to run without WANDB you must pass `--wandb.enable=false` for all your jobs.

Package Dependencies:
- docker
- conda
- gcc

## Project Structure

```
rho/
├── rho/                        # Core package
│   ├── common/                 # Shared constants, types, and utilities
│   ├── datasets/               # Dataset loading and preprocessing
│   ├── environment/            # Base environment wrapper and evaluation logic
│   ├── eval/                   # Policy interface for inference
│   ├── models/                 # Optimizer and LR scheduler configs
│   ├── policies/               # Policy implementations
│   │   ├── base.py             # PolicyConfig base class
│   │   ├── diffusion/          # Diffusion Policy (DDPM-based)
│   │   ├── rhoalpha/           # Rho-Alpha (Phi-4 multimodal + flow matching)
│   │   └── BC/                 # Behavioral Cloning baseline
│   ├── training/               # Training loop and utilities
│   └── utils/                  # General helpers
├── config/                     # Top-level training configs
│   ├── datasets/               # Dataset YAML configs (e.g. pusht.yaml)
│   └── policies/               # Policy YAML configs (diffusion.yaml, rhoalpha.yaml)
├── environments/               # Environment-specific extensions
│   └── libero/                 # Libero benchmark (env, train, eval, docker)
├── docker/                     # Docker build files
│   └── training/               # Base training image
├── tests/                      # Test suite (pytest)
└── outputs/                    # Training outputs (git-ignored)
```

---
## Installation
```
conda create -y -n rhoalpha python=3.10.16
conda activate rhoalpha
pip install -e ".[dev,test,server]"
```

Unfortunately flash-attn must be installed as a separate step.
```
pip install flash-attn==2.8.3 --no-build-isolation --no-cache-dir
```


## Getting Started

### 1. Build the Training Docker Image

```bash
docker build -t rho-training:latest -f docker/training/Dockerfile .
```

### 2. Run in Interactive Mode

```bash
export RHOALPHA_DIR=`pwd`
docker run --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --rm -v ~/.cache/huggingface:/hf_home \
  -v $RHOALPHA_DIR/rho:/workspace/rho \
  -v $RHOALPHA_DIR/config:/workspace/config \
  -v $RHOALPHA_DIR/environments:/workspace/environments \
  -v $RHOALPHA_DIR/outputs:/workspace/outputs \
  -v $RHOALPHA_DIR/tests:/workspace/tests \
  -v /data/:/data \
  -e WANDB_BASE_URL="$WANDB_BASE_URL" \
  -e WANDB_API_KEY="$WANDB_API_KEY" \
  -e HF_TOKEN="$HF_TOKEN" \
  --name rhoalpha-interactive \
  -d rho-training:latest \
  tail -f /dev/null
```

In addition to mounting all local source folders, this mounts the local user's Hugging Face cache directory so that downloaded models persist between runs. It also mounts `/data` if available, as many sandbox VMs have a larger storage drive at that location.

### 3. Quick Diffusion Policy Test (PushT)

By default the container does not have PushT installed, so install it first:

```bash
pip install gym-pusht
```

Then run a short training job to verify everything works (WANDB disabled):

```bash
python -m rho.training.train \
  --config_path=config/train_pusht_diffusion.yaml \
  --steps=1000 \
  --wandb.enable=false
```

This will typically require many more steps to fully train, but confirms the training pipeline functions.

### 4. Multi-GPU Training

```bash
export NUM_GPU=2
accelerate launch --multi-gpu \
  --num_processes=${NUM_GPU} \
  -m rho.training.train_accelerate \
  --config_path=config/train_pusht_diffusion.yaml \
  --steps=1000 \
  --wandb.enable=true
```

---

## Checkpoints

Control checkpoint frequency with the `keep_interval` and `save_interval` settings. Checkpoints are saved with this structure:

```
${OUTPUT_DIR}/checkpoints/checkpoint_latest.pt
${OUTPUT_DIR}/manual_eval/
${OUTPUT_DIR}/train_config.json
```

> **Important:** The relative position of `checkpoint_latest.pt` and `train_config.json` is critical — the code looks for `train_config.json` when loading a checkpoint to recreate the same training settings.

## Configuration

Training configs use YAML with `!include` directives to compose dataset, policy, and environment settings. For example:

```yaml
# config/train_pusht_diffusion.yaml
dataset: !include datasets/pusht.yaml
policy: !include policies/diffusion.yaml
environment:
  type: "GymEnvironment"
  env_name: "gym_pusht/PushT-v0"
  ...
batch_size: 64
steps: 50000
```

**Any config field can be overridden from the command line:**

```bash
python -m rho.training.train \
  --config_path=config/train_pusht_diffusion.yaml \
  --batch_size=32 \
  --steps=10000 \
  --policy.horizon=32 \
  --wandb.enable=false
```

Nested fields use dot notation (e.g. `--policy.horizon=32`, `--wandb.project="my_experiment"`).

---

## Libero Environment

A Libero environment has been set up to make testing more convenient and demonstrate how to extend the rho environment classes.

### 1. Build the Libero Docker Image

> **Requires `rho-training:latest` to have already been built.**

```bash
docker build -t rho-libero:latest -f environments/libero/docker/Dockerfile .
```

### 2. Run in Interactive Mode

```bash
export RHOALPHA_DIR=`pwd`
docker stop rho-libero-interactive 2>/dev/null || true
sleep 1
docker run --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --rm -v ~/.cache/huggingface:/hf_home \
  -v $RHOALPHA_DIR/rho:/workspace/rho \
  -v $RHOALPHA_DIR/config:/workspace/config \
  -v $RHOALPHA_DIR/environments:/workspace/environments \
  -v $RHOALPHA_DIR/outputs:/workspace/outputs \
  -v $RHOALPHA_DIR/tests:/workspace/tests \
  -v /data/:/data \
  -e WANDB_BASE_URL="$WANDB_BASE_URL" \
  -e WANDB_API_KEY="$WANDB_API_KEY" \
  -e HF_TOKEN="$HF_TOKEN" \
  --name rho-libero-interactive \
  -d rho-libero:latest \
  tail -f /dev/null
```

This command is also saved in `environments/libero/docker/run_interactive.sh`.

Then exec into the container:

```bash
docker exec -it rho-libero-interactive /bin/bash
```

### 3. Training & Evaluation

Once inside the container, run Libero training and evaluation (requires at least 20 GB VRAM, e.g. NVIDIA RTX 4090):

```bash
python environments/libero/train.py \
  --config_path=environments/libero/configs/train_libero_rhoalpha.yaml \
  --wandb.enable=false \
  --batch_size=4 \
  --steps=20 \
  --eval_interval=20
```

This will automatically download all assets necessary for Libero evaluation and the training dataset. If you have already downloaded the dataset elsewhere, add:

```
--dataset.root_dir=${PATH_TO_DATASET}
```

#### Evaluate a Pre-existing Checkpoint

```bash
python environments/libero/eval.py \
  --config_path=environments/libero/configs/eval_libero_rhoalpha.yaml \
  --pretrained_checkpoint=${PATH_TO_CHECKPOINT}
```

#### Evaluate Against All Task Variants

```bash
./environments/libero/eval_all_envs.sh ${PATH_TO_CHECKPOINT}
```

### Known Issues

- Training writes files as root to the mounted output folder, so you need elevated permissions to delete them afterwards.

---


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
