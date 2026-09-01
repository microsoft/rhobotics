# Rho

Rho is a vision-language-action policy for robot learning. This repository
provides the code needed to finetune Rho on LeRobot datasets, evaluate it in
simulation, and serve it over a websocket for deployment. It also includes
FlowDAgger support for human-in-the-loop adaptation of a frozen Rho policy.

The public release focuses on Rho finetuning and inference. Pretraining code,
internal datasets, and unrelated experimental policies are intentionally not
part of the supported surface.

## Features

- Rho flow-matching policy with a Phi vision-language backbone.
- Single- and multi-GPU finetuning with PyTorch and Accelerate.
- Portable sharded-safetensors checkpoints.
- LeRobot dataset loading, transforms, and normalization.
- LIBERO and RoboEval training and evaluation examples.
- Websocket policy server and lightweight Python client.
- FlowDAgger human-in-the-loop adaptation.

## Requirements

- Linux
- Python 3.12
- A CUDA-capable GPU for practical training and inference
- Git

Docker is recommended for a reproducible GPU environment. Weights and datasets
hosted on Hugging Face may require `HF_TOKEN`. Weights & Biases is optional;
disable it with `--wandb.enabled=false`.

## Installation

Create a Python environment and install the repository:

```bash
conda create -y -n rho python=3.12
conda activate rho
pip install -e rho_client
pip install -e ".[dev,test,server]"
```

The websocket client is also independently installable:

```bash
pip install -e rho_client
```

For supported NVIDIA GPUs, install FlashAttention separately:

```bash
pip install flash-attn==2.8.3 --no-build-isolation --no-cache-dir
```

## Docker

Build the base training image:

```bash
./docker/training/build.sh
```

Start an interactive container from the repository root:

```bash
./docker/training/interactive.sh
```

The launcher mounts the repository at `/workspace`. Set `RHO_DATA_DIR` to mount
a host dataset directory at `/data`:

```bash
RHO_DATA_DIR=/path/to/datasets ./docker/training/interactive.sh
```

## Checkpoints

Rho supports local checkpoint paths and Hugging Face repository IDs. Public
pretrained model checkpoints contain:

- Model configuration.
- Feature schema.
- Sharded safetensors weights.

Pretrained checkpoints may omit dataset configuration because finetuning
supplies it from the selected training YAML. Finetuned checkpoints contain the
dataset configuration and normalization statistics needed for evaluation.
Evaluation YAML files should not override `dataset` unless deliberately
evaluating with different preprocessing. Public model checkpoints do not
contain optimizer, scheduler, or other training state.

A full training checkpoint may additionally contain `training_state.pt` for
trusted local resume:

```text
checkpoint_step_0010000/
├── manifest.json
├── policy.json
├── features.json
├── model-00001-of-00003.safetensors
├── model-00002-of-00003.safetensors
├── model-00003-of-00003.safetensors
├── model.safetensors.index.json
└── training_state.pt
```

Rho downloads the default pretrained weights from the Hugging Face repository
configured by the policy. Set `pretrained_checkpoint` only to select a
different Hugging Face repository or a locally available checkpoint. Set
`resume=true` only when loading trusted training state.

## Configuration

Training and evaluation use YAML configuration files. Configurations can
compose shared files with `!include`, and any field can be overridden from the
command line with dotted names:

The shared `rho.train` entry point automatically selects single-process
training or the Accelerate implementation based on the launch environment.
Environment-specific training scripts register their adapters and then
delegate to this entry point, so use the corresponding script for LIBERO and
RoboEval configurations.

```bash
python environments/libero/train.py \
  --config_path=environments/libero/configs/train_libero_rho.yaml \
  --steps=10 \
  --batch_size=1 \
  --policy.num_flow_samples=1 \
  --wandb.enabled=false
```

For a distributed smoke test:

```bash
accelerate launch --multi-gpu \
  --num_processes=2 \
  environments/libero/train.py \
  --config_path=environments/libero/configs/train_libero_rho.yaml \
  --steps=10 \
  --batch_size=1 \
  --policy.num_flow_samples=1 \
  --wandb.enabled=false
```

## LIBERO

Build and launch the LIBERO image after building `rho-training:latest`:

```bash
./environments/libero/docker/build.sh
./environments/libero/docker/run_interactive.sh
```

Run a short single-GPU smoke test:

```bash
python environments/libero/train.py \
  --config_path=environments/libero/configs/train_libero_rho.yaml \
  --steps=10 \
  --batch_size=1 \
  --policy.num_flow_samples=1 \
  --wandb.enabled=false
```

### Reproducing the published LIBERO finetuning recipe

The canonical configuration trains for 40,000 optimizer steps with a global
effective batch size of 128. The published-result topology uses four H100
GPUs, per-device batch size 32, and no gradient accumulation:

```bash
accelerate launch --multi-gpu \
  --num_processes=4 \
  environments/libero/train.py \
  --config_path=environments/libero/configs/train_libero_rho.yaml
```

The effective batch size is:

```text
batch_size × number of processes × gradient_accumulation_steps
```

Users do not need a multi-GPU machine. On one GPU, `--batch_size=128` with no
accumulation is mathematically equivalent if the GPU has enough memory. In
practice, use a smaller per-device batch with gradient accumulation. For one
GPU with batch size 32 and accumulation 4:

```bash
accelerate launch \
  --num_processes=1 \
  environments/libero/train.py \
  --config_path=environments/libero/configs/train_libero_rho.yaml \
  --batch_size=32 \
  --gradient_accumulation_steps=4
```

If batch size 32 does not fit, use batch size 16 with accumulation 8, or batch
size 8 with accumulation 16. In general:

```text
gradient_accumulation_steps = 128 ÷ (batch_size × number of processes)
```

The canonical configuration also uses BF16, a learning rate of `1e-4`, 2,500
warmup steps, cosine decay to `5e-6` over 40,000 steps, and checkpoints every
5,000 steps. It trains with the vision and language backbone unfrozen, predicts
16-step action chunks, executes 8 actions per inference, and draws 8 flow
samples per training example (`policy.num_flow_samples=8`). The flow-sample
setting matches the published recipe but is not required for ordinary
finetuning, which can use the policy default of 1. On an H100, the published
global batch of 128 with 8 flow samples takes approximately 7.2 seconds per
optimizer step; exact throughput depends on hardware and launch topology.
A checkpoint for this model is roughly 10.5 GB for model weights alone or
20.6 GB when it also retains the optimizer and scheduler state required to
resume training. Plan output storage accordingly.

On Python 3.12, worker startup may print a warning about forking a
multithreaded process. The validated container completed normally; if worker
startup stalls on another host, use `--num_workers=0`.

Run evaluation separately and explicitly select the resulting finetuned
checkpoint. Leaving `pretrained_checkpoint` unset would evaluate the hosted
base checkpoint instead:

```bash
python environments/libero/eval.py \
  --config_path=environments/libero/configs/eval_libero_rho.yaml \
  --pretrained_checkpoint=/path/to/training-run/checkpoints/checkpoint_step_0040000
```

For a short four-suite validation, run one episode for each of the 10 tasks in
each suite, for 40 episodes total:

```bash
python environments/libero/eval.py \
  --config_path=environments/libero/configs/multieval_libero_rho_smoke.yaml \
  --pretrained_checkpoint=/path/to/training-run/checkpoints/checkpoint_step_0040000
```

The full published evaluation runs 50 episodes for each of 10 tasks in each
suite: 500 episodes per suite and 2,000 episodes total.

```bash
python environments/libero/eval.py \
  --config_path=environments/libero/configs/multieval_libero_rho_50.yaml \
  --pretrained_checkpoint=/path/to/training-run/checkpoints/checkpoint_step_0040000
```

## RoboEval

See [`environments/roboeval/README.md`](environments/roboeval/README.md) for
container setup, dataset layout, finetuning, and checkpoint evaluation
instructions for end-effector and joint-position policies.

## Serving

The policy server communicates with robot or simulator clients over websockets.
Install the server dependencies and the standalone client:

```bash
pip install -e rho_client
pip install -e ".[server]"
```

Environment integrations provide the observation and action conversion needed
between a policy server and a robot or simulator client.

## FlowDAgger

FlowDAgger wraps a frozen Rho policy with a deterministic noise policy. Human
interventions are inverted through Rho's flow model, and the noise policy is
updated online with supervised regression. FlowDAgger uses the HIL transport
and trainer modules under `rho/hil` and the selected implementation under
`rho/policies/dsrl`.

The public workflow requires a Rho checkpoint, dataset-specific preprocessing,
and a robot-side client that publishes intervention transitions.

## Dataset utilities

Validate that a training configuration can construct and sample its dataset:

```bash
python -m rho.utils.check_dataset \
  --config_path=environments/libero/configs/train_libero_rho.yaml
```

Recompute LeRobot statistics directly from parquet files:

```bash
python -m rho.utils.recompute_lerobot_stats_parquet \
  --dataset_path=/path/to/lerobot_dataset \
  --stats_type=quantile
```

## Development

Run the test suite:

```bash
pytest
```

Run static checks:

```bash
ruff check rho tests environments
```

The repository is organized around:

```text
rho/                 Core package
rho_client/          Standalone websocket client
config/              Shared training and dataset configurations
environments/libero  LIBERO integration
environments/roboeval RoboEval integration
docker/training      Base training container
tests/               Unit and integration tests
```

## License

This project is released under the MIT License.

## Contributing

This project welcomes contributions and suggestions. Most contributions
require you to agree to a Contributor License Agreement (CLA) declaring that
you have the right to, and actually do, grant us the rights to use your
contribution. For details, visit
[Contributor License Agreements](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether
you need to provide a CLA and decorate the pull request appropriately. Follow
the instructions provided by the bot. You only need to do this once across all
repositories using the CLA.

This project has adopted the
[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information, see the
[Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com).

## Trademarks

This project may contain trademarks or logos for projects, products, or
services. Authorized use of Microsoft trademarks or logos is subject to and
must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must
not cause confusion or imply Microsoft sponsorship. Any use of third-party
trademarks or logos is subject to those third parties' policies.
