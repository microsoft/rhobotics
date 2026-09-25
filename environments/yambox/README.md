# Yambox (BusyBox) finetuning

This directory contains a Rho training configuration for the **Yambox / BusyBox**
manipulation task on the [I2RT YAM](https://github.com/i2rt-robotics) dual-arm
robot (30 Hz, end-effector control). It reproduces the recipe we use internally
for the Yambox models, using only the public Rho training code and a publicly
hosted dataset.

## Download the training dataset

The configuration trains on the LeRobot dataset hosted at
[`microsoft/BusyBox`](https://huggingface.co/datasets/microsoft/BusyBox), under
the `I2RT_YAM_Box/` subtree. Download that subtree into your `HF_HOME`:

```bash
export HF_HOME=/path/to/large/storage/huggingface
mkdir -p "$HF_HOME/datasets/microsoft/BusyBox"

hf download microsoft/BusyBox \
  --repo-type dataset \
  --include "I2RT_YAM_Box/**" \
  --local-dir "$HF_HOME/datasets/microsoft/BusyBox"
```

After downloading, the LeRobot dataset root is:

```text
$HF_HOME/datasets/microsoft/BusyBox/I2RT_YAM_Box/
├── data/
├── meta/
└── videos/
```

This is the path referenced by `configs/yambox_dataset.yaml`
(`root_dir: "${HF_HOME}/datasets/microsoft/BusyBox/I2RT_YAM_Box"`), so make sure
`HF_HOME` is exported in the shell you launch training from. The dataset has
**2,766 episodes / ~606k frames at 30 Hz** (≈5.6 hours).

## Training

We train the Yambox models on **4× H100 (80 GB) GPUs**. Launch a full run with
`accelerate`:

```bash
export HF_HOME=/path/to/large/storage/huggingface

accelerate launch --multi-gpu \
  --num_processes=4 \
  environments/yambox/train.py \
  --config_path=environments/yambox/configs/train_yambox_rho.yaml
```

For a quick single-GPU smoke test (override the step count / batch size):

```bash
python environments/yambox/train.py \
  --config_path=environments/yambox/configs/train_yambox_rho.yaml \
  --steps=200 --batch_size=1
```

Checkpoints are written under `output_dir` (default `outputs/yambox_busybox_rho`).

### Recipe summary (`configs/train_yambox_rho.yaml`)

| Setting | Value |
|---|---|
| Policy | `rho` (published base model), fully unfrozen |
| Action chunk | `chunk_size=32`, `n_action_steps=32` |
| Flow matching | `num_flow_samples=8` |
| Dropout | `0.1` |
| LR schedule | cosine, warmup 2500, peak `1e-4` → `2.5e-6` over 50k |
| Steps | 50,000 |
| Precision | bf16, `grad_clip_norm=1.0` |
| Global batch | `batch_size 16 × 4 GPUs × grad_accum 2 = 128` |

### Data recipe (`configs/yambox_dataset.yaml`)

- **Action space:** dual-arm end-effector `EE_QUAT_POS_XYZW` → converted to 6D
  rotation and trained as **state-relative deltas with absolute grippers**.
- **Normalization:** `QUANTILE` (state) / `ACTIONCHUNK_QUANTILE` (action) — the
  dataset's `meta/stats.json` already contains the required quantiles.
- **Image augmentation (per camera):** center-crop 480 → resize-pad 448 →
  random-resized-crop 448 (scale [0.9,1.0], ratio [0.98,1.02]) → color-jitter
  (brightness 0.15, contrast/saturation [0.85,1.15], no hue) → resize-pad 256.
- **Cameras:** `cam_scene` → image.0, `cam_left_wrist` → image.1,
  `cam_right_wrist` → image.2.

## Notes

- The config uses Rho's **hosted pretrained checkpoint by default**. To start
  from a different base (or resume a full training checkpoint), pass
  `--pretrained_checkpoint=<repository-or-path>`.
- **Fewer/more GPUs:** keep the reference global batch (128) by adjusting
  `--gradient_accumulation_steps` and `--num_processes`. For example, on 8 GPUs
  set `--gradient_accumulation_steps=1`; on 1 GPU raise it to `8` (and expect a
  proportionally longer wall-clock).
- Full finetuning (unfrozen VLM backbone) is memory-heavy; if you hit OOM, lower
  `--batch_size` and raise `--gradient_accumulation_steps`, or freeze the
  backbone with `--policy.freeze_vlm_backbone=true --policy.train_expert_only=true`.
