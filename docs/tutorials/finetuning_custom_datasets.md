# Finetuning on a Custom Dataset

This tutorial covers the steps necessary to finetune Rho on a custom dataset.
Rho uses the hosted checkpoint configured by `RhoConfig.pretrained_repo_id` by
default, so `pretrained_checkpoint` is optional.

## Choosing a starting checkpoint

The default pretrained checkpoint is
[`microsoft/rho-base`](https://huggingface.co/microsoft/rho-base). We also
provide robot-specific midtrained models for the following platforms:

| Robot platform | Midtrained checkpoint |
| --- | --- |
| [UR AI Trainer](https://www.universal-robots.com/products/ur-ai-trainer/) | [`microsoft/rho-ur-ai-trainer`](https://huggingface.co/microsoft/rho-ur-ai-trainer) |
| [Franka FR3 Duo](https://franka.de/fr3-duo) | [`microsoft/rho-fr3-duo`](https://huggingface.co/microsoft/rho-fr3-duo) |
| [YAM Box](https://i2rt.com/products/yam-box) | [`microsoft/rho-yam-box`](https://huggingface.co/microsoft/rho-yam-box) |

Use the checkpoint for your platform as a starting point for task-specific
finetuning. Set the top-level fields in your training YAML, for example:

```yaml
pretrained_checkpoint: microsoft/rho-ur-ai-trainer
resume: false
```

Alternatively, pass `--pretrained_checkpoint=microsoft/rho-ur-ai-trainer`
when launching training. Keep `resume=false` to start fresh training state.
Configure your dataset's observations, actions, and normalization as described
below, and match any policy architecture overrides to the selected checkpoint.

## 1. Preparing the dataset

### 1a. LeRobot V3 dataset format

The Rho library uses the LeRobot v3 data format for training. One advantage of our dataset format is that we do not require specific key names — we can define a key remapping when loading the data. There are two caveats:

- There must be separate keys for observations and actions.
- For best results, all action keys should use the `action.` prefix in order to take full advantage of the action chunk behavior available in the LeRobot dataset class.

### 1b. Computing custom stats

We use the LeRobot v3 format for our stats files and can use the normalization stats for any LeRobot dataset without alteration. In addition to the standard options provided in LeRobot, we also support `ACTIONCHUNK` normalization. This form of normalization works best when working with longer action chunks and is recommended for most systems.

To compute custom stats you must complete the following steps:

**A. Create an `action_mapping.yaml` file.**

This file defines the relationship between observations and actions in the target dataset and is used to compute observation-relative action chunk statistics. For example, for a bimanual dataset with six joints followed by one gripper channel per arm:

```yaml
action_eef:
  state_key: observation.state_eef
  action_type: EE_EULER_POS
action_original:
  state_key: observation.state_original
  action_type: POSITION
  absolute_idx: [6, 13]
```

**B. Run the stats computation script.**

```bash
python rho/utils/recompute_lerobot_stats_parquet.py \
    --dataset_path /path/to/your/dataset \
    --stats_type both \
    --action_mapping config/datasets/action_mapping.yaml \
    --output_path config/datasets/your_dataset_stats
```

Once you have created a new stats file you can either overwrite the original stats file or save it in a separate directory. It is safe to overwrite the original stats file since all the original statistics are preserved — this script only appends new fields in the same format that is ignored by the standard LeRobot dataloader. If you save the statistics in a separate file you can use the `dataset.stats` parameter in your configuration file to load them separately.

Chunk statistics preserve end-effector gripper channels as absolute values by
default, matching `delta_actions.use_absolute_grippers: true`. For a dataset
that explicitly uses delta grippers, pass `--no-use_absolute_grippers` to the
stats command and set `use_absolute_grippers: false` in the transform. Existing
statistics must match the selected convention; changing defaults does not
regenerate them.

`absolute_idx` is optional and applies independently to each action key in the
mapping. It preserves the listed channels even with
`--no-use_absolute_grippers`; omit it or use `[]` to select no additional
absolute channels. Both stats scripts honor it. Indices are zero-based and
refer to the raw action vector in the dataset, not observation-remapped keys
or a later converted representation. Match these selections in the training
transform below. Regenerate chunk statistics, including first-pass bounds,
when changing the selections; do not reuse bounds from the old convention.

## 2. Create your dataset configuration file

Here is an example of a minimal dataset configuration file that contains all required fields:

```yaml
repo_id: "my_new_dataset"  # An arbitrary name that doesn't exist on HuggingFace
root_dir: /path/to/my/dataset  # Load the dataset from a local directory
num_workers: 4  # Number of parallel dataloading workers

observation_mapping:
  image: observation.image.0         # Rho uses the 'observation.image' prefix for images
  wrist_image: observation.image.1   # Second camera maps to observation.image.1
  actions: action                    # Rho expects a single action key called 'action'
  state: observation.state           # Rho expects proprioceptive state as 'observation.state'
  prompt: task                       # Rho expects the language instruction as 'task'

features:  # Uses the post-mapping key names
  action:
    shape: (2,)          # Post-transform shape
    type: ACTION
  observation.image.0:
    shape: (3, 96, 96)   # Post-transform shape
    type: VISUAL
  observation.image.1:
    shape: (3, 96, 96)
    type: VISUAL
  observation.state:
    shape: (2,)
    type: STATE

normalization_mapping:
  ACTION: MIN_MAX
  ENV: MIN_MAX
  STATE: MIN_MAX
  VISUAL: MEAN_STD
```

At minimum the user must provide a `repo_id` and a `root_dir`. We always use local copies of datasets when possible, so we typically set `repo_id` to an arbitrary string that does not exist on HuggingFace (to prevent it from downloading something unintended if it can't find your `root_dir`), with `root_dir` set to a folder on your local system. In most example training commands you will see `--dataset.root_dir=` overridden on the command line, as the mounted data directory may change depending on how your server or container is configured.

> **Most common failure:** The most common failure when launching training is an error about not being able to parse the `DatasetConfig`/`MultiDatasetConfig` or download the HuggingFace dataset. In the vast majority of cases this is because the `root_dir` cannot be found. Check that path first.

While it is possible to load a dataset with only `repo_id` and `root_dir` defined, we recommend that users define the following parameters for greater control:

- **`observation_mapping`** — A dictionary mapping between the original field names in the LeRobot dataset and the names used for training. The keys are the original dataset names and the values are the names given to the policy. Rho expects all images to use the `observation.image` prefix (e.g. `observation.image.0`, `observation.image.1`), proprioceptive states to use `observation.state`, actions to use `action`, and language instructions to use `task`.

- **`features`** — Overwrites the default features dictionary in the LeRobot metadata. Each entry is keyed by the post-mapping name and defines the tensor shape and its type (`ACTION`, `STATE`, `VISUAL`). The type determines which form of normalization is applied.

- **`normalization_mapping`** — Defines the type of normalization applied to each feature type. We support all the same normalizations as a standard LeRobot dataset. There are also special normalizations for `ACTION` type features that use the `ACTIONCHUNK_` prefix — these are used in conjunction with state-relative delta actions and require custom stats computed via the script in section 1b.


### Additional dataset parameters

There are three additional parameters that we use for our hardware demonstrations:

```yaml
stats: !include my_custom_stats.json  # Import custom stats computed in step 1b

action_type: EE_EULER_POS  # Required if using the convert_to_6d_actions transform

transform_mapping:
  "action":
    - type: convert_to_6d_actions
      action_key: "observation.state"
      action_type: "EE_EULER_POS"
      post_norm: false

    - type: convert_to_6d_actions
      action_key: "action"
      action_type: "EE_EULER_POS"
      post_norm: false

    - type: delta_actions
      action_type: "EE_6D_POS"
      action_key: "action"
      state_key: "observation.state"
      relative_to_state: true
      use_absolute_grippers: true
      post_norm: false

  "observation.image.0":
    - type: random_resized_crop
      height: 224
      width: 224
      scale: [0.9, 0.9]
      ratio: [1.0, 1.0]

    - type: color_jitter
      brightness: 0.2
      contrast: [0.8, 1.2]
      saturation: [0.8, 1.2]
      hue: 0.05
```

- **`stats`** — Overwrites the stats included in the LeRobot dataset. This is useful for replacing the default stats with ones you computed yourself, such as when you have ground truth stats defined from a larger dataset than what you are finetuning against, or when you have used our scripts to compute custom `ACTIONCHUNK` stats. Our configuration file parsing supports loading stats directly using a draccus-compatible `!include` statement.

- **`action_type`** — Defines the action space you will be training against (for options see `rho.common.types.ActionType`). The default is `POSITION`, a generic format used for joint positions and other spaces. If you want to use end effector control and its associated transforms, select the correct end effector action type (e.g. `EE_EULER_POS`, `EE_6D_POS`).

- **`transform_mapping`** — Allows users to apply transforms from `rho.common.transforms` to different features in the dataset. Transforms are keyed by the post-mapping feature name and are applied in the order listed. We cover the two main categories below.

#### Action transforms

Our model is pretrained using a bimanual end effector action space with orientation represented in a 6D format. For best results when using end effector space, we use the `convert_to_6d_actions` transform to convert from the current action space into the 6D representation. When you serve the policy, any transforms defined here are automatically reversed (including denormalization) to return actions in the original space found in the dataset. Converting to 6D actions also has the secondary effect of deactivating normalization for the orientation features — we therefore use it even when the original data is already in `EE_6DOF` format.

For datasets with absolute end-effector targets, we recommend the
`delta_actions` transform with matching `ACTIONCHUNK_` statistics. Its defaults
are `relative_to_state: true`, `use_absolute_grippers: true`, and
`post_norm: false`: poses are relative to the current observation, grippers
remain absolute, and conversion happens before normalization.
`convert_to_6d_actions` also defaults to `post_norm: false`.
All these flags remain explicitly overridable; use
`relative_to_state: false` for temporal differences and
`use_absolute_grippers: false` for delta grippers. Generated inverse transforms
use the same settings.

For `action_type: POSITION`, joint/gripper layout cannot be inferred. Specify
`absolute_idx` to keep selected channels absolute while making the remaining
channels relative. For the six-joint-per-arm bimanual layout in the stats
example above:

```yaml
transform_mapping:
  action:
    - type: delta_actions
      action_type: POSITION
      absolute_idx: [6, 13]
```

`absolute_idx` defaults to `null` (no explicit selection) and is propagated to
the generated `absolute_actions` inverse. It works with both state-relative
and temporal deltas and is independent of `use_absolute_grippers`. For EEF
actions, it adds to the automatically selected gripper channels unless that
flag is disabled. To make all channels relative, use
`use_absolute_grippers: false` with `absolute_idx: []`.

Use non-negative integer indices within the action width **at the point
`delta_actions` runs**. For example, converting a single-arm quaternion EEF
vector to 6D moves its gripper from index 7 to index 9, so explicit indices in
the raw stats mapping and converted training transform differ. Automatic EEF
gripper selection already accounts for this change. For rotation-bearing
action types, select an entire rotation block or none of it; partial
quaternion, Euler, or 6D selections are rejected because they are not
independently invertible. The example above is not a universal gripper
layout; existing robot YAML recipes are unchanged.

These transforms are not automatically added to datasets. In particular, do
not add the absolute-EEF conversion chain to the standard LIBERO controller
actions. Observation-relative actions can also drift under out-of-distribution
states, so the action convention must match the system.

#### Image transforms

Image transforms are applied per-camera by keying on the post-mapping image name (e.g. `observation.image.0`). If you want the same transforms on multiple cameras, define entries for each one. The following image transforms are available:

| Transform | Description | Key parameters |
|-----------|-------------|----------------|
| `random_resized_crop` | Randomly crops a region and resizes to a target size. Useful for spatial augmentation. | `height`, `width`, `scale` (crop area range), `ratio` (aspect ratio range) |
| `color_jitter` | Randomly adjusts brightness, contrast, saturation, and hue. | `brightness`, `contrast`, `saturation`, `hue` |
| `center_crop` | Deterministic center crop to a target size. | `height`, `width` |
| `resize_with_padding` | Resizes the image while preserving aspect ratio, padding the shorter dimension with zeros. | `height`, `width`, `mode` (interpolation) |
| `random_flip_left_right` | Randomly flips the image horizontally. | `p` (probability, default 0.5) |
| `random_flip_up_down` | Randomly flips the image vertically. | `p` (probability, default 0.5) |
| `random_rot90` | Randomly rotates the image by 90-degree multiples. | `p` (probability, default 0.5) |
| `channel_reorder` | Reorders image channels (e.g. BGR to RGB). | (see `rho.common.transforms`) |

When explicitly selected, `random_resized_crop` defaults to
`scale: [0.9, 1.0]` and `ratio: [0.98, 1.02]`; `height` and `width` remain
required. Override these ranges for the camera's field of view, or choose
`resize_with_padding` to preserve the full frame. No image transform is
automatically inserted into the dataset pipeline. Evaluation/serving uses the
deterministic centered counterpart of a configured random crop and drops color
jitter. Remember that the `shape` in your `features` block must reflect the
post-transform image dimensions.


### Using MultiDatasetConfig

Rho supports training on multiple datasets simultaneously using the `MultiDatasetConfig`. This is useful when you want to co-train on data from different robots, environments, or task distributions. Rather than referencing a single dataset config with `!include`, you define a `datasets` list directly in your training config where each entry contains a `dataset` block and a sampling `weight`.

Here is an example that combines two datasets:

```yaml
dataset:
  datasets:
    - dataset:
        repo_id: "my_robot_a_data"
        root_dir: /data/robot_a/
        observation_mapping:
          "observations.joint_position": "observation.state"
          "observations.images.cam_high": "observation.image.0"
          "observations.images.cam_wrist": "observation.image.1"
          "action.joint_position": "action"
      weight: 1.0

    - dataset:
        repo_id: "my_robot_b_data"
        root_dir: /data/robot_b/
        observation_mapping:
          "observation.images.image": "observation.image.0"
          "observation.images.wrist_image": "observation.image.1"
        <<: !include my_robot_b_features.yaml  # Merge in features/normalization/stats
      weight: 0.5

  # Top-level features define the unified feature space across all datasets.
  # Shapes must be the superset — shorter state/action vectors are zero-padded.
  features:
    observation.state:
      type: STATE
      shape: (28,)
    observation.image.0:
      type: VISUAL
      shape: (3,256,256)
    observation.image.1:
      type: VISUAL
      shape: (3,256,256)
    action:
      type: ACTION
      shape: (28,)
```

#### Key points

- **`weight`**: Controls the relative sampling frequency. A dataset with `weight: 2.0` will be sampled twice as often as one with `weight: 1.0`. Weights do not need to sum to 1 — they are normalized internally.

- **Top-level `features`**: When combining datasets with different state/action dimensions, define a top-level `features` block on the `MultiDatasetConfig` that represents the unified (padded) feature space. Shorter state and action vectors from individual datasets are automatically zero-padded to match `max_state_dim` and `max_action_dim` in the policy config.

- **Per-dataset features and stats**: Each dataset entry can include its own `features`, `normalization_mapping`, and `stats` (via `<<: !include`). These are used for per-dataset normalization before padding to the unified shape.

- **`observation_mapping`**: Each dataset defines its own observation mapping to remap its native keys to the shared key names (e.g. `observation.image.0`, `observation.state`, `action`). Use numbered image keys (`observation.image.0`, `observation.image.1`, etc.) to align cameras across datasets.

- **`observation_whitelist`**: When combining datasets with different sets of keys, you can add an `observation_whitelist` at the multi-dataset level to explicitly list which keys should be passed to the policy. This prevents extraneous keys from one dataset from causing errors.

- **Nesting**: `MultiDatasetConfig` supports nesting — a dataset entry can itself be another `MultiDatasetConfig`. Nested configs are flattened by default (`flatten_nested: true`), and their weights are normalized proportionally.

#### Example configs

The repository includes working multi-dataset configs you can use as references:

- **`config/datasets/multidataset.yaml`** — A minimal example showing how to combine multiple weighted datasets using `!include` for each entry:
  ```yaml
  datasets:
    - dataset: !include pusht.yaml
      weight: 1.5
    - dataset: !include pusht.yaml
      weight: 0.25
  ```

- **`environments/roboeval/configs/ee_6d_pos/roboeval_multidataset.yaml`** — A
  RoboEval example combining eight tasks with equal sampling weights. Each
  dataset merges shared features with `<<: !include features.yaml` and uses
  `${ROBOEVAL_DATA_ROOT}` for its dataset root. The configuration also defines
  an `observation_whitelist` for the shared model inputs.


## 3. Create a training configuration file

The training configuration file ties together the dataset config, the policy config, and the training hyperparameters. Training configs use YAML with `!include` directives to reference your dataset configuration. Any field can be overridden from the command line using dot notation (e.g. `--policy.embed_dim=2048`).

Here is a minimal Rho finetuning configuration. When
`pretrained_checkpoint` is unset, training downloads the policy's configured
hosted default.

```yaml
# my_custom_train.yaml

# WandB logging configuration
wandb:
  project: "my_project"
  enabled: false  # Set to true once you have WandB configured

# Dataset configuration - include the file you created in step 2
dataset: !include "my_custom_dataset.yaml"

# Policy configuration
policy:
  type: "rho"

# Training parameters
batch_size: 4
num_workers: 4
learning_rate: 1e-4
steps: 100000

# Checkpoint and output settings
output_dir: "outputs/my_custom_training"
save_checkpoint_every: 1000
keep_checkpoint_interval: 10000
logging_interval: 200
mixed_precision: "bf16"     # Default; alternatives: "no", "fp16"
grad_clip_norm: 10.0        # Default; null disables clipping
gradient_accumulation_steps: 1

# Evaluation settings
eval_interval: 10000
record_videos: false        # Set to true if you have a simulation environment

# Resume / pretrained checkpoint
resume: false
pretrained_checkpoint: null  # Optional override; null uses Rho's hosted default
```

### Key Parameters

- **`seed`**: Seeds Python, NumPy, and PyTorch. Strict mode also propagates it
  to dataset samplers, including children of dataset mixtures.
- **`deterministic_training`**: Opt-in single-process strict-kernel mode.
  Set `num_workers: 0` for every active dataloader; incompatible worker counts
  now raise an error instead of being silently overridden. Loader RNGs remain
  separate from the model RNG, and TF32 stays disabled for regression precision.
  See `environments/libero/configs/train_libero_rho_deterministic.yaml`.
- **`batch_size`**: This is the per-GPU batch size and overwrites the
  placeholder value in the dataset config. Effective batch size is
  `batch_size × number of processes × gradient_accumulation_steps`.
- **`gradient_accumulation_steps`**: Both the plain Python and Accelerate
  trainers accumulate this many microbatches per update. Step counts,
  scheduler updates, and checkpoint intervals use optimizer-update boundaries.
- **`mixed_precision`**: Defaults to `"bf16"`; `"no"` or `"fp16"` explicitly overrides it. Controls training
  autocast independently of the policy's stored parameter dtype. The plain
  trainer saves and restores its FP16 loss scaler with training state.
- **`grad_clip_norm`**: Defaults to `10.0`. Clip accumulated gradients once per update, after
  unscaling when using FP16. Set to `null` to disable clipping.
- **`learning_rate`**: This is overwritten by the policy. Ignore it.
- **`save_checkpoint_every`**: This determines how frequently checkpoints are created.
- **`keep_checkpoint_interval`**: If set all checkpoints that are not modulo this interval are deleted when the next chekpoint is created.
- **`resume`**: Restore trusted optimizer, scheduler, sampler, and step state
  from a full training checkpoint. Leave this false for ordinary finetuning.
  New sampler states preserve the next yielded index, including child
  positions and dataset-selection RNG state for multi-dataset sampling.
  Older states without positions or usable RNG state warn when exact sampler
  continuation is unavailable. Sampler state does not include batches already
  prefetched by dataloader workers, Accelerate, or lookahead monitors, nor
  arbitrary iterable-dataset state; it is not by itself a guarantee of
  identical training continuation.
- **`pretrained_checkpoint`**: Optional Hugging Face repository ID or local
  checkpoint override. When unset, Rho uses `policy.pretrained_repo_id`.

#### Policy Parameters
- **`policy.type`**: Use `"rho"` for the public Rho policy.
- **`policy.pretrained_repo_id`**: Hosted checkpoint used when
  `pretrained_checkpoint` is unset.
- **`policy.embed_dim`**: The action expert embedding dimension. Use `1024` for a smaller model that fits on ~20 GB VRAM GPUs (e.g. RTX 4090). Use `2048` for larger models that require at least 40 GB VRAM (e.g. A100).
- **`policy.attention_type`**: The cross-attention mechanism. `layerwise_cross` provides the best performance for language-rich tasks but is slower to train.
- **`policy.chunk_size`**: The total length of the action sequence predicted at each step. This value is automatically propagated to the dataset config.
- **`policy.num_flow_samples`**: Defaults to `8` time/noise samples per
  observation, reusing the VLM context. Set to `1` for lower action-expert memory
  and compute cost. This does not multiply the number of distinct observations
  in the batch.
- **`pretrained_checkpoint`**: Use this only to override the hosted default
  with another Hugging Face repository or a local checkpoint.

### Matching Pretrained Checkpoint Parameters

When overriding the hosted default, the policy architecture must match the
checkpoint. Portable checkpoints include `policy.json` and `features.json`;
verify any explicitly overridden architecture fields against that metadata.





## 4. Running training

### Single-GPU Training

For training with a custom dataset (not tied to a specific simulation environment), use the generic training entry point:

```bash
python -m rho.train \
  --config_path=configs/my_custom_train.yaml \
  --wandb.enabled=false \
  --batch_size=4 \
  --steps=50000
```

If you are working within a specific environment such as Libero, you can use the environment-specific training script instead:

Note that you only need to use the custom entrypoint if you are attempting to perform inline evaluation.

```bash
python environments/libero/train.py \
  --config_path=environments/libero/configs/my_custom_train.yaml \
  --wandb.enabled=false \
  --batch_size=4 \
  --steps=50000
```

Any config field can be overridden from the command line using dot notation. For example:

```bash
--policy.embed_dim=1024 \
--policy.train_expert_only=true \
--dataset.root_dir=/data/my_dataset
```

### Multi-GPU Training

For multi-GPU training with the generic entry point, use `accelerate launch`;
`rho.train` selects the Accelerate implementation automatically:

```bash
export NUM_GPU=2
accelerate launch --multi-gpu \
  --num_processes=${NUM_GPU} \
  -m rho.train \
  --config_path=configs/my_custom_train.yaml \
  --wandb.enabled=false
```

Environment-specific training scripts (like `environments/libero/train.py`) automatically detect whether they are launched under accelerate and will switch to the accelerate training path:

```bash
export NUM_GPU=2
accelerate launch --multi-gpu \
  --num_processes=${NUM_GPU} \
  environments/libero/train.py \
  --config_path=environments/libero/configs/my_custom_train.yaml
```

### Resuming Training

There are three checkpoint-selection modes:

- **Hosted-default finetuning** (`resume=false`, `pretrained_checkpoint=null`):
  Downloads `policy.pretrained_repo_id`, loads model weights, and starts fresh
  optimizer, scheduler, and step state.

- **Explicit-source finetuning** (`resume=false`,
  `pretrained_checkpoint=...`): Overrides the hosted default with another
  Hugging Face repository or local checkpoint while starting fresh training
  state.

- **Resuming** (`resume=true`, `pretrained_checkpoint=...`): Restores full
  training state. Only resume from a trusted checkpoint containing
  `training_state.pt`.

You can also use the `run_name` parameter for automatic resume behavior:

```bash
python -m rho.train \
  --config_path=configs/my_custom_train.yaml \
  --run_name=my_experiment
```

When `run_name` is set, the output directory becomes `<output_dir>/<run_name>/checkpoints`. If an existing checkpoint is found in that folder, the job will automatically resume from it without needing to specify `--resume=true` or `--pretrained_checkpoint`.


## 5. Checkpoints and monitoring

### Checkpoint Directory Structure

Training outputs are organized under `output_dir` with either a timestamp or `run_name`:

```
output_dir/
  0604_1430/                    # Timestamp-based (default)
    train_config.json           # Full training config for reference
    checkpoints/
      checkpoint_step_0010000/  # Versioned portable checkpoint bundle
```

Or with `run_name`:

```
output_dir/
  my_experiment/
    train_config.json
    checkpoints/
      checkpoint_step_0010000/
```

- **`save_checkpoint_every`**: How often to save a checkpoint (in steps). These are rolling — older ones are overwritten unless they fall on a `keep_checkpoint_interval` boundary.
- **`keep_checkpoint_interval`**: Checkpoints on these step boundaries are kept permanently.
- **`train_config.json`**: Saved alongside the checkpoints directory. This file records the full training configuration and is used by evaluation and serving utilities to reconstruct the policy and dataset settings.

### WandB Monitoring

When WandB is enabled (`--wandb.enabled=true`), training metrics (loss, learning rate, gradient norms) are logged at each `logging_interval`. Make sure the `WANDB_API_KEY` and `WANDB_BASE_URL` environment variables are set.


## 6. Preflight checklist

Before launching training, verify the following:

- [ ] **`root_dir` is accessible** — The dataset `root_dir` must be reachable from within your container or environment. This is the most common failure. If running in Docker, ensure the data directory is mounted correctly.
- [ ] **`repo_id` does not match a real HuggingFace dataset** — When using a local dataset, set `repo_id` to an arbitrary string that doesn't exist on HuggingFace to prevent accidental downloads.
- [ ] **`features` shapes match post-transform dimensions** — The shape entries in your dataset config must reflect the final shapes after any observation mapping and transforms are applied.
- [ ] **Policy parameters match an explicit checkpoint override** — If
  overriding the hosted default, check architecture fields against the
  checkpoint's `policy.json`.
- [ ] **Custom stats file is reachable** — If you use a separate stats file via the `stats` field, ensure it is accessible from your training environment.
- [ ] **VRAM is sufficient** — `embed_dim=1024` typically fits on 20 GB GPUs; `embed_dim=2048` requires at least 40 GB. Reducing `batch_size` or enabling `gradient_accumulation_steps` can help if you are close to the limit.
