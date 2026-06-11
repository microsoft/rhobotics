# Finetuning on a Custom Dataset

This tutorial covers the steps necessary to finetune RhoAlpha on a custom dataset. These instructions assume you will be using the Phi5 backbone with a pretrained checkpoint.

## 1. Preparing the dataset

### 1a. LeRobot V3 dataset format

The Rho library uses the LeRobot v3 data format for training. One advantage of our dataset format is that we do not require specific key names — we can define a key remapping when loading the data. There are two caveats:

- There must be separate keys for observations and actions.
- For best results, all action keys should use the `action.` prefix in order to take full advantage of the action chunk behavior available in the LeRobot dataset class.

### 1b. Computing custom stats

We use the LeRobot v3 format for our stats files and can use the normalization stats for any LeRobot dataset without alteration. In addition to the standard options provided in LeRobot, we also support `ACTIONCHUNK` normalization. This form of normalization works best when working with longer action chunks and is recommended for most systems.

To compute custom stats you must complete the following steps:

**A. Create an `action_mapping.yaml` file.**

This file defines the relationship between observations and actions in the target dataset and is used to compute observation-relative action chunk statistics. Here is an example:

```yaml
action_eef:
  state_key: observation.state_eef
  action_type: EE_EULER_POS
action_original:
  state_key: observation.state_original
  action_type: POSITION
```

**B. Run the stats computation script.**

```bash
python rho/utils/recompute_lerobot_stats_parquet.py \
    --dataset_path /path/to/your/dataset \
    --stats_type both \
    --action_mapping config/datasets/your_dataset_config.yaml \
    --output_path config/datasets/your_dataset_stats
```

Once you have created a new stats file you can either overwrite the original stats file or save it in a separate directory. It is safe to overwrite the original stats file since all the original statistics are preserved — this script only appends new fields in the same format that is ignored by the standard LeRobot dataloader. If you save the statistics in a separate file you can use the `dataset.stats` parameter in your configuration file to load them separately.

## 2. Create your dataset configuration file

Here is an example of a minimal dataset configuration file that contains all required fields:

```yaml
repo_id: "my_new_dataset"  # An arbitrary name that doesn't exist on HuggingFace
root_dir: /path/to/my/dataset  # Load the dataset from a local directory
num_workers: 4  # Number of parallel dataloading workers

observation_mapping:
  image: observation.image.0         # RhoAlpha expects the 'observation.image' prefix for images
  wrist_image: observation.image.1   # Second camera maps to observation.image.1
  actions: action                    # RhoAlpha expects a single action key called 'action'
  state: observation.state           # RhoAlpha expects proprioceptive state as 'observation.state'
  prompt: task                       # RhoAlpha expects the language instruction as 'task'

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

- **`observation_mapping`** — A dictionary mapping between the original field names in the LeRobot dataset and the names used for training. The keys are the original dataset names and the values are the names given to the policy. RhoAlpha expects all images to use the `observation.image` prefix (e.g. `observation.image.0`, `observation.image.1`), proprioceptive states to use `observation.state`, actions to use `action`, and language instructions to use `task`.

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

We also recommend using the `delta_actions` transform. This leads to the policy learning more precise control than learning absolute positions, although it can lead to drift when encountering out-of-distribution states. This transform must be matched with `ACTIONCHUNK_` statistics for the best performance.

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

For most finetuning tasks we recommend at minimum a `random_resized_crop` with a narrow scale range (e.g. `[0.9, 0.9]`) to add slight spatial variation, paired with a mild `color_jitter` to improve robustness to lighting changes. Use `center_crop` or `resize_with_padding` instead of `random_resized_crop` if you do not want random augmentation (e.g. during evaluation). Remember that the `shape` in your `features` block must reflect the post-transform image dimensions.


### Using MultiDatasetConfig

RhoAlpha supports training on multiple datasets simultaneously using the `MultiDatasetConfig`. This is useful when you want to co-train on data from different robots, environments, or task distributions. Rather than referencing a single dataset config with `!include`, you define a `datasets` list directly in your training config where each entry contains a `dataset` block and a sampling `weight`.

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

- **`config/datasets/aloha_taskbox.yaml`** — A real-world example that combines four Aloha taskbox datasets. It demonstrates using `<<: !include` to merge a shared base config (features, observation mapping, normalization) into each dataset entry while overriding `repo_id` and `root_dir` per dataset. It also uses environment variables (e.g. `${ALOHA_TASKBOX_DATA_ROOT}`) for portable root paths, and defines a top-level `features` block, `observation_whitelist`, and `shuffle_buffer_size` at the multi-dataset level.


## 3. Create a training configuration file

The training configuration file ties together the dataset config, the policy config, and the training hyperparameters. Training configs use YAML with `!include` directives to reference your dataset configuration. Any field can be overridden from the command line using dot notation (e.g. `--policy.embed_dim=2048`).

Here is a minimal example training configuration for finetuning from a pretrained checkpoint with a Phi5 backbone.

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
  type: "phi4mm"           # The registered RhoAlpha policy type
  vlm_backend: phi5        # Select the Phi5 VLM backbone
  vlm_backbone_folder: /path/to/Phi-4-vision-5B  # Path to Phi5 model code/weights if available. Set to null otherwise
  embed_dim: 2048          # Action expert embedding dimension (1024 or 2048)
  hidden_state_idx: 12     # Which VLM transformer layer to tap
  n_obs_steps: 1           # Number of observation steps
  chunk_size: 16           # Length of the predicted action sequence
  n_action_steps: 8        # How many steps from the chunk to execute
  attention_type: layerwise_cross  # Cross-attention type (self, cross, layerwise_cross)
  freeze_vision_encoder: true
  freeze_vision_transformer: true
  freeze_vision_projector: true
  train_expert_only: false

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
mixed_precision: "no"       # Options: "no", "fp16", "bf16"
gradient_accumulation_steps: 1

# Evaluation settings
eval_interval: 10000
record_videos: false        # Set to true if you have a simulation environment

# Resume / pretrained checkpoint
resume: false
pretrained_checkpoint: null  # Override on the command line with --pretrained_checkpoint=...
```

### Key Parameters

- **`batch_size`**: This is the value used by training and defines the per-gpu batch size. It overwrites the placeholder batch_size defined in the dataset config. Multiply this number by the total number GPUs being used for training to determine your effective batch size
- **`learning_rate`**: This is overwritten by the policy. Ignore it.
- **`save_checkpoint_every`**: This determines how frequently checkpoints are created.
- **`keep_checkpoint_interval`**: If set all checkpoints that are not modulo this interval are deleted when the next chekpoint is created.
- **`resume`**: Pick up training from the last state saved in the pretrained checkpoint. This includes the dataloader state, optimizer state, and learning rate schedule state in addition the the model weights at that training step. 
- **`pretrained_checkpoint`**: The training checkpoint used as the intial weights for training. The code will automatically look for a train_config.json file located one directory above the location of this checkpoint to load any previous policy settings not defined in this file. 

### NOTE:
Don't trust the checkpoint_latest.pkl, the process for saving and copying to that filename is unreliable. 


#### Policy Parameters
- **`policy.type`**: Use `"phi4mm"` for all RhoAlpha models. This is the registered policy name regardless of which VLM backbone you use.
- **`policy.vlm_backend`**: Selects the VLM backbone. Options include `phi5`, `phi5_tactile`, `phi4mm`, `phi4mm_tactile`, `qwen25vl`, `qwen3vl`.
- **`policy.vlm_backbone_folder`**: Path to the folder containing the Phi5 model code (and optionally weights). When using the default path `rho/models/Phi-4-vision-5B` without weights, a `pretrained_checkpoint` is required.
- **`policy.embed_dim`**: The action expert embedding dimension. Use `1024` for a smaller model that fits on ~20 GB VRAM GPUs (e.g. RTX 4090). Use `2048` for larger models that require at least 40 GB VRAM (e.g. A100).
- **`policy.attention_type`**: The cross-attention mechanism. `layerwise_cross` provides the best performance for language-rich tasks but is slower to train.
- **`policy.chunk_size`**: The total length of the action sequence predicted at each step. This value is automatically propagated to the dataset config.
- **`pretrained_checkpoint`**: Absolute path to a pretrained checkpoint file. When finetuning, ensure the policy parameters (embed_dim, attention_type, etc.) match those of the checkpoint.

### Matching Pretrained Checkpoint Parameters

When loading a pretrained checkpoint, the policy configuration **must** match the parameters used when the checkpoint was trained. Each pretrained checkpoint directory includes a `train_config.json` file containing the full set of training parameters. Verify that `embed_dim`, `attention_type`, `hidden_state_idx`, `chunk_size`, and feature shapes all match before launching training.





## 4. Running training

### Single-GPU Training

For training with a custom dataset (not tied to a specific simulation environment), use the generic training entry point:

```bash
python -m rho.training.train \
  --config_path=configs/my_custom_train.yaml \
  --pretrained_checkpoint=/path/to/checkpoint/checkpoint_latest.pt \
  --wandb.enable=false \
  --batch_size=4 \
  --steps=50000
```

If you are working within a specific environment such as Libero, you can use the environment-specific training script instead:

Note that you only need to use the custom entrypoint if you are attempting to perform inline evaluation. 

```bash
python environments/libero/train.py \
  --config_path=environments/libero/configs/my_custom_train.yaml \
  --pretrained_checkpoint=/path/to/checkpoint/checkpoint_latest.pt \
  --wandb.enable=false \
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

For multi-GPU training with the generic training entry point, use `accelerate launch` with the accelerate training module:

```bash
export NUM_GPU=2
accelerate launch --multi_gpu \
  --num_processes=${NUM_GPU} \
  -m rho.training.train_accelerate \
  --config_path=configs/my_custom_train.yaml \
  --pretrained_checkpoint=/path/to/checkpoint/checkpoint_latest.pt \
  --wandb.enable=false
```

Environment-specific training scripts (like `environments/libero/train.py`) automatically detect whether they are launched under accelerate and will switch to the accelerate training path:

```bash
export NUM_GPU=2
accelerate launch --multi_gpu \
  --num_processes=${NUM_GPU} \
  environments/libero/train.py \
  --config_path=environments/libero/configs/my_custom_train.yaml \
  --pretrained_checkpoint=/path/to/checkpoint/checkpoint_latest.pt
```

### Resuming Training

There are two modes for loading a pretrained checkpoint:

- **Finetuning** (`resume=false`, `pretrained_checkpoint=...`): Loads the policy weights only and starts a fresh optimizer, scheduler, and step counter. This is the default when you provide `--pretrained_checkpoint`.

- **Resuming** (`resume=true`, `pretrained_checkpoint=...`): Restores the full training state including the optimizer, scheduler, and step counter. Use this to continue a previously interrupted training run.

You can also use the `run_name` parameter for automatic resume behavior:

```bash
python -m rho.training.train \
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
      checkpoint_latest.pt      # Most recent checkpoint
      checkpoint_step_01000.pt  # Periodic checkpoint (save_checkpoint_every)
      checkpoint_step_10000.pt  # Kept checkpoint (keep_checkpoint_interval)
```

Or with `run_name`:

```
output_dir/
  my_experiment/
    train_config.json
    checkpoints/
      checkpoint_latest.pt
```

- **`save_checkpoint_every`**: How often to save a checkpoint (in steps). These are rolling — older ones are overwritten unless they fall on a `keep_checkpoint_interval` boundary.
- **`keep_checkpoint_interval`**: Checkpoints on these step boundaries are kept permanently.
- **`train_config.json`**: Saved alongside the checkpoints directory. This file records the full training configuration and is used by evaluation and serving utilities to reconstruct the policy and dataset settings.

### WandB Monitoring

When WandB is enabled (`--wandb.enable=true`), training metrics (loss, learning rate, gradient norms) are logged at each `logging_interval`. Make sure the `WANDB_API_KEY` and `WANDB_BASE_URL` environment variables are set.


## 6. Preflight checklist

Before launching training, verify the following:

- [ ] **`root_dir` is accessible** — The dataset `root_dir` must be reachable from within your container or environment. This is the most common failure. If running in Docker, ensure the data directory is mounted correctly.
- [ ] **`repo_id` does not match a real HuggingFace dataset** — When using a local dataset, set `repo_id` to an arbitrary string that doesn't exist on HuggingFace to prevent accidental downloads.
- [ ] **`features` shapes match post-transform dimensions** — The shape entries in your dataset config must reflect the final shapes after any observation mapping and transforms are applied.
- [ ] **Policy parameters match the pretrained checkpoint** — If loading a pretrained checkpoint, check `embed_dim`, `attention_type`, `hidden_state_idx`, and `chunk_size` against the checkpoint's `train_config.json`.
- [ ] **`vlm_backbone_folder` exists** — When using `vlm_backend: phi5`, the path set by `vlm_backbone_folder` must contain the model code. If it doesn't contain safetensor weight files, a `pretrained_checkpoint` is required.
- [ ] **Custom stats file is reachable** — If you use a separate stats file via the `stats` field, ensure it is accessible from your training environment.
- [ ] **VRAM is sufficient** — `embed_dim=1024` typically fits on 20 GB GPUs; `embed_dim=2048` requires at least 40 GB. Reducing `batch_size` or enabling `gradient_accumulation_steps` can help if you are close to the limit.