# Pi0 / Pi0.5 Policy

This directory contains the Pi0 and Pi0.5 policy implementations, based on [Physical Intelligence's OpenPI](https://github.com/Physical-Intelligence/openpi).

## Prerequisites

### 1. Install Dependencies

Pi0 requires `sentencepiece` for the PaliGemma tokenizer and a specific version of `transformers`:

```bash
pip install sentencepiece
pip install transformers==4.53.2
```

### 2. Patch Transformers with `transformers_replace`

Pi0 uses custom modifications to the HuggingFace `transformers` library (Gemma, PaliGemma, and SigLIP models). These must be copied over the installed package:

```bash
TRANSFORMERS_PATH=$(python -c "import transformers; import os; print(os.path.dirname(transformers.__file__))")

cp rho/policies/pi0/transformers_replace/models/gemma/configuration_gemma.py $TRANSFORMERS_PATH/models/gemma/configuration_gemma.py
cp rho/policies/pi0/transformers_replace/models/gemma/modeling_gemma.py $TRANSFORMERS_PATH/models/gemma/modeling_gemma.py
cp rho/policies/pi0/transformers_replace/models/paligemma/modeling_paligemma.py $TRANSFORMERS_PATH/models/paligemma/modeling_paligemma.py
cp rho/policies/pi0/transformers_replace/models/siglip/modeling_siglip.py $TRANSFORMERS_PATH/models/siglip/modeling_siglip.py
cp rho/policies/pi0/transformers_replace/models/siglip/check.py $TRANSFORMERS_PATH/models/siglip/check.py
```

> **Warning:** This overwrites files in the installed `transformers` package. This may conflict with other policies (e.g., Phi4MM) that depend on the unmodified library. Consider using separate environments or containers if running both.

### 3. PaliGemma Tokenizer

Pi0 requires a PaliGemma sentencepiece tokenizer model file. Set the path in your training config:

```yaml
policy:
  tokenizer_model_path: /path/to/paligemma_tokenizer.model
```

A known location on the shared data mount is:
```bash
# on sandbox 544
/data/dean/models/paligemma_models/paligemma_tokenizer.model
# in blob storage at Subscription MSRX Development AOAI
https://msrxworkspace1sa.blob.core.windows.net/models/paligemma/paligemma_tokenizer.model
```

### 4. Pi05 pretrained checkpoint

Optional (but recommended) pretrained checkpoint from PI
```
https://msrxworkspace1sa.blob.core.windows.net/models/pi05_base_checkpoint/
```

## Training

### Configuration

Training configs are in `environments/libero/configs/`. See `train_libero_pi05.yaml` for a Pi0.5 example.

Key policy config fields:

| Field | Description | Default |
|-------|-------------|---------|
| `type` | Must be `"pi0"` | — |
| `pi05` | Enable Pi0.5 mode | `false` |
| `chunk_size` | Action chunk length | `50` |
| `n_action_steps` | Action steps to execute | `50` |
| `tokenizer_model_path` | Path to PaliGemma sentencepiece model | `null` (required) |
| `scheduler_decay_steps` | LR scheduler decay steps | `30000` |

> **Note:** `freeze_vision_encoder` and `train_expert_only` are **not** valid for `PI0Config` (those are Phi4MM-specific).

### Launch Command (Local Multi-GPU)

```bash
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --multi-gpu --num_processes=2 \
      environments/libero/train.py \
      --config_path environments/libero/configs/train_libero_pi05.yaml \
      --dataset.root_dir /data/dean/data/libero_lerobot_resim/v1.1-video-resim_v30/ \
      --output_dir output_pi05 \
      --set_static_graph True \
      --wandb.enable=true --batch_size 4
```

Adjust `CUDA_VISIBLE_DEVICES` and `--num_processes` to match the number of GPUs you want to use.

### Full Setup Script (Inside Container)

For convenience, here is the complete setup sequence to run before training:

```bash
# Install dependencies
pip install sentencepiece
pip install transformers==4.53.2

# Patch transformers
TRANSFORMERS_PATH=$(python -c "import transformers; import os; print(os.path.dirname(transformers.__file__))")
cp rho/policies/pi0/transformers_replace/models/gemma/configuration_gemma.py $TRANSFORMERS_PATH/models/gemma/configuration_gemma.py
cp rho/policies/pi0/transformers_replace/models/gemma/modeling_gemma.py $TRANSFORMERS_PATH/models/gemma/modeling_gemma.py
cp rho/policies/pi0/transformers_replace/models/paligemma/modeling_paligemma.py $TRANSFORMERS_PATH/models/paligemma/modeling_paligemma.py
cp rho/policies/pi0/transformers_replace/models/siglip/modeling_siglip.py $TRANSFORMERS_PATH/models/siglip/modeling_siglip.py
cp rho/policies/pi0/transformers_replace/models/siglip/check.py $TRANSFORMERS_PATH/models/siglip/check.py

# Launch training
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --multi-gpu --num_processes=2 \
      environments/libero/train.py \
      --config_path environments/libero/configs/train_libero_pi05.yaml \
      --dataset.root_dir /data/dean/data/libero_lerobot_resim/v1.1-video-resim_v30/ \
      --output_dir output_pi05 \
      --set_static_graph True \
      --wandb.enable=true --batch_size 4
```

## References

- [OpenPI GitHub](https://github.com/Physical-Intelligence/openpi)
- [OpenPI PyTorch Support](https://github.com/Physical-Intelligence/openpi/tree/main?tab=readme-ov-file#pytorch-support)
