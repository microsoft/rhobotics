# First Steps when using RhoAlpha with Phi5

This section covers the particulars of training using the Phi5 backbone and repeats the instructions provided in the README.md file for evaluating and 

## Important Note on the Phi-5 Backbone

RhoAlpha supports multiple VLM backbones such as Phi4MM, Qwen 2.5VL, Qwen 3VL, and Phi4-Vision-5B. We refer to Phi-4-Vision-5B as Phi5 in our codebase and it is handled differently from our other backbones because it is not readily availble on huggingface. We have added it's code to our repository under rho/models/Phi-4-vision-5B. We also included a slightly newer version of this code under rho/models/Phi-4-vision-reasoning-5B but we do not provide pretrained checkpoints for that version at this time. 

When you select `policy.vlm_backend=phi5` or `policy.vlm_backend=phi5_tactile` the policy will load the VLM backbone from the folder defined by `policy.vlm_backbone_folder`. If you have access to the folders containing the weights for Phi-4-Vision-5B or Phi-4-vision-reasoning-5B you can point the `policy.vlm_backbone_folder` parameter at this folder and it will detect and load the weights. Otherwise it will default to `policy.vlm_backbone_folder=rho/models/Phi-4-vision-5B`. 

If the folder set by `policy.vlm_backbone_folder` does not contain safetensor files then it will create the VLM backbone with uninitialized weights. This is what will occur when using the default of rho/models/Phi-4-vision-5B. In this case the user **must** define a `pretrained_checkpoint` for training or evaluation. Otherwise an error will be thrown to prevent users from accidentally running with a completely uninitialized model.

## Updated Libero Instructions

This section provides a new set of instructions adapted from the README.md file and adapted to use the Phi5 backbone

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
  -v $RHOALPHA_DIR/configs:/workspace/configs \
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

### 3a. Finetuning from scratch

Note on training from scratch 
If you have access to the Phi-5 weights you can train without a pretrained checkpoint. We call this "from scratch" since this Phi-5 has not yet been trained on any robotics data and the action expert is uninitialized. Phi5 is particularly sensitive when training in this mode and doing so can damage its language understanding. We recommend using the `policy.train_expert_only=true` option when operating in this mode to precondition the action expert network. 

For this example we use a smaller embedding dimension of 1024 which is fine for environments such as Libero. Ths allows the run to fit on a machine with only 20 GB VRAM, such as an NVIDIA GTX 4090.

```bash
python environments/libero/train.py \
  --config_path=environments/libero/configs/train_libero_phi5_demo.yaml \
  --policy.vlm_backbone_folder=${PATH_TO_PHI5_CODE_AND_WEIGHTS} \
  --wandb.enable=false \
  --policy.train_expert_only= true \
  --batch_size=4 \
  --steps=20 \
  --eval_interval=20
```

### 3b. Finetuning from a pretrained checkpoint 

We recommend using a pretrained checkpoint for fine tuning as this is our intended method of interacting with the model.

When using a pretrained checkpoint is necessary to make sure the policy parameters you are using for finetuning match those of the pretrained checkpoint. Otherwise the network will not be initialized with the correct state_dict and won't be able to load the checkpoint. Each pretrained checkpoint includes a train_config.json file that includes the full dictionary of training parameters that can be used to verify the settings of the pretrained checkpoint.

In this example we will assume the pretrained checkpoint is using the layerwise cross attention mechanism and an embedding dimension of 2048. This leads to a larger slower model to train but also provides the best performance for language rich tasks with high amounts of variability in the workspace confgiurations. When the embedding dimension is set to 2048 the model requires at last 40GB of VRAM to train (i.e. an A100 40GB GPU). 

```bash
python environments/libero/train.py \
  --config_path=environments/libero/configs/train_libero_phi5_demo.yaml \
  --pretrained_checkpoint=${PATH_TO_PRETRAINED_CHECKPOINT} \
  --wandb.enable=false \
  --policy.train_expert_only=true \
  --policy.embedding_dim=2048 \
  --policy.attention_type=layerwise_cross \
  --batch_size=4 \
  --steps=20 \
  --eval_interval=20
```


#### Evaluate a Pre-existing Checkpoint

Loading Phi-5 based models for evaluation should not differ from our methods for evaluating other models. As the pretrained checkpoint is already provided.

```bash
python environments/libero/eval.py \
  --config_path=environments/libero/configs/eval_libero_rhoalpha.yaml \
  --pretrained_checkpoint=${PATH_TO_CHECKPOINT}
```

