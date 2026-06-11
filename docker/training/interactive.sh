
export PHI4ROBOTICS_DIR=`pwd`
# Server port to forward host <-> container; matches the `port` field in the
# environment yaml (e.g. environments/fr3_duo/serve_duo.yaml).
export RHO_SERVER_PORT="${RHO_SERVER_PORT:-9999}"

docker stop rho-training-interactive 2>/dev/null || true
sleep 1
docker run --gpus all --ipc=host \
--ulimit memlock=-1 --ulimit stack=67108864 \
--rm -v ~/.cache/huggingface:/hf_home  \
-v $PHI4ROBOTICS_DIR/rho:/workspace/rho \
-v $PHI4ROBOTICS_DIR/rho_client:/workspace/rho_client \
-v $PHI4ROBOTICS_DIR/config:/workspace/config \
-v $PHI4ROBOTICS_DIR/environments:/workspace/environments \
-v $PHI4ROBOTICS_DIR/outputs:/workspace/outputs \
-v $PHI4ROBOTICS_DIR/tests:/workspace/tests \
-v $PHI4ROBOTICS_DIR/notebooks:/workspace/notebooks \
-v /data/:/data \
-e WANDB_BASE_URL="$WANDB_BASE_URL" \
-e WANDB_API_KEY="$WANDB_API_KEY" \
-e HF_TOKEN="$HF_TOKEN" \
-p $RHO_SERVER_PORT:$RHO_SERVER_PORT \
--name rho-training-interactive \
-d rho-training:latest \
tail -f /dev/null

docker exec -it rho-training-interactive /bin/bash\
