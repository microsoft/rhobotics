export PHI4ROBOTICS_DIR=`pwd`
export CONTAINER_NAME=alku-libero-interactive

docker stop $CONTAINER_NAME 2>/dev/null || true
sleep 1
docker run --gpus all --ipc=host \
--ulimit memlock=-1 --ulimit stack=67108864 \
--rm -v ~/.cache/huggingface:/hf_home  \
-v $PHI4ROBOTICS_DIR/rho:/workspace/rho \
-v $PHI4ROBOTICS_DIR/config:/workspace/config \
-v $PHI4ROBOTICS_DIR/environments:/workspace/environments \
-v $PHI4ROBOTICS_DIR/outputs:/workspace/outputs \
-v $PHI4ROBOTICS_DIR/tests:/workspace/tests \
-v $PHI4ROBOTICS_DIR/notebooks:/workspace/notebooks \
-v $PHI4ROBOTICS_DIR/scratch:/workspace/scratch \
-v $PHI4ROBOTICS_DIR/rho_client:/workspace/rho_client \
-v /data/:/data \
-e WANDB_BASE_URL="$WANDB_BASE_URL" \
-e WANDB_API_KEY="$WANDB_API_KEY" \
-e HF_TOKEN="$HF_TOKEN" \
--name $CONTAINER_NAME \
-d msrxworkspace1acr.azurecr.io/phi4robotics/rho-libero:latest \
tail -f /dev/null

docker exec -it $CONTAINER_NAME /bin/bash\
