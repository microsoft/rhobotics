prerequisites:
```bash
cd phi-4-robotics
conda create -y -n phi4robotics python=3.10.16  # if not already created
conda activate phi4robotics
pip install -e .
pip install --no-deps "openpi-client @ git+https://github.com/Physical-Intelligence/openpi.git#subdirectory=packages/openpi-client"
pip install pin  # Python bindings for the Pinocchio IK library; may require additional system dependencies (see Pinocchio docs)
```

running server:
```bash
conda activate phi4robotics
python3 environments/aloha/serve_real.py --config_path environments/aloha/new_server.yaml
```

running real client:
```bash
# copy aloha_client.py to the aloha machine
python3 aloha_client.py -l "Pull the red wire."
```

running dataset client (testing-purposes):
```bash
# start server in a separate terminal
conda activate phi4robotics
python3 environments/aloha/dataset_client.py \
    --dataset_root /datadrive/datasets/aloha-busybox_lerobot_v3_wave3_combined \
    --port 7000 \
    --episode 37 \
    --num_samples 100
```
