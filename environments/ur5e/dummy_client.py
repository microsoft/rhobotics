import numpy as np

from rho_client import websocket_client_policy

client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=9999)
metadata = client.get_server_metadata()
print(metadata)
obs = {
    "joint_positions": np.random.randn(1, 7).astype(np.float64),
    "zed_scene_bgr": np.random.randn(180, 320, 3).astype(np.uint8),
    "zed_wrist_left_bgr": np.random.randn(180, 320, 3).astype(np.uint8),
    "zed_wrist_right_bgr": np.random.randn(180, 320, 3).astype(np.uint8),
    "actual_tcp_forces": np.random.randn(1, 6).astype(np.float64),
    "tactile": np.random.randn(1, 19).astype(np.float64),
    "tcp_forces_history": np.random.randn(10, 1, 6).astype(np.float64),
    "tactile_history": np.tile(np.linspace(32, 17, 16, dtype=np.float32)[:, np.newaxis], (1, 19)).reshape(
        1, 16, 19
    ),
    "task": ["put the plug into the socket"],
    "_reset_": 1,
    # "joint_velocities": np.random.randn(14,).astype(np.float64),
    # "ee_pos_quat": np.random.randn(14,).astype(np.float64),
    # "gripper_position": np.random.randn(2,).astype(np.float64),
    # "action": np.random.randn(16, 14).astype(np.float32),
}
action = client.infer(obs)
print("time to infer: ", action["infer_ms"])
print("Received action shape:", action["action"].shape)
