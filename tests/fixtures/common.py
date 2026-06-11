import torch


def create_dummy_batch(batch_size=4, device="cpu"):
    """Create a dummy batch for testing"""
    return create_dummy_pusht_batch(batch_size=batch_size, device=device)


def create_dummy_pusht_batch(batch_size=4, device="cpu"):
    """Create a dummy batch that matches PushT environment structure"""
    return {
        "observation.image": torch.randn(batch_size, 3, 96, 96).to(device),
        "observation.state": torch.randn(batch_size, 5).to(device),  # PushT state dimension
        "action": torch.randn(batch_size, 2).to(device),  # PushT action dimension
        "episode_index": torch.arange(batch_size).to(device),
        "frame_index": torch.arange(batch_size).to(device),
        "timestamp": torch.arange(batch_size, dtype=torch.float32).to(device),
        "index": torch.arange(batch_size).to(device),
    }
