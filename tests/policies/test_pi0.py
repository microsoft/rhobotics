import pytest
import torch

from rho.common.types import FeatureType, PolicyFeature
from rho.policies import POLICY_REGISTRY, list_available_policies, make_policy
from rho.policies.pi0 import PI0Config


def test_pi0_is_registered():
    available = list_available_policies()
    assert "pi0" in available
    assert "pi0" in POLICY_REGISTRY


def test_pi0_config_validate_features_minimal():
    feature_dict = {
        "observation.image.cam0": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }

    cfg = PI0Config(
        name="pi0",
        feature_dict=feature_dict,
        device="cpu",
        dtype=torch.float32,
        chunk_size=10,
        n_action_steps=2,
        max_state_dim=32,
        max_action_dim=32,
        empty_cameras=0,
    )

    cfg.validate_features()


@pytest.mark.resource_intensive
def test_pi0_config_validate_features_minimal_gpu():
    feature_dict = {
        "observation.image.cam0": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }

    cfg = PI0Config(
        name="pi0",
        feature_dict=feature_dict,
        device="cuda",
        dtype=torch.bfloat16,
        chunk_size=10,
        n_action_steps=2,
        max_state_dim=32,
        max_action_dim=32,
        empty_cameras=0,
    )

    cfg.validate_features()


@pytest.mark.resource_intensive
def test_pi0_policy_instantiation_smoke():
    """Smoke test that PI0Policy can be instantiated.

    This does NOT run a forward pass (which would require tokenization and can be very heavy).

    To run:
      pytest --all -q tests/policies/test_pi0.py

    Optionally gate via env var to avoid accidental runs in some environments.
    """
    feature_dict = {
        "observation.image.cam0": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }

    cfg = PI0Config(
        name="pi0",
        feature_dict=feature_dict,
        device="cuda",
        dtype=torch.bfloat16,
        chunk_size=10,
        n_action_steps=2,
        max_state_dim=32,
        max_action_dim=32,
        empty_cameras=0,
        require_transformers_replace=False,
        compile_model=False,
        enable_gradient_checkpointing=False,
    )

    policy = make_policy(cfg)
    assert policy is not None
