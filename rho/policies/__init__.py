"""
Policy module initialization with automatic registration and factory function.
"""

from rho.policies.base import PolicyConfig, PreTrainedPolicy
from rho.policies.BC.behavioral_cloning import BehavioralCloningPolicy
from rho.policies.diffusion.diffusion import DiffusionPolicy
from rho.policies.dsrl.dsrl_policy import DSRLPolicy, DSRLPolicyConfig
from rho.policies.dsrl.flowdagger_policy import FlowDAggerPolicy, FlowDAggerPolicyConfig
from rho.policies.pi0.modeling_pi0 import PI0Policy
from rho.policies.pi0fast.pi0fast_policy import PI0FASTPolicy
from rho.policies.rhoalpha.configuration_rhoalpha import RhoAlphaConfig
from rho.policies.rhoalpha.rhoalpha_policy import RhoAlphaPolicy
from rho.policies.rhoalpha.rhoalpha_tactile import RhoAlphaTactilePolicy

# Registry to store policy classes
POLICY_REGISTRY: dict[str, type[PreTrainedPolicy]] = {}


def register_policy(name: str):
    """
    Decorator to register a policy class in the global registry.

    Args:
        name: The name to register the policy under

    Example:
        @register_policy("BehavioralCloning")
        class BehavioralCloningPolicy(PreTrainedPolicy):
            pass
    """

    def decorator(cls: type[PreTrainedPolicy]):
        POLICY_REGISTRY[name] = cls
        cls.name = name  # Set the name attribute on the class
        return cls

    return decorator


def make_policy(policy_cfg: PolicyConfig) -> PreTrainedPolicy:
    """
    Create a policy instance from the given configuration.

    Args:
        policy_cfg: PolicyConfig containing the policy name and parameters

    Returns:
        An instance of the requested policy class

    Raises:
        ValueError: If the policy name is not found in the registry

    Example:
        config = PolicyConfig(name="BehavioralCloning", feature_dict=features)
        policy = make_policy(config)
    """
    if policy_cfg.name is None:
        raise ValueError("PolicyConfig.name must be specified")

    if policy_cfg.name not in POLICY_REGISTRY:
        available_policies = list(POLICY_REGISTRY.keys())
        raise ValueError(
            f"Policy '{policy_cfg.name}' not found in registry. Available policies: {available_policies}"
        )

    policy_class = POLICY_REGISTRY[policy_cfg.name]
    return policy_class(policy_cfg)


def make_policy_from_checkpoint(checkpoint_path: str, policy_cfg: PolicyConfig = None) -> PreTrainedPolicy:
    """
    Create a policy instance directly from a checkpoint.

    This function loads the policy configuration and feature_dict from the checkpoint,
    avoiding the need to load the dataset during evaluation.

    Args:
        checkpoint_path: Path to the checkpoint file
        policy_cfg: Optional policy config. If None, tries to load from checkpoint

    Returns:
        An instance of the requested policy class with weights loaded

    Raises:
        ValueError: If policy config cannot be determined
        FileNotFoundError: If checkpoint doesn't exist
    """
    from pathlib import Path

    import torch

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Try to get policy config from checkpoint
    if policy_cfg is None:
        if "policy_config" in checkpoint:
            policy_cfg = checkpoint["policy_config"]
        else:
            raise ValueError(
                "No policy_config provided and none found in checkpoint. Please provide policy_cfg parameter."
            )

    # Set feature_dict from checkpoint if available and not already set
    if policy_cfg.feature_dict is None:
        if "feature_dict" in checkpoint:
            policy_cfg.feature_dict = checkpoint["feature_dict"]
        elif "policy_config" in checkpoint and hasattr(checkpoint["policy_config"], "feature_dict"):
            policy_cfg.feature_dict = checkpoint["policy_config"].feature_dict
        else:
            raise ValueError(
                "feature_dict not found in checkpoint and not provided in policy_cfg. "
                "Cannot create policy without feature_dict."
            )

    # Create policy
    policy = make_policy(policy_cfg)

    # Load weights
    if "policy_state_dict" in checkpoint:
        # Handle DDP wrapper case
        state_dict = checkpoint["policy_state_dict"]
        if any(key.startswith("module.") for key in state_dict):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        policy.load_state_dict(state_dict)
    else:
        raise ValueError("No policy_state_dict found in checkpoint")

    return policy


def list_available_policies() -> list[str]:
    """
    Get a list of all registered policy names.

    Returns:
        List of policy names available in the registry
    """
    return list(POLICY_REGISTRY.keys())


# Register all available policies
register_policy("behavioral_cloning")(BehavioralCloningPolicy)
register_policy("diffusion")(DiffusionPolicy)
register_policy("rhoalpha")(RhoAlphaPolicy)
register_policy("rhoalpha_tactile")(RhoAlphaTactilePolicy)
register_policy("dsrl")(DSRLPolicy)
register_policy("flowdagger")(FlowDAggerPolicy)

# Backward-compat aliases — these all resolve to RhoAlphaPolicy with
# vlm_backend set accordingly by the training config / YAML.
POLICY_REGISTRY["phi4mm"] = RhoAlphaPolicy
POLICY_REGISTRY["phi4mm_tactile"] = RhoAlphaTactilePolicy
POLICY_REGISTRY["qwen25vl"] = RhoAlphaPolicy
POLICY_REGISTRY["qwen3vl"] = RhoAlphaPolicy

# Register RhoAlphaConfig under Qwen names so draccus can deserialize
# YAML configs with type: "qwen25vl" / "qwen3vl" into RhoAlphaConfig.
PolicyConfig.register_subclass("qwen25vl", RhoAlphaConfig)
PolicyConfig.register_subclass("qwen3vl", RhoAlphaConfig)

register_policy("pi0")(PI0Policy)
register_policy("pi0fast")(PI0FASTPolicy)

# Add more policies here as they are implemented:
# register_policy("TransformerPolicy")(TransformerPolicy)


# Export public API
__all__ = [
    "PreTrainedPolicy",
    "PolicyConfig",
    "BehavioralCloningPolicy",
    "DiffusionPolicy",
    "DSRLPolicy",
    "DSRLPolicyConfig",
    "FlowDAggerPolicy",
    "FlowDAggerPolicyConfig",
    "RhoAlphaPolicy",
    "RhoAlphaTactilePolicy",
    "PI0Policy",
    "PI0FASTPolicy",
    "make_policy",
    "make_policy_from_checkpoint",
    "register_policy",
    "list_available_policies",
    "POLICY_REGISTRY",
]
