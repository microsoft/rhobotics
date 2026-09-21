"""
Policy module initialization with automatic registration and factory function.
"""

from rho.models.schedule import migrate_legacy_scheduler_config
from rho.policies.base import PolicyConfig, PreTrainedPolicy
from rho.policies.rho import RhoPolicy

# Registry to store policy classes
POLICY_REGISTRY: dict[str, type[PreTrainedPolicy]] = {}
POLICY_CONFIG_REGISTRY: dict[str, type[PolicyConfig]] = {}


def register_policy(name: str):
    """
    Decorator to register a policy class in the global registry.

    Args:
        name: The name to register the policy under
    """

    def decorator(cls: type[PreTrainedPolicy]):
        try:
            config_class = PolicyConfig.get_choice_class(name)
        except KeyError:
            raise ValueError(
                f"Cannot register policy type {name!r} without a matching "
                "PolicyConfig.register_subclass() registration"
            ) from None
        POLICY_REGISTRY[name] = cls
        POLICY_CONFIG_REGISTRY[name] = config_class
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
    """
    policy_type = policy_cfg.type

    if policy_type not in POLICY_REGISTRY:
        available_policies = list(POLICY_REGISTRY.keys())
        raise ValueError(
            f"Policy type '{policy_type}' not found in registry. Available policies: {available_policies}"
        )

    config_class = POLICY_CONFIG_REGISTRY.get(policy_type)
    if config_class is not None and not isinstance(policy_cfg, config_class):
        raise TypeError(
            f"Policy type {policy_type!r} requires {config_class.__name__}, "
            f"got {policy_cfg.__class__.__name__}"
        )

    policy_class = POLICY_REGISTRY[policy_type]
    return policy_class(policy_cfg)


def make_policy_from_checkpoint(
    checkpoint_path: str,
    policy_cfg: PolicyConfig = None,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
) -> PreTrainedPolicy:
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
    import torch

    from rho.checkpoints import (
        checkpoint_read_lease,
        is_checkpoint_bundle,
        load_bundle_metadata,
        resolve_checkpoint,
    )

    checkpoint_path = resolve_checkpoint(
        checkpoint_path,
        revision=revision,
        cache_dir=cache_dir,
        include_training_state=False,
    )
    if is_checkpoint_bundle(checkpoint_path):
        with checkpoint_read_lease(checkpoint_path):
            if policy_cfg is None:
                import draccus

                metadata = load_bundle_metadata(checkpoint_path)
                policy_dict = metadata["policy"]
                if policy_dict is None:
                    raise ValueError(f"Policy metadata not found in checkpoint bundle: {checkpoint_path}")
                policy_dict = dict(policy_dict)
                if metadata["features"] is not None:
                    policy_dict["feature_dict"] = metadata["features"]
                if "type" not in policy_dict and "name" in policy_dict:
                    policy_dict["type"] = policy_dict["name"]
                if isinstance(policy_dict.get("lr_scheduler"), dict):
                    policy_dict["lr_scheduler"] = migrate_legacy_scheduler_config(policy_dict["lr_scheduler"])
                if policy_dict.get("type") == "rho":
                    policy_dict.pop("empty_cameras", None)
                policy_cfg = draccus.decode(PolicyConfig, policy_dict)
            policy = make_policy(policy_cfg)
            policy.load_from_pretrained(checkpoint_path)
            return policy

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
register_policy("rho")(RhoPolicy)


# Export public API
__all__ = [
    "PreTrainedPolicy",
    "PolicyConfig",
    "RhoPolicy",
    "make_policy",
    "make_policy_from_checkpoint",
    "register_policy",
    "list_available_policies",
    "POLICY_REGISTRY",
    "POLICY_CONFIG_REGISTRY",
]
