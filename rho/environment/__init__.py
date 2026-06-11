import logging

from rho.environment.dataset_env import DatasetEnvironment, DatasetEnvironmentConfig  # noqa: F401
from rho.environment.env import EnvironmentConfig, EnvironmentWrapper, GymEnvironment

logger = logging.getLogger(__name__)

ENV_REGISTRY: dict[str, type[EnvironmentWrapper]] = {}


def register_environment(name: str):
    """
    Decorator to register an environment class in the global registry.

    Args:
        name: The name to register the environment under
    """

    def decorator(cls: type[EnvironmentWrapper]):
        ENV_REGISTRY[name] = cls
        cls.name = name  # Set the name attribute on the class
        return cls

    return decorator


def make_environment(env_config: EnvironmentConfig) -> EnvironmentWrapper:
    """
    Create an environment instance from the given configuration.

    Args:
        env_config: EnvironmentConfig containing the environment name and parameters

    Returns:
        An instance of the requested environment class

    Raises:
        ValueError: If the environment name is not found in the registry

    Example:
        config = EnvironmentConfig(name="GymEnvironment", params={...})
        env = make_environment(config)
    """
    if env_config is None:
        return None

    if env_config.name is None:
        raise ValueError("EnvironmentConfig.name must be specified")

    logger.info(f"EnvironmentConfig Subclass is {env_config.name}")
    if env_config.name not in ENV_REGISTRY:
        available_envs = list(ENV_REGISTRY.keys())
        raise ValueError(
            f"Environment '{env_config.name}' not found in registry. Available environments: {available_envs}"
        )

    env_class = ENV_REGISTRY[env_config.name]
    return env_class(env_config)


register_environment("GymEnvironment")(GymEnvironment)
register_environment("DatasetEnvironment")(DatasetEnvironment)
__all__ = ["make_environment", "register_environment", "ENV_REGISTRY", "check_dataset_loading"]
