import logging

from rho.environment.dataset_env import DatasetEnvironment, DatasetEnvironmentConfig  # noqa: F401
from rho.environment.env import EnvironmentConfig, EnvironmentWrapper, GymEnvironment

logger = logging.getLogger(__name__)

ENV_REGISTRY: dict[str, type[EnvironmentWrapper]] = {}
ENV_CONFIG_REGISTRY: dict[str, type[EnvironmentConfig]] = {}


def register_environment(name: str):
    """
    Decorator to register an environment class in the global registry.

    Args:
        name: The name to register the environment under
    """

    def decorator(cls: type[EnvironmentWrapper]):
        try:
            config_class = EnvironmentConfig.get_choice_class(name)
        except KeyError:
            raise ValueError(
                f"Cannot register environment type {name!r} without a matching "
                "EnvironmentConfig.register_subclass() registration"
            ) from None
        ENV_REGISTRY[name] = cls
        ENV_CONFIG_REGISTRY[name] = config_class
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

    environment_type = env_config.type

    logger.info(f"EnvironmentConfig type is {environment_type}")
    if environment_type not in ENV_REGISTRY:
        available_envs = list(ENV_REGISTRY.keys())
        raise ValueError(
            f"Environment type '{environment_type}' not found in registry. "
            f"Available environments: {available_envs}"
        )

    config_class = ENV_CONFIG_REGISTRY.get(environment_type)
    if config_class is not None and not isinstance(env_config, config_class):
        raise TypeError(
            f"Environment type {environment_type!r} requires {config_class.__name__}, "
            f"got {env_config.__class__.__name__}"
        )

    env_class = ENV_REGISTRY[environment_type]
    return env_class(env_config)


register_environment("GymEnvironment")(GymEnvironment)
register_environment("DatasetEnvironment")(DatasetEnvironment)
__all__ = [
    "make_environment",
    "register_environment",
    "ENV_REGISTRY",
    "ENV_CONFIG_REGISTRY",
    "check_dataset_loading",
]
