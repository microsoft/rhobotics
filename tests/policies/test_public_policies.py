from rho.policies import POLICY_CONFIG_REGISTRY, POLICY_REGISTRY, list_available_policies
from rho.policies.rho import RhoPolicy
from rho.policies.rho.configuration_rho import RhoConfig


def test_public_policy_registry_contains_supported_policies():
    assert set(list_available_policies()) == {"rho"}
    assert POLICY_REGISTRY["rho"] is RhoPolicy
    assert POLICY_CONFIG_REGISTRY["rho"] is RhoConfig


def test_rho_uses_default_optional_delta_schedules():
    config = RhoConfig()

    assert config.reward_delta_indices is None
    assert config.tactile_observation_delta_indices is None
