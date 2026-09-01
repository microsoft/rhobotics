from rho.policies import POLICY_CONFIG_REGISTRY, POLICY_REGISTRY, list_available_policies
from rho.policies.dsrl.dsrl_policy import DSRLPolicy, DSRLPolicyConfig
from rho.policies.dsrl.flowdagger_policy import FlowDAggerPolicy, FlowDAggerPolicyConfig
from rho.policies.rho import RhoPolicy
from rho.policies.rho.configuration_rho import RhoConfig


def test_public_policy_registry_contains_supported_policies():
    assert {"rho", "dsrl", "flowdagger"} <= set(list_available_policies())
    assert POLICY_REGISTRY["rho"] is RhoPolicy
    assert POLICY_REGISTRY["dsrl"] is DSRLPolicy
    assert POLICY_REGISTRY["flowdagger"] is FlowDAggerPolicy
    assert POLICY_CONFIG_REGISTRY["rho"] is RhoConfig
    assert POLICY_CONFIG_REGISTRY["dsrl"] is DSRLPolicyConfig
    assert POLICY_CONFIG_REGISTRY["flowdagger"] is FlowDAggerPolicyConfig


def test_public_wrapper_defaults_use_rho():
    assert DSRLPolicyConfig().base_policy_name == "rho"
    assert FlowDAggerPolicyConfig().base_policy_name == "rho"


def test_rho_uses_default_optional_delta_schedules():
    config = RhoConfig()

    assert config.reward_delta_indices is None
    assert config.tactile_observation_delta_indices is None
