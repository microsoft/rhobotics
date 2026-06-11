from dataclasses import dataclass

from rho.policies.base import PolicyConfig, PreTrainedPolicy


@dataclass
class BasicFlowMatchingConfig(PolicyConfig):
    """Config for BasicFlowMatchingPolicy."""

    pass


class BasicFlowMatchingPolicy(PreTrainedPolicy):
    """Basic policy that matches flow with a simple neural network."""

    def __init__(self, cfg: BasicFlowMatchingConfig):
        super().__init__(cfg)
        # ...initialize your model here...

    def forward(self, obs):
        # ...define the forward pass...
        pass
