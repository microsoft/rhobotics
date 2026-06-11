import abc


class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: dict) -> dict:
        """Infer actions from observations."""

    def reset(self) -> None:
        """Reset the policy to its initial state."""
        pass
