import jax.numpy as jnp
from abc import ABC, abstractmethod
from flax import nnx
from typing import Dict


class BaseAgent(nnx.Module, ABC):
    """Abstract base class for RL agents."""
    step = 0  # Number of updates we've done
    gamma = 0.99

    @abstractmethod
    def select_action(self, state: jnp.ndarray, **kwargs) -> int:
        """
        Select an action given an observation.

        Args:
            state: Observation from environment
            **kwargs: Algorithm-specific args (e.g., epsilon for DQN, deterministic for SAC)

        Returns:
            action: Selected action
        """
        pass

    @abstractmethod
    def update(self, batch: Dict[str, jnp.ndarray]) -> Dict[str, float]:
        """
        Update agent from a batch of data.

        Args:
            batch: Dictionary containing transitions
                   (states, actions, rewards, next_states, dones)

        Returns:
            metrics: Dictionary of training metrics (e.g., loss, Q-values)
        """
        pass

    def save(self, path: str):
        """Save agent to disk."""
        raise NotImplementedError("Save not implemented for this agent")

    def load(self, path: str):
        """Load agent from disk."""
        raise NotImplementedError("Load not implemented for this agent")
