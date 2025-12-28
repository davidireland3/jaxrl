import jax.numpy as jnp
import numpy as np
from typing import Dict, Optional, Union


class ReplayBuffer:
    """Experience replay buffer for off-policy RL algorithms."""

    def __init__(
            self,
            state_dim: int,
            capacity: int,
            discrete_actions: bool = True,
            action_dim: Optional[int] = None,
    ):
        """
        Args:
            state_dim: Observation dimension
            capacity: Maximum buffer size
            discrete_actions: Whether actions are discrete (int) or continuous (float)
            action_dim: Action dimension (unused if discrete_actions=True)
        """
        self.capacity = capacity
        self.discrete_actions = discrete_actions
        self.size = 0
        self.ptr = 0

        # Preallocate arrays
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

        if discrete_actions:
            self.actions = np.zeros(capacity, dtype=np.int32)
        else:
            self.actions = np.zeros((capacity, action_dim), dtype=np.float32)

    def add(
            self,
            obs: np.ndarray,
            action: Union[np.ndarray, int],
            reward: float,
            next_obs: np.ndarray,
            done: bool,
    ):
        """Add a transition to the buffer."""
        self.states[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_obs
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, jnp.ndarray]:
        """
        Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            batch: Dictionary of JAX arrays
        """
        indices = np.random.randint(0, self.size, size=batch_size)

        return {
            'states': jnp.array(self.states[indices]),
            'actions': jnp.array(self.actions[indices]),
            'rewards': jnp.array(self.rewards[indices]),
            'next_states': jnp.array(self.next_states[indices]),
            'dones': jnp.array(self.dones[indices], dtype=jnp.float32),
        }

    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size
