import jax.numpy as jnp
import numpy as np
from typing import Dict, Optional


class RolloutBuffer:
    """Rollout buffer for on-policy RL algorithms like PPO."""

    def __init__(
            self,
            state_dim: int,
            capacity: int,
            discrete_actions: bool = True,
            action_dim: Optional[int] = None,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
    ):
        """
        Args:
            state_dim: Observation dimension
            capacity: Maximum buffer size (typically n_steps * n_envs)
            discrete_actions: Whether actions are discrete (int) or continuous (float)
            action_dim: Action dimension (unused if discrete_actions=True)
            gamma: Discount factor for computing returns
            gae_lambda: Lambda parameter for GAE (Generalised Advantage Estimation)
        """
        self.capacity = capacity
        self.discrete_actions = discrete_actions
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.size = 0
        self.ptr = 0

        # Preallocate arrays
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)

        # These will be computed when get() is called
        self.advantages = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)

        if discrete_actions:
            self.actions = np.zeros(capacity, dtype=np.int32)
        else:
            self.actions = np.zeros((capacity, action_dim), dtype=np.float32)

    def add(
            self,
            state: np.ndarray,
            action: np.ndarray,
            reward: float,
            done: bool,
            value: float,
            log_prob: float,
    ):
        """
        Add a transition to the buffer.

        Args:
            state: Current observation
            action: Action taken
            reward: Reward received - note that if episode was truncated we assume that we have added gamma * V(s') to r
            done: Whether episode truly ended
            value: Value estimate V(s) from critic
            log_prob: Log probability of the action
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob

        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)

    def compute_advantages_and_returns(self, last_value: float = 0.0):
        """
        Compute advantages using GAE and returns.

        This should be called after collecting a full rollout and before sampling.

        Args:
            last_value: Value estimate for the last state (V(s_T)).
                        Use 0.0 if episode terminated, otherwise use critic value.
        """
        # GAE (Generalised Advantage Estimation)
        # A_t = δ_t + (γλ)δ_{t+1} + (γλ)^2 δ_{t+2} + ...
        # where δ_t = r_t + γV(s_{t+1}) - V(s_t)

        last_gae_lambda = 0
        for step in reversed(range(self.ptr)):
            if step == self.ptr - 1:
                next_value = last_value
            elif self.dones[step]:
                next_value = 0.0
            else:
                next_value = self.values[step + 1]

            next_non_terminal = 1.0 - self.dones[step]

            # TD residual: δ_t = r_t + γV(s_{t+1}) - V(s_t)
            delta = self.rewards[step] + self.gamma * next_value * next_non_terminal - self.values[step]

            # GAE: A_t = δ_t + (γλ)(1 - done)A_{t+1}
            # Reset trace on episode boundaries
            if self.dones[step]:
                last_gae_lambda = delta
            else:
                last_gae_lambda = delta + self.gamma * self.gae_lambda * last_gae_lambda

            self.advantages[step] = last_gae_lambda

        # Returns = Advantages + Values
        self.returns[:self.ptr] = self.advantages[:self.ptr] + self.values[:self.ptr]

    def sample(self, last_value: float = 0.0) -> Dict[str, jnp.ndarray]:
        """
        Sample all the transitions in the buffer as JAX arrays.

        Args:
            last_value: Value estimate for the last state. Pass 0.0 if terminated,
                        otherwise pass critic's value estimate.

        Returns:
            batch: Dictionary of JAX arrays
        """
        self.compute_advantages_and_returns(last_value)
        return {
            'states': jnp.array(self.states[:self.ptr]),
            'actions': jnp.array(self.actions[:self.ptr]),
            'log_probs': jnp.array(self.log_probs[:self.ptr]),
            'advantages': jnp.array(self.advantages[:self.ptr]),
            'returns': jnp.array(self.returns[:self.ptr]),
            'values': jnp.array(self.values[:self.ptr]),
        }

    def reset(self):
        """Clear the buffer. Should be called after each PPO update epoch."""
        self.ptr = 0
        self.size = 0

    def __len__(self) -> int:
        """Return current buffer size."""
        return self.ptr
