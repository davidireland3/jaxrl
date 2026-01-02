import jax.numpy as jnp
from flax import nnx

from .simple_mlp import MLP


class CriticNetwork(nnx.Module):
    """
    Critic network for DDPG that estimates Q(s, a).
    Concatenates state and action, then passes through MLP.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, rngs: nnx.Rngs):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.mlp = MLP(input_dim=state_dim + action_dim, hidden_dim=hidden_dim, output_dim=1, rngs=rngs)

    def __call__(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass of the critic network.

        Args:
            state: State observation
            action: Continuous action

        Returns:
            Q-value estimate (scalar)
        """
        x = jnp.concatenate([state, action], axis=-1)
        return self.mlp(x).squeeze(-1)
