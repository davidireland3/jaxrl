import jax.numpy as jnp
from flax import nnx

from .simple_mlp import MLP


class ActorNetwork(nnx.Module):
    """
    Actor network for DDPG that outputs deterministic continuous actions.
    Uses tanh activation to bound outputs to [-1, 1].
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, rngs: nnx.Rngs):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.mlp = MLP(input_dim=state_dim, hidden_dim=hidden_dim, output_dim=action_dim, rngs=rngs)

    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass of the actor network.

        Args:
            state: State observation

        Returns:
            Continuous action bounded to [-1, 1]
        """
        return jnp.tanh(self.mlp(state))
