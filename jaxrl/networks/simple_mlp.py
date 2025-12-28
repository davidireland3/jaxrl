import jax.numpy as jnp
from flax import nnx


class MLP(nnx.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, rngs: nnx.Rngs):
        self.input_layer = nnx.Linear(input_dim, hidden_dim, rngs=rngs)
        self.hidden_layer = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.output_layer = nnx.Linear(hidden_dim, output_dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nnx.relu(self.input_layer(x))
        x = nnx.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x
