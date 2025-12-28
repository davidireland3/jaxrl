import jax
import jax.numpy as jnp
import optax
from flax import nnx

from .dqn import DQN

class DDQN(DQN):
    """DDQN algorithm."""
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @staticmethod
    @nnx.jit
    def jit_update(model, target_model, states, actions, rewards, next_states, dones, gamma, optimiser):
        def loss_fn(model):
            q_values = jnp.take_along_axis(model(states), actions.reshape(-1, 1), axis=-1).squeeze()

            online_q_values_next = model(next_states)
            argmax_actions = jnp.argmax(online_q_values_next, axis=-1, keepdims=True)
            q_values_next = jnp.take_along_axis(target_model(next_states), argmax_actions, axis=-1).squeeze()
            targets = rewards + gamma * q_values_next * (1 - dones)
            targets = jax.lax.stop_gradient(targets)
            loss = optax.losses.huber_loss(q_values, targets).mean()
            return loss

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimiser.update(model, grads)
        return loss
