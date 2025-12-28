import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from typing import Dict

from jaxrl.common.base_agent import BaseAgent
from jaxrl.utils.schedules import LinearSchedule
from jaxrl.networks.q_network import QNetwork


class DQN(BaseAgent):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, learning_rate: float, epsilon_duration: int,
                 epsilon_min: float, tau: float = 0.005, gamma: float = 0.99, rngs: nnx.Rngs = None) -> None:
        self.qnet = QNetwork(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim, rngs=rngs)
        self.target_qnet = QNetwork(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim, rngs=rngs)
        target_state = nnx.state(self.qnet, nnx.Param)
        nnx.update(self.target_qnet, target_state)

        self.optimiser = nnx.Optimizer(self.qnet, optax.adam(learning_rate), wrt=nnx.Param)
        self.gamma = gamma
        self.epsilon_schedule = LinearSchedule(1, epsilon_min, epsilon_duration)
        self.num_actions = action_dim
        self.tau = jnp.array(tau)
        self.loss_fn = optax.losses.huber_loss

    def select_action(self, state: jnp.ndarray, **kwargs) -> int:
        if np.random.uniform() < self.epsilon_schedule(self.step):
            return np.random.randint(self.num_actions)
        else:
            return self._network_action_selection(self.qnet, state).item()

    @staticmethod
    @nnx.jit
    def _network_action_selection(model, state: jnp.ndarray) -> jnp.ndarray:
        q_values = model(state)
        return jnp.argmax(q_values)

    @staticmethod
    @nnx.jit
    def jit_update(model, target_model, states, actions, rewards, next_states, dones, gamma, optimiser):
        def loss_fn(model):
            q_values = jnp.take_along_axis(model(states), actions.reshape(-1, 1), axis=-1).squeeze()
            q_values_next = jnp.max(target_model(next_states), axis=-1)
            targets = rewards + gamma * q_values_next * (1 - dones)
            targets = jax.lax.stop_gradient(targets)
            loss = optax.losses.huber_loss(q_values, targets).mean()
            return loss

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimiser.update(model, grads)
        return loss

    def update(self, batch: Dict[str, jnp.ndarray]) -> Dict[str, float]:
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        dones = batch['dones']
        next_states = batch['next_states']
        loss = self.jit_update(self.qnet, self.target_qnet, states, actions, rewards, next_states, dones, self.gamma, self.optimiser)
        self.soft_update_target_network(self.qnet, self.target_qnet, self.tau)
        return {'loss': loss}

    @staticmethod
    # @nnx.jit
    def soft_update_target_network(model, target_model, tau):
        """
        Soft update (Polyak averaging):
        target = tau * qnet + (1 - tau) * target

        Args:
            tau: Interpolation parameter (typically 0.001 - 0.01)
            :param tau:
            :param target_model:
            :param model:
        """
        target_state = nnx.state(target_model, nnx.Param)
        qnet_state = nnx.state(model, nnx.Param)

        # Use jax.tree.map to interpolate between the two states
        new_target_state = jax.tree.map(
            lambda target, source: tau * source + (1 - tau) * target,
            target_state,
            qnet_state
        )

        # Update the target network with the new state
        nnx.update(target_model, new_target_state)
