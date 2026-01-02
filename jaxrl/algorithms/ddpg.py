import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from typing import Any, Dict

from jaxrl.common.base_agent import BaseAgent
from jaxrl.networks.actor_network import ActorNetwork
from jaxrl.networks.critic_network import CriticNetwork


class DDPG(BaseAgent):
    """Deep Deterministic Policy Gradient algorithm."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        noise_std: float = 0.1,
        noise_decay: float = 0.9999,
        noise_min: float = 0.01,
        tau: float = 0.005,
        gamma: float = 0.99,
        rngs: nnx.Rngs = None,
    ) -> None:
        # Create actor and critic networks
        self.actor = ActorNetwork(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim, rngs=rngs)
        self.critic = CriticNetwork(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim, rngs=rngs)

        # Create target networks
        self.target_actor = ActorNetwork(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim, rngs=rngs)
        self.target_critic = CriticNetwork(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim, rngs=rngs)

        # Initialise target networks with same weights as main networks
        target_actor_state = nnx.state(self.actor, nnx.Param)
        nnx.update(self.target_actor, target_actor_state)
        target_critic_state = nnx.state(self.critic, nnx.Param)
        nnx.update(self.target_critic, target_critic_state)

        # Create optimisers
        self.actor_optimiser = nnx.Optimizer(self.actor, optax.adam(actor_learning_rate), wrt=nnx.Param)
        self.critic_optimiser = nnx.Optimizer(self.critic, optax.adam(critic_learning_rate), wrt=nnx.Param)

        # Store hyperparameters
        self.gamma = gamma
        self.tau = jnp.array(tau)
        self.action_dim = action_dim
        self.noise_std = noise_std
        self.noise_decay = noise_decay
        self.noise_min = noise_min
        self.current_noise_std = noise_std

    def select_action(self, state: jnp.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Select action using the actor network with exploration noise.

        Args:
            state: Current state observation
            deterministic: If True, return deterministic action without noise

        Returns:
            action: Continuous action vector
        """
        action = self._network_action_selection(self.actor, state)

        if not deterministic:
            # Add Gaussian exploration noise
            noise = np.random.normal(0, self.current_noise_std, size=self.action_dim)
            action = np.clip(action + noise, -1.0, 1.0)

            # Decay noise
            self.current_noise_std = max(self.noise_min, self.current_noise_std * self.noise_decay)

        return action

    @staticmethod
    @nnx.jit
    def _network_action_selection(actor: ActorNetwork, state: jnp.ndarray) -> jnp.ndarray:
        """Get deterministic action from actor network."""
        return actor(state)

    @staticmethod
    @nnx.jit
    def jit_critic_update(
        critic: CriticNetwork,
        target_critic: CriticNetwork,
        target_actor: ActorNetwork,
        states: jnp.ndarray,
        actions: jnp.ndarray,
        rewards: jnp.ndarray,
        next_states: jnp.ndarray,
        dones: jnp.ndarray,
        gamma: float,
        optimiser: nnx.Optimizer,
    ) -> float:
        """Update critic network using TD error."""

        def critic_loss_fn(critic):
            # Current Q-values
            q_values = critic(states, actions)

            # Target Q-values
            next_actions = target_actor(next_states)
            target_q_values = target_critic(next_states, next_actions)
            targets = rewards + gamma * target_q_values * (1 - dones)
            targets = jax.lax.stop_gradient(targets)

            # MSE loss
            loss = optax.losses.squared_error(q_values, targets).mean()
            return loss

        loss, grads = nnx.value_and_grad(critic_loss_fn)(critic)
        optimiser.update(critic, grads)
        return loss

    @staticmethod
    @nnx.jit
    def jit_actor_update(
        actor: ActorNetwork,
        critic: CriticNetwork,
        states: jnp.ndarray,
        optimiser: nnx.Optimizer,
    ) -> float:
        """Update actor network using policy gradient."""

        def actor_loss_fn(actor):
            # Actor loss: negative mean Q-value
            actions = actor(states)
            q_values = critic(states, actions)
            loss = -q_values.mean()
            return loss

        loss, grads = nnx.value_and_grad(actor_loss_fn)(actor)
        optimiser.update(actor, grads)
        return loss

    def update(self, batch: Dict[str, jnp.ndarray]) -> Dict[str, float]:
        """
        Update both actor and critic networks.

        Args:
            batch: Dictionary containing transitions

        Returns:
            metrics: Dictionary of training metrics
        """
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        dones = batch["dones"]
        next_states = batch["next_states"]

        # Update critic
        critic_loss = self.jit_critic_update(
            self.critic,
            self.target_critic,
            self.target_actor,
            states,
            actions,
            rewards,
            next_states,
            dones,
            self.gamma,
            self.critic_optimiser,
        )

        # Update actor
        actor_loss = self.jit_actor_update(self.actor, self.critic, states, self.actor_optimiser)

        # Soft update target networks
        self.soft_update_target_network(self.actor, self.target_actor, self.tau)
        self.soft_update_target_network(self.critic, self.target_critic, self.tau)

        return {"critic_loss": critic_loss, "actor_loss": actor_loss}

    @staticmethod
    def soft_update_target_network(model: nnx.Module, target_model: nnx.Module, tau: float) -> None:
        """
        Soft update (Polyak averaging):
        target = tau * model + (1 - tau) * target

        Args:
            model: Source network
            target_model: Target network to update
            tau: Interpolation parameter (typically 0.001 - 0.01)
        """
        target_state = nnx.state(target_model, nnx.Param)
        model_state = nnx.state(model, nnx.Param)

        # Use jax.tree.map to interpolate between the two states
        new_target_state = jax.tree.map(
            lambda target, source: tau * source + (1 - tau) * target, target_state, model_state
        )

        # Update the target network with the new state
        nnx.update(target_model, new_target_state)

    def _get_state(self, inference_only: bool = False) -> Dict[str, Any]:
        """Get DDPG state for checkpointing."""
        if inference_only:
            return {"actor": nnx.state(self.actor), "critic": nnx.state(self.critic)}
        else:
            return {
                "actor": nnx.state(self.actor),
                "critic": nnx.state(self.critic),
                "target_actor": nnx.state(self.target_actor),
                "target_critic": nnx.state(self.target_critic),
                "actor_optimiser": nnx.state(self.actor_optimiser),
                "critic_optimiser": nnx.state(self.critic_optimiser),
            }

    def _restore_state(self, state: Dict[str, Any], inference_mode: bool = False) -> None:
        """Restore DDPG state from checkpoint."""
        nnx.update(self.actor, state["actor"])
        nnx.update(self.critic, state["critic"])
        if not inference_mode:
            nnx.update(self.target_actor, state["target_actor"])
            nnx.update(self.target_critic, state["target_critic"])
            nnx.update(self.actor_optimiser, state["actor_optimiser"])
            nnx.update(self.critic_optimiser, state["critic_optimiser"])
