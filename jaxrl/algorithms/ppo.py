import jax
import jax.numpy as jnp
import optax
from flax import nnx
from typing import Dict, Tuple, Any

from jaxrl.common.base_agent import BaseAgent
from jaxrl.networks import MLP, PolicyNetwork


class PPO(BaseAgent):
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dim: int,
            policy_learning_rate: float,
            value_learning_rate: float,
            gamma: float = 0.99,
            clip_epsilon: float = 0.2,
            entropy_coef: float = 0.01,
            max_grad_norm: float = 0.5,
            n_epochs: int = 10,
            batch_size: int = 64,
            rngs: nnx.Rngs = None,
    ) -> None:
        self.policy_network = PolicyNetwork(state_dim, action_dim, hidden_dim, rngs)
        self.value_network = MLP(state_dim, hidden_dim, 1, rngs)

        self.policy_optimiser = nnx.Optimizer(
            self.policy_network,
            optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(policy_learning_rate),
            ),
            wrt=nnx.Param
        )
        self.value_optimiser = nnx.Optimizer(
            self.value_network,
            optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(value_learning_rate),
            ),
            wrt=nnx.Param
        )
        self.gamma = gamma
        self.num_actions = action_dim

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.rngs = rngs
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef

    def select_action(self, state: jnp.ndarray, **kwargs) -> Tuple[int, float, float]:
        logits, log_probs, value = self._network_action_selection(self.policy_network, self.value_network,
                                                                  state, **kwargs)

        action = jax.random.categorical(self.rngs(), logits, -1)
        action_log_prob = log_probs[action]

        return int(action), float(action_log_prob), float(value.item())

    @staticmethod
    @nnx.jit
    def _network_action_selection(
            policy_model,
            value_model,
            state: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        logits = policy_model(state)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        value = value_model(state)
        return logits, log_probs, value

    @staticmethod
    @nnx.jit
    def jit_update(
            policy_net,
            value_net,
            policy_optimiser,
            value_optimiser,
            states,
            actions,
            old_log_probs,
            advantages,
            returns,
            clip_eps,
            entropy_coef,
    ):
        # Policy loss
        def policy_loss_fn(policy_net):
            logits = policy_net(states)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            action_log_probs = jnp.take_along_axis(log_probs, actions[:, None], axis=1).squeeze(-1)

            ratio = jnp.exp(action_log_probs - old_log_probs)
            clipped_ratio = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
            policy_loss = -jnp.minimum(ratio * advantages, clipped_ratio * advantages).mean()

            entropy = -(jnp.exp(log_probs) * log_probs).sum(axis=-1).mean()
            total_policy_loss = policy_loss - entropy_coef * entropy

            return total_policy_loss, {'policy': policy_loss, 'entropy': entropy}

        # Value loss
        def value_loss_fn(value_net):
            values = value_net(states).squeeze(-1)
            value_loss = optax.losses.huber_loss(values, returns).mean()
            return value_loss

        (policy_loss, policy_metrics), policy_grads = nnx.value_and_grad(policy_loss_fn, has_aux=True)(policy_net)
        value_loss, value_grads = nnx.value_and_grad(value_loss_fn)(value_net)

        policy_optimiser.update(policy_net, policy_grads)
        value_optimiser.update(value_net, value_grads)

        return {'policy': policy_metrics['policy'], 'value': value_loss, 'entropy': policy_metrics['entropy']}

    def update(self, batch: Dict[str, jnp.ndarray]) -> Dict[str, float]:
        states = batch['states']
        actions = batch['actions']
        olg_log_probs = batch['log_probs']
        old_values = batch['values']
        advantages = batch['advantages']
        returns = advantages + old_values

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_losses = {'policy': 0., 'value': 0., 'entropy': 0.}
        num_updates = 0

        for epoch in range(self.n_epochs):
            perm = jax.random.permutation(self.rngs(), len(states))

            for start in range(0, len(states), self.batch_size):
                idx = perm[start:start + self.batch_size]

                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_old_log_probs = olg_log_probs[idx]
                mb_advantages = advantages[idx]
                mb_returns = returns[idx]

                losses = self.jit_update(
                    policy_net=self.policy_network,
                    value_net=self.value_network,
                    policy_optimiser=self.policy_optimiser,
                    value_optimiser=self.value_optimiser,
                    states=mb_states,
                    actions=mb_actions,
                    old_log_probs=mb_old_log_probs,
                    advantages=mb_advantages,
                    returns=mb_returns,
                    clip_eps=self.clip_epsilon,
                    entropy_coef=self.entropy_coef,
                )
                for k, v in losses.items():
                    total_losses[k] += float(v)
                num_updates += 1

        return {k: v / num_updates for k, v in total_losses.items()}

    def _get_state(self, inference_only: bool = False) -> Dict[str, Any]:
        """Get PPO state for checkpointing."""
        if inference_only:
            return {
                'policy_network': nnx.state(self.policy_network),
                'value_network': nnx.state(self.value_network),
            }
        else:
            return {
                'policy_network': nnx.state(self.policy_network),
                'value_network': nnx.state(self.value_network),
                'policy_optimiser': nnx.state(self.policy_optimiser),
                'value_optimiser': nnx.state(self.value_optimiser),
                'rngs': nnx.state(self.rngs),
            }

    def _restore_state(self, state: Dict[str, Any], inference_mode: bool = False) -> None:
        """Restore PPO state from checkpoint."""
        nnx.update(self.policy_network, state['policy_network'])
        nnx.update(self.value_network, state['value_network'])
        if not inference_mode:
            nnx.update(self.policy_optimiser, state['policy_optimiser'])
            nnx.update(self.value_optimiser, state['value_optimiser'])
            nnx.update(self.rngs, state['rngs'])
