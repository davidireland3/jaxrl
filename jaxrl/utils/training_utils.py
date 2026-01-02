from flax import nnx

from jaxrl.algorithms import DDPG, DQN, DDQN, PPO
from jaxrl.buffers import RolloutBuffer
from jaxrl.buffers.replay_buffer import ReplayBuffer
from jaxrl.common import off_policy_train, on_policy_train


def create_agent_from_config(AgentClass, config, state_dim: int, action_dim: int):
    """
    Create an agent instance from config.

    Args:
        AgentClass: Agent class to instantiate
        config: Configuration object
        state_dim: State dimension
        action_dim: Action dimension

    Returns:
        Instantiated agent
    """
    if AgentClass in [DQN, DDQN]:
        return AgentClass(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=config.hidden_dim,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            tau=config.tau,
            epsilon_min=config.epsilon_min,
            epsilon_duration=config.epsilon_duration,
            rngs=nnx.Rngs(config.seed),
        )
    elif AgentClass == DDPG:
        return AgentClass(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=config.hidden_dim,
            actor_learning_rate=config.actor_learning_rate,
            critic_learning_rate=config.critic_learning_rate,
            gamma=config.gamma,
            tau=config.tau,
            noise_std=config.noise_std,
            noise_decay=config.noise_decay,
            noise_min=config.noise_min,
            rngs=nnx.Rngs(config.seed),
        )
    elif AgentClass == PPO:
        return AgentClass(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=config.hidden_dim,
            policy_learning_rate=config.policy_learning_rate,
            value_learning_rate=config.value_learning_rate,
            gamma=config.gamma,
            clip_epsilon=config.clip_epsilon,
            entropy_coef=config.entropy_coef,
            max_grad_norm=config.max_grad_norm,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            rngs=nnx.Rngs(config.seed),
        )
    else:
        raise ValueError(f"Unknown agent class: {AgentClass}")


def create_buffer(AgentClass, config, state_dim: int, action_dim: int, discrete_actions: bool):
    """
    Create a replay/rollout buffer for the agent.

    Args:
        AgentClass: Agent class
        config: Configuration object
        state_dim: State dimension
        action_dim: Action dimension
        discrete_actions: Whether actions are discrete

    Returns:
        Buffer instance
    """
    if AgentClass in [DQN, DDQN, DDPG]:
        return ReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            capacity=config.buffer_size,
            discrete_actions=discrete_actions,
        )
    elif AgentClass == PPO:
        return RolloutBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            capacity=config.train_freq,
            discrete_actions=discrete_actions,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )
    else:
        raise ValueError(f"Unknown agent class: {AgentClass}")


def get_training_function(AgentClass):
    """
    Get the appropriate training function for the agent class.

    Args:
        AgentClass: Agent class

    Returns:
        Training function (off_policy_train or on_policy_train)
    """
    if AgentClass in [DQN, DDQN, DDPG]:
        return off_policy_train
    elif AgentClass == PPO:
        return on_policy_train
    else:
        raise ValueError(f"Unknown agent class: {AgentClass}")
