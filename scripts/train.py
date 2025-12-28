import argparse
import gymnasium as gym
from loguru import logger
from flax import nnx

from jaxrl.buffers.replay_buffer import ReplayBuffer
from jaxrl.common.runner import train
from jaxrl.utils.agent_registry import get_agent_info
from jaxrl.utils.config_utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, required=True, help="Algorithm name (dqn, ppo, sac)")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML (optional)")
    args = parser.parse_args()

    # Get agent class and config
    agent_info = get_agent_info(args.algo)
    AgentClass = agent_info['class']
    ConfigClass = agent_info['config']

    config = load_config(ConfigClass, args.config)
    logger.info(f"Algorithm: {args.algo.upper()}")
    logger.info(f"Config: {config}")

    # Create environment
    env = gym.make(config.env_name)
    eval_env = gym.make(config.env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Create agent
    agent = AgentClass(
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

    # Create buffer
    buffer = ReplayBuffer(
        state_dim=state_dim,
        capacity=config.buffer_size,
        discrete_actions=True,
    )

    logger.info("Starting training...")

    # Train
    metrics = train(
        agent=agent,
        env=env,
        buffer=buffer,
        total_timesteps=config.total_timesteps,
        learning_starts=config.learning_starts,
        train_freq=config.train_freq,
        eval_freq=config.eval_freq,
        eval_env=eval_env,
        eval_episodes=config.eval_episodes,
        log_freq=config.log_freq,
        seed=config.seed,
    )

    logger.info("Training complete!")
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
