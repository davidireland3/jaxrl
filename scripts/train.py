import argparse
import gymnasium as gym
import wandb
from loguru import logger
from flax import nnx

from jaxrl.algorithms import *
from jaxrl.buffers import RolloutBuffer
from jaxrl.buffers.replay_buffer import ReplayBuffer
from jaxrl.common import off_policy_train, on_policy_train
from jaxrl.utils.agent_registry import get_agent_info
from jaxrl.utils.config_utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, required=True, help="Algorithm name (dqn, ppo, sac)")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML (optional)")
    parser.add_argument("--env", type=str, default=None, help="Environment name")
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="jaxrl", help="W&B project name")
    args = parser.parse_args()

    # Get agent class and config
    agent_info = get_agent_info(args.algo)
    AgentClass = agent_info['class']
    ConfigClass = agent_info['config']

    config = load_config(ConfigClass, args.config)
    logger.info(f"Algorithm: {args.algo.upper()}")
    logger.info(f"Config: {config}")

    # Create environment
    if args.env is not None:
        config.env_name = args.env
    env = gym.make(config.env_name)
    eval_env = gym.make(config.env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Initialise wandb if enabled
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"{args.algo}_{config.env_name}",
            config=vars(config),
        )
        # Define custom x-axes for different metric types
        wandb.define_metric("agent_step")
        wandb.define_metric("loss/*", step_metric="agent_step")
        wandb.define_metric("train/*", step_metric="env_step")
        wandb.define_metric("eval/*", step_metric="env_step")
        logger.info("Weights & Biases logging enabled")

    logger.info("Starting training...")

    # Train
    if AgentClass in [DQN, DDQN]:
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
        metrics = off_policy_train(
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
            use_wandb=args.use_wandb,
        )
    elif AgentClass == PPO:
        agent = AgentClass(
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
        buffer = RolloutBuffer(
            state_dim=state_dim,
            capacity=config.train_freq,
            discrete_actions=True,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda
        )
        metrics = on_policy_train(
            agent=agent,
            env=env,
            buffer=buffer,
            total_timesteps=config.total_timesteps,
            train_freq=config.train_freq,
            eval_freq=config.eval_freq,
            eval_env=eval_env,
            eval_episodes=config.eval_episodes,
            log_freq=config.log_freq,
            seed=config.seed,
            use_wandb=args.use_wandb,
        )

    logger.info("Training complete!")
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
