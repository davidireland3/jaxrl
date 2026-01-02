import argparse
import gymnasium as gym
import wandb
from loguru import logger

from jaxrl.utils.agent_registry import get_agent_info
from jaxrl.utils.config_utils import load_config
from jaxrl.utils.env_utils import get_action_space_info
from jaxrl.utils.training_utils import create_agent_from_config, create_buffer, get_training_function


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
    action_dim, discrete_actions = get_action_space_info(env.action_space)

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

    # Create agent and buffer
    agent = create_agent_from_config(AgentClass, config, state_dim, action_dim)
    buffer = create_buffer(AgentClass, config, state_dim, action_dim, discrete_actions)
    train_fn = get_training_function(AgentClass)

    # Train
    metrics = train_fn(
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
        config=config,
        algo_name=args.algo,
    )

    logger.info("Training complete!")
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
