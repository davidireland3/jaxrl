import argparse
import gymnasium as gym
import pickle
from flax import nnx
from loguru import logger
from pathlib import Path

from jaxrl.algorithms import DDPG, DQN, DDQN, PPO
from jaxrl.utils.agent_registry import get_agent_info
from jaxrl.utils.env_utils import get_action_space_info


def test_agent(
    agent,
    env,
    n_episodes=1,
    seed=42,
    algo_class=None
):
    """
    Run test episodes with the agent.

    Args:
        agent: RL agent
        env: Gymnasium environment
        n_episodes: Number of episodes to run
        seed: Random seed
        algo_class: Agent class (for greedy action selection)

    Returns:
        Dictionary with episode statistics
    """
    episode_rewards = []
    episode_lengths = []

    for episode in range(n_episodes):
        state, _ = env.reset(seed=seed + episode)
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            # Select action (greedy for DQN/DDQN, deterministic for DDPG, normal for PPO)
            if algo_class in [DQN, DDQN]:
                # Use greedy action selection for DQN/DDQN
                action = agent._network_action_selection(agent.qnet, state).item()
            elif algo_class == DDPG:
                # Use deterministic action selection for DDPG
                action = agent.select_action(state, deterministic=True)
            elif algo_class == PPO:
                # PPO returns (action, log_prob, value)
                action, _, _ = agent.select_action(state)
            else:
                action = agent.select_action(state)

            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        logger.info(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")

    mean_reward = sum(episode_rewards) / len(episode_rewards)
    mean_length = sum(episode_lengths) / len(episode_lengths)

    logger.info(f"\n{'='*50}")
    logger.info(f"Results over {n_episodes} episode(s):")
    logger.info(f"Mean Reward: {mean_reward:.2f}")
    logger.info(f"Mean Length: {mean_length:.2f}")
    logger.info(f"{'='*50}")

    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'mean_reward': mean_reward,
        'mean_length': mean_length
    }


def main():
    parser = argparse.ArgumentParser(description="Test a trained RL agent")
    parser.add_argument("--env", type=str, required=True, help="Environment name")
    parser.add_argument("--algo", type=str, required=True, help="Algorithm name (dqn, ddqn, ppo)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used during training")
    parser.add_argument("--checkpoint", type=str, default="best.pkl", help="Checkpoint filename (best.pkl or latest.pkl)")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument("--n-episodes", type=int, default=1, help="Number of episodes to run")
    args = parser.parse_args()

    # Build checkpoint path
    checkpoint_path = Path("results") / args.env / args.algo / str(args.seed) / args.checkpoint

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.error(f"Make sure you've trained a model first with: python scripts/train.py --algo {args.algo} --env {args.env}")
        return

    # Load checkpoint to get config
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)

    config_dict = checkpoint['metadata'].get('config')
    if config_dict is None:
        logger.error("Checkpoint was saved without config. Please retrain the model with the updated code.")
        return

    # Get agent class
    agent_info = get_agent_info(args.algo)
    AgentClass = agent_info['class']

    logger.info(f"Testing {args.algo.upper()} on {args.env}")
    logger.info(f"Checkpoint timestamp: {checkpoint['metadata']['timestamp']}")
    logger.info(f"Checkpoint step: {checkpoint['step']}")

    # Create environment
    render_mode = "human" if args.render else None
    env = gym.make(args.env, render_mode=render_mode)

    state_dim = env.observation_space.shape[0]
    action_dim, discrete_actions = get_action_space_info(env.action_space)

    logger.info(f"State dim: {state_dim}, Action dim: {action_dim}, Discrete: {discrete_actions}")

    # Create agent with saved config
    if AgentClass in [DQN, DDQN]:
        agent = AgentClass(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=config_dict['hidden_dim'],
            learning_rate=config_dict['learning_rate'],
            gamma=config_dict['gamma'],
            tau=config_dict['tau'],
            epsilon_min=config_dict['epsilon_min'],
            epsilon_duration=config_dict['epsilon_duration'],
            rngs=nnx.Rngs(args.seed),
        )
    elif AgentClass == DDPG:
        agent = AgentClass(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=config_dict['hidden_dim'],
            actor_learning_rate=config_dict['actor_learning_rate'],
            critic_learning_rate=config_dict['critic_learning_rate'],
            gamma=config_dict['gamma'],
            tau=config_dict['tau'],
            noise_std=config_dict['noise_std'],
            noise_decay=config_dict['noise_decay'],
            noise_min=config_dict['noise_min'],
            rngs=nnx.Rngs(args.seed),
        )
    elif AgentClass == PPO:
        agent = AgentClass(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=config_dict['hidden_dim'],
            policy_learning_rate=config_dict['policy_learning_rate'],
            value_learning_rate=config_dict['value_learning_rate'],
            gamma=config_dict['gamma'],
            clip_epsilon=config_dict['clip_epsilon'],
            entropy_coef=config_dict['entropy_coef'],
            max_grad_norm=config_dict['max_grad_norm'],
            batch_size=config_dict['batch_size'],
            n_epochs=config_dict['n_epochs'],
            rngs=nnx.Rngs(args.seed),
        )
    else:
        logger.error(f"Unknown agent class: {AgentClass}")
        return

    # Load weights
    agent.load_checkpoint(checkpoint_path, inference_mode=True)

    # Test agent
    results = test_agent(
        agent=agent,
        env=env,
        n_episodes=args.n_episodes,
        seed=args.seed,
        algo_class=AgentClass
    )

    env.close()
    logger.info("Testing complete!")


if __name__ == "__main__":
    main()
