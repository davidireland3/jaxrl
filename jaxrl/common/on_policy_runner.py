import gymnasium as gym
from loguru import logger
from typing import Optional, Dict, Any
from tqdm import tqdm

from jaxrl.common.base_agent import BaseAgent
from jaxrl.buffers import RolloutBuffer


def train(
        agent: BaseAgent,
        env: gym.Env,
        buffer: RolloutBuffer,
        total_timesteps: int,
        train_freq: int = 1,
        eval_freq: Optional[int] = None,
        eval_env: Optional[gym.Env] = None,
        eval_episodes: int = 10,
        log_freq: int = 1000,
        seed: int = 0,
) -> Dict[str, Any]:
    """
    Generic training loop for RL agents.

    Args:
        agent: RL agent (must inherit from BaseAgent)
        env: Training environment
        buffer: Experience buffer (ReplayBuffer or RolloutBuffer)
        total_timesteps: Total training steps
        train_freq: Update agent every N steps
        eval_freq: Evaluate every N steps (None = no eval)
        eval_env: Evaluation environment (if None, uses env)
        eval_episodes: Number of eval episodes
        log_freq: Print metrics every N steps
        seed: Random seed

    Returns:
        metrics: Dictionary of training metrics
    """
    state, _ = env.reset(seed=seed)
    episode_reward = 0
    episode_count = 0
    all_episode_rewards = []

    for step in tqdm(range(total_timesteps)):
        # Select action
        action, log_prob, value = agent.select_action(state)

        # Step environment
        next_state, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        done = terminated or truncated
        if truncated:
            _, _, next_value = agent.select_action(next_state)
            reward += agent.gamma * next_value

        # Store transition
        buffer.add(state, action, reward, done, value, log_prob)

        state = next_state

        if done:
            state, _ = env.reset()
            all_episode_rewards.append(episode_reward)
            episode_count += 1

            if episode_count % 10 == 0:
                logger.info(f"Episode {episode_count}, Reward: {episode_reward:.2f}")

            episode_reward = 0

        # Training
        if (step + 1) % train_freq == 0:
            next_value = 0
            if not terminated:
                _, _, next_value = agent.select_action(state)
            batch = buffer.sample(next_value)
            metrics = agent.update(batch)
            buffer.reset()
            agent.step += 1

            if step % log_freq == 0:
                logger.info(f"Step {step}, Metrics: {metrics}")

        # Evaluation
        if eval_freq is not None and step % eval_freq == 0 and step > 0:
            eval_env = eval_env or env
            eval_reward = evaluate(agent, eval_env, n_episodes=eval_episodes, seed=seed)
            logger.info(f"[Eval] Step {step}, Mean Reward: {eval_reward:.2f}")

    return {
        "episode_rewards": all_episode_rewards,
        "final_eval": evaluate(agent, eval_env or env, eval_episodes) if eval_env else None,
    }


def evaluate(agent: BaseAgent, env: gym.Env, n_episodes: int = 10, seed: int = 42) -> float:
    """
    Evaluate agent without exploration.

    Args:
        agent: RL agent
        env: Evaluation environment
        n_episodes: Number of episodes to run

    Returns:
        mean_reward: Average episode reward
    """
    total_reward = 0
    for idx in range(n_episodes):
        state, _ = env.reset(seed=seed + 100 * idx)
        done = False
        while not done:
            action, _, _ = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
    return total_reward / n_episodes
