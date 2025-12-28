from dataclasses import dataclass


@dataclass
class BaseConfig:
    # Environment
    env_name: str = "CartPole-v1"
    seed: int = 42

    # Network
    hidden_dim: int = 512

    # Training params
    total_timesteps: int = 50_000
    train_freq: int = 1
    eval_freq: int = 2_500
    eval_episodes: int = 10
    log_freq: int = 1000
    batch_size: int = 256
    learning_starts: int = 1
    gamma: float = 0.99
