from dataclasses import dataclass

from .base import BaseConfig


@dataclass
class DQNConfig(BaseConfig):
    # Agent params
    hidden_dim: int = 512
    learning_rate: float = 1e-4
    gamma: float = 0.99
    tau: float = 0.005
    epsilon_min: float = 0.01
    epsilon_duration: int = 10_000

    # Training params
    buffer_size: int = 100_000
    learning_starts: int = 1000
