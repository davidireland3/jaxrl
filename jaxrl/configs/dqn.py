from dataclasses import dataclass

from .base import BaseConfig


@dataclass
class DQNConfig(BaseConfig):
    # Agent params
    learning_rate: float = 1e-4
    tau: float = 0.005
    epsilon_min: float = 0.01
    epsilon_duration: int = 10_000

    # Training params
    buffer_size: int = 100_000
    learning_starts: int = 1000
