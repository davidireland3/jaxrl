from dataclasses import dataclass

from .base import BaseConfig


@dataclass
class DDPGConfig(BaseConfig):
    # Agent params
    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 1e-3
    tau: float = 0.005

    # Exploration noise params
    noise_std: float = 0.1
    noise_decay: float = 0.9999
    noise_min: float = 0.01

    # Training params
    buffer_size: int = 100_000
    learning_starts: int = 1000
