from dataclasses import dataclass

from .base import BaseConfig


@dataclass
class PPOConfig(BaseConfig):
    train_freq: int = 128
    batch_size: int = 32
    policy_learning_rate: float = 1e-4
    value_learning_rate: float = 1e-3
    max_grad_norm: float = 2.0
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    n_epochs: int = 3
    gae_lambda: float = 0.95