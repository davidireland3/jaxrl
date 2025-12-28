from jaxrl.algorithms import *
from jaxrl.configs import *


AGENT_REGISTRY = {
    'dqn': {
        'class': DQN,
        'config': DQNConfig,
    },
    'ddqn': {
        'class': DDQN,
        'config': DQNConfig,
    },
    'ppo': {
        'class': PPO,
        'config': PPOConfig,
    }
}


def get_agent_info(algo_name: str):
    """Get agent class and config class for algorithm."""
    if algo_name.lower() not in AGENT_REGISTRY:
        raise ValueError(f"Unknown algorithm: {algo_name}")
    return AGENT_REGISTRY[algo_name.lower()]
