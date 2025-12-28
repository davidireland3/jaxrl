from jaxrl.algorithms.dqn import DQN
from jaxrl.algorithms.ddqn import DDQN
from jaxrl.configs.dqn import DQNConfig


AGENT_REGISTRY = {
    'dqn': {
        'class': DQN,
        'config': DQNConfig,
    },
    'ddqn': {
        'class': DDQN,
        'config': DQNConfig,
    }
}


def get_agent_info(algo_name: str):
    """Get agent class and config class for algorithm."""
    if algo_name.lower() not in AGENT_REGISTRY:
        raise ValueError(f"Unknown algorithm: {algo_name}")
    return AGENT_REGISTRY[algo_name.lower()]
