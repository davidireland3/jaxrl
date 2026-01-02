import gymnasium as gym
from typing import Tuple


def get_action_space_info(action_space: gym.Space) -> Tuple[int, bool]:
    """
    Get action dimension and whether the action space is discrete.

    Args:
        action_space: Gymnasium action space

    Returns:
        Tuple of (action_dim, discrete_actions)

    Raises:
        ValueError: If action space type is not supported
    """
    if isinstance(action_space, gym.spaces.Discrete):
        action_dim = action_space.n
        discrete_actions = True
    elif isinstance(action_space, gym.spaces.Box):
        action_dim = action_space.shape[0]
        discrete_actions = False
    else:
        raise ValueError(f"Unsupported action space type: {type(action_space)}")

    return action_dim, discrete_actions
