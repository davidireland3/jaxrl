import yaml
from dataclasses import replace
from pathlib import Path
from typing import Optional, Union

from jaxrl.configs.base import BaseConfig
from jaxrl.configs.dqn import DQNConfig


def load_config(config_class, config_path: Optional[str] = None) -> Union[BaseConfig, DQNConfig]:
    """
    Load config with defaults, optionally overriding with YAML.

    Args:
        config_class: Dataclass with default values
        config_path: Optional path to YAML file. Relative to `configs`

    Returns:
        Config object with merged values
    """
    # Start with defaults
    config = config_class()

    # Override with YAML if provided
    if config_path is not None:
        with open(Path('configs', config_path), 'r') as f:
            yaml_config = yaml.safe_load(f)

        config = replace(config, **yaml_config)

    return config
