import jax.numpy as jnp
import pickle
import logging
from abc import ABC, abstractmethod
from flax import nnx
from typing import Dict, Any
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseAgent(nnx.Module, ABC):
    """Abstract base class for RL agents."""
    step = 0  # Number of updates we've done
    gamma = 0.99

    @abstractmethod
    def select_action(self, state: jnp.ndarray, **kwargs) -> int:
        """
        Select an action given an observation.

        Args:
            state: Observation from environment
            **kwargs: Algorithm-specific args (e.g., epsilon for DQN, deterministic for SAC)

        Returns:
            action: Selected action
        """
        pass

    @abstractmethod
    def update(self, batch: Dict[str, jnp.ndarray]) -> Dict[str, float]:
        """
        Update agent from a batch of data.

        Args:
            batch: Dictionary containing transitions
                   (states, actions, rewards, next_states, dones)

        Returns:
            metrics: Dictionary of training metrics (e.g., loss, Q-values)
        """
        pass

    @abstractmethod
    def _get_state(self, inference_only: bool = False) -> Dict[str, Any]:
        """
        Get agent-specific state for checkpointing.

        Args:
            inference_only: If True, only return model weights (no optimizer state)

        Returns:
            Dictionary containing agent state
        """
        pass

    @abstractmethod
    def _restore_state(self, state: Dict[str, Any], inference_mode: bool = False) -> None:
        """
        Restore agent-specific state from checkpoint.

        Args:
            state: Dictionary containing agent state
            inference_mode: If True, only restore model weights (skip optimizer state)
        """
        pass

    def save_checkpoint(self, path: str, inference_only: bool = False, config: Dict[str, Any] = None) -> None:
        """
        Save agent checkpoint to disk.

        Args:
            path: Path to save checkpoint file
            inference_only: If True, only save model weights (no optimizer state)
            config: Optional config dict to save for reconstruction
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'agent_state': self._get_state(inference_only=inference_only),
            'step': self.step,
            'metadata': {
                'agent_class': self.__class__.__name__,
                'timestamp': datetime.now().isoformat(),
                'config': config,
            }
        }

        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)

    def load_checkpoint(self, path: str, inference_mode: bool = True) -> Dict[str, Any]:
        """
        Load agent from checkpoint.

        Args:
            path: Path to checkpoint file
            inference_mode: If True, only load model weights (default: True)

        Returns:
            Dictionary containing metadata from checkpoint
        """
        path = Path(path)

        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)

        # Validate agent class
        if checkpoint['metadata']['agent_class'] != self.__class__.__name__:
            raise ValueError(
                f"Agent class mismatch: checkpoint is {checkpoint['metadata']['agent_class']}, "
                f"but trying to load into {self.__class__.__name__}"
            )

        # Restore state
        self._restore_state(checkpoint['agent_state'], inference_mode=inference_mode)

        # Restore step count if not in inference mode
        if not inference_mode:
            self.step = checkpoint['step']

        # Log what was loaded
        mode_str = "inference mode (weights only)" if inference_mode else "training mode (full state)"
        logger.info(f"Loaded checkpoint from {path} in {mode_str}")
        logger.info(f"Checkpoint metadata: {checkpoint['metadata']}")

        return checkpoint['metadata']
