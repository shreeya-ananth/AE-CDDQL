"""
Base class for all cloud task schedulers.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np


class BaseScheduler(ABC):
    def __init__(self, name: str, num_vms: int = 4):
        self.name = name
        self.num_vms = num_vms

    @abstractmethod
    def select_action(self, state: np.ndarray, env: Optional[Any] = None) -> int:
        """Select a VM index for scheduling the current task."""
        pass

    def update(self, *args, **kwargs) -> Optional[Dict[str, float]]:
        """Optional learning step for RL schedulers."""
        return None

    def reset(self):
        """Reset internal episode state if any."""
        pass
