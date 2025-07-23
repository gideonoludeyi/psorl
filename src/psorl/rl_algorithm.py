from abc import ABC, abstractmethod
import numpy as np
from typing import Any

class RLAlgorithm(ABC):
    @abstractmethod
    def select_action(self, state: Any) -> Any:
        pass

    @abstractmethod
    def update(self, replay_buffer: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def get_actor_parameters(self) -> np.ndarray:
        pass

    @abstractmethod
    def set_actor_parameters(self, params: np.ndarray) -> None:
        pass
