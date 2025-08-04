from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any

import torch


class AuxiliaryObjective(ABC):
    @abstractmethod
    def __call__(self, segments: torch.Tensor, preferences: torch.Tensor, weights: torch.Tensor, segment_metas: list, mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        pass
