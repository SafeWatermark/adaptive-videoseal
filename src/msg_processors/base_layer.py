from abc import ABC, abstractmethod
import torch


class MsgLayer(ABC):
    """
    A message transformation layer.
    """

    name: str = "Base"

    @abstractmethod
    def __call__(self, msg_bits: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
