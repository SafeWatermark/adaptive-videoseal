from abc import ABC, abstractmethod
import torch


class MACProvider(ABC):
    """Base class for MAC algorithms."""

    @abstractmethod
    def build_authenticated_payload_bits(
        self, message_bits: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """Compute MAC tag for message bits."""
        raise NotImplementedError

    @abstractmethod
    def verify_tag(self, payload_bits: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Verify MAC tag. Returns boolean tensor (batch-wise)."""
        raise NotImplementedError
