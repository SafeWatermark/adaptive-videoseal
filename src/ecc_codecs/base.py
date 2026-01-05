from abc import ABC, abstractmethod
import torch


class ECCCodec(ABC):
    """
    Base class to inherit from when implementing an error correction code codec.
    """

    def __init__(
        self, total_bits: int, message_bits: int, threshold: float, *args, **kwargs
    ):
        raise NotImplementedError

    # TODO: Conisder cache
    @abstractmethod
    def encode(self, message_bits: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Encode a message bit array into an error-correcting codeword.

        Args:
            message_bits: (B, message_bits,) array of 0s and 1s

        Returns:
            (total_bits,) array of 0s and 1s with RS parity and 0 padding.

        Raises:
            ValueError: If message_bits length doesn't match expected.
        """
        raise NotImplementedError

    # TODO: Conisder cache
    @abstractmethod
    def decode(
        self,
        watermark_bits: torch.Tensor,
        return_confidence: bool = False,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, list[int]] | tuple[torch.Tensor, list[int], list[float]]:
        """
        Decode a received bit array into the original message bits.

        Args:
            watermark_bits(B, total_bits,): tensor of values (0-1 or -1-1 range, soft or hard)
            return_confidence: If True, also return confidence score

        Returns:
            message_bits: (B, message_bits,) corrected message bits (0s and 1s)
            errors_corrected: Number of byte errors corrected, or -1 if
                             correction failed (too many errors)
            confidence: (optional) if return_confidence: Confidence score based on soft values
        """
        raise NotImplementedError
