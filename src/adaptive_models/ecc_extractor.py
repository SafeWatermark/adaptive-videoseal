import torch
from videoseal.models.extractor import Extractor
from ecc_codecs.rs_codec import RSWatermarkCodec
import copy
from typing import Literal


class ECCExtractor(Extractor):
    @classmethod
    def from_extractor(
        cls,
        model: Extractor,
        total_bits: int,
        message_bits: int,
        threshold: float = 0.5,
        input_type: Literal["auto", "sigmoid", "logits"] = "auto",
    ) -> "ECCExtractor":
        """
        Create an ECCExtractor from an existing Extractor.

        Args:
            model: Base extractor to wrap
            total_bits: Total watermark bits (codeword length)
            message_bits: Number of message bits before ECC encoding
            threshold: Decision threshold for soft values (used for confidence)
            input_type: Input type for Polar decoder:
                - "auto": Auto-detect based on value range
                - "sigmoid": Values in [0, 1] (e.g., RevMark)
                - "logits": Unbounded values (e.g., VideoSeal)
        """
        new = copy.deepcopy(model)
        new.__class__ = cls

        new.extractor = copy.deepcopy(model)
        new.total_bits = total_bits
        new.message_bits = message_bits
        new.codec = RSWatermarkCodec(total_bits, message_bits, threshold)
        return new

    def _apply_ecc_decoding(
        self, raw_bits: torch.Tensor
    ) -> tuple[torch.Tensor, list[int]]:
        """Apply ECC decoding to extract and error-correct message bits.

        Args:
            raw_bits: (B, total_bits) tensor of soft bit values.

        Returns:
            decoded_msgs: (B, message_bits) tensor of corrected message bits.
            errors_list: List of error counts per batch item.
        """
        decoded_tensor, errors_list = self.codec.decode(raw_bits)
        return decoded_tensor, errors_list

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Extract and ECC-decode watermark from images.

        Note: For better performance with video, consider using
        extract_soft_bits() followed by temporal averaging and
        decode_averaged_bits() instead.

        Args:
            imgs: (B, C, H, W) input images

        Returns:
            (B, 1 + message_bits) tensor with detection probability and decoded bits
        """
        raw_bits = self.extractor(imgs)

        probability = raw_bits[:, 0].unsqueeze(1)
        decoded_msgs, _ = self._apply_ecc_decoding(raw_bits[:, 1:])
        return torch.cat([probability, decoded_msgs], dim=1)
