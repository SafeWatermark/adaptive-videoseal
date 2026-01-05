import torch
from videoseal.models.extractor import Extractor
from ecc_codecs.rs_codec import RSWatermarkCodec
import copy


class ECCExtractor(Extractor):
    @classmethod
    def from_extractor(
        cls,
        model: Extractor,
        total_bits: int,
        message_bits: int,
        threshold: float = 0.5,
    ) -> "ECCExtractor":
        """
        Create an ECCExtractor from an existing Extractor.

        Args:
            model: Base extractor to wrap
            total_bits: Total watermark bits (codeword length)
            message_bits: Number of message bits before ECC encoding
            threshold: Decision threshold for soft values (used for confidence)
        """
        new = copy.deepcopy(model)
        new.__class__ = cls

        new.total_bits = total_bits
        new.message_bits = message_bits
        new.codec = RSWatermarkCodec(total_bits, message_bits, threshold)
        new.register_forward_hook(new._ecc_forward_hook)

        new.forward = model.forward
        return new

    def _ecc_forward_hook(self, module, input, output):
        probability = output[:, 0].unsqueeze(1)
        decoded_msgs, _ = self._apply_ecc_decoding(output[:, 1:])
        return torch.cat([probability, decoded_msgs], dim=1)

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
