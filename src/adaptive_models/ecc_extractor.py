import torch
from videoseal.models.extractor import Extractor
from ecc_codecs.rs_codec import RSWatermarkCodec
import numpy as np
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
            raw_bits: (B, 100) tensor of soft bit values (0-1 range).

        Returns:
            decoded_msgs: (B, message_bits) tensor of corrected message bits.
            errors_list: List of error counts per batch item (-1 if failed).
        """
        device = raw_bits.device
        batch_size = raw_bits.shape[0]
        decoded_msgs = []
        errors_list = []

        for i in range(batch_size):
            bits_np = raw_bits[i].detach().cpu().numpy()
            msg_bits, errors = self.codec.decode(bits_np)
            decoded_msgs.append(msg_bits)
            errors_list.append(errors)

        decoded_tensor = torch.from_numpy(np.stack(decoded_msgs)).to(device)
        return decoded_tensor.float(), errors_list

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        raw_bits = self.extractor(imgs)

        probability = raw_bits[:, 0].unsqueeze(1)
        decoded_msgs, _ = self._apply_ecc_decoding(raw_bits[:, 1:])
        return torch.cat([probability, decoded_msgs], dim=1)
