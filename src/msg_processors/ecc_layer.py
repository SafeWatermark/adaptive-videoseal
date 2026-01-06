from msg_processors.base_layer import MsgLayer

import torch
from ecc_codecs.base import ECCCodec


class ECCMsgProcessor(MsgLayer):
    """
    Apply the ECC encoding message to recovery the data even though some bits flipped.
    """

    name: str = "Error Correction Code"

    def __init__(
        self,
        ecc_codec: ECCCodec,
    ):
        # super().__init__(nbits, hidden_size, msg_processor_type, msg_mult)
        self.ecc_codec = ecc_codec

    def __call__(self, message_bits: torch.Tensor) -> torch.Tensor:
        """message_bits → ecc_codec.encode(message_bits)"""
        return self.ecc_codec.encode(message_bits)
