import torch
from mac.base import MACProvider
from msg_processors.base_layer import MsgLayer

VIDEOSEAL_MSG_PROCESSOR_HIDDEN_SIZE = 256


class MACMsgProcessor(MsgLayer):
    """
    Apply the MAC encoding message to authenticate the watermark.
    """

    name: str = "Message Authentication Code"

    def __init__(
        self,
        mac_provider: MACProvider,
    ):
        self.mac_provider = mac_provider

    def __call__(self, msg_bits: torch.Tensor) -> torch.Tensor:
        """msg_bits → msg_bits || tag_bits"""
        return self.mac_provider.build_authenticated_payload_bits(msg_bits)
