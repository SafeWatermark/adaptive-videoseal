from mac.base import MACProvider
import hmac
import hashlib
import torch
from mac.configs import mac_settings
from pydantic_settings import BaseSettings
from typing import Literal
from schemas.converters import ValueConverter


class HMACProvider(MACProvider):
    def __init__(
        self, message_bits: int, tag_bits: int, config: BaseSettings = mac_settings
    ):
        self.message_bits = message_bits
        self.tag_bits = tag_bits
        self.config = config
        self.hmac_tag = self._compute_truncated_hmac_tag()

        self.verification_results: list = []

        if self.tag_bits <= 0:
            raise ValueError("tag_bits must be positive")

        BYTE = 8
        max_tag_bits = len(config.MAC_SECRET_KEY) * BYTE

        if tag_bits > max_tag_bits:
            raise ValueError(
                f"tag_bits must be less than or equal to the MAC secret key size in bits. "
                f"tag_bits={tag_bits}, MAC_SECRET_KEY_bits={max_tag_bits}"
            )

    def clear_results(self):
        self.verification_results = []

    def build_authenticated_payload_bits(
        self, message_bits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute and Insert HMAC Tag into watermark message bits.

        Args:
            message_bits: (B, message_bits) binary tensor of 0s and 1s

        Returns:
            (B, message_bits + tag_bits)

        Raises:
            ValueError: If input length are invalid
        """
        if message_bits.size(-1) != self.message_bits:
            raise ValueError(
                f"Expected {self.message_bits} bits, got {message_bits.size(-1)}"
            )

        if self.hmac_tag.size(0) != self.tag_bits:
            raise RuntimeError("Tag size mismatch (internal error)")

        hmac_tag = self.hmac_tag.expand(
            message_bits.size(0), -1
        )  # (tag_bits,) -> # (B, tag_bits)
        return torch.concat(
            [message_bits, hmac_tag], dim=1
        )  # (B, message_bits + tag_bits)

    def _bytes_to_bits(self, byte_tensor: torch.Tensor) -> torch.Tensor:
        """Convert bytes to bit tensor (MSB first).
        Args:
            byte_tensor(torch.Tensor[B, n_bytes]): uint8 tensor of shape
        Returns:
            torch.Tensor[B, n_bytes * 8]: uint8 tensor of shape
        """
        device = byte_tensor.device
        shifts = torch.arange(7, -1, -1, dtype=torch.uint8, device=device)
        bits = ((byte_tensor[..., None] >> shifts) & 1).to(
            torch.uint8
        )  # (B, n_bytes, 8)
        return bits.reshape(byte_tensor.size(0), -1)

    def _compute_truncated_hmac_tag(
        self,
    ) -> torch.Tensor:
        """
        Compute truncated HMAC tag bits using shared password.
        """

        digest = hmac.new(
            self.config.MAC_SECRET_KEY, self.config.PASSWORD_BYTES, hashlib.sha256
        ).digest()
        return self.truncate_bits_from_digest(digest, self.tag_bits)

    def truncate_bits_from_digest(self, digest: bytes, tag_bits: int) -> torch.Tensor:
        """
        Take the left-most tag_bits from digest (big-endian) and return as bit array.
        tag_bits does not need to be divisible by 8.
        """
        digest_tensor = torch.frombuffer(bytearray(digest), dtype=torch.uint8).clone()
        # digest is SHA-256 = 32 bytes = 256 bits
        digest_tensor = digest_tensor.unsqueeze(0)  # (32,) -> (1, 32)
        digest_bits = self._bytes_to_bits(
            digest_tensor
        ).squeeze()  #  (1, 32) -> (1, 256) -> (256,)
        if tag_bits > digest_bits.size(-1):
            raise ValueError(
                f"tag_bits exceeds digest size, expected {tag_bits}, got {digest_bits.size(-1)}"
            )
        return digest_bits[:tag_bits]  # (tag_bits,)

    def verify_tag(
        self,
        payload_bits: torch.Tensor,
        input_type: Literal["auto", "sigmoid", "logits", "binary"] = "auto",
    ) -> torch.Tensor:
        """
        Verify payload authenticity:
        - Extract (message_bits, tag_bits)
        - Recompute tag from shared password
        - Constant-time compare in bit space

        Args:
            payload_bits (torch.Tensor): shape [B, message_bits + tag_bits]
            input_type: Input interpretation mode ("auto", "sigmoid", "logits", "binary")

        Returns:
            torch.Tensor: shape [B], dtype=torch.bool
        """

        _, tag_bits = self.split_payload_bits(payload_bits)

        # Detect input type using unified ValueConverter
        detected_type = ValueConverter.detect_value_type(tag_bits, input_type)

        # Convert to binary using ValueConverter
        tag_bits = ValueConverter.to_binary(tag_bits, detected_type)

        expected_tag = self.hmac_tag

        if expected_tag.ndim == 1:
            expected_tag = expected_tag.expand(tag_bits.size(0), -1)

        # If shapes still don't match, fail closed with all-False.
        if tag_bits.size() != expected_tag.size():
            return torch.zeros(
                tag_bits.size(0), dtype=torch.bool, device=payload_bits.device
            )
        # TODO: Threshold if soft message
        diff = (tag_bits.to(torch.uint8) ^ expected_tag.to(torch.uint8)).sum(
            dim=-1
        )  # [B]
        return diff == 0

    def split_payload_bits(
        self,
        payload_bits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split payload_bits into (message_bits, tag_bits)."""
        message = payload_bits[:, : self.message_bits]
        tag = payload_bits[:, self.message_bits : self.message_bits + self.tag_bits]
        return message, tag
