"""Reed-Solomon Error Correction for Watermark Systems.

This module provides Reed-Solomon (RS) encoding/decoding utilities designed
for fixed-size watermark systems (e.g., 100-bit StegaStamp).

RS Configuration:
- Uses GF(2^8) field (byte-level symbols)
- Works with 96 bits (12 bytes) for RS coding
- 4 remaining bits reserved for metadata/padding
- Error correction capacity: (parity_bytes // 2) byte errors

Example:
    For a 16-bit message with 100-bit total capacity:
    - Message: 2 bytes (16 bits)
    - Parity: 10 bytes (80 bits)
    - Reserved: 4 bits
    - Can correct up to 5 byte errors
"""

import torch
from reedsolo import RSCodec, ReedSolomonError
from torch.nn import functional as F
from ecc_codecs.base import ECCCodec
from schemas.converters import ValueConverter


class RSWatermarkCodec(ECCCodec):
    """Reed-Solomon codec for watermark error correction.

    Designed for fixed watermark capacity.

    RS operates on GF(2^8), so we work with byte-level symbols.
    The codec can correct up to (parity_bytes // 2) byte errors.
    """

    def __init__(
        self,
        total_bits,
        message_bits: int,
        threshold: float = 0.5,
        device: torch.device = torch.device("cpu"),
        *args,
        **kwargs,
    ):
        """Initialize RS codec for watermark.

        Args:
            message_bits: Number of bits for user message.
                          Must be divisible by 8 and leave room for parity.

        Raises:
            ValueError: If message_bits is not divisible by 8 or
                       doesn't leave enough room for parity (min 2 bytes).
        """

        self.total_bits = total_bits

        if message_bits % 8 != 0:
            raise ValueError(f"message_bits must be divisible by 8, got {message_bits}")

        self.message_bits = message_bits
        self.message_bytes = message_bits // 8

        self.parity_bytes = (self.total_bits // 8) - self.message_bytes
        self.parity_bits = 8 * self.parity_bytes

        self.rs_bits = self.parity_bits + self.message_bits
        self.rs_bytes = self.rs_bits // 8
        self.reserved_bits = self.total_bits - self.rs_bits
        self.device = device

        if self.parity_bytes < 2:
            raise ValueError(
                f"Not enough parity bytes. message_bits={message_bits} "
                f"leaves only {self.parity_bytes} parity bytes. Need at least 2."
            )

        if self.message_bytes < 1:
            raise ValueError("message_bits must be at least 8 (1 byte)")

        # RS codec: can correct up to parity_bytes/2 byte errors
        self.rs = RSCodec(self.parity_bytes)
        self.error_correction_capacity = self.parity_bytes // 2

        self.threshold = threshold

    @property
    def info(self) -> dict:
        """Return codec configuration info."""
        return {
            "message_bits": self.message_bits,
            "message_bytes": self.message_bytes,
            "parity_bytes": self.parity_bytes,
            "parity_bits": self.parity_bytes * 8,
            "rs_total_bytes": self.rs_bytes,
            "error_correction_capacity_bytes": self.error_correction_capacity,
            "total_bits": self.total_bits,
        }

    def encode(self, message_bits: torch.Tensor) -> torch.Tensor:
        """Encode message bits with Reed-Solomon parity.

        Args:
            message_bits: (B, message_bits,) array of 0s and 1s

        Returns:
            (total_bits,) array of 0s and 1s with RS parity and 0 padding.

        Raises:
            ValueError: If message_bits length doesn't match expected.
        """
        B = message_bits.size(0)
        if message_bits.ndim != 2:
            raise ValueError(f"Expected 2 dimensions, got {message_bits.ndim}")
        if message_bits.size(-1) != self.message_bits:
            raise ValueError(
                f"Expected {self.message_bits} bits, got {message_bits.size(-1)}"
            )

        # Convert bits to bytes
        message_bytes_list = self._bits_to_bytes(message_bits)

        bytes_buffer = torch.empty(
            (B, self.rs_bytes), dtype=torch.uint8, device=torch.device("cpu")
        )
        for i, message_bytes in enumerate(message_bytes_list):
            # RS encode: message_bytes -> message_bytes + parity_bytes
            encoded_bytes: bytearray = self.rs.encode(message_bytes)
            bytes_buffer[i] = torch.frombuffer(
                bytearray(encoded_bytes), dtype=torch.uint8
            ).clone()
            # Convert back to bits
        bytes_buffer = bytes_buffer.to(message_bits.device)
        encoded_bits = self._bytes_to_bits(bytes_buffer)

        pad_bits = self.total_bits - encoded_bits.size(1)
        encoded_bits = F.pad(encoded_bits, (0, pad_bits), "constant", value=0)
        return encoded_bits

    def decode(
        self,
        watermark_bits: torch.Tensor,
        return_confidence: bool = False,
        input_type: str = "auto",
    ) -> tuple[torch.Tensor, list[int]] | tuple[torch.Tensor, list[int], list[float]]:
        """Decode watermark bits with RS error correction.

        Args:
            watermark_bits(B, total_bits,): tensor of values (0-1 range, soft or hard)
            return_confidence: If True, also return confidence score
            input_type: Input interpretation mode ("auto", "sigmoid", "logits", "binary")

        Returns:
            message_bits: (B, message_bits,) corrected message bits (0s and 1s)
            errors_corrected: Number of byte errors corrected, or -1 if
                             correction failed (too many errors)
            confidence: (optional) Confidence score based on soft values
        """
        # Take RS portion
        if watermark_bits.ndim != 2:
            raise ValueError(f"Exepcted 2 dimensions, got {watermark_bits.ndim}")
        if watermark_bits.size(-1) != self.total_bits:
            raise ValueError(
                f"Expected {self.total_bits} bits, got {watermark_bits.size(-1)}"
            )
        B = watermark_bits.size(0)

        rs_bits = watermark_bits[:, : self.rs_bits]

        # Detect input type using unified ValueConverter
        detected_type = ValueConverter.detect_value_type(rs_bits, input_type)

        # Compute confidence from soft values before hard decision
        if return_confidence:
            # Confidence = how far from threshold the values are (average)
            threshold = ValueConverter.get_threshold(detected_type)
            confidence = torch.mean(torch.abs(rs_bits - threshold) * 2, dim=0).tolist()

        # Convert to hard decisions using ValueConverter
        hard_bits = ValueConverter.to_binary(rs_bits, detected_type)

        # Convert bits to bytes
        encoded_bytes_list = self._bits_to_bytes(hard_bits)

        bytes_buffer = torch.empty(
            (B, self.message_bytes), dtype=torch.uint8, device=torch.device("cpu")
        )
        errors_corrected_list = []
        for i, encoded_bytes in enumerate(encoded_bytes_list):
            try:
                # RS decode: returns (decoded_msg, decoded_msgecc, errata_pos)
                decoded_bytes, _, errata_pos = self.rs.decode(encoded_bytes)
                errors_corrected = len(errata_pos)
                errors_corrected_list.append(errors_corrected)
                message_bytes = bytes(decoded_bytes)
            except ReedSolomonError:
                # Too many errors to correct - return raw message (may be corrupted)
                message_bytes = encoded_bytes[: self.message_bytes]
                errors_corrected_list.append(-1)  # Indicates correction failure

            bytes_buffer[i] = torch.frombuffer(
                bytearray(message_bytes), dtype=torch.uint8
            ).clone()

        bytes_buffer = bytes_buffer.to(watermark_bits.device)
        message_bits = self._bytes_to_bits(bytes_buffer)
        if return_confidence:
            return message_bits, errors_corrected_list, confidence
        return message_bits, errors_corrected_list

    def _bits_to_bytes(self, bits: torch.Tensor) -> list[bytes]:
        """Convert bit tensor to bytes (MSB first).
        Args:
            bits(torch.Tensor[B, message_bits]): uint8/bool tensor
        Returns:
            list[bytes]: length B, each element contains floor(N/8) bytes
        """

        B, N = bits.size()
        n_bytes = N // 8
        if n_bytes == 0:
            return [b""] * B

        # Truncate to multiple of 8 and reshape per-row into bytes
        bits = bits[:, : n_bytes * 8].view(B, n_bytes, 8)  # (B, n_bytes, 8)

        weights = (
            1 << torch.arange(7, -1, -1, dtype=torch.uint8, device=bits.device)
        ).view(1, 1, 8)
        byte_vals = (bits * weights).sum(dim=-1)  # (B, n_bytes), uint8

        # Convert each row to bytes
        return [bytes(row.tolist()) for row in byte_vals]

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
