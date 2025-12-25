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


class RSWatermarkCodec:
    """Reed-Solomon codec for watermark error correction.

    Designed for fixed 100-bit watermark capacity:
    - Uses 96 bits (12 bytes) for RS coding
    - 4 remaining bits reserved for metadata/padding

    RS operates on GF(2^8), so we work with byte-level symbols.
    The codec can correct up to (parity_bytes // 2) byte errors.
    """

    def __init__(self, total_bits, message_bits: int, threshold: float = 0.5):
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
        self.reserved_bits = self.total_bits - self.rs_bits

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
            "rs_total_bytes": self.RS_BYTES,
            "error_correction_capacity_bytes": self.error_correction_capacity,
            "total_bits": self.TOTAL_BITS,
        }

    def encode(self, message_bits: torch.Tensor) -> torch.Tensor:
        """Encode message bits with Reed-Solomon parity.

        Args:
            message_bits: (message_bits,) array of 0s and 1s

        Returns:
            (100,) array of 0s and 1s with RS parity

        Raises:
            ValueError: If message_bits length doesn't match expected.
        """
        if len(message_bits) != self.message_bits:
            raise ValueError(
                f"Expected {self.message_bits} bits, got {len(message_bits)}"
            )

        # Convert bits to bytes
        message_bytes = self._bits_to_bytes(message_bits)

        # RS encode: message_bytes -> message_bytes + parity_bytes
        encoded_bytes: bytearray = self.rs.encode(message_bytes)

        # Convert back to bits (96 bits)
        encoded_bits = self._bytes_to_bits(bytes(encoded_bytes))

        # Create 100-bit output with reserved bits set to 0
        result = torch.zeros(
            self.total_bits, dtype=torch.uint8, device=message_bits.device
        )
        result[: self.rs_bits] = encoded_bits

        return result

    def decode(
        self, watermark_bits: torch.Tensor, return_confidence: bool = False
    ) -> tuple[torch.Tensor, int] | tuple[torch.Tensor, int, float]:
        """Decode watermark bits with RS error correction.

        Args:
            watermark_bits: tensor of values (0-1 range, soft or hard)
            return_confidence: If True, also return confidence score

        Returns:
            message_bits: (message_bits,) corrected message bits (0s and 1s)
            errors_corrected: Number of byte errors corrected, or -1 if
                             correction failed (too many errors)
            confidence: (optional) Confidence score based on soft values
        """
        # Take RS portion
        rs_bits = watermark_bits[: self.rs_bits]
        device = watermark_bits.device

        # Compute confidence from soft values before hard decision
        if return_confidence:
            # Confidence = how far from 0.5 the values are (average)
            confidence = torch.mean(torch.abs(rs_bits - self.threshold) * 2).item()

        # Convert to hard decisions
        hard_bits = rs_bits > self.threshold

        # Convert bits to bytes
        encoded_bytes = self._bits_to_bytes(hard_bits)

        try:
            # RS decode: returns (decoded_msg, decoded_msgecc, errata_pos)
            decoded_bytes, _, errata_pos = self.rs.decode(encoded_bytes)
            errors_corrected = len(errata_pos)
            message_bytes = bytes(decoded_bytes)
        except ReedSolomonError:
            # Too many errors to correct - return raw message (may be corrupted)
            message_bytes = encoded_bytes[: self.message_bytes]
            errors_corrected = -1  # Indicates correction failure

        # Convert back to bits
        message_bits = self._bytes_to_bits(message_bytes, device)

        if return_confidence:
            return message_bits, errors_corrected, confidence
        return message_bits, errors_corrected

    def _bits_to_bytes(self, bits: torch.Tensor) -> bytes:
        """Convert bit tensor to bytes (MSB first)."""

        # Ensure uint8 on CPU (bytes() expects CPU-accessible data)
        bits = bits.to(dtype=torch.uint8, device="cpu").flatten()

        n_bytes = bits.numel() // 8
        if n_bytes == 0:
            return b""

        bits = bits[: n_bytes * 8].view(n_bytes, 8)
        weights = 1 << torch.arange(7, -1, -1, dtype=torch.uint8)

        byte_vals = (bits * weights).sum(dim=1)  # uint8
        return bytes(byte_vals.tolist())

    def _bytes_to_bits(
        self, data: bytes, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """Convert bytes to bit tensor (MSB first)."""
        byte_tensor = torch.frombuffer(data, dtype=torch.uint8).to(device)

        # Bit positions MSB -> LSB: 7..0
        shifts = torch.arange(7, -1, -1, dtype=torch.uint8, device=device)

        # Extract bits: shape (n_bytes, 8)
        bits = ((byte_tensor[:, None] >> shifts[None, :]) & 1).to(torch.uint8)

        # Flatten to 1D: (n_bytes*8,)
        return bits.reshape(-1)
