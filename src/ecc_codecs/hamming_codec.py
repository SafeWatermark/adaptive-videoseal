"""Hamming Code Error Correction for Watermark Systems.

This module provides Hamming encoding/decoding via PyTorch for watermark systems.

Hamming Code Configuration:
- Hamming(n, k) where n = 2^r - 1 (codeword length), k = n - r (message bits per block)
- Single-bit error correction capability per block
- Efficient hardware-friendly implementation using XOR operations
- Soft-input decoding support via LLR-based syndrome decoding

Supported Hamming Configurations:
- Hamming(7, 4): r=3, 4 message bits, 3 parity bits per block
    - k=4 → message_bits must be divisible by 4 (e.g., 16, 20, 24)
- Hamming(15, 11): r=4, 11 message bits, 4 parity bits per block
    - k=11 → message_bits must be divisible by 11 (e.g., 11, 22, 33)
- Hamming(31, 26): r=5, 26 message bits, 5 parity bits per block
    - k=26 → message_bits must be divisible by 26 (e.g., 26, 52)
- Hamming(63, 57): r=6, 57 message bits, 6 parity bits per block
    - k=57 → message_bits must be divisible by 57 (e.g., 57, 114)

Input Conventions:
- Binary output (0 or 1): e.g., StegaStamp
- Sigmoid output [0, 1]: probability that bit = 1 (e.g., RevMark)
- Logits (unbounded): raw neural network output (e.g., VideoSeal)

The codec auto-detects input type based on value range, or you can explicitly
specify via the `input_type` parameter in decode().

Performance Considerations:
- Hamming codes are simple and fast to encode/decode
- Limited to single-bit error correction per block
- Best suited for low bit-error-rate channels
- For higher error rates, consider Polar or Convolutional codes
"""

from typing import Literal

import torch

from ecc_codecs.base import ECCCodec
from schemas.converters import ValueConverter
from schemas.model_outputs import OutputValueType


class HammingWatermarkCodec(ECCCodec):
    """
    Hamming codec for fixed-size watermark error correction (pure PyTorch).

    Supports both sigmoid probability inputs and raw logit inputs from
    various watermarking neural networks.
    """

    # Valid (n, k, r) configurations: n = 2^r - 1, k = n - r
    VALID_CONFIGS = {
        3: (7, 4),  # Hamming(7, 4)
        4: (15, 11),  # Hamming(15, 11)
        5: (31, 26),  # Hamming(31, 26)
        6: (63, 57),  # Hamming(63, 57)
    }

    def __init__(
        self,
        total_bits: int,
        message_bits: int,
        r: int = 4,
        threshold: float = 0.5,
        device: torch.device = torch.device("cpu"),
        *args,
        **kwargs,
    ):
        """
        Initialize Hamming codec for watermark error correction.

        Args:
            total_bits: Total watermark capacity (must accommodate encoded blocks).
            message_bits: Number of message bits (must be divisible by k).
            r: Number of parity bits per block (3-6). Determines Hamming code size.
                - r=3: Hamming(7,4) - 4 message bits per block
                - r=4: Hamming(15,11) - 11 message bits per block (default)
                - r=5: Hamming(31,26) - 26 message bits per block
                - r=6: Hamming(63,57) - 57 message bits per block
            threshold: Decision threshold for sigmoid inputs (default: 0.5)
            device: torch device for computation

        Raises:
            ValueError: If configuration is invalid
        """
        self.r = int(r)
        self._validate_r()

        self.n, self.k = self.VALID_CONFIGS[self.r]
        self.total_bits = int(total_bits)
        self.message_bits = int(message_bits)
        self.threshold = float(threshold)
        self.device = device

        self._validate_bits()

        # Number of blocks needed
        self.num_blocks = self.message_bits // self.k
        self.codeword_bits = self.num_blocks * self.n

        # Build generator and parity-check matrices
        self.H = self._build_parity_check_matrix()  # (r, n)
        self.G = self._build_generator_matrix()  # (k, n)

        # Precompute syndrome lookup table for fast decoding
        self.syndrome_table = self._build_syndrome_table()

    @property
    def info(self) -> dict:
        """Return codec configuration info."""
        return {
            "codec": f"Hamming({self.n},{self.k})",
            "message_bits": self.message_bits,
            "codeword_bits": self.codeword_bits,
            "total_bits": self.total_bits,
            "r": self.r,
            "n": self.n,
            "k": self.k,
            "num_blocks": self.num_blocks,
            "error_correction_capacity": 1,  # per block
        }

    def _validate_r(self):
        """Validate r parameter."""
        if self.r not in self.VALID_CONFIGS:
            valid = list(self.VALID_CONFIGS.keys())
            raise ValueError(f"r must be in {valid}, got {self.r}")

    def _validate_bits(self):
        """Validate bit configuration."""
        if self.message_bits % self.k != 0:
            raise ValueError(
                f"message_bits must be divisible by k={self.k} for Hamming({self.n},{self.k}), "
                f"got {self.message_bits}. Consider message_bits={self.message_bits - (self.message_bits % self.k)} "
                f"or {self.message_bits + (self.k - self.message_bits % self.k)}"
            )

        expected_codeword_bits = (self.message_bits // self.k) * self.n
        if self.total_bits < expected_codeword_bits:
            raise ValueError(
                f"total_bits={self.total_bits} is too small. "
                f"For message_bits={self.message_bits} with Hamming({self.n},{self.k}), "
                f"need at least {expected_codeword_bits} bits."
            )

        if self.message_bits < self.k:
            raise ValueError(
                f"message_bits must be at least {self.k} for Hamming({self.n},{self.k})"
            )

    def _build_parity_check_matrix(self) -> torch.Tensor:
        """
        Build the parity-check matrix H for Hamming code.

        For systematic Hamming code, H has columns that are binary representations
        of 1 to n (non-zero r-bit patterns).

        Returns:
            H: (r, n) tensor representing the parity-check matrix
        """
        H = torch.zeros((self.r, self.n), dtype=torch.uint8, device=self.device)

        for i in range(1, self.n + 1):
            # Column i-1 is binary representation of i
            for j in range(self.r):
                H[j, i - 1] = (i >> j) & 1

        return H

    def _build_generator_matrix(self) -> torch.Tensor:
        """
        Build the generator matrix G for systematic Hamming code.

        For systematic code: G = [I_k | P] where P is derived from H.
        The data positions are non-power-of-2 positions.

        Returns:
            G: (k, n) tensor representing the generator matrix
        """
        # Identify data positions (non-powers of 2) and parity positions (powers of 2)
        data_positions = []
        parity_positions = []

        for i in range(1, self.n + 1):
            if (i & (i - 1)) == 0:  # Power of 2
                parity_positions.append(i - 1)  # 0-indexed
            else:
                data_positions.append(i - 1)

        self.data_positions = torch.tensor(
            data_positions, dtype=torch.long, device=self.device
        )
        self.parity_positions = torch.tensor(
            parity_positions, dtype=torch.long, device=self.device
        )

        # Build G: for each data bit position, encode it
        G = torch.zeros((self.k, self.n), dtype=torch.uint8, device=self.device)

        for i, data_pos in enumerate(data_positions):
            # Set the data bit
            G[i, data_pos] = 1

            # Compute parity bits from H
            # Each parity bit j covers positions where bit j of position index is 1
            col = self.H[:, data_pos]  # Which parity bits this data position affects
            for j, parity_pos in enumerate(parity_positions):
                if col[j] == 1:
                    G[i, parity_pos] = 1

        return G

    def _build_syndrome_table(self) -> torch.Tensor:
        """
        Build syndrome lookup table for single-bit error correction.

        Returns:
            syndrome_table: (2^r,) tensor mapping syndrome to error position (0 = no error)
        """
        table_size = 1 << self.r
        syndrome_table = torch.zeros(table_size, dtype=torch.long, device=self.device)

        # Syndrome i corresponds to error at position i (1-indexed in Hamming)
        # Syndrome 0 means no error
        for i in range(1, self.n + 1):
            syndrome = i  # Binary representation is the syndrome
            if syndrome < table_size:
                syndrome_table[syndrome] = i  # Error at position i (1-indexed)

        return syndrome_table

    def encode(self, message_bits: torch.Tensor) -> torch.Tensor:
        """
        Encode message bits into Hamming codeword bits.

        Args:
            message_bits: (B, message_bits) binary tensor of 0s and 1s

        Returns:
            (B, total_bits) uint8 tensor containing the encoded codeword

        Raises:
            ValueError: If input length or values are invalid
        """
        if message_bits.size(-1) != self.message_bits:
            raise ValueError(
                f"Expected {self.message_bits} bits, got {message_bits.size(-1)}"
            )

        # Validate binary input
        if not torch.all((message_bits == 0) | (message_bits == 1)):
            raise ValueError("Input must be binary (0 or 1)")

        B = message_bits.size(0)
        message_bits = message_bits.to(self.device)

        # Reshape into blocks: (B, num_blocks, k)
        blocks = message_bits.view(B, self.num_blocks, self.k)

        # Encode each block: codeword = message @ G (mod 2)
        # (B, num_blocks, k) @ (k, n) -> (B, num_blocks, n)
        G_float = self.G.float()
        codewords = torch.matmul(blocks.float(), G_float)
        codewords = (codewords % 2).to(torch.uint8)

        # Flatten blocks: (B, num_blocks * n)
        encoded = codewords.view(B, -1)

        # Pad to total_bits if needed
        if encoded.size(1) < self.total_bits:
            pad_size = self.total_bits - encoded.size(1)
            encoded = torch.nn.functional.pad(encoded, (0, pad_size), value=0)

        return encoded

    def decode(
        self,
        watermark_bits: torch.Tensor,
        return_confidence: bool = False,
        input_type: Literal["auto", "sigmoid", "logits"] = "auto",
    ) -> tuple[torch.Tensor, list[int]] | tuple[torch.Tensor, list[int], list[float]]:
        """
        Decode watermark bits using Hamming syndrome decoding.

        Args:
            watermark_bits: (B, total_bits) tensor of soft or hard values.
                - Sigmoid: values in [0, 1], probability of bit=1
                - Logits: unbounded real values, positive -> bit=1 more likely
            return_confidence: If True, also return confidence score
            input_type: Input interpretation mode:
                - "auto": Detect based on value range (default)
                - "sigmoid": Treat as sigmoid probabilities [0, 1]
                - "logits": Treat as raw logits (unbounded)

        Returns:
            u_hat: (B, message_bits) decoded message bits as uint8
            bit_errors: List of corrected errors per batch (count across all blocks)
            confidence: (optional) Confidence score if return_confidence=True

        Raises:
            ValueError: If input shape is invalid
        """
        if watermark_bits.ndim != 2 or watermark_bits.size(-1) < self.codeword_bits:
            raise ValueError(
                f"watermark_bits must be 2D with length >= {self.codeword_bits}. "
                f"Got shape {watermark_bits.shape}."
            )

        B = watermark_bits.size(0)
        watermark_bits = watermark_bits.to(self.device)

        # Take only the bits used for Hamming coding
        hamming_bits = watermark_bits[:, : self.codeword_bits]

        # Detect input type
        detected_type = ValueConverter.detect_value_type(hamming_bits, input_type)

        # Compute confidence based on value type
        confidence = self._compute_confidence(hamming_bits, detected_type)

        # Convert to hard decisions
        hard_bits = ValueConverter.to_binary(hamming_bits, detected_type)

        # Reshape into blocks: (B, num_blocks, n)
        blocks = hard_bits.view(B, self.num_blocks, self.n)

        # For soft decoding, also get soft values per block
        soft_blocks = None
        if detected_type != OutputValueType.BINARY:
            soft_values = hamming_bits.float()
            if detected_type == OutputValueType.SIGMOID:
                # Convert to reliability: distance from 0.5
                soft_values = torch.abs(soft_values - 0.5) * 2
            else:  # LOGITS
                # Use absolute value as reliability
                soft_values = torch.abs(soft_values)
            soft_blocks = soft_values.view(B, self.num_blocks, self.n)

        # Decode each block with syndrome decoding
        decoded_blocks, errors_per_sample = self._syndrome_decode(
            blocks, soft_blocks, hard_bits.view(B, self.num_blocks, self.n)
        )

        # Extract message bits from decoded blocks
        # (B, num_blocks, n) -> select data positions -> (B, num_blocks, k)
        message_blocks = decoded_blocks[:, :, self.data_positions]
        decoded_message = message_blocks.view(B, -1)

        if return_confidence:
            return decoded_message, errors_per_sample, confidence
        return decoded_message, errors_per_sample

    def _syndrome_decode(
        self,
        blocks: torch.Tensor,
        soft_blocks: torch.Tensor | None,
        original_blocks: torch.Tensor,
    ) -> tuple[torch.Tensor, list[int]]:
        """
        Perform syndrome decoding on blocks.

        Args:
            blocks: (B, num_blocks, n) hard decision blocks
            soft_blocks: (B, num_blocks, n) reliability values (optional)
            original_blocks: (B, num_blocks, n) original hard bits for correction

        Returns:
            corrected_blocks: (B, num_blocks, n) error-corrected blocks
            errors_per_sample: list of error counts per batch sample
        """
        B = blocks.size(0)
        corrected = blocks.clone()
        errors_per_sample = []

        H_float = self.H.float()

        for b in range(B):
            total_errors = 0
            for block_idx in range(self.num_blocks):
                block = corrected[b, block_idx].float()

                # Compute syndrome: s = H @ r (mod 2)
                syndrome = torch.matmul(H_float, block)
                syndrome = (syndrome % 2).to(torch.long)

                # Convert syndrome to integer
                syndrome_int = 0
                for i in range(self.r):
                    syndrome_int += int(syndrome[i].item()) << i

                if syndrome_int != 0:
                    # Error detected
                    error_pos = self.syndrome_table[syndrome_int].item()

                    if error_pos > 0 and error_pos <= self.n:
                        # Flip the error bit (1-indexed to 0-indexed)
                        corrected[b, block_idx, error_pos - 1] ^= 1
                        total_errors += 1

            errors_per_sample.append(total_errors)

        return corrected, errors_per_sample

    def _compute_confidence(
        self, bits: torch.Tensor, value_type: OutputValueType
    ) -> list[float]:
        """
        Compute confidence score for decoded values.

        Args:
            bits: (B, codeword_bits) tensor of input values
            value_type: Detected input value type

        Returns:
            List of confidence scores per batch sample
        """
        if value_type == OutputValueType.SIGMOID:
            # Distance from 0.5, normalized to [0, 1]
            return torch.mean(torch.abs(bits - self.threshold) * 2, dim=1).tolist()
        elif value_type == OutputValueType.LOGITS:
            # Sigmoid of absolute logits gives confidence
            abs_logits = torch.abs(bits)
            normalized = torch.tanh(abs_logits / 4.0)
            return torch.mean(normalized, dim=1).tolist()
        else:  # BINARY
            return [1.0] * bits.size(0)
