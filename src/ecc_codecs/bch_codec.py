"""BCH Code Error Correction for Watermark Systems.

This module provides BCH (Bose-Chaudhuri-Hocquenghem) encoding/decoding via PyTorch
for watermark systems.

BCH Code Configuration:
- BCH(n, k, t) where n = 2^m - 1 (codeword length), t = error correction capability
- Multi-bit error correction capability per block
- Uses Galois Field GF(2^m) arithmetic
- Berlekamp-Massey algorithm for syndrome decoding

Supported BCH Configurations:
- BCH(15, 7, 2): m=4, corrects 2 errors per block
- BCH(15, 5, 3): m=4, corrects 3 errors per block
- BCH(31, 21, 2): m=5, corrects 2 errors per block
- BCH(31, 16, 3): m=5, corrects 3 errors per block
- BCH(31, 11, 5): m=5, corrects 5 errors per block
- BCH(63, 51, 2): m=6, corrects 2 errors per block
- BCH(63, 45, 3): m=6, corrects 3 errors per block
- BCH(63, 36, 5): m=6, corrects 5 errors per block

Input Conventions:
- Binary output (0 or 1): e.g., StegaStamp
- Sigmoid output [0, 1]: probability that bit = 1 (e.g., RevMark)
- Logits (unbounded): raw neural network output (e.g., VideoSeal)

The codec auto-detects input type based on value range, or you can explicitly
specify via the `input_type` parameter in decode().

Performance Considerations:
- BCH codes provide stronger error correction than Hamming codes
- Decoding complexity is O(t^2) where t is error correction capability
- Best suited for moderate bit-error-rate channels
- For very high error rates, consider Polar or LDPC codes
"""

from typing import Literal

import torch

from ecc_codecs.base import ECCCodec
from schemas.converters import ValueConverter
from schemas.model_outputs import OutputValueType


class GaloisField:
    """
    Galois Field GF(2^m) arithmetic implementation.

    Uses lookup tables for efficient multiplication and division.
    """

    # Primitive polynomials for GF(2^m)
    # These are irreducible polynomials used to construct the field
    PRIMITIVE_POLYS = {
        3: 0b1011,  # x^3 + x + 1
        4: 0b10011,  # x^4 + x + 1
        5: 0b100101,  # x^5 + x^2 + 1
        6: 0b1000011,  # x^6 + x + 1
        7: 0b10000011,  # x^7 + x + 1
        8: 0b100011101,  # x^8 + x^4 + x^3 + x^2 + 1
    }

    def __init__(self, m: int):
        """
        Initialize Galois Field GF(2^m).

        Args:
            m: Field order exponent (field has 2^m elements)
        """
        if m not in self.PRIMITIVE_POLYS:
            raise ValueError(f"m must be in {list(self.PRIMITIVE_POLYS.keys())}")

        self.m = m
        self.n = (1 << m) - 1  # 2^m - 1
        self.field_size = 1 << m  # 2^m
        self.prim_poly = self.PRIMITIVE_POLYS[m]

        # Build lookup tables
        self._build_tables()

    def _build_tables(self):
        """Build exp and log tables for fast arithmetic."""
        self.exp_table = [0] * (2 * self.n + 1)
        self.log_table = [0] * (self.field_size)

        x = 1
        for i in range(self.n):
            self.exp_table[i] = x
            self.log_table[x] = i
            x <<= 1
            if x & self.field_size:
                x ^= self.prim_poly

        # Extend exp table for easy modular arithmetic
        for i in range(self.n, 2 * self.n + 1):
            self.exp_table[i] = self.exp_table[i - self.n]

    def mul(self, a: int, b: int) -> int:
        """Multiply two field elements."""
        if a == 0 or b == 0:
            return 0
        return self.exp_table[(self.log_table[a] + self.log_table[b]) % self.n]

    def div(self, a: int, b: int) -> int:
        """Divide two field elements."""
        if b == 0:
            raise ZeroDivisionError("Division by zero in GF")
        if a == 0:
            return 0
        return self.exp_table[(self.log_table[a] - self.log_table[b] + self.n) % self.n]

    def pow(self, a: int, power: int) -> int:
        """Raise field element to a power."""
        if a == 0:
            return 0 if power > 0 else 1
        return self.exp_table[(self.log_table[a] * power) % self.n]

    def inv(self, a: int) -> int:
        """Multiplicative inverse of field element."""
        if a == 0:
            raise ZeroDivisionError("Inverse of zero")
        return self.exp_table[self.n - self.log_table[a]]

    def alpha(self, i: int) -> int:
        """Get alpha^i (primitive element to power i)."""
        return self.exp_table[i % self.n]


class BCHWatermarkCodec(ECCCodec):
    """
    BCH codec for fixed-size watermark error correction (pure PyTorch).

    Supports both sigmoid probability inputs and raw logit inputs from
    various watermarking neural networks.
    """

    # Valid BCH configurations: (m, t) -> (n, k)
    # n = 2^m - 1, k depends on t and minimal polynomial degrees
    VALID_CONFIGS = {
        # m=4: n=15
        (4, 1): (15, 11),  # BCH(15, 11, 1) - same as Hamming
        (4, 2): (15, 7),  # BCH(15, 7, 2)
        (4, 3): (15, 5),  # BCH(15, 5, 3)
        # m=5: n=31
        (5, 1): (31, 26),  # BCH(31, 26, 1)
        (5, 2): (31, 21),  # BCH(31, 21, 2)
        (5, 3): (31, 16),  # BCH(31, 16, 3)
        (5, 5): (31, 11),  # BCH(31, 11, 5)
        (5, 7): (31, 6),  # BCH(31, 6, 7)
        # m=6: n=63
        (6, 1): (63, 57),  # BCH(63, 57, 1)
        (6, 2): (63, 51),  # BCH(63, 51, 2)
        (6, 3): (63, 45),  # BCH(63, 45, 3)
        (6, 4): (63, 39),  # BCH(63, 39, 4)
        (6, 5): (63, 36),  # BCH(63, 36, 5)
        (6, 6): (63, 30),  # BCH(63, 30, 6)
        (6, 7): (63, 24),  # BCH(63, 24, 7)
        (6, 10): (63, 18),  # BCH(63, 18, 10)
        (6, 11): (63, 10),  # BCH(63, 10, 11)
        (6, 13): (63, 7),  # BCH(63, 7, 13)
    }

    def __init__(
        self,
        total_bits: int,
        message_bits: int,
        m: int = 5,
        t: int = 2,
        threshold: float = 0.5,
        device: torch.device = torch.device("cpu"),
        *args,
        **kwargs,
    ):
        """
        Initialize BCH codec for watermark error correction.

        Args:
            total_bits: Total watermark capacity (must accommodate encoded blocks).
            message_bits: Number of message bits (must be divisible by k).
            m: Galois field order exponent (4-6). n = 2^m - 1.
            t: Error correction capability (number of correctable errors per block).
            threshold: Decision threshold for sigmoid inputs (default: 0.5)
            device: torch device for computation

        Raises:
            ValueError: If configuration is invalid
        """
        self.m = int(m)
        self.t = int(t)
        self._validate_config()

        self.n, self.k = self.VALID_CONFIGS[(self.m, self.t)]
        self.total_bits = int(total_bits)
        self.message_bits = int(message_bits)
        self.threshold = float(threshold)
        self.device = device

        self._validate_bits()

        # Initialize Galois Field
        self.gf = GaloisField(self.m)

        # Number of blocks needed
        self.num_blocks = self.message_bits // self.k
        self.codeword_bits = self.num_blocks * self.n

        # Build generator polynomial
        self.gen_poly = self._build_generator_polynomial()

    @property
    def info(self) -> dict:
        """Return codec configuration info."""
        return {
            "codec": f"BCH({self.n},{self.k},{self.t})",
            "message_bits": self.message_bits,
            "codeword_bits": self.codeword_bits,
            "total_bits": self.total_bits,
            "m": self.m,
            "n": self.n,
            "k": self.k,
            "t": self.t,
            "num_blocks": self.num_blocks,
            "error_correction_capacity": self.t,
        }

    def _validate_config(self):
        """Validate m and t parameters."""
        if (self.m, self.t) not in self.VALID_CONFIGS:
            valid_m = sorted(set(m for m, _ in self.VALID_CONFIGS.keys()))
            valid_t_for_m = {
                m: sorted(t for m2, t in self.VALID_CONFIGS.keys() if m2 == m)
                for m in valid_m
            }
            raise ValueError(
                f"Invalid (m={self.m}, t={self.t}) combination. "
                f"Valid combinations: {valid_t_for_m}"
            )

    def _validate_bits(self):
        """Validate bit configuration."""
        if self.message_bits % self.k != 0:
            raise ValueError(
                f"message_bits must be divisible by k={self.k} for BCH({self.n},{self.k},{self.t}), "
                f"got {self.message_bits}. Consider message_bits="
                f"{self.message_bits - (self.message_bits % self.k)} "
                f"or {self.message_bits + (self.k - self.message_bits % self.k)}"
            )

        expected_codeword_bits = (self.message_bits // self.k) * self.n
        if self.total_bits < expected_codeword_bits:
            raise ValueError(
                f"total_bits={self.total_bits} is too small. "
                f"For message_bits={self.message_bits} with BCH({self.n},{self.k},{self.t}), "
                f"need at least {expected_codeword_bits} bits."
            )

        if self.message_bits < self.k:
            raise ValueError(
                f"message_bits must be at least {self.k} for BCH({self.n},{self.k},{self.t})"
            )

    def _build_generator_polynomial(self) -> list[int]:
        """
        Build the generator polynomial for BCH code.

        The generator polynomial is the LCM of minimal polynomials of
        alpha^1, alpha^3, ..., alpha^(2t-1).

        Returns:
            Generator polynomial coefficients (lowest degree first)
        """
        # Start with g(x) = 1
        g = [1]

        # Collect roots: alpha^1, alpha^2, ..., alpha^2t
        roots_needed = set()
        for i in range(1, 2 * self.t + 1):
            roots_needed.add(i % self.gf.n)

        # For each root, find its minimal polynomial and multiply
        roots_processed = set()

        for root_power in sorted(roots_needed):
            if root_power in roots_processed:
                continue

            # Find the conjugacy class of this root
            conjugates = []
            power = root_power
            while power not in conjugates:
                conjugates.append(power)
                power = (power * 2) % self.gf.n

            # Mark all conjugates as processed
            for c in conjugates:
                roots_processed.add(c)

            # Build minimal polynomial for this conjugacy class
            min_poly = [1]
            for c in conjugates:
                # Multiply by (x - alpha^c) = (x + alpha^c) in GF(2^m)
                alpha_c = self.gf.alpha(c)
                new_poly = [0] * (len(min_poly) + 1)
                for i, coef in enumerate(min_poly):
                    new_poly[i + 1] ^= coef
                    new_poly[i] ^= self.gf.mul(coef, alpha_c)
                min_poly = new_poly

            # Multiply generator by minimal polynomial
            g = self._poly_mul(g, min_poly)

        return g

    def _poly_mul(self, p1: list[int], p2: list[int]) -> list[int]:
        """Multiply two polynomials over GF(2^m)."""
        result = [0] * (len(p1) + len(p2) - 1)
        for i, c1 in enumerate(p1):
            for j, c2 in enumerate(p2):
                result[i + j] ^= self.gf.mul(c1, c2)
        return result

    def _poly_mod(self, dividend: list[int], divisor: list[int]) -> list[int]:
        """Compute polynomial remainder over GF(2^m)."""
        dividend = dividend.copy()

        while len(dividend) >= len(divisor):
            if dividend[-1] != 0:
                coef = self.gf.div(dividend[-1], divisor[-1])
                for i in range(len(divisor)):
                    dividend[len(dividend) - len(divisor) + i] ^= self.gf.mul(
                        coef, divisor[i]
                    )
            dividend.pop()

        return dividend

    def _encode_block(self, message: list[int]) -> list[int]:
        """
        Encode a single block using systematic encoding.

        Args:
            message: k message bits

        Returns:
            n codeword bits (message + parity)
        """
        # Systematic encoding: c(x) = m(x) * x^(n-k) + r(x)
        # where r(x) = m(x) * x^(n-k) mod g(x)

        # Shift message by (n-k) positions
        parity_len = self.n - self.k
        shifted = [0] * parity_len + message

        # Compute remainder
        remainder = self._poly_mod(shifted, self.gen_poly)

        # Pad remainder to parity length
        while len(remainder) < parity_len:
            remainder.append(0)

        # Codeword = parity + message (systematic form)
        return remainder + message

    def _compute_syndromes(self, received: list[int]) -> list[int]:
        """
        Compute syndromes S_1, S_2, ..., S_2t.

        Args:
            received: n received bits

        Returns:
            2t syndromes (S_1 to S_2t)
        """
        syndromes = []
        for i in range(1, 2 * self.t + 1):
            s = 0
            for j, bit in enumerate(received):
                if bit:
                    s ^= self.gf.pow(self.gf.alpha(i), j)
            syndromes.append(s)
        return syndromes

    def _berlekamp_massey(self, syndromes: list[int]) -> list[int]:
        """
        Berlekamp-Massey algorithm to find error locator polynomial.

        Args:
            syndromes: 2t syndromes

        Returns:
            Error locator polynomial coefficients (lowest degree first)
        """
        # Initialize
        sigma = [1]  # Error locator polynomial
        B = [1]  # Auxiliary polynomial
        L = 0  # Current number of errors
        m = 1  # Step counter

        for n in range(2 * self.t):
            # Compute discrepancy
            d = syndromes[n]
            for i in range(1, min(L + 1, len(sigma))):
                if i < len(sigma) and n - i >= 0:
                    d ^= self.gf.mul(sigma[i], syndromes[n - i])

            if d == 0:
                m += 1
            elif 2 * L <= n:
                # Update sigma
                T = sigma.copy()
                # sigma = sigma - d * x^m * B
                while len(sigma) < len(B) + m:
                    sigma.append(0)
                for i in range(len(B)):
                    sigma[i + m] ^= self.gf.mul(d, B[i])
                L = n + 1 - L
                B = [self.gf.div(t, d) for t in T]
                m = 1
            else:
                # Update sigma without changing L
                while len(sigma) < len(B) + m:
                    sigma.append(0)
                for i in range(len(B)):
                    sigma[i + m] ^= self.gf.mul(d, B[i])
                m += 1

        return sigma

    def _chien_search(self, sigma: list[int]) -> list[int]:
        """
        Chien search to find error locations.

        Args:
            sigma: Error locator polynomial

        Returns:
            List of error positions
        """
        error_positions = []

        for i in range(self.n):
            # Evaluate sigma at alpha^(-i) = alpha^(n-i)
            val = 0
            for j, coef in enumerate(sigma):
                val ^= self.gf.mul(coef, self.gf.pow(self.gf.alpha(self.n - i), j))

            if val == 0:
                error_positions.append(i)

        return error_positions

    def _decode_block(self, received: list[int]) -> tuple[list[int], int]:
        """
        Decode a single block using syndrome decoding.

        Args:
            received: n received bits

        Returns:
            Tuple of (decoded message bits, number of errors corrected)
        """
        # Compute syndromes
        syndromes = self._compute_syndromes(received)

        # Check if all syndromes are zero (no errors)
        if all(s == 0 for s in syndromes):
            # Extract message (last k bits in systematic form)
            return received[-(self.k) :], 0

        # Find error locator polynomial using Berlekamp-Massey
        sigma = self._berlekamp_massey(syndromes)

        # Find error locations using Chien search
        error_positions = self._chien_search(sigma)

        # Verify we found the correct number of errors
        num_errors = len(sigma) - 1
        if len(error_positions) != num_errors:
            # Decoding failure - return received message as is
            return received[-(self.k) :], -1

        # Correct errors
        corrected = received.copy()
        for pos in error_positions:
            corrected[pos] ^= 1

        return corrected[-(self.k) :], len(error_positions)

    def encode(self, message_bits: torch.Tensor) -> torch.Tensor:
        """
        Encode message bits into BCH codeword bits.

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

        # Process each batch and block
        encoded_list = []
        for b in range(B):
            batch_encoded = []
            for block_idx in range(self.num_blocks):
                start = block_idx * self.k
                end = start + self.k
                block = message_bits[b, start:end].tolist()
                encoded_block = self._encode_block(block)
                batch_encoded.extend(encoded_block)
            encoded_list.append(batch_encoded)

        encoded = torch.tensor(encoded_list, dtype=torch.uint8, device=self.device)

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
        Decode watermark bits using BCH syndrome decoding.

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
            bit_errors: List of total corrected errors per batch
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

        # Take only the bits used for BCH coding
        bch_bits = watermark_bits[:, : self.codeword_bits]

        # Detect input type
        detected_type = ValueConverter.detect_value_type(bch_bits, input_type)

        # Compute confidence based on value type
        confidence = self._compute_confidence(bch_bits, detected_type)

        # Convert to hard decisions
        hard_bits = ValueConverter.to_binary(bch_bits, detected_type)

        # Process each batch
        decoded_list = []
        errors_per_sample = []

        for b in range(B):
            batch_decoded = []
            total_errors = 0

            for block_idx in range(self.num_blocks):
                start = block_idx * self.n
                end = start + self.n
                block = hard_bits[b, start:end].tolist()
                decoded_block, errors = self._decode_block(block)
                batch_decoded.extend(decoded_block)

                if errors >= 0:
                    total_errors += errors
                else:
                    total_errors = -1  # Indicate decoding failure

            decoded_list.append(batch_decoded)
            errors_per_sample.append(total_errors)

        decoded = torch.tensor(decoded_list, dtype=torch.uint8, device=self.device)

        if return_confidence:
            return decoded, errors_per_sample, confidence
        return decoded, errors_per_sample

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
