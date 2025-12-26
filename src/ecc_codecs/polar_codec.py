"""Polar Code Error Correction for Watermark Systems.

This module provides 5G NR Polar encoding/decoding via Sionna for watermark systems.

Polar Code Configuration:
- Uses 5G NR Polar codes with rate-matching
- Supports SC, SCL, and BP decoding
- Soft-input decoding via LLRs (Log-Likelihood Ratios)
- GPU/CPU acceleration via TensorFlow

Input Conventions:
- Binary output (0 or 1): (e.g., StegaStamp)
- Sigmoid output [0, 1]: probability that bit = 1 (e.g., RevMark)
- Logits (unbounded): raw neural network output (e.g., VideoSeal)
  - Positive logit → bit more likely to be 1
  - Negative logit → bit more likely to be 0

The codec auto-detects input type based on value range, or you can explicitly
specify via the `input_type` parameter in decode().

Performance Considerations:
- SHORT CODEWORDS & SEVERE BIT ERRORS: Performance degrades significantly when
  bit error rate is high relative to the error correction capacity. With short
  bit lengths (e.g., 32-64 bits), even a few corrupted bits can exceed the
  correction threshold, causing complete decoding failure.

- REVMARK "per_block" MODE: When using REVMark with per_block detection mode,
  each 8-frame block produces only once. This can makes damage REVMark's
  detection performance

    - RECOMMENDED APPROACH: Use "per_frame" mode with ECCExtractor to enhance
    REVMark's Detection performance.

- ERROR CORRECTION CAPACITY: 5G Polar codes with SCL decoding can typically
  correct ~10-15% bit errors depending on code rate. Beyond this threshold,
  decoding may produce incorrect results silently (no error flag).
"""

from typing import Literal
import tensorflow as tf

from sionna.fec.polar import Polar5GEncoder, Polar5GDecoder
from ecc_codecs.base import ECCCodec
import torch
import logging

logger = logging.getLogger(__name__)


class PolarWatermarkCodec(ECCCodec):
    """
    Polar codec for fixed-size watermark error correction (5G NR Polar via Sionna).

    Supports both sigmoid probability inputs and raw logit inputs from
    various watermarking neural networks.
    """

    MINIMUM_MESSAGE_BITS = 12

    def __init__(
        self,
        total_bits: int,
        message_bits: int,
        num_iter: int = 20,
        threshold: float = 0.5,
        device: torch.device = torch.device("cpu"),
        dec_type: str = "SCL",
        list_size: int = 8,
        *args,
        **kwargs,
    ):
        """
        Initialize Polar codec for watermark error correction.

        Args:
            total_bits: Total watermark capacity (codeword length after rate-matching)
            message_bits: Number of message bits (k). Must be >= 12 for 5G Polar.
            num_iter: BP iterations if dec_type="BP" (default: 20)
            threshold: Decision threshold for sigmoid inputs (default: 0.5)
            device: "cpu" or "cuda"
            dec_type: Polar decoder type: "SC", "SCL", or "BP" (default: "SCL")
            list_size: List size for SCL decoding (default: 8)

        Raises:
            ValueError: If message_bits < 12 (5G Polar minimum)
        """
        self.total_bits = int(total_bits)
        self.message_bits = int(message_bits)
        self.threshold = float(threshold)
        self.num_iter = int(num_iter)
        self.dec_type = dec_type.upper()
        self.list_size = int(list_size)

        self._validation()

        self.device = device
        if device == torch.device("cpu"):
            tf.config.set_visible_devices([], "GPU")
        elif device == torch.device("cuda"):
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                tf.config.set_visible_devices(gpus, "GPU")

        # 5G Polar uses rate-matching, so n can be the target length
        self.n = self.total_bits

        # Initialize 5G Polar Encoder
        self.encoder = Polar5GEncoder(k=self.message_bits, n=self.n)

        # Initialize 5G Polar Decoder with appropriate settings
        decoder_kwargs = {
            "enc_polar": self.encoder,
            "dec_type": self.dec_type,
        }

        if self.dec_type == "SCL":
            decoder_kwargs["list_size"] = self.list_size
        elif self.dec_type == "BP":
            decoder_kwargs["num_iter"] = self.num_iter

        self.decoder = Polar5GDecoder(**decoder_kwargs)

        # Internal constants
        self._llr_max = 20.0  # Clip LLRs to avoid numerical issues
        self._sigmoid_range_margin = 0.05  # Margin for sigmoid detection

    @property
    def info(self) -> dict:
        """Return codec configuration info."""
        return {
            "codec": f"Polar 5G ({self.dec_type})",
            "message_bits": self.message_bits,
            "codeword_bits": self.n,
            "total_bits": self.total_bits,
            "num_iter": self.num_iter if self.dec_type == "BP" else None,
            "list_size": self.list_size if self.dec_type == "SCL" else None,
        }

    def is_power_of_two(self, n: int) -> bool:
        return n > 0 and (n & (n - 1)) == 0

    def _validation(self):
        if self.message_bits < self.MINIMUM_MESSAGE_BITS:
            raise ValueError(
                f"5G Polar requires message_bits >= 12, got {self.message_bits}"
            )

        if not self.is_power_of_two(self.total_bits):
            logger.warning(
                f"total_bits={self.total_bits} is not a power of two. Polar decoding expects N=2^n; "
                f"results may be unreliable."
            )

        if self.total_bits < self.message_bits:
            raise ValueError(
                f"total_bits={self.total_bits} must be >= message_bits={self.message_bits}"
            )

    def encode(self, message_bits: torch.Tensor) -> torch.Tensor:
        """
        Encode message bits into watermark codeword bits.

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

        u = self._torch_to_tf(message_bits)
        u = tf.cast(u, tf.float32)

        c = self.encoder(u)  # (B, n)

        return self._tf_to_torch(c).to(torch.uint8)

    def decode(
        self,
        watermark_bits: torch.Tensor,
        return_confidence: bool = False,
        input_type: Literal["auto", "sigmoid", "logits"] = "auto",
    ) -> tuple[torch.Tensor, list[int]] | tuple[torch.Tensor, list[int], list[float]]:
        """
        Decode watermark bits using Polar 5G soft decoding.

        Args:
            watermark_bits: (B, total_bits) tensor of soft or hard values.
                - Sigmoid: values in [0, 1], probability of bit=1
                - Logits: unbounded real values, positive → bit=1 more likely
            return_confidence: If True, also return confidence score
            input_type: Input interpretation mode:
                - "auto": Detect based on value range (default)
                - "sigmoid": Treat as sigmoid probabilities [0, 1]
                - "logits": Treat as raw logits (unbounded)

        Returns:
            u_hat: (B, message_bits) decoded message bits as uint8
            bit_errors: Estimated bit errors (heuristic, always 0 for now)
            confidence: (optional) Confidence score if return_confidence=True

        Raises:
            ValueError: If input shape is invalid
        """
        if watermark_bits.ndim != 2 or watermark_bits.size(-1) < self.n:
            raise ValueError(
                f"watermark_bits must be 1D with length >= {self.n}. "
                f"Got shape {watermark_bits.shape}."
            )

        # Take only the bits used for Polar coding
        polar_bits = watermark_bits[:, : self.n]

        # Detect or use specified input type
        detected_type = self._detect_input_type(polar_bits, input_type)

        # Convert to LLRs based on input type
        if detected_type == "sigmoid":
            llr = self._sigmoid_to_llr(polar_bits)
            confidence = self._compute_confidence_sigmoid(polar_bits)
        else:  # logits
            llr = self._clip_llr(polar_bits)
            confidence = self._compute_confidence_logits(polar_bits)

        # torch -> tf
        y = self._torch_to_tf(llr)
        y = tf.cast(y, tf.float32)

        # Decode - returns hard decisions directly
        u_hat = self.decoder(y)  # (1, k)
        u_hat = self._tf_to_torch(u_hat).to(torch.uint8)

        bit_errors = [0] * polar_bits.size(
            0
        )  # Polar decoder doesn't provide error count

        if return_confidence:
            return u_hat, bit_errors, confidence
        return u_hat, bit_errors

    def _detect_input_type(
        self, values: torch.Tensor, specified: str
    ) -> Literal["sigmoid", "logits"]:
        """
        Detect whether input values are sigmoid probabilities or raw logits.

        Args:
            values: Input array
            specified: User-specified type ("auto", "sigmoid", "logits")

        Returns:
            Detected type: "sigmoid" or "logits"
        """
        if specified == "sigmoid":
            return "sigmoid"
        elif specified == "logits":
            return "logits"
        else:  # auto
            min_val, max_val = float(values.min()), float(values.max())

            # Check if values are within sigmoid range [0, 1] with margin
            if (
                min_val >= -self._sigmoid_range_margin
                and max_val <= 1.0 + self._sigmoid_range_margin
            ):
                # Additional check: if all values are very close to 0 or 1,
                # they might still be hard logits
                if min_val >= 0.0 and max_val <= 1.0:
                    return "sigmoid"

            return "logits"

    def _sigmoid_to_llr(self, p: torch.Tensor) -> torch.Tensor:
        """
        Convert sigmoid probabilities to LLRs.

        LLR = log(p / (1 - p)) where p = P(bit = 1)
        - p close to 1 → large positive LLR → bit=1 likely
        - p close to 0 → large negative LLR → bit=0 likely

        Args:
            p: Sigmoid probabilities in [0, 1]

        Returns:
            LLR values (clipped for numerical stability)
        """
        eps = 1e-7
        p_clipped = torch.clamp(p, eps, 1.0 - eps)
        llr = torch.log(p_clipped / (1.0 - p_clipped))
        return self._clip_llr(llr)

    def _clip_llr(self, llr: torch.Tensor) -> torch.Tensor:
        """Clip LLR values to avoid numerical issues."""
        return torch.clamp(llr, -self._llr_max, self._llr_max)

    def _compute_confidence_sigmoid(self, p: torch.Tensor) -> list[float]:
        """
        Compute confidence score for sigmoid inputs.

        Confidence is based on how far values are from the decision boundary (0.5).
        Values close to 0 or 1 indicate high confidence.
        """
        # Distance from 0.5, normalized to [0, 1]
        return torch.mean(torch.abs(p - self.threshold) * 2, dim=1).tolist()

    def _compute_confidence_logits(self, logits: torch.Tensor) -> list[float]:
        """
        Compute confidence score for logit inputs.

        Confidence is based on the magnitude of logits.
        Large absolute values indicate high confidence.
        """
        # Sigmoid of absolute logits gives confidence
        # Large |logit| → sigmoid close to 1 → high confidence
        abs_logits = torch.abs(logits)
        # Normalize by expected max to get [0, 1] range
        normalized = torch.tanh(abs_logits / 4.0)  # tanh saturates around |x|=4
        return torch.mean(normalized, dim=1).tolist()

    def _torch_to_tf(self, x: torch.Tensor) -> tf.Tensor:
        x_detached = x.detach()
        return tf.experimental.dlpack.from_dlpack(
            torch.utils.dlpack.to_dlpack(x_detached)
        )

    def _tf_to_torch(self, x: tf.Tensor) -> torch.Tensor:
        dlpack = tf.experimental.dlpack.to_dlpack(x)
        return torch.utils.dlpack.from_dlpack(dlpack)
