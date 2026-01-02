"""Convolutional Code Error Correction for Watermark Systems.

This module provides Convolutional encoding/decoding via Sionna for watermark systems.

Convolutional Code Configuration:
- Uses Viterbi decoding for optimal soft-decision decoding
- Supports rate 1/2 and 1/3 codes
- Constraint lengths from 3 to 8
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
- Convolutional codes are well-suited for continuous error correction
- Viterbi decoding provides optimal Maximum Likelihood decoding
- Performance depends on constraint length (longer = stronger but more complex)
- Rate 1/2 provides good balance between redundancy and capacity
"""

from typing import Literal
import tensorflow as tf

from sionna.fec.conv import ConvEncoder, ViterbiDecoder
from ecc_codecs.base import ECCCodec
from schemas.converters import ValueConverter
from schemas.model_outputs import OutputValueType
import torch
import logging

logger = logging.getLogger(__name__)


class ConvWatermarkCodec(ECCCodec):
    """
    Convolutional codec for fixed-size watermark error correction (Viterbi via Sionna).

    Supports both sigmoid probability inputs and raw logit inputs from
    various watermarking neural networks.
    """

    VALID_RATES = (1 / 2, 1 / 3)
    VALID_CONSTRAINT_LENGTHS = (3, 4, 5, 6, 7, 8)

    def __init__(
        self,
        total_bits: int,
        message_bits: int,
        rate: float = 1 / 2,
        constraint_length: int = 7,
        threshold: float = 0.5,
        device: torch.device = torch.device("cpu"),
        method: str = "soft_llr",
        gen_poly: tuple[str, ...] | None = None,
        *args,
        **kwargs,
    ):
        """
        Initialize Convolutional codec for watermark error correction.

        Args:
            total_bits: Total watermark capacity (codeword length after encoding).
                        Must satisfy: total_bits = message_bits / rate
            message_bits: Number of message bits (k).
            rate: Code rate, either 1/2 or 1/3 (default: 1/2).
                  Ignored if gen_poly is provided.
            constraint_length: Constraint length (3-8). Determines code strength.
                              Ignored if gen_poly is provided. (default: 7)
            threshold: Decision threshold for sigmoid inputs (default: 0.5)
            device: "cpu" or "cuda"
            method: Viterbi decoding method:
                - "soft_llr": Soft LLR input (default, best for neural network outputs)
                - "soft": Soft bit values centered on 0/1
                - "hard": Hard binary decisions
            gen_poly: Optional custom generator polynomials as tuple of binary strings.
                      Example: ('1011011', '1111001') for constraint_length=7, rate=1/2

        Raises:
            ValueError: If configuration is invalid
        """
        self.total_bits = int(total_bits)
        self.message_bits = int(message_bits)
        self.threshold = float(threshold)
        self.method = method.lower()
        self.gen_poly = gen_poly

        # Validate and determine rate from gen_poly if provided
        if gen_poly is not None:
            # Early validation of gen_poly format
            if not all(isinstance(poly, str) for poly in gen_poly):
                raise ValueError("Each element of gen_poly must be a string.")
            if not all(len(poly) == len(gen_poly[0]) for poly in gen_poly):
                raise ValueError("Each polynomial must be of same length.")
            if not all(all(char in ["0", "1"] for char in poly) for poly in gen_poly):
                raise ValueError("Each polynomial must be a string of 0's and 1's.")
            self.rate = 1 / len(gen_poly)
            self.constraint_length = len(gen_poly[0])
        else:
            self.rate = rate
            self.constraint_length = int(constraint_length)

        self._validation()

        self.device = device
        if device == torch.device("cpu"):
            tf.config.set_visible_devices([], "GPU")
        elif device == torch.device("cuda"):
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                tf.config.set_visible_devices(gpus, "GPU")

        # Initialize Convolutional Encoder
        encoder_kwargs = {"output_dtype": tf.float32}
        if gen_poly is not None:
            encoder_kwargs["gen_poly"] = gen_poly
        else:
            encoder_kwargs["rate"] = self.rate
            encoder_kwargs["constraint_length"] = self.constraint_length

        self.encoder = ConvEncoder(**encoder_kwargs)

        # Initialize Viterbi Decoder with same settings
        decoder_kwargs = {
            "method": self.method,
            "output_dtype": tf.float32,
        }
        if gen_poly is not None:
            decoder_kwargs["gen_poly"] = gen_poly
        else:
            decoder_kwargs["rate"] = self.rate
            decoder_kwargs["constraint_length"] = self.constraint_length

        self.decoder = ViterbiDecoder(**decoder_kwargs)

    @property
    def info(self) -> dict:
        """Return codec configuration info."""
        return {
            "codec": f"Convolutional (Viterbi, {self.method})",
            "message_bits": self.message_bits,
            "codeword_bits": self.total_bits,
            "total_bits": self.total_bits,
            "rate": self.rate,
            "constraint_length": self.constraint_length,
            "gen_poly": self.gen_poly or "default",
            "method": self.method,
        }

    def _validation(self):
        """Validate codec configuration."""
        # Validate method
        if self.method not in ("soft_llr", "soft", "hard"):
            raise ValueError(
                f"method must be 'soft_llr', 'soft', or 'hard', got '{self.method}'"
            )

        # Validate rate and constraint_length (only if not using custom gen_poly)
        if self.gen_poly is None:
            if self.rate not in self.VALID_RATES:
                raise ValueError(
                    f"rate must be 1/2 or 1/3, got {self.rate}. "
                    f"For other rates, provide custom gen_poly."
                )

            if self.constraint_length not in self.VALID_CONSTRAINT_LENGTHS:
                raise ValueError(
                    f"constraint_length must be in {self.VALID_CONSTRAINT_LENGTHS}, "
                    f"got {self.constraint_length}"
                )
        # gen_poly validation already done in __init__ before calling this method

        # Validate total_bits vs message_bits relationship
        expected_total = int(self.message_bits / self.rate)
        if self.total_bits != expected_total:
            raise ValueError(
                f"For rate={self.rate}, message_bits={self.message_bits} requires "
                f"total_bits={expected_total}, but got {self.total_bits}. "
                f"Ensure: total_bits = message_bits / rate"
            )

        if self.message_bits < 1:
            raise ValueError(f"message_bits must be >= 1, got {self.message_bits}")

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

        c = self.encoder(u, training=False)  # (B, n)

        return self._tf_to_torch(c).to(torch.uint8)

    def decode(
        self,
        watermark_bits: torch.Tensor,
        return_confidence: bool = False,
        input_type: Literal["auto", "sigmoid", "logits"] = "auto",
    ) -> tuple[torch.Tensor, list[int]] | tuple[torch.Tensor, list[int], list[float]]:
        """
        Decode watermark bits using Viterbi soft decoding.

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
            bit_errors: Estimated bit errors (always 0 for Viterbi, no error count)
            confidence: (optional) Confidence score if return_confidence=True

        Raises:
            ValueError: If input shape is invalid
        """
        if watermark_bits.ndim != 2 or watermark_bits.size(-1) < self.total_bits:
            raise ValueError(
                f"watermark_bits must be 2D with length >= {self.total_bits}. "
                f"Got shape {watermark_bits.shape}."
            )

        # Take only the bits used for convolutional coding
        conv_bits = watermark_bits[:, : self.total_bits]

        # Detect or use specified input type using unified ValueConverter
        detected_type = ValueConverter.detect_value_type(conv_bits, input_type)

        # Compute confidence based on value type
        if detected_type == OutputValueType.SIGMOID:
            confidence = self._compute_confidence_sigmoid(conv_bits)
        else:  # logits
            confidence = self._compute_confidence_logits(conv_bits)

        # Convert to appropriate format for Viterbi decoder
        if self.method == "soft_llr":
            # Convert to LLRs
            y = ValueConverter.to_llr(conv_bits, detected_type)
        elif self.method == "soft":
            # Convert to probability space [0, 1]
            y = ValueConverter.to_probability(conv_bits, detected_type)
        else:  # hard
            # Convert to binary
            y = ValueConverter.to_binary(conv_bits, detected_type).float()

        # torch -> tf
        y_tf = self._torch_to_tf(y)
        y_tf = tf.cast(y_tf, tf.float32)

        # Decode - Viterbi returns hard decisions
        u_hat = self.decoder(y_tf, training=False)  # (B, k)
        u_hat = self._tf_to_torch(u_hat).to(torch.uint8)

        # Viterbi decoder doesn't provide error count
        bit_errors = [0] * conv_bits.size(0)

        if return_confidence:
            return u_hat, bit_errors, confidence
        return u_hat, bit_errors

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
