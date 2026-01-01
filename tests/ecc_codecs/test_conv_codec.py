import torch
import pytest

from ecc_codecs.conv_codec import ConvWatermarkCodec


class TestInitialization:
    def test_normal_initialization(self):
        """Normal initialization with default parameters."""
        codec = ConvWatermarkCodec(
            total_bits=64, message_bits=32, rate=1 / 2, constraint_length=7
        )

        assert codec.message_bits == 32
        assert codec.total_bits == 64
        assert codec.rate == 0.5
        assert codec.constraint_length == 7
        assert codec.threshold == 0.5
        assert codec.method == "soft_llr"

    def test_initialization_with_rate_third(self):
        """Initialization with rate 1/3."""
        codec = ConvWatermarkCodec(
            total_bits=96, message_bits=32, rate=1 / 3, constraint_length=5
        )

        assert codec.rate == 1 / 3
        assert codec.constraint_length == 5
        assert codec.total_bits == 96

    def test_initialization_with_soft_method(self):
        """Initialization with soft decoding method."""
        codec = ConvWatermarkCodec(total_bits=64, message_bits=32, method="soft")

        assert codec.method == "soft"

    def test_initialization_with_hard_method(self):
        """Initialization with hard decoding method."""
        codec = ConvWatermarkCodec(total_bits=64, message_bits=32, method="hard")

        assert codec.method == "hard"

    def test_initialization_with_custom_gen_poly(self):
        """Initialization with custom generator polynomials."""
        gen_poly = ("101", "111")  # rate 1/2, constraint_length 3
        codec = ConvWatermarkCodec(total_bits=64, message_bits=32, gen_poly=gen_poly)

        assert codec.gen_poly == gen_poly
        assert codec.rate == 0.5
        assert codec.constraint_length == 3

    def test_invalid_rate_raises(self):
        """Invalid rate should raise ValueError."""
        with pytest.raises(ValueError, match="rate must be"):
            ConvWatermarkCodec(total_bits=128, message_bits=32, rate=1 / 4)

    def test_invalid_constraint_length_raises(self):
        """Invalid constraint_length should raise ValueError."""
        with pytest.raises(ValueError, match="constraint_length must be"):
            ConvWatermarkCodec(total_bits=64, message_bits=32, constraint_length=2)

    def test_invalid_method_raises(self):
        """Invalid method should raise ValueError."""
        with pytest.raises(ValueError, match="method must be"):
            ConvWatermarkCodec(total_bits=64, message_bits=32, method="invalid")

    def test_total_bits_mismatch_raises(self):
        """Mismatched total_bits for given rate should raise ValueError."""
        with pytest.raises(ValueError, match="total_bits="):
            ConvWatermarkCodec(total_bits=100, message_bits=32, rate=1 / 2)

    def test_info_property(self):
        """Test info property returns correct values."""
        codec = ConvWatermarkCodec(
            total_bits=64, message_bits=32, rate=1 / 2, constraint_length=7
        )
        info = codec.info

        assert info["message_bits"] == 32
        assert info["codeword_bits"] == 64
        assert info["total_bits"] == 64
        assert info["rate"] == 0.5
        assert info["constraint_length"] == 7
        assert "Convolutional" in info["codec"]


class TestEncode:
    @pytest.fixture
    def codec(self):
        return ConvWatermarkCodec(total_bits=64, message_bits=32)

    def test_encode_output_length_and_format(self, codec):
        """Encode output length and format validation."""
        message_bits = torch.randint(0, 2, size=(1, 32), dtype=torch.uint8)
        result = codec.encode(message_bits)

        assert result.shape == (1, 64)
        assert set(result.unique().tolist()).issubset({0, 1})

    def test_encode_batch(self, codec):
        """Encode with batch input."""
        message_bits = torch.randint(0, 2, size=(4, 32), dtype=torch.uint8)
        result = codec.encode(message_bits)

        assert result.shape == (4, 64)

    def test_encode_deterministic(self, codec):
        """Same input should produce same output."""
        message_bits = torch.tensor([[1, 0] * 16], dtype=torch.uint8)
        result1 = codec.encode(message_bits)
        result2 = codec.encode(message_bits)

        torch.testing.assert_close(result1, result2)

    def test_encode_wrong_length_raises(self, codec):
        """Encode with wrong input length should raise ValueError."""
        wrong_length = torch.zeros(1, 16, dtype=torch.uint8)
        with pytest.raises(ValueError, match="Expected 32 bits, got 16"):
            codec.encode(wrong_length)

    def test_encode_non_binary_raises(self, codec):
        """Encode with non-binary input should raise ValueError."""
        non_binary = torch.tensor([[0, 1, 2] + [0] * 29], dtype=torch.uint8)
        with pytest.raises(ValueError, match="must be binary"):
            codec.encode(non_binary)


class TestDecode:
    @pytest.fixture
    def codec(self):
        return ConvWatermarkCodec(total_bits=64, message_bits=32)

    def test_decode_no_error_roundtrip(self, codec):
        """No-error round-trip with random message."""
        message_bits = torch.randint(0, 2, size=(1, 32), dtype=torch.uint8)
        encoded = codec.encode(message_bits)
        decoded, _ = codec.decode(encoded.float())

        torch.testing.assert_close(decoded, message_bits)

    def test_decode_batch_roundtrip(self, codec):
        """Batch round-trip with random messages."""
        message_bits = torch.randint(0, 2, size=(4, 32), dtype=torch.uint8)
        encoded = codec.encode(message_bits)
        decoded, bit_errors = codec.decode(encoded.float())

        assert decoded.shape == (4, 32)
        torch.testing.assert_close(decoded, message_bits)
        assert len(bit_errors) == 4

    def test_decode_with_soft_values(self, codec):
        """Decode with soft values (0.1 for 0, 0.9 for 1)."""
        message_bits = torch.randint(0, 2, size=(1, 32), dtype=torch.uint8)
        encoded = codec.encode(message_bits)

        # Convert to soft values
        soft_watermark = torch.where(encoded == 1, 0.9, 0.1).float()
        decoded, _ = codec.decode(soft_watermark)

        torch.testing.assert_close(decoded, message_bits)

    def test_decode_wrong_shape_raises(self, codec):
        """Decode with wrong shape should raise ValueError."""
        wrong_shape = torch.zeros(1, 32, dtype=torch.float)
        with pytest.raises(ValueError, match="must be 2D with length >= 64"):
            codec.decode(wrong_shape)

    def test_decode_1d_array_raises(self, codec):
        """Decode with 1D array should raise ValueError."""
        wrong_dim = torch.zeros(64, dtype=torch.float)
        with pytest.raises(ValueError):
            codec.decode(wrong_dim)


class TestSoftInputAndConfidence:
    @pytest.fixture
    def codec(self):
        return ConvWatermarkCodec(total_bits=64, message_bits=32)

    def _to_soft_values(
        self, hard_bits: torch.Tensor, low: float = 0.1, high: float = 0.9
    ) -> torch.Tensor:
        """Convert hard bits to soft values (0 -> low, 1 -> high)."""
        return torch.where(hard_bits == 1, high, low).float()

    def test_confidence_return_value(self, codec):
        """Confidence return value range and type."""
        message_bits = torch.randint(0, 2, size=(1, 32), dtype=torch.uint8)
        encoded = codec.encode(message_bits)

        soft_watermark = self._to_soft_values(encoded, low=0.1, high=0.9)
        _, _, confidence = codec.decode(soft_watermark, return_confidence=True)

        assert isinstance(confidence, list)
        assert len(confidence) == 1
        assert isinstance(confidence[0], float)
        assert 0.0 <= confidence[0] <= 1.0
        # With 0.1/0.9 values, confidence should be high
        assert confidence[0] >= 0.7, f"Expected high confidence, got {confidence[0]}"

    def test_confidence_batch(self, codec):
        """Confidence return value for batch inputs."""
        message_bits = torch.randint(0, 2, size=(4, 32), dtype=torch.uint8)
        encoded = codec.encode(message_bits)

        soft_watermark = self._to_soft_values(encoded, low=0.1, high=0.9)
        _, _, confidence = codec.decode(soft_watermark, return_confidence=True)

        assert isinstance(confidence, list)
        assert len(confidence) == 4
        for conf in confidence:
            assert 0.0 <= conf <= 1.0

    def test_low_confidence_values(self, codec):
        """Values near threshold produce low confidence."""
        message_bits = torch.randint(0, 2, size=(1, 32), dtype=torch.uint8)
        encoded = codec.encode(message_bits)

        # Low confidence soft values (close to threshold 0.5)
        soft_watermark = self._to_soft_values(encoded, low=0.45, high=0.55)
        _, _, confidence = codec.decode(soft_watermark, return_confidence=True)

        assert confidence[0] < 0.2, f"Expected low confidence, got {confidence[0]}"


class TestDecodingMethods:
    """Test different decoding methods produce valid output."""

    @pytest.fixture
    def message_bits(self):
        return torch.randint(0, 2, size=(1, 32), dtype=torch.uint8)

    def test_soft_llr_method_roundtrip(self, message_bits):
        """soft_llr method round-trip."""
        codec = ConvWatermarkCodec(total_bits=64, message_bits=32, method="soft_llr")
        encoded = codec.encode(message_bits)
        decoded, _ = codec.decode(encoded.float())

        torch.testing.assert_close(decoded, message_bits)

    def test_soft_method_roundtrip(self, message_bits):
        """soft method round-trip."""
        codec = ConvWatermarkCodec(total_bits=64, message_bits=32, method="soft")
        encoded = codec.encode(message_bits)
        decoded, _ = codec.decode(encoded.float())

        torch.testing.assert_close(decoded, message_bits)

    def test_hard_method_roundtrip(self, message_bits):
        """hard method round-trip."""
        codec = ConvWatermarkCodec(total_bits=64, message_bits=32, method="hard")
        encoded = codec.encode(message_bits)
        decoded, _ = codec.decode(encoded.float())

        torch.testing.assert_close(decoded, message_bits)


class TestConstraintLengths:
    """Test different constraint lengths."""

    @pytest.fixture
    def message_bits(self):
        return torch.randint(0, 2, size=(1, 32), dtype=torch.uint8)

    @pytest.mark.parametrize("constraint_length", [3, 4, 5, 6, 7, 8])
    def test_constraint_length_roundtrip(self, message_bits, constraint_length):
        """Round-trip with different constraint lengths."""
        codec = ConvWatermarkCodec(
            total_bits=64,
            message_bits=32,
            rate=1 / 2,
            constraint_length=constraint_length,
        )
        encoded = codec.encode(message_bits)
        decoded, _ = codec.decode(encoded.float())

        torch.testing.assert_close(decoded, message_bits)


class TestRates:
    """Test different code rates."""

    @pytest.fixture
    def message_bits(self):
        return torch.randint(0, 2, size=(1, 32), dtype=torch.uint8)

    def test_rate_half_roundtrip(self, message_bits):
        """Round-trip with rate 1/2."""
        codec = ConvWatermarkCodec(total_bits=64, message_bits=32, rate=1 / 2)
        encoded = codec.encode(message_bits)
        decoded, _ = codec.decode(encoded.float())

        assert encoded.shape == (1, 64)
        torch.testing.assert_close(decoded, message_bits)

    def test_rate_third_roundtrip(self, message_bits):
        """Round-trip with rate 1/3."""
        codec = ConvWatermarkCodec(total_bits=96, message_bits=32, rate=1 / 3)
        encoded = codec.encode(message_bits)
        decoded, _ = codec.decode(encoded.float())

        assert encoded.shape == (1, 96)
        torch.testing.assert_close(decoded, message_bits)


class TestDecodeSigmoidInputs:
    """Test decoding with sigmoid probability inputs [0, 1]."""

    @pytest.fixture
    def codec(self):
        return ConvWatermarkCodec(total_bits=64, message_bits=32)

    @pytest.fixture
    def message_bits(self):
        return torch.randint(0, 2, size=(1, 32), dtype=torch.uint8)

    def test_decode_hard_binary_values(self, codec, message_bits):
        """Decode with exact 0 and 1 values (sigmoid)."""
        encoded = codec.encode(message_bits)
        decoded, _ = codec.decode(encoded.float(), input_type="sigmoid")

        torch.testing.assert_close(decoded, message_bits)

    def test_decode_extreme_soft_values(self, codec, message_bits):
        """Decode with extreme soft values (0.0 and 1.0)."""
        encoded = codec.encode(message_bits)
        soft = torch.where(encoded == 1, 1.0, 0.0).float()
        decoded, _ = codec.decode(soft, input_type="sigmoid")

        torch.testing.assert_close(decoded, message_bits)

    def test_decode_near_extreme_values(self, codec, message_bits):
        """Decode with values near extremes (0.01 and 0.99)."""
        encoded = codec.encode(message_bits)
        soft = torch.where(encoded == 1, 0.99, 0.01).float()
        decoded, _ = codec.decode(soft, input_type="sigmoid")

        torch.testing.assert_close(decoded, message_bits)

    def test_decode_moderate_soft_values(self, codec, message_bits):
        """Decode with moderate soft values (0.2 and 0.8)."""
        encoded = codec.encode(message_bits)
        soft = torch.where(encoded == 1, 0.8, 0.2).float()
        decoded, _ = codec.decode(soft, input_type="sigmoid")

        torch.testing.assert_close(decoded, message_bits)

    def test_decode_weak_soft_values(self, codec, message_bits):
        """Decode with weak soft values (0.3 and 0.7)."""
        encoded = codec.encode(message_bits)
        soft = torch.where(encoded == 1, 0.7, 0.3).float()
        decoded, _ = codec.decode(soft, input_type="sigmoid")

        torch.testing.assert_close(decoded, message_bits)

    def test_decode_noisy_soft_values(self, codec, message_bits):
        """Decode with noisy soft values (adding small random noise)."""
        encoded = codec.encode(message_bits)
        # Base soft values with small noise
        base = torch.where(encoded == 1, 0.85, 0.15).float()
        noise = torch.empty_like(base).uniform_(-0.05, 0.05)
        soft = torch.clamp(base + noise, 0.0, 1.0)
        decoded, _ = codec.decode(soft, input_type="sigmoid")

        torch.testing.assert_close(decoded, message_bits)


class TestDecodeLogitInputs:
    """Test decoding with raw logit inputs (unbounded real values)."""

    @pytest.fixture
    def codec(self):
        return ConvWatermarkCodec(total_bits=64, message_bits=32)

    @pytest.fixture
    def message_bits(self):
        return torch.randint(0, 2, size=(1, 32), dtype=torch.uint8)

    def test_decode_strong_logits(self, codec, message_bits):
        """Decode with strong logits (+5 for 1, -5 for 0)."""
        encoded = codec.encode(message_bits)
        logits = torch.where(encoded == 1, 5.0, -5.0).float()
        decoded, _ = codec.decode(logits, input_type="logits")

        torch.testing.assert_close(decoded, message_bits)

    def test_decode_moderate_logits(self, codec, message_bits):
        """Decode with moderate logits (+2 for 1, -2 for 0)."""
        encoded = codec.encode(message_bits)
        logits = torch.where(encoded == 1, 2.0, -2.0).float()
        decoded, _ = codec.decode(logits, input_type="logits")

        torch.testing.assert_close(decoded, message_bits)

    def test_decode_weak_logits(self, codec, message_bits):
        """Decode with weak logits (+0.5 for 1, -0.5 for 0)."""
        encoded = codec.encode(message_bits)
        logits = torch.where(encoded == 1, 0.5, -0.5).float()
        decoded, _ = codec.decode(logits, input_type="logits")

        torch.testing.assert_close(decoded, message_bits)

    def test_decode_noisy_logits(self, codec, message_bits):
        """Decode with noisy logits."""
        encoded = codec.encode(message_bits)
        base = torch.where(encoded == 1, 3.0, -3.0).float()
        noise = torch.randn_like(base) * 0.5
        logits = base + noise
        decoded, _ = codec.decode(logits, input_type="logits")

        torch.testing.assert_close(decoded, message_bits)

    def test_logit_confidence(self, codec, message_bits):
        """Test confidence calculation for logit inputs."""
        encoded = codec.encode(message_bits)

        # Strong logits should give high confidence
        strong_logits = torch.where(encoded == 1, 5.0, -5.0).float()
        _, _, strong_conf = codec.decode(
            strong_logits, return_confidence=True, input_type="logits"
        )

        # Weak logits should give lower confidence
        weak_logits = torch.where(encoded == 1, 0.5, -0.5).float()
        _, _, weak_conf = codec.decode(
            weak_logits, return_confidence=True, input_type="logits"
        )

        assert strong_conf[0] > weak_conf[0]
        assert 0.0 <= strong_conf[0] <= 1.0
        assert 0.0 <= weak_conf[0] <= 1.0


class TestAutoInputTypeDetection:
    """Test automatic input type detection."""

    @pytest.fixture
    def codec(self):
        return ConvWatermarkCodec(total_bits=64, message_bits=32)

    @pytest.fixture
    def message_bits(self):
        return torch.randint(0, 2, size=(1, 32), dtype=torch.uint8)

    def test_auto_detect_sigmoid(self, codec, message_bits):
        """Auto-detect sigmoid inputs in [0, 1] range."""
        encoded = codec.encode(message_bits)
        soft = torch.where(encoded == 1, 0.9, 0.1).float()

        # Auto should detect as sigmoid
        decoded, _ = codec.decode(soft, input_type="auto")
        torch.testing.assert_close(decoded, message_bits)

    def test_auto_detect_logits_negative(self, codec, message_bits):
        """Auto-detect logits when values are negative."""
        encoded = codec.encode(message_bits)
        logits = torch.where(encoded == 1, 3.0, -3.0).float()

        # Auto should detect as logits (contains negative values)
        decoded, _ = codec.decode(logits, input_type="auto")
        torch.testing.assert_close(decoded, message_bits)

    def test_auto_detect_logits_greater_than_one(self, codec, message_bits):
        """Auto-detect logits when values are > 1."""
        encoded = codec.encode(message_bits)
        # Use values that clearly indicate logits (one value > 1)
        logits = torch.where(encoded == 1, 2.0, -2.0).float()
        # Add offset to make one value > 1 while keeping distinction
        logits = logits + 1.5  # Now values are 3.5 for 1, -0.5 for 0

        # Auto should detect as logits (contains values > 1)
        decoded, _ = codec.decode(logits, input_type="auto")
        torch.testing.assert_close(decoded, message_bits)

    def test_default_is_auto(self, codec, message_bits):
        """Default input_type should be 'auto'."""
        encoded = codec.encode(message_bits)
        soft = torch.where(encoded == 1, 0.9, 0.1).float()

        # Should work without specifying input_type
        decoded, _ = codec.decode(soft)
        torch.testing.assert_close(decoded, message_bits)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def codec(self):
        return ConvWatermarkCodec(total_bits=64, message_bits=32)

    @pytest.fixture
    def message_bits(self):
        return torch.randint(0, 2, size=(1, 32), dtype=torch.uint8)

    def test_decode_with_longer_input(self, codec, message_bits):
        """Decode accepts input longer than n (extra bits ignored)."""
        encoded = codec.encode(message_bits)
        # Pad with extra values
        extended = torch.cat([encoded.float(), torch.zeros(1, 32)], dim=1)
        decoded, _ = codec.decode(extended)

        torch.testing.assert_close(decoded, message_bits)

    def test_all_zeros_message(self, codec):
        """Encode and decode all-zeros message."""
        message_bits = torch.zeros(1, 32, dtype=torch.uint8)
        encoded = codec.encode(message_bits)
        decoded, _ = codec.decode(encoded.float())

        torch.testing.assert_close(decoded, message_bits)

    def test_all_ones_message(self, codec):
        """Encode and decode all-ones message."""
        message_bits = torch.ones(1, 32, dtype=torch.uint8)
        encoded = codec.encode(message_bits)
        decoded, _ = codec.decode(encoded.float())

        torch.testing.assert_close(decoded, message_bits)

    def test_alternating_bits_message(self, codec):
        """Encode and decode alternating bits message."""
        message_bits = torch.tensor([[0, 1] * 16], dtype=torch.uint8)
        encoded = codec.encode(message_bits)
        decoded, _ = codec.decode(encoded.float())

        torch.testing.assert_close(decoded, message_bits)

    def test_small_message_bits(self):
        """Test with small message_bits."""
        codec = ConvWatermarkCodec(total_bits=16, message_bits=8, rate=1 / 2)
        message_bits = torch.randint(0, 2, size=(1, 8), dtype=torch.uint8)
        encoded = codec.encode(message_bits)
        decoded, _ = codec.decode(encoded.float())

        torch.testing.assert_close(decoded, message_bits)


class TestCustomGeneratorPolynomials:
    """Test custom generator polynomials."""

    @pytest.fixture
    def message_bits(self):
        return torch.randint(0, 2, size=(1, 32), dtype=torch.uint8)

    def test_custom_rate_half_poly(self, message_bits):
        """Custom rate 1/2 generator polynomial."""
        gen_poly = ("101", "111")  # (5, 7) in octal
        codec = ConvWatermarkCodec(total_bits=64, message_bits=32, gen_poly=gen_poly)

        encoded = codec.encode(message_bits)
        decoded, _ = codec.decode(encoded.float())

        assert encoded.shape == (1, 64)
        torch.testing.assert_close(decoded, message_bits)

    def test_custom_rate_third_poly(self, message_bits):
        """Custom rate 1/3 generator polynomial."""
        gen_poly = ("101", "111", "111")  # (5, 7, 7)
        codec = ConvWatermarkCodec(total_bits=96, message_bits=32, gen_poly=gen_poly)

        encoded = codec.encode(message_bits)
        decoded, _ = codec.decode(encoded.float())

        assert encoded.shape == (1, 96)
        torch.testing.assert_close(decoded, message_bits)

    def test_invalid_gen_poly_format_raises(self):
        """Invalid gen_poly format should raise ValueError."""
        with pytest.raises(ValueError, match="string"):
            ConvWatermarkCodec(
                total_bits=64,
                message_bits=32,
                gen_poly=(101, 111),  # Not strings
            )

    def test_invalid_gen_poly_chars_raises(self):
        """gen_poly with invalid characters should raise ValueError."""
        with pytest.raises(ValueError, match="0's and 1's"):
            ConvWatermarkCodec(total_bits=64, message_bits=32, gen_poly=("102", "111"))

    def test_mismatched_poly_lengths_raises(self):
        """gen_poly with different lengths should raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            ConvWatermarkCodec(total_bits=64, message_bits=32, gen_poly=("101", "11"))
