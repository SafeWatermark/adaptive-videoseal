import torch
import pytest

from ecc_codecs.bch_codec import BCHWatermarkCodec, GaloisField


class TestGaloisField:
    """Test Galois Field arithmetic."""

    def test_gf_init_m4(self):
        """Initialize GF(2^4)."""
        gf = GaloisField(4)
        assert gf.m == 4
        assert gf.n == 15
        assert gf.field_size == 16

    def test_gf_init_m5(self):
        """Initialize GF(2^5)."""
        gf = GaloisField(5)
        assert gf.m == 5
        assert gf.n == 31
        assert gf.field_size == 32

    def test_gf_init_m6(self):
        """Initialize GF(2^6)."""
        gf = GaloisField(6)
        assert gf.m == 6
        assert gf.n == 63
        assert gf.field_size == 64

    def test_gf_invalid_m(self):
        """Invalid m should raise ValueError."""
        with pytest.raises(ValueError):
            GaloisField(2)

    def test_gf_mul_identity(self):
        """Multiplication by 1 gives identity."""
        gf = GaloisField(4)
        for a in range(1, gf.field_size):
            assert gf.mul(a, 1) == a

    def test_gf_mul_zero(self):
        """Multiplication by 0 gives 0."""
        gf = GaloisField(4)
        for a in range(gf.field_size):
            assert gf.mul(a, 0) == 0
            assert gf.mul(0, a) == 0

    def test_gf_mul_commutative(self):
        """Multiplication is commutative."""
        gf = GaloisField(4)
        for a in range(1, gf.field_size):
            for b in range(1, gf.field_size):
                assert gf.mul(a, b) == gf.mul(b, a)

    def test_gf_div_inverse(self):
        """a / a = 1 for non-zero a."""
        gf = GaloisField(4)
        for a in range(1, gf.field_size):
            assert gf.div(a, a) == 1

    def test_gf_div_by_one(self):
        """a / 1 = a."""
        gf = GaloisField(4)
        for a in range(1, gf.field_size):
            assert gf.div(a, 1) == a

    def test_gf_pow_identity(self):
        """a^1 = a."""
        gf = GaloisField(4)
        for a in range(1, gf.field_size):
            assert gf.pow(a, 1) == a

    def test_gf_pow_zero(self):
        """a^0 = 1 for non-zero a."""
        gf = GaloisField(4)
        for a in range(1, gf.field_size):
            assert gf.pow(a, 0) == 1

    def test_gf_inv(self):
        """a * inv(a) = 1."""
        gf = GaloisField(4)
        for a in range(1, gf.field_size):
            assert gf.mul(a, gf.inv(a)) == 1


class TestInitialization:
    def test_normal_initialization_m5_t2(self):
        """Normal initialization with BCH(31, 21, 2)."""
        codec = BCHWatermarkCodec(total_bits=62, message_bits=42, m=5, t=2)

        assert codec.message_bits == 42
        assert codec.total_bits == 62
        assert codec.m == 5
        assert codec.t == 2
        assert codec.n == 31
        assert codec.k == 21
        assert codec.num_blocks == 2
        assert codec.codeword_bits == 62

    def test_initialization_m4_t2(self):
        """Initialization with BCH(15, 7, 2)."""
        codec = BCHWatermarkCodec(total_bits=30, message_bits=14, m=4, t=2)

        assert codec.m == 4
        assert codec.t == 2
        assert codec.n == 15
        assert codec.k == 7
        assert codec.num_blocks == 2

    def test_initialization_m5_t3(self):
        """Initialization with BCH(31, 16, 3)."""
        codec = BCHWatermarkCodec(total_bits=62, message_bits=32, m=5, t=3)

        assert codec.m == 5
        assert codec.t == 3
        assert codec.n == 31
        assert codec.k == 16
        assert codec.num_blocks == 2

    def test_initialization_m6_t2(self):
        """Initialization with BCH(63, 51, 2)."""
        codec = BCHWatermarkCodec(total_bits=126, message_bits=102, m=6, t=2)

        assert codec.m == 6
        assert codec.t == 2
        assert codec.n == 63
        assert codec.k == 51
        assert codec.num_blocks == 2

    def test_invalid_config_raises(self):
        """Invalid (m, t) should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid"):
            BCHWatermarkCodec(total_bits=64, message_bits=32, m=4, t=5)

    def test_message_bits_not_divisible_raises(self):
        """message_bits not divisible by k should raise ValueError."""
        with pytest.raises(ValueError, match="must be divisible by k"):
            BCHWatermarkCodec(total_bits=64, message_bits=20, m=5, t=2)

    def test_total_bits_too_small_raises(self):
        """total_bits too small should raise ValueError."""
        with pytest.raises(ValueError, match="too small"):
            BCHWatermarkCodec(total_bits=30, message_bits=42, m=5, t=2)

    def test_info_property(self):
        """Test info property returns correct values."""
        codec = BCHWatermarkCodec(total_bits=62, message_bits=42, m=5, t=2)
        info = codec.info

        assert info["message_bits"] == 42
        assert info["codeword_bits"] == 62
        assert info["total_bits"] == 62
        assert info["m"] == 5
        assert info["n"] == 31
        assert info["k"] == 21
        assert info["t"] == 2
        assert info["error_correction_capacity"] == 2
        assert "BCH(31,21,2)" in info["codec"]


class TestEncode:
    @pytest.fixture
    def codec_m4_t2(self):
        """BCH(15, 7, 2) codec with 14 message bits."""
        return BCHWatermarkCodec(total_bits=30, message_bits=14, m=4, t=2)

    @pytest.fixture
    def codec_m5_t2(self):
        """BCH(31, 21, 2) codec with 42 message bits."""
        return BCHWatermarkCodec(total_bits=62, message_bits=42, m=5, t=2)

    def test_encode_output_length_and_format(self, codec_m5_t2):
        """Encode output length and format validation."""
        message_bits = torch.randint(0, 2, size=(1, 42), dtype=torch.uint8)
        result = codec_m5_t2.encode(message_bits)

        assert result.shape == (1, 62)
        assert set(result.unique().tolist()).issubset({0, 1})

    def test_encode_batch(self, codec_m5_t2):
        """Encode with batch input."""
        message_bits = torch.randint(0, 2, size=(4, 42), dtype=torch.uint8)
        result = codec_m5_t2.encode(message_bits)

        assert result.shape == (4, 62)

    def test_encode_deterministic(self, codec_m5_t2):
        """Same input should produce same output."""
        message_bits = torch.randint(0, 2, size=(1, 42), dtype=torch.uint8)
        result1 = codec_m5_t2.encode(message_bits)
        result2 = codec_m5_t2.encode(message_bits)

        torch.testing.assert_close(result1, result2)

    def test_encode_wrong_length_raises(self, codec_m5_t2):
        """Encode with wrong input length should raise ValueError."""
        wrong_length = torch.zeros(1, 30, dtype=torch.uint8)
        with pytest.raises(ValueError, match="Expected 42 bits"):
            codec_m5_t2.encode(wrong_length)

    def test_encode_non_binary_raises(self, codec_m5_t2):
        """Encode with non-binary input should raise ValueError."""
        non_binary = torch.tensor([[0, 1, 2] + [0] * 39], dtype=torch.uint8)
        with pytest.raises(ValueError, match="must be binary"):
            codec_m5_t2.encode(non_binary)

    def test_encode_with_padding(self):
        """Encode should pad to total_bits if needed."""
        codec = BCHWatermarkCodec(total_bits=80, message_bits=42, m=5, t=2)
        message_bits = torch.randint(0, 2, size=(1, 42), dtype=torch.uint8)
        result = codec.encode(message_bits)

        assert result.shape == (1, 80)
        # Last bits should be padding zeros
        assert torch.all(result[:, 62:] == 0)


class TestDecode:
    @pytest.fixture
    def codec_m4_t2(self):
        return BCHWatermarkCodec(total_bits=30, message_bits=14, m=4, t=2)

    @pytest.fixture
    def codec_m5_t2(self):
        return BCHWatermarkCodec(total_bits=62, message_bits=42, m=5, t=2)

    def test_decode_no_error_roundtrip(self, codec_m5_t2):
        """No-error round-trip."""
        message_bits = torch.randint(0, 2, size=(1, 42), dtype=torch.uint8)
        encoded = codec_m5_t2.encode(message_bits)
        decoded, errors = codec_m5_t2.decode(encoded.float())

        torch.testing.assert_close(decoded, message_bits)
        assert errors[0] == 0

    def test_decode_batch_roundtrip(self, codec_m5_t2):
        """Batch round-trip with random messages."""
        message_bits = torch.randint(0, 2, size=(4, 42), dtype=torch.uint8)
        encoded = codec_m5_t2.encode(message_bits)
        decoded, bit_errors = codec_m5_t2.decode(encoded.float())

        assert decoded.shape == (4, 42)
        torch.testing.assert_close(decoded, message_bits)
        assert len(bit_errors) == 4

    def test_decode_with_soft_values(self, codec_m5_t2):
        """Decode with soft values (0.1 for 0, 0.9 for 1)."""
        message_bits = torch.randint(0, 2, size=(1, 42), dtype=torch.uint8)
        encoded = codec_m5_t2.encode(message_bits)

        # Convert to soft values
        soft_watermark = torch.where(encoded == 1, 0.9, 0.1).float()
        decoded, _ = codec_m5_t2.decode(soft_watermark)

        torch.testing.assert_close(decoded, message_bits)

    def test_decode_wrong_shape_raises(self, codec_m5_t2):
        """Decode with wrong shape should raise ValueError."""
        wrong_shape = torch.zeros(1, 30, dtype=torch.float)
        with pytest.raises(ValueError, match="must be 2D with length >= 62"):
            codec_m5_t2.decode(wrong_shape)

    def test_decode_1d_array_raises(self, codec_m5_t2):
        """Decode with 1D array should raise ValueError."""
        wrong_dim = torch.zeros(62, dtype=torch.float)
        with pytest.raises(ValueError):
            codec_m5_t2.decode(wrong_dim)


class TestMultiBitErrorCorrection:
    """Test BCH code's multi-bit error correction capability."""

    @pytest.fixture
    def codec_m4_t2(self):
        return BCHWatermarkCodec(total_bits=30, message_bits=14, m=4, t=2)

    @pytest.fixture
    def codec_m5_t2(self):
        return BCHWatermarkCodec(total_bits=62, message_bits=42, m=5, t=2)

    @pytest.fixture
    def codec_m5_t3(self):
        return BCHWatermarkCodec(total_bits=62, message_bits=32, m=5, t=3)

    def test_correct_single_error_per_block(self, codec_m5_t2):
        """BCH(31, 21, 2) should correct 1-bit error per block."""
        message_bits = torch.randint(0, 2, size=(1, 42), dtype=torch.uint8)
        encoded = codec_m5_t2.encode(message_bits)

        # Flip one bit in each block
        corrupted = encoded.clone().float()
        corrupted[0, 5] = 1.0 - corrupted[0, 5]  # Error in first block
        corrupted[0, 35] = 1.0 - corrupted[0, 35]  # Error in second block

        decoded, errors = codec_m5_t2.decode(corrupted)

        torch.testing.assert_close(decoded, message_bits)
        assert errors[0] == 2

    def test_correct_two_errors_per_block(self, codec_m5_t2):
        """BCH(31, 21, 2) should correct 2-bit errors per block."""
        message_bits = torch.randint(0, 2, size=(1, 42), dtype=torch.uint8)
        encoded = codec_m5_t2.encode(message_bits)

        # Flip two bits in first block
        corrupted = encoded.clone().float()
        corrupted[0, 3] = 1.0 - corrupted[0, 3]
        corrupted[0, 10] = 1.0 - corrupted[0, 10]

        decoded, errors = codec_m5_t2.decode(corrupted)

        torch.testing.assert_close(decoded, message_bits)
        assert errors[0] == 2

    def test_correct_three_errors_t3(self, codec_m5_t3):
        """BCH(31, 16, 3) should correct 3-bit errors per block."""
        message_bits = torch.randint(0, 2, size=(1, 32), dtype=torch.uint8)
        encoded = codec_m5_t3.encode(message_bits)

        # Flip three bits in first block
        corrupted = encoded.clone().float()
        corrupted[0, 2] = 1.0 - corrupted[0, 2]
        corrupted[0, 8] = 1.0 - corrupted[0, 8]
        corrupted[0, 15] = 1.0 - corrupted[0, 15]

        decoded, errors = codec_m5_t3.decode(corrupted)

        torch.testing.assert_close(decoded, message_bits)
        assert errors[0] == 3


class TestSoftInputAndConfidence:
    @pytest.fixture
    def codec(self):
        return BCHWatermarkCodec(total_bits=62, message_bits=42, m=5, t=2)

    def _to_soft_values(
        self, hard_bits: torch.Tensor, low: float = 0.1, high: float = 0.9
    ) -> torch.Tensor:
        """Convert hard bits to soft values (0 -> low, 1 -> high)."""
        return torch.where(hard_bits == 1, high, low).float()

    def test_confidence_return_value(self, codec):
        """Confidence return value range and type."""
        message_bits = torch.randint(0, 2, size=(1, 42), dtype=torch.uint8)
        encoded = codec.encode(message_bits)

        soft_watermark = self._to_soft_values(encoded, low=0.1, high=0.9)
        _, _, confidence = codec.decode(soft_watermark, return_confidence=True)

        assert isinstance(confidence, list)
        assert len(confidence) == 1
        assert isinstance(confidence[0], float)
        assert 0.0 <= confidence[0] <= 1.0
        assert confidence[0] >= 0.7

    def test_confidence_batch(self, codec):
        """Confidence return value for batch inputs."""
        message_bits = torch.randint(0, 2, size=(4, 42), dtype=torch.uint8)
        encoded = codec.encode(message_bits)

        soft_watermark = self._to_soft_values(encoded, low=0.1, high=0.9)
        _, _, confidence = codec.decode(soft_watermark, return_confidence=True)

        assert len(confidence) == 4
        for conf in confidence:
            assert 0.0 <= conf <= 1.0


class TestDecodeSigmoidInputs:
    """Test decoding with sigmoid probability inputs [0, 1]."""

    @pytest.fixture
    def codec(self):
        return BCHWatermarkCodec(total_bits=62, message_bits=42, m=5, t=2)

    @pytest.fixture
    def message_bits(self):
        return torch.randint(0, 2, size=(1, 42), dtype=torch.uint8)

    def test_decode_hard_binary_values(self, codec, message_bits):
        """Decode with exact 0 and 1 values (sigmoid)."""
        encoded = codec.encode(message_bits)
        decoded, _ = codec.decode(encoded.float(), input_type="sigmoid")

        torch.testing.assert_close(decoded, message_bits)

    def test_decode_moderate_soft_values(self, codec, message_bits):
        """Decode with moderate soft values (0.2 and 0.8)."""
        encoded = codec.encode(message_bits)
        soft = torch.where(encoded == 1, 0.8, 0.2).float()
        decoded, _ = codec.decode(soft, input_type="sigmoid")

        torch.testing.assert_close(decoded, message_bits)


class TestDecodeLogitInputs:
    """Test decoding with raw logit inputs."""

    @pytest.fixture
    def codec(self):
        return BCHWatermarkCodec(total_bits=62, message_bits=42, m=5, t=2)

    @pytest.fixture
    def message_bits(self):
        return torch.randint(0, 2, size=(1, 42), dtype=torch.uint8)

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


class TestAutoInputTypeDetection:
    """Test automatic input type detection."""

    @pytest.fixture
    def codec(self):
        return BCHWatermarkCodec(total_bits=62, message_bits=42, m=5, t=2)

    @pytest.fixture
    def message_bits(self):
        return torch.randint(0, 2, size=(1, 42), dtype=torch.uint8)

    def test_auto_detect_sigmoid(self, codec, message_bits):
        """Auto-detect sigmoid inputs in [0, 1] range."""
        encoded = codec.encode(message_bits)
        soft = torch.where(encoded == 1, 0.9, 0.1).float()

        decoded, _ = codec.decode(soft, input_type="auto")
        torch.testing.assert_close(decoded, message_bits)

    def test_auto_detect_logits(self, codec, message_bits):
        """Auto-detect logits when values are outside [0, 1]."""
        encoded = codec.encode(message_bits)
        logits = torch.where(encoded == 1, 3.0, -3.0).float()

        decoded, _ = codec.decode(logits, input_type="auto")
        torch.testing.assert_close(decoded, message_bits)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def codec(self):
        return BCHWatermarkCodec(total_bits=62, message_bits=42, m=5, t=2)

    def test_all_zeros_message(self, codec):
        """Encode and decode all-zeros message."""
        message_bits = torch.zeros(1, 42, dtype=torch.uint8)
        encoded = codec.encode(message_bits)
        decoded, _ = codec.decode(encoded.float())

        torch.testing.assert_close(decoded, message_bits)

    def test_all_ones_message(self, codec):
        """Encode and decode all-ones message."""
        message_bits = torch.ones(1, 42, dtype=torch.uint8)
        encoded = codec.encode(message_bits)
        decoded, _ = codec.decode(encoded.float())

        torch.testing.assert_close(decoded, message_bits)

    def test_alternating_bits_message(self, codec):
        """Encode and decode alternating bits message."""
        message_bits = torch.tensor([[0, 1] * 21], dtype=torch.uint8)
        encoded = codec.encode(message_bits)
        decoded, _ = codec.decode(encoded.float())

        torch.testing.assert_close(decoded, message_bits)


class TestDifferentConfigurations:
    """Test different BCH configurations."""

    def test_bch_15_7_2_roundtrip(self):
        """Roundtrip with BCH(15, 7, 2)."""
        codec = BCHWatermarkCodec(total_bits=30, message_bits=14, m=4, t=2)
        message_bits = torch.randint(0, 2, size=(1, 14), dtype=torch.uint8)
        encoded = codec.encode(message_bits)
        decoded, _ = codec.decode(encoded.float())

        assert encoded.shape == (1, 30)
        torch.testing.assert_close(decoded, message_bits)

    def test_bch_31_21_2_roundtrip(self):
        """Roundtrip with BCH(31, 21, 2)."""
        codec = BCHWatermarkCodec(total_bits=62, message_bits=42, m=5, t=2)
        message_bits = torch.randint(0, 2, size=(1, 42), dtype=torch.uint8)
        encoded = codec.encode(message_bits)
        decoded, _ = codec.decode(encoded.float())

        assert encoded.shape == (1, 62)
        torch.testing.assert_close(decoded, message_bits)

    def test_bch_31_16_3_roundtrip(self):
        """Roundtrip with BCH(31, 16, 3)."""
        codec = BCHWatermarkCodec(total_bits=62, message_bits=32, m=5, t=3)
        message_bits = torch.randint(0, 2, size=(1, 32), dtype=torch.uint8)
        encoded = codec.encode(message_bits)
        decoded, _ = codec.decode(encoded.float())

        assert encoded.shape == (1, 62)
        torch.testing.assert_close(decoded, message_bits)

    def test_bch_63_51_2_roundtrip(self):
        """Roundtrip with BCH(63, 51, 2)."""
        codec = BCHWatermarkCodec(total_bits=126, message_bits=102, m=6, t=2)
        message_bits = torch.randint(0, 2, size=(1, 102), dtype=torch.uint8)
        encoded = codec.encode(message_bits)
        decoded, _ = codec.decode(encoded.float())

        assert encoded.shape == (1, 126)
        torch.testing.assert_close(decoded, message_bits)
