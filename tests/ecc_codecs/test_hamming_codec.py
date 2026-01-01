import torch
import pytest

from ecc_codecs.hamming_codec import HammingWatermarkCodec


class TestInitialization:
    def test_normal_initialization_r4(self):
        """Normal initialization with default parameters (r=4, Hamming(15,11))."""
        codec = HammingWatermarkCodec(total_bits=30, message_bits=22, r=4)

        assert codec.message_bits == 22
        assert codec.total_bits == 30
        assert codec.r == 4
        assert codec.n == 15
        assert codec.k == 11
        assert codec.num_blocks == 2
        assert codec.codeword_bits == 30
        assert codec.threshold == 0.5

    def test_initialization_r3(self):
        """Initialization with r=3 (Hamming(7,4))."""
        codec = HammingWatermarkCodec(total_bits=28, message_bits=16, r=3)

        assert codec.r == 3
        assert codec.n == 7
        assert codec.k == 4
        assert codec.num_blocks == 4
        assert codec.codeword_bits == 28

    def test_initialization_r5(self):
        """Initialization with r=5 (Hamming(31,26))."""
        codec = HammingWatermarkCodec(total_bits=62, message_bits=52, r=5)

        assert codec.r == 5
        assert codec.n == 31
        assert codec.k == 26
        assert codec.num_blocks == 2
        assert codec.codeword_bits == 62

    def test_initialization_r6(self):
        """Initialization with r=6 (Hamming(63,57))."""
        codec = HammingWatermarkCodec(total_bits=126, message_bits=114, r=6)

        assert codec.r == 6
        assert codec.n == 63
        assert codec.k == 57
        assert codec.num_blocks == 2
        assert codec.codeword_bits == 126

    def test_invalid_r_raises(self):
        """Invalid r should raise ValueError."""
        with pytest.raises(ValueError, match="r must be in"):
            HammingWatermarkCodec(total_bits=64, message_bits=32, r=2)

        with pytest.raises(ValueError, match="r must be in"):
            HammingWatermarkCodec(total_bits=64, message_bits=32, r=7)

    def test_message_bits_not_divisible_raises(self):
        """message_bits not divisible by k should raise ValueError."""
        # r=4 -> k=11, so message_bits must be divisible by 11
        with pytest.raises(ValueError, match="must be divisible by k=11"):
            HammingWatermarkCodec(total_bits=64, message_bits=20, r=4)

    def test_total_bits_too_small_raises(self):
        """total_bits too small should raise ValueError."""
        # r=4 -> n=15, k=11, so 22 message bits need 2 blocks = 30 codeword bits
        with pytest.raises(ValueError, match="total_bits=.*is too small"):
            HammingWatermarkCodec(total_bits=25, message_bits=22, r=4)

    def test_message_bits_too_small_raises(self):
        """message_bits smaller than k should raise ValueError."""
        # r=3 -> k=4, use message_bits=4 which equals k but satisfies divisibility
        # then use message_bits=3 which is < k and also not divisible
        with pytest.raises(ValueError, match="must be divisible by k=4"):
            HammingWatermarkCodec(total_bits=64, message_bits=3, r=3)

    def test_info_property(self):
        """Test info property returns correct values."""
        codec = HammingWatermarkCodec(total_bits=30, message_bits=22, r=4)
        info = codec.info

        assert info["message_bits"] == 22
        assert info["codeword_bits"] == 30
        assert info["total_bits"] == 30
        assert info["r"] == 4
        assert info["n"] == 15
        assert info["k"] == 11
        assert info["num_blocks"] == 2
        assert info["error_correction_capacity"] == 1
        assert "Hamming(15,11)" in info["codec"]


class TestEncode:
    @pytest.fixture
    def codec_r3(self):
        """Hamming(7,4) codec with 16 message bits."""
        return HammingWatermarkCodec(total_bits=28, message_bits=16, r=3)

    @pytest.fixture
    def codec_r4(self):
        """Hamming(15,11) codec with 22 message bits."""
        return HammingWatermarkCodec(total_bits=30, message_bits=22, r=4)

    def test_encode_output_length_and_format_r3(self, codec_r3):
        """Encode output length and format validation for r=3."""
        message_bits = torch.randint(0, 2, size=(1, 16), dtype=torch.uint8)
        result = codec_r3.encode(message_bits)

        assert result.shape == (1, 28)
        assert set(result.unique().tolist()).issubset({0, 1})

    def test_encode_output_length_and_format_r4(self, codec_r4):
        """Encode output length and format validation for r=4."""
        message_bits = torch.randint(0, 2, size=(1, 22), dtype=torch.uint8)
        result = codec_r4.encode(message_bits)

        assert result.shape == (1, 30)
        assert set(result.unique().tolist()).issubset({0, 1})

    def test_encode_batch(self, codec_r4):
        """Encode with batch input."""
        message_bits = torch.randint(0, 2, size=(4, 22), dtype=torch.uint8)
        result = codec_r4.encode(message_bits)

        assert result.shape == (4, 30)

    def test_encode_deterministic(self, codec_r4):
        """Same input should produce same output."""
        message_bits = torch.randint(0, 2, size=(1, 22), dtype=torch.uint8)
        result1 = codec_r4.encode(message_bits)
        result2 = codec_r4.encode(message_bits)

        torch.testing.assert_close(result1, result2)

    def test_encode_wrong_length_raises(self, codec_r4):
        """Encode with wrong input length should raise ValueError."""
        wrong_length = torch.zeros(1, 16, dtype=torch.uint8)
        with pytest.raises(ValueError, match="Expected 22 bits, got 16"):
            codec_r4.encode(wrong_length)

    def test_encode_non_binary_raises(self, codec_r4):
        """Encode with non-binary input should raise ValueError."""
        non_binary = torch.tensor([[0, 1, 2] + [0] * 19], dtype=torch.uint8)
        with pytest.raises(ValueError, match="must be binary"):
            codec_r4.encode(non_binary)

    def test_encode_with_padding(self):
        """Encode should pad to total_bits if needed."""
        codec = HammingWatermarkCodec(total_bits=40, message_bits=22, r=4)
        message_bits = torch.randint(0, 2, size=(1, 22), dtype=torch.uint8)
        result = codec.encode(message_bits)

        assert result.shape == (1, 40)
        # Last 10 bits should be padding zeros
        assert torch.all(result[:, 30:] == 0)


class TestDecode:
    @pytest.fixture
    def codec_r3(self):
        return HammingWatermarkCodec(total_bits=28, message_bits=16, r=3)

    @pytest.fixture
    def codec_r4(self):
        return HammingWatermarkCodec(total_bits=30, message_bits=22, r=4)

    def test_decode_no_error_roundtrip_r3(self, codec_r3):
        """No-error round-trip with Hamming(7,4)."""
        message_bits = torch.randint(0, 2, size=(1, 16), dtype=torch.uint8)
        encoded = codec_r3.encode(message_bits)
        decoded, errors = codec_r3.decode(encoded.float())

        torch.testing.assert_close(decoded, message_bits)
        assert errors[0] == 0

    def test_decode_no_error_roundtrip_r4(self, codec_r4):
        """No-error round-trip with Hamming(15,11)."""
        message_bits = torch.randint(0, 2, size=(1, 22), dtype=torch.uint8)
        encoded = codec_r4.encode(message_bits)
        decoded, errors = codec_r4.decode(encoded.float())

        torch.testing.assert_close(decoded, message_bits)
        assert errors[0] == 0

    def test_decode_batch_roundtrip(self, codec_r4):
        """Batch round-trip with random messages."""
        message_bits = torch.randint(0, 2, size=(4, 22), dtype=torch.uint8)
        encoded = codec_r4.encode(message_bits)
        decoded, bit_errors = codec_r4.decode(encoded.float())

        assert decoded.shape == (4, 22)
        torch.testing.assert_close(decoded, message_bits)
        assert len(bit_errors) == 4

    def test_decode_with_soft_values(self, codec_r4):
        """Decode with soft values (0.1 for 0, 0.9 for 1)."""
        message_bits = torch.randint(0, 2, size=(1, 22), dtype=torch.uint8)
        encoded = codec_r4.encode(message_bits)

        # Convert to soft values
        soft_watermark = torch.where(encoded == 1, 0.9, 0.1).float()
        decoded, _ = codec_r4.decode(soft_watermark)

        torch.testing.assert_close(decoded, message_bits)

    def test_decode_wrong_shape_raises(self, codec_r4):
        """Decode with wrong shape should raise ValueError."""
        wrong_shape = torch.zeros(1, 20, dtype=torch.float)
        with pytest.raises(ValueError, match="must be 2D with length >= 30"):
            codec_r4.decode(wrong_shape)

    def test_decode_1d_array_raises(self, codec_r4):
        """Decode with 1D array should raise ValueError."""
        wrong_dim = torch.zeros(30, dtype=torch.float)
        with pytest.raises(ValueError):
            codec_r4.decode(wrong_dim)


class TestSingleBitErrorCorrection:
    """Test Hamming code's single-bit error correction capability."""

    @pytest.fixture
    def codec_r3(self):
        return HammingWatermarkCodec(total_bits=28, message_bits=16, r=3)

    @pytest.fixture
    def codec_r4(self):
        return HammingWatermarkCodec(total_bits=30, message_bits=22, r=4)

    def test_correct_single_bit_error_per_block_r3(self, codec_r3):
        """Hamming(7,4) should correct 1-bit error per block."""
        message_bits = torch.randint(0, 2, size=(1, 16), dtype=torch.uint8)
        encoded = codec_r3.encode(message_bits)

        # Flip one bit in each block (4 blocks of 7 bits each)
        corrupted = encoded.clone().float()
        for block_idx in range(4):
            error_pos = block_idx * 7 + (
                block_idx % 7
            )  # Different position in each block
            corrupted[0, error_pos] = 1.0 - corrupted[0, error_pos]

        decoded, errors = codec_r3.decode(corrupted)

        torch.testing.assert_close(decoded, message_bits)
        assert errors[0] == 4  # 4 errors corrected (one per block)

    def test_correct_single_bit_error_per_block_r4(self, codec_r4):
        """Hamming(15,11) should correct 1-bit error per block."""
        message_bits = torch.randint(0, 2, size=(1, 22), dtype=torch.uint8)
        encoded = codec_r4.encode(message_bits)

        # Flip one bit in each block (2 blocks of 15 bits each)
        corrupted = encoded.clone().float()
        corrupted[0, 3] = 1.0 - corrupted[0, 3]  # Error in first block
        corrupted[0, 18] = 1.0 - corrupted[0, 18]  # Error in second block

        decoded, errors = codec_r4.decode(corrupted)

        torch.testing.assert_close(decoded, message_bits)
        assert errors[0] == 2  # 2 errors corrected

    def test_error_at_different_positions(self, codec_r4):
        """Test error correction at various positions within a block."""
        message_bits = torch.randint(0, 2, size=(1, 22), dtype=torch.uint8)
        encoded = codec_r4.encode(message_bits)

        # Test error at each position in first block
        for pos in range(15):
            corrupted = encoded.clone().float()
            corrupted[0, pos] = 1.0 - corrupted[0, pos]

            decoded, errors = codec_r4.decode(corrupted)
            torch.testing.assert_close(
                decoded, message_bits, msg=f"Failed at position {pos}"
            )
            assert errors[0] == 1


class TestSoftInputAndConfidence:
    @pytest.fixture
    def codec(self):
        return HammingWatermarkCodec(total_bits=30, message_bits=22, r=4)

    def _to_soft_values(
        self, hard_bits: torch.Tensor, low: float = 0.1, high: float = 0.9
    ) -> torch.Tensor:
        """Convert hard bits to soft values (0 -> low, 1 -> high)."""
        return torch.where(hard_bits == 1, high, low).float()

    def test_confidence_return_value(self, codec):
        """Confidence return value range and type."""
        message_bits = torch.randint(0, 2, size=(1, 22), dtype=torch.uint8)
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
        message_bits = torch.randint(0, 2, size=(4, 22), dtype=torch.uint8)
        encoded = codec.encode(message_bits)

        soft_watermark = self._to_soft_values(encoded, low=0.1, high=0.9)
        _, _, confidence = codec.decode(soft_watermark, return_confidence=True)

        assert isinstance(confidence, list)
        assert len(confidence) == 4
        for conf in confidence:
            assert 0.0 <= conf <= 1.0

    def test_low_confidence_values(self, codec):
        """Values near threshold produce low confidence."""
        message_bits = torch.randint(0, 2, size=(1, 22), dtype=torch.uint8)
        encoded = codec.encode(message_bits)

        # Low confidence soft values (close to threshold 0.5)
        soft_watermark = self._to_soft_values(encoded, low=0.45, high=0.55)
        _, _, confidence = codec.decode(soft_watermark, return_confidence=True)

        assert confidence[0] < 0.2, f"Expected low confidence, got {confidence[0]}"


class TestDecodeSigmoidInputs:
    """Test decoding with sigmoid probability inputs [0, 1]."""

    @pytest.fixture
    def codec(self):
        return HammingWatermarkCodec(total_bits=30, message_bits=22, r=4)

    @pytest.fixture
    def message_bits(self):
        return torch.randint(0, 2, size=(1, 22), dtype=torch.uint8)

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


class TestDecodeLogitInputs:
    """Test decoding with raw logit inputs (unbounded real values)."""

    @pytest.fixture
    def codec(self):
        return HammingWatermarkCodec(total_bits=30, message_bits=22, r=4)

    @pytest.fixture
    def message_bits(self):
        return torch.randint(0, 2, size=(1, 22), dtype=torch.uint8)

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
        return HammingWatermarkCodec(total_bits=30, message_bits=22, r=4)

    @pytest.fixture
    def message_bits(self):
        return torch.randint(0, 2, size=(1, 22), dtype=torch.uint8)

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
        logits = torch.where(encoded == 1, 2.0, -2.0).float()
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
        return HammingWatermarkCodec(total_bits=30, message_bits=22, r=4)

    def test_decode_with_longer_input(self, codec):
        """Decode accepts input longer than codeword_bits (extra bits ignored)."""
        message_bits = torch.randint(0, 2, size=(1, 22), dtype=torch.uint8)
        encoded = codec.encode(message_bits)
        # Pad with extra values
        extended = torch.cat([encoded.float(), torch.zeros(1, 20)], dim=1)
        decoded, _ = codec.decode(extended)

        torch.testing.assert_close(decoded, message_bits)

    def test_all_zeros_message(self, codec):
        """Encode and decode all-zeros message."""
        message_bits = torch.zeros(1, 22, dtype=torch.uint8)
        encoded = codec.encode(message_bits)
        decoded, _ = codec.decode(encoded.float())

        torch.testing.assert_close(decoded, message_bits)

    def test_all_ones_message(self, codec):
        """Encode and decode all-ones message."""
        message_bits = torch.ones(1, 22, dtype=torch.uint8)
        encoded = codec.encode(message_bits)
        decoded, _ = codec.decode(encoded.float())

        torch.testing.assert_close(decoded, message_bits)

    def test_alternating_bits_message(self, codec):
        """Encode and decode alternating bits message."""
        message_bits = torch.tensor([[0, 1] * 11], dtype=torch.uint8)
        encoded = codec.encode(message_bits)
        decoded, _ = codec.decode(encoded.float())

        torch.testing.assert_close(decoded, message_bits)


class TestDifferentConfigurations:
    """Test different Hamming configurations."""

    @pytest.fixture
    def message_bits_16(self):
        return torch.randint(0, 2, size=(1, 16), dtype=torch.uint8)

    @pytest.fixture
    def message_bits_22(self):
        return torch.randint(0, 2, size=(1, 22), dtype=torch.uint8)

    @pytest.fixture
    def message_bits_52(self):
        return torch.randint(0, 2, size=(1, 52), dtype=torch.uint8)

    def test_hamming_7_4_roundtrip(self, message_bits_16):
        """Roundtrip with Hamming(7,4)."""
        codec = HammingWatermarkCodec(total_bits=28, message_bits=16, r=3)
        encoded = codec.encode(message_bits_16)
        decoded, _ = codec.decode(encoded.float())

        assert encoded.shape == (1, 28)
        torch.testing.assert_close(decoded, message_bits_16)

    def test_hamming_15_11_roundtrip(self, message_bits_22):
        """Roundtrip with Hamming(15,11)."""
        codec = HammingWatermarkCodec(total_bits=30, message_bits=22, r=4)
        encoded = codec.encode(message_bits_22)
        decoded, _ = codec.decode(encoded.float())

        assert encoded.shape == (1, 30)
        torch.testing.assert_close(decoded, message_bits_22)

    def test_hamming_31_26_roundtrip(self, message_bits_52):
        """Roundtrip with Hamming(31,26)."""
        codec = HammingWatermarkCodec(total_bits=62, message_bits=52, r=5)
        encoded = codec.encode(message_bits_52)
        decoded, _ = codec.decode(encoded.float())

        assert encoded.shape == (1, 62)
        torch.testing.assert_close(decoded, message_bits_52)


class TestMatrixProperties:
    """Test mathematical properties of Hamming matrices."""

    def test_generator_matrix_shape(self):
        """Generator matrix G should have shape (k, n)."""
        codec = HammingWatermarkCodec(total_bits=30, message_bits=22, r=4)
        assert codec.G.shape == (11, 15)  # (k, n) for Hamming(15,11)

    def test_parity_check_matrix_shape(self):
        """Parity-check matrix H should have shape (r, n)."""
        codec = HammingWatermarkCodec(total_bits=30, message_bits=22, r=4)
        assert codec.H.shape == (4, 15)  # (r, n) for Hamming(15,11)

    def test_syndrome_table_size(self):
        """Syndrome table should have 2^r entries."""
        codec = HammingWatermarkCodec(total_bits=30, message_bits=22, r=4)
        assert codec.syndrome_table.shape == (16,)  # 2^4 = 16

    def test_data_positions_count(self):
        """Data positions should have k entries."""
        codec = HammingWatermarkCodec(total_bits=30, message_bits=22, r=4)
        assert len(codec.data_positions) == 11  # k = 11

    def test_parity_positions_count(self):
        """Parity positions should have r entries."""
        codec = HammingWatermarkCodec(total_bits=30, message_bits=22, r=4)
        assert len(codec.parity_positions) == 4  # r = 4

    def test_hg_transpose_is_zero(self):
        """H @ G^T should be zero matrix (mod 2) for valid Hamming code."""
        codec = HammingWatermarkCodec(total_bits=30, message_bits=22, r=4)
        H = codec.H.float()
        G = codec.G.float()

        # H @ G^T mod 2 should be zero
        result = torch.matmul(H, G.T) % 2
        assert torch.all(result == 0), "H @ G^T should be zero matrix mod 2"
