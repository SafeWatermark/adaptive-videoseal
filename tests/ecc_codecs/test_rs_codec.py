import pytest
import torch

from ecc_codecs.rs_codec import RSWatermarkCodec


class TestInitialization:
    def test_normal_initialization(self):
        """Normal initialization with representative configuration."""
        codec = RSWatermarkCodec(total_bits=100, message_bits=64, threshold=0.5)

        assert codec.message_bytes == 8
        assert codec.parity_bytes == (100 // 8) - 8  # 12 - 8 = 4
        assert codec.error_correction_capacity == 2
        assert codec.rs_bits == 64 + 4 * 8  # 96
        assert codec.reserved_bits == 100 - 96  # 4
        assert codec.threshold == 0.5

    def test_message_bits_not_divisible_by_8_raises(self):
        """message_bits not divisible by 8 should raise ValueError."""
        with pytest.raises(ValueError, match="message_bits must be divisible by 8"):
            RSWatermarkCodec(total_bits=100, message_bits=63)

    def test_insufficient_parity_bytes_raises(self):
        """Insufficient parity bytes (< 2) should raise ValueError."""
        # total_bits=100 -> 12 bytes, message_bits=88 -> 11 bytes -> parity=1
        with pytest.raises(ValueError, match="Not enough parity bytes"):
            RSWatermarkCodec(total_bits=100, message_bits=88)

    def test_message_bits_zero_raises(self):
        """message_bits=0 should raise ValueError (divisible by 8 check)."""
        with pytest.raises(ValueError, match="message_bits must be divisible by 8"):
            RSWatermarkCodec(total_bits=100, message_bits=4)

    def test_various_valid_configurations(self):
        """Test various valid configurations."""
        configs = [
            (100, 16, 0.5),  # 2 msg bytes, 10 parity bytes, capacity=5
            (100, 32, 0.5),  # 4 msg bytes, 8 parity bytes, capacity=4
            (100, 48, 0.5),  # 6 msg bytes, 6 parity bytes, capacity=3
            (96, 64, 0.0),  # 8 msg bytes, 4 parity bytes, capacity=2
        ]
        for total, msg, thresh in configs:
            codec = RSWatermarkCodec(
                total_bits=total, message_bits=msg, threshold=thresh
            )
            assert codec.message_bits == msg
            assert codec.threshold == thresh


class TestEncode:
    @pytest.fixture
    def codec(self):
        return RSWatermarkCodec(total_bits=100, message_bits=64, threshold=0.5)

    def test_encode_with_random_message(self, codec):
        """encode output length and format validation."""
        message_bits = torch.randint(0, 2, (64,), dtype=torch.uint8)
        result = codec.encode(message_bits)

        assert len(result) == 100
        assert result.dtype == torch.uint8
        assert set(torch.unique(result).tolist()).issubset({0, 1})

        torch.testing.assert_close(
            result[codec.rs_bits :], torch.zeros(codec.reserved_bits, dtype=torch.uint8)
        )

    def test_encode_wrong_length_raises(self, codec):
        """TC-B2: encode with wrong input length should raise ValueError."""
        wrong_length = torch.zeros(32, dtype=torch.uint8)
        with pytest.raises(ValueError, match="Expected 64 bits, got 32"):
            codec.encode(wrong_length)


class TestDecode:
    @pytest.fixture
    def codec(self):
        return RSWatermarkCodec(total_bits=100, message_bits=64, threshold=0.5)

    def _corrupt_bytes(self, encoded: torch.Tensor, n_bytes: int) -> torch.Tensor:
        """Corrupt n_bytes worth of bits in the RS region (byte-level corruption)."""
        corrupted = encoded.clone()

        # RS region is the first rs_bits
        rs_bytes = len(encoded) // 8
        byte_indices = torch.randperm(rs_bytes)[: min(n_bytes, rs_bytes)]

        for byte_idx in byte_indices:
            # Flip all 8 bits in this byte
            start = byte_idx * 8
            for i in range(8):
                if start + i < len(corrupted):
                    corrupted[start + i] = 1 - corrupted[start + i]

        return corrupted

    def test_decode_no_error_roundtrip_random(self, codec):
        """TC-C1: No-error round-trip with random message."""
        message_bits = torch.randint(0, 2, (64,), dtype=torch.uint8)
        encoded = codec.encode(message_bits)
        decoded, errors_corrected = codec.decode(encoded.float())

        torch.testing.assert_close(decoded, message_bits)
        assert errors_corrected == 0

    def test_decode_correctable_errors(self, codec):
        """Errors within correction capacity should be corrected."""
        message_bits = torch.randint(0, 2, (64,), dtype=torch.uint8)
        encoded = codec.encode(message_bits)

        # Corrupt up to capacity bytes (capacity = 2 for this config)
        for n_errors in range(1, codec.error_correction_capacity + 1):
            corrupted = self._corrupt_bytes(encoded, n_errors)
            decoded, errors_corrected = codec.decode(corrupted.float())

            torch.testing.assert_close(
                decoded, message_bits, msg=f"Failed with {n_errors} byte errors"
            )
            assert (
                errors_corrected >= 0
            ), f"Expected successful correction with {n_errors} errors"

    def test_decode_uncorrectable_errors(self, codec):
        """Errors exceeding correction capacity should fail."""
        message_bits = torch.randint(0, 2, (64,), dtype=torch.uint8)
        encoded = codec.encode(message_bits)

        # Corrupt more than capacity bytes
        n_errors = codec.error_correction_capacity + 1
        corrupted = self._corrupt_bytes(encoded, n_errors)
        _, errors_corrected = codec.decode(corrupted.float())

        assert errors_corrected == -1, "Expected correction failure"

    def test_decode_high_capacity_codec(self):
        """Test decode with higher error correction capacity."""
        # 16-bit message = 2 bytes, 10 parity bytes, capacity = 5
        codec = RSWatermarkCodec(total_bits=100, message_bits=16, threshold=0.5)
        message_bits = torch.randint(0, 2, (16,), dtype=torch.uint8)
        encoded = codec.encode(message_bits)

        # Should correct up to 5 byte errors
        for n_errors in [1, 3, 5]:
            corrupted = self._corrupt_bytes(encoded, n_errors)
            decoded, errors_corrected = codec.decode(corrupted.float())
            torch.testing.assert_close(decoded, message_bits)
            assert errors_corrected >= 0


class TestSoftInputAndConfidence:
    @pytest.fixture
    def codec(self):
        return RSWatermarkCodec(total_bits=100, message_bits=64, threshold=0.5)

    def _to_soft_values(
        self, hard_bits: torch.Tensor, low: float = 0.1, high: float = 0.9
    ) -> torch.Tensor:
        """Convert hard bits to soft values (0 -> low, 1 -> high)."""
        return torch.where(hard_bits == 1, high, low).float()

    def test_soft_input_hard_decision(self, codec):
        """TC-D1: Soft input with clear hard decision values."""
        message_bits = torch.randint(0, 2, (64,), dtype=torch.uint8)
        encoded = codec.encode(message_bits)

        # Convert to soft values (0 -> 0.1, 1 -> 0.9)
        soft_watermark = self._to_soft_values(encoded)

        decoded, errors = codec.decode(soft_watermark)

        torch.testing.assert_close(decoded, message_bits)
        assert errors == 0

    def test_confidence_return_and_range(self, codec):
        """Confidence return value range and consistency."""
        message_bits = torch.randint(0, 2, (64,), dtype=torch.uint8)
        encoded = codec.encode(message_bits)

        # High confidence soft values (far from threshold)
        soft_watermark = self._to_soft_values(encoded, low=0.1, high=0.9)
        _, _, confidence = codec.decode(soft_watermark, return_confidence=True)

        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
        # With 0.1/0.9 values, confidence should be high (|0.1-0.5|*2=0.8, |0.9-0.5|*2=0.8)
        assert confidence >= 0.7, f"Expected high confidence, got {confidence}"

    def test_low_confidence_values(self, codec):
        """Test that values near threshold produce low confidence."""
        message_bits = torch.randint(0, 2, (64,), dtype=torch.uint8)
        encoded = codec.encode(message_bits)

        # Low confidence soft values (close to threshold)
        soft_watermark = self._to_soft_values(encoded, low=0.45, high=0.55)
        decoded, _, confidence = codec.decode(soft_watermark, return_confidence=True)

        assert decoded is not None
        assert confidence < 0.2, f"Expected low confidence, got {confidence}"

    def test_threshold_boundary_values(self):
        """Test threshold at boundary values."""
        threshold = 0.5
        codec = RSWatermarkCodec(total_bits=100, message_bits=64, threshold=threshold)

        message_bits = torch.randint(0, 2, (64,), dtype=torch.uint8)
        encoded = codec.encode(message_bits)

        # Values exactly at threshold edge
        soft_watermark = self._to_soft_values(
            encoded, low=threshold - 0.01, high=threshold + 0.01
        )
        decoded, errors_corrected = codec.decode(soft_watermark)

        # Should still decode correctly (0.49 < 0.5 -> 0, 0.51 > 0.5 -> 1)
        torch.testing.assert_close(decoded, message_bits)
        assert errors_corrected == 0


class TestBitsToBytes:
    """Test internal bit/byte conversion methods."""

    @pytest.fixture
    def codec(self):
        return RSWatermarkCodec(total_bits=100, message_bits=64, threshold=0.5)

    def test_bits_to_bytes_roundtrip(self, codec):
        """Test _bits_to_bytes and _bytes_to_bits are inverses."""
        original_bits = torch.tensor(
            [1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.uint8
        )
        bytes_result = codec._bits_to_bytes(original_bits)
        bits_result = codec._bytes_to_bits(bytes_result)

        torch.testing.assert_close(bits_result, original_bits)

    def test_bits_to_bytes_msb_first(self, codec):
        """Verify MSB-first byte ordering."""
        # 10000000 = 128, 00000001 = 1
        bits = torch.tensor(
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=torch.uint8
        )
        bytes_result = codec._bits_to_bytes(bits)

        assert bytes_result == bytes([128, 1])
