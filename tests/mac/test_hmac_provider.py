import pytest
import torch
from pydantic_settings import BaseSettings

from mac.hmac_provider import HMACProvider


class DummyConfig(BaseSettings):
    MAC_SECRET_KEY: bytes = b"0123456789abcdef"  # 16 bytes = 128 bits
    PASSWORD_BYTES: bytes = b"test_password"


class TestConstructorValidation:
    def test_tag_bits_zero_raises(self):
        """tag_bits <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="tag_bits must be positive"):
            HMACProvider(message_bits=8, tag_bits=0, config=DummyConfig())

    def test_tag_bits_negative_raises(self):
        """Negative tag_bits should raise ValueError."""
        with pytest.raises(ValueError, match="tag_bits must be positive"):
            HMACProvider(message_bits=8, tag_bits=-1, config=DummyConfig())

    def test_tag_bits_exceeds_key_size_raises(self):
        """tag_bits > secret_key_bits should raise ValueError."""
        # DummyConfig has 16-byte key = 128 bits
        with pytest.raises(ValueError, match="tag_bits must be less than or equal"):
            HMACProvider(message_bits=8, tag_bits=129, config=DummyConfig())


class TestTruncateBitsFromDigest:
    @pytest.fixture
    def provider(self):
        return HMACProvider(message_bits=8, tag_bits=13, config=DummyConfig())

    def test_non_multiple_of_8_tag_bits(self, provider):
        """Returns correct length for non-multiple-of-8 tag_bits."""
        digest = b"\xff" * 32  # 256 bits
        result = provider.truncate_bits_from_digest(digest, tag_bits=9)

        assert result.shape == (9,)

    def test_another_non_multiple_of_8(self, provider):
        """Returns correct length for tag_bits=13."""
        digest = b"\xff" * 32
        result = provider.truncate_bits_from_digest(digest, tag_bits=13)

        assert result.shape == (13,)

    def test_tag_bits_exceeds_digest_raises(self, provider):
        """tag_bits > digest size in bits should raise ValueError."""
        digest = b"\xff" * 4  # 32 bits
        with pytest.raises(ValueError, match="tag_bits exceeds digest size"):
            provider.truncate_bits_from_digest(digest, tag_bits=33)


class TestBuildAuthenticatedPayloadBits:
    @pytest.fixture
    def provider(self):
        return HMACProvider(message_bits=8, tag_bits=13, config=DummyConfig())

    def test_message_length_mismatch_raises(self, provider):
        """Raises ValueError if message length doesn't match."""
        wrong_length = torch.zeros(2, 10, dtype=torch.uint8)
        with pytest.raises(ValueError, match="Expected 8 bits, got 10"):
            provider.build_authenticated_payload_bits(wrong_length)

    def test_output_shape(self, provider):
        """Returns correct shape [B, message_bits + tag_bits]."""
        message = torch.zeros(2, 8, dtype=torch.uint8)
        result = provider.build_authenticated_payload_bits(message)

        assert result.shape == (2, 8 + 13)

    def test_appended_tag_equals_provider_tag(self, provider):
        """Appended tag matches provider's computed tag."""
        message = torch.tensor(
            [[1, 0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 1]], dtype=torch.uint8
        )
        result = provider.build_authenticated_payload_bits(message)

        appended_tag = result[:, 8:]
        expected_tag = provider.hmac_tag.expand(2, -1)

        assert torch.equal(appended_tag, expected_tag)


class TestSplitPayloadBits:
    @pytest.fixture
    def provider(self):
        return HMACProvider(message_bits=8, tag_bits=13, config=DummyConfig())

    def test_split_shapes(self, provider):
        """Splits into correct shapes."""
        payload = torch.zeros(2, 21, dtype=torch.uint8)
        message, tag = provider.split_payload_bits(payload)

        assert message.shape == (2, 8)
        assert tag.shape == (2, 13)

    def test_split_values(self, provider):
        """Splits correctly preserving values."""
        message_part = torch.tensor(
            [[1, 0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 1]], dtype=torch.uint8
        )
        tag_part = torch.ones(2, 13, dtype=torch.uint8)
        payload = torch.cat([message_part, tag_part], dim=1)

        message, tag = provider.split_payload_bits(payload)

        assert torch.equal(message, message_part)
        assert torch.equal(tag, tag_part)


class TestVerifyTag:
    @pytest.fixture
    def provider(self):
        return HMACProvider(message_bits=8, tag_bits=13, config=DummyConfig())

    def test_valid_payload_returns_true(self, provider):
        """Returns all True for payloads created by build_authenticated_payload_bits."""
        message = torch.tensor(
            [[1, 0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 1]], dtype=torch.uint8
        )
        payload = provider.build_authenticated_payload_bits(message)

        result = provider.verify_tag(payload)

        assert result.dtype == torch.bool
        assert result.shape == (2,)
        assert torch.all(result)

    def test_flipped_tag_bit_returns_false(self, provider):
        """Returns False if one bit of tag is flipped."""
        message = torch.tensor([[1, 0, 1, 0, 1, 0, 1, 0]], dtype=torch.uint8)
        payload = provider.build_authenticated_payload_bits(message)

        # Flip first bit of tag (position 8)
        payload[0, 8] = 1 - payload[0, 8]

        result = provider.verify_tag(payload)

        assert not result[0]


class TestComputeTagMatchRatio:
    @pytest.fixture
    def provider(self):
        return HMACProvider(message_bits=8, tag_bits=16, config=DummyConfig())

    def test_perfect_match_returns_one(self, provider):
        """Returns 1.0 when all tag bits match."""
        message = torch.tensor(
            [[1, 0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 1]], dtype=torch.float32
        )
        payload = provider.build_authenticated_payload_bits(message)

        result = provider.compute_tag_match_ratio(payload.float())

        assert result.shape == (2,)
        assert torch.allclose(result, torch.ones(2))

    def test_all_bits_flipped_returns_zero(self, provider):
        """Returns 0.0 when all tag bits are flipped."""
        message = torch.tensor([[1, 0, 1, 0, 1, 0, 1, 0]], dtype=torch.float32)
        payload = provider.build_authenticated_payload_bits(message)

        # Flip all tag bits (positions 8 to 23)
        payload[:, 8:24] = 1 - payload[:, 8:24]

        result = provider.compute_tag_match_ratio(payload.float())

        assert torch.allclose(result, torch.zeros(1))

    def test_partial_match_returns_correct_ratio(self, provider):
        """Returns correct ratio for partial bit matches."""
        message = torch.tensor([[1, 0, 1, 0, 1, 0, 1, 0]], dtype=torch.float32)
        payload = provider.build_authenticated_payload_bits(message)

        # Flip 4 out of 16 tag bits → ratio = 12/16 = 0.75
        payload[:, 8:12] = 1 - payload[:, 8:12]

        result = provider.compute_tag_match_ratio(payload.float())

        expected = torch.tensor([0.75])
        assert torch.allclose(result, expected)

    def test_batch_processing(self, provider):
        """Works correctly with batched inputs having different match ratios."""
        message = torch.zeros(3, 8, dtype=torch.float32)
        payload = provider.build_authenticated_payload_bits(message)

        # Batch 0: perfect match (16/16)
        # Batch 1: flip 4 bits (12/16 = 0.75)
        payload[1, 8:12] = 1 - payload[1, 8:12]
        # Batch 2: flip 8 bits (8/16 = 0.5)
        payload[2, 8:16] = 1 - payload[2, 8:16]

        result = provider.compute_tag_match_ratio(payload.float())

        expected = torch.tensor([1.0, 0.75, 0.5])
        assert torch.allclose(result, expected)

    def test_single_bit_flip(self, provider):
        """Returns correct ratio when single bit is flipped."""
        message = torch.tensor([[1, 0, 1, 0, 1, 0, 1, 0]], dtype=torch.float32)
        payload = provider.build_authenticated_payload_bits(message)

        # Flip 1 out of 16 tag bits → ratio = 15/16 = 0.9375
        payload[0, 8] = 1 - payload[0, 8]

        result = provider.compute_tag_match_ratio(payload.float())

        expected = torch.tensor([15.0 / 16.0])
        assert torch.allclose(result, expected)
