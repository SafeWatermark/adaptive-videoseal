from unittest.mock import MagicMock

from schemas.model_outputs import OutputValueType
from hooks.utils import infer_input_type_by_module


class TestInferInputTypeByModule:
    """Tests for infer_input_type_by_module function."""

    def test_stegastamp_extractor_returns_binary(self):
        """StegaStampExtractor should return BINARY output type."""
        from models.extractor import StegaStampExtractor

        mock_module = MagicMock(spec=StegaStampExtractor)
        mock_module.__class__ = StegaStampExtractor

        result = infer_input_type_by_module(mock_module)
        assert result == OutputValueType.BINARY

    def test_revmark_extractor_returns_sigmoid(self):
        """REVMarkExtractor should return SIGMOID output type."""
        from models.extractor import REVMarkExtractor

        mock_module = MagicMock(spec=REVMarkExtractor)
        mock_module.__class__ = REVMarkExtractor

        result = infer_input_type_by_module(mock_module)
        assert result == OutputValueType.SIGMOID

    def test_convnext_extractor_returns_logits(self):
        """ConvnextExtractor should return LOGITS output type."""
        from videoseal.models.extractor import ConvnextExtractor

        mock_module = MagicMock(spec=ConvnextExtractor)
        mock_module.__class__ = ConvnextExtractor

        result = infer_input_type_by_module(mock_module)
        assert result == OutputValueType.LOGITS

    def test_unknown_module_returns_auto(self):
        """Unknown module types should return 'auto'."""
        from videoseal.models.extractor import Extractor

        # Create a mock of base Extractor (not a known subclass)
        mock_module = MagicMock(spec=Extractor)
        mock_module.__class__ = Extractor

        result = infer_input_type_by_module(mock_module)
        assert result == "auto"

    def test_custom_extractor_returns_auto(self):
        """Custom extractor subclass not in mapping should return 'auto'."""

        class CustomExtractor:
            pass

        mock_module = MagicMock()
        mock_module.__class__ = CustomExtractor

        result = infer_input_type_by_module(mock_module)
        assert result == "auto"
