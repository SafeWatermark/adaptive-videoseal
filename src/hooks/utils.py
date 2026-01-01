from videoseal.models.extractor import Extractor, ConvnextExtractor
from models.extractor import StegaStampExtractor, REVMarkExtractor
from schemas.model_outputs import MODEL_TO_VALUE_TYPE, ModelType, OutputValueType


def infer_input_type_by_module(module: Extractor) -> OutputValueType | str:
    if isinstance(module, StegaStampExtractor):
        return MODEL_TO_VALUE_TYPE[ModelType.STEGASTAMP]
    elif isinstance(module, REVMarkExtractor):
        return MODEL_TO_VALUE_TYPE[ModelType.REVMARK]
    elif isinstance(module, ConvnextExtractor):  # Videoseal V1.0 extractor
        return MODEL_TO_VALUE_TYPE[ModelType.VIDEOSEAL]
    return "auto"
