from src.utils import setup_seed
import pytest
from utils import get_videoseal, get_stega_stamp_videoseal, get_revmark_videoseal
from videoseal.models.videoseal import Videoseal
from adaptive_models.adaptive_videoseal import AdaptiveVideoseal
import os
from models.embedder import REVMarkEmbedder
from models.extractor import REVMarkExtractor
from REVMark import revmark_path

# Common paths for REVMark models
ME_SPYNET_PATH = os.path.join(os.getcwd(), "ME_Spynet_Full.pth")
REVMARK_ENCODER_PATH = os.path.join(revmark_path, "checkpoints/Encoder.pth")
REVMARK_DECODER_PATH = os.path.join(revmark_path, "checkpoints/Decoder.pth")


@pytest.fixture(scope="session", autouse=True)
def set_global_seed():
    """Automatically set the random seed before all tests."""
    seed = 42
    setup_seed(seed)
    print(f"\nRandom seed set globally to {seed}")


@pytest.fixture
def videoseal_model() -> Videoseal:
    return get_videoseal()


@pytest.fixture
def stega_stamp_videoseal_model() -> Videoseal:
    path = "onnx/stega_stamp.onnx"

    if not os.path.exists(path):
        pytest.skip(f"Skipping test: required ONNX file not found at {path!r}")

    return get_stega_stamp_videoseal(path)


@pytest.fixture
def revmark_videoseal_model() -> Videoseal:
    if not os.path.exists(REVMARK_ENCODER_PATH):
        pytest.skip(
            f"Skipping test: required .pth file not found at {REVMARK_ENCODER_PATH!r}"
        )
    if not os.path.exists(REVMARK_DECODER_PATH):
        pytest.skip(
            f"Skipping test: required .pth file not found at {REVMARK_DECODER_PATH!r}"
        )
    if not os.path.exists(ME_SPYNET_PATH):
        pytest.skip(
            f"Skipping test: required .pth file not found at {ME_SPYNET_PATH!r}"
        )

    return get_revmark_videoseal(
        encoder_path=REVMARK_ENCODER_PATH, decoder_path=REVMARK_DECODER_PATH
    )


@pytest.fixture
def make_adaptive_videoseal():
    """
    Factory fixture that converts any Videoseal-like model into an AdaptiveVideoseal.
    """

    def _make(model):
        return AdaptiveVideoseal.from_videoseal(model)

    return _make


@pytest.fixture
def default_revmark_embedder():
    if not os.path.exists(ME_SPYNET_PATH):
        pytest.skip(
            f"Skipping test: required ME Spynet file not found at {ME_SPYNET_PATH!r}"
        )
    if not os.path.exists(REVMARK_ENCODER_PATH):
        pytest.skip(
            f"Skipping test: required encoder file not found at {REVMARK_ENCODER_PATH!r}"
        )
    return REVMarkEmbedder(REVMARK_ENCODER_PATH)


@pytest.fixture
def default_revmark_extractor():
    if not os.path.exists(ME_SPYNET_PATH):
        pytest.skip(
            f"Skipping test: required ME Spynet file not found at {ME_SPYNET_PATH!r}"
        )
    if not os.path.exists(REVMARK_DECODER_PATH):
        pytest.skip(
            f"Skipping test: required decoder file not found at {REVMARK_DECODER_PATH!r}"
        )
    return REVMarkExtractor(REVMARK_DECODER_PATH)
