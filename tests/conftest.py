from src.utils import setup_seed
import pytest
from utils import get_videoseal, get_stega_stamp_videoseal
from videoseal.models.videoseal import Videoseal
from adaptive_models.adaptive_videoseal import AdaptiveVideoseal
import os


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
def make_adaptive_videoseal():
    """
    Factory fixture that converts any Videoseal-like model into an AdaptiveVideoseal.
    """

    def _make(model):
        return AdaptiveVideoseal.from_videoseal(model)

    return _make
