from src.utils import setup_seed
import pytest


@pytest.fixture(scope="session", autouse=True)
def set_global_seed():
    """Automatically set the random seed before all tests."""
    seed = 42
    setup_seed(seed)
    print(f"\nRandom seed set globally to {seed}")
