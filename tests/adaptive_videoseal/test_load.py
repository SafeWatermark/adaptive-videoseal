from src.utils import get_videoseal
import torch.nn as nn
import torch


def test_get_videoseal():
    """Test that the videoseal model loads and can perform a basic embed pass."""
    F, C, H, W = 2, 3, 224, 224

    # Load model
    model = get_videoseal()
    assert isinstance(model, nn.Module), "get_videoseal() did not return an nn.Module"

    # Prepare dummy image batch
    imgs = torch.randn(F, C, H, W).clamp_(0.0, 1.0)

    # Generate random message
    msgs = model.get_random_msg()

    # Forward pass
    outputs = model.embed(
        imgs,
        msgs=msgs,
        is_video=True,
        lowres_attenuation=True,
    )

    # Validate output
    assert isinstance(outputs, dict), f"Expected dict output, got {type(outputs)}"
