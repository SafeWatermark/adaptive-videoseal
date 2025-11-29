from src.utils import get_videoseal
from src.adaptive_models.adaptive_videoseal import AdaptiveVideoseal
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


def test_from_videoseal():
    """
    Test that AdaptiveVideoseal.from_videoseal(model) correctly clones all weights
    and internal buffers from the original Videoseal model.

    This test performs:
        1. Detection of mismatched or missing keys in the state dict.
        2. Detailed tensor-by-tensor comparison with informative error messages.
        3. Reporting of any extra keys in the adaptive model.

    The goal is to ensure that converting a Videoseal instance into an
    AdaptiveVideoseal via `from_videoseal()` preserves *all* learned parameters
    exactly, making AdaptiveVideoseal a drop-in replacement.
    """
    model = get_videoseal()
    adaptive = AdaptiveVideoseal.from_videoseal(model)
    # Detect extra keys
    extra_keys = set(model.state_dict()) - set(adaptive.state_dict())
    if extra_keys:
        print(f"[WARNING] Keys in parent model but not in adaptive model: {extra_keys}")
    missing_keys = set(adaptive.state_dict()) - set(model.state_dict())
    if missing_keys:
        print(f"[WARNING] Extra keys in adaptive model: {missing_keys}")

    # Compare tensors
    for k, v in model.state_dict().items():
        a = adaptive.state_dict()[k]

        if not torch.allclose(a, v, atol=1e-6, rtol=1e-6):
            diff = (a - v).abs().max().item()
            raise AssertionError(
                f"[MISMATCH] Key '{k}' does not match.\n"
                f"  • Parent shape:   {tuple(v.shape)}\n"
                f"  • Child shape:    {tuple(a.shape)}\n"
                f"  • Parent dtype:   {v.dtype}\n"
                f"  • Child dtype:    {a.dtype}\n"
                f"  • Max diff:       {diff}\n"
                f"  • Mean diff:      {(a - v).abs().mean().item()}\n"
            )


def test_adaptive_videoseal_embed_matches_parent():
    """
    Ensure that AdaptiveVideoseal produces the same watermarking output as the
    original Videoseal model when no adaptive logic is applied.

    This test compares:
        • imgs_w (watermarked images)
        • msgs   (embedded/extracted messages)

    If the adaptive model diverges from the parent model, the assertions will
    show detailed mismatch information to make debugging easier.
    """

    # Load parent & converted model
    model = get_videoseal()
    adaptive = AdaptiveVideoseal.from_videoseal(model)

    # Prepare random image + message(intentionally F = 1 to deactivate scene detection)
    F, C, H, W = 1, 3, 224, 224
    imgs = torch.randn(F, C, H, W).clamp_(0.0, 1.0)
    msgs = model.get_random_msg()

    # Forward pass
    outputs = model.embed(
        imgs,
        msgs=msgs,
        is_video=True,
        lowres_attenuation=True,
    )
    adaptive_outputs = adaptive.embed(
        imgs,
        msgs=msgs,
        is_video=True,
        lowres_attenuation=True,
    )

    # imgs_w comparison
    diff = (outputs["imgs_w"] - adaptive_outputs["imgs_w"]).abs()
    assert torch.allclose(
        outputs["imgs_w"], adaptive_outputs["imgs_w"], atol=1e-6, rtol=1e-6
    ), (
        "[ERROR] imgs_w mismatch between Videoseal and AdaptiveVideoseal.\n"
        f"  • Max diff:       {diff.max().item()}\n"
    )

    # msgs comparison
    diff = (outputs["msgs"] - adaptive_outputs["msgs"]).abs()
    assert torch.allclose(
        outputs["msgs"], adaptive_outputs["msgs"], atol=1e-6, rtol=1e-6
    ), (
        "[ERROR] msgs mismatch between Videoseal and AdaptiveVideoseal.\n"
        f"  • Max diff:       {diff.max().item()}\n"
    )
