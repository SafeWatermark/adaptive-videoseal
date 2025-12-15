import torch
from videoseal.models.videoseal import Videoseal
import pytest
from models.embedder import StegaStampEmbedder
from models.extractor import StegaStampExtractor
from models.stega_stamp import (
    STEGA_STAMP_ATTENUATION,
    STEGA_STAMP_IMAGE_WIDTH,
    STEGA_STAMP_SCALING_I,
    STEGA_STAMP_SCALING_W,
)
import logging


@pytest.mark.parametrize(
    "model_name",
    ["videoseal_model", "stega_stamp_videoseal_model", "revmark_videoseal_model"],
)
def test_get_videoseal(request, model_name: str):
    """Test that the videoseal model loads and can perform a basic embed pass."""
    F, C, H, W = 2, 3, 224, 224

    # Load model
    # convert the model_name(str) into the actual fixture by calling
    model = request.getfixturevalue(model_name)
    assert isinstance(
        model, Videoseal
    ), f"getter({model_name}) did not return an Videoseal, got {type(model).__name__}"

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


@pytest.mark.parametrize(
    "model_name",
    ["videoseal_model", "stega_stamp_videoseal_model", "revmark_videoseal_model"],
)
def test_from_videoseal(request, model_name: str, make_adaptive_videoseal):
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
    # convert the model_name(str) into the actual fixture by calling
    model = request.getfixturevalue(model_name)
    adaptive = make_adaptive_videoseal(model)

    # Detect extra keys
    extra_keys = set(model.state_dict()) - set(adaptive.state_dict())
    if extra_keys:
        logging.warning(
            f"[WARNING] Keys in parent model but not in adaptive model: {extra_keys}"
        )
    missing_keys = set(adaptive.state_dict()) - set(model.state_dict())
    if missing_keys:
        logging.warning(f"[WARNING] Extra keys in adaptive model: {missing_keys}")

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


@pytest.mark.parametrize(
    "model_name",
    ["videoseal_model", "stega_stamp_videoseal_model", "revmark_videoseal_model"],
)
def test_adaptive_videoseal_embed_matches_parent(
    request, model_name: str, make_adaptive_videoseal
):
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
    # convert the model_name(str) into the actual fixture by calling
    model = request.getfixturevalue(model_name)
    adaptive = make_adaptive_videoseal(model)

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


@pytest.mark.parametrize(
    "model_name",
    ["videoseal_model", "stega_stamp_videoseal_model", "revmark_videoseal_model"],
)
def test_adaptive_videoseal_detached_from_parent(
    request, model_name: str, make_adaptive_videoseal
):
    """
    AdaptiveVideoseal should behave independently from its parent Videoseal
    when their attributes are modified separately.

    In this test:
      • We deliberately set different blender.scaling_w values in the parent
        and the adapted child model.
      • Therefore, the *watermarked images (imgs_w)* should differ.
      • However, the *embedded message (msgs)* should remain identical,
        because the same message is passed as input and message generation
        is independent of the watermarked-image differences.

    Notes:
        - Test-only modification(Stega Stamp):
        We intentionally change scaling_w/scaling_i here to validate behavior.
        In real models, BOTH must remain 1.0 to avoid degrading StegaStamp decoding.
    """

    # convert the model_name(str) into the actual fixture by calling
    model = request.getfixturevalue(model_name)
    adaptive = make_adaptive_videoseal(model)

    # Modify attributes independently
    model.blender.scaling_w = 0.4
    adaptive.blender.scaling_w = 0.2

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

    # ----------------------------------------------------------------------
    # 1. Watermarked image comparison
    # Because scaling_w was changed independently, the watermarked outputs
    # MUST differ between parent and child models.
    # ----------------------------------------------------------------------
    diff_w = (outputs["imgs_w"] - adaptive_outputs["imgs_w"]).abs()
    assert not torch.allclose(
        outputs["imgs_w"], adaptive_outputs["imgs_w"], atol=1e-6, rtol=1e-6
    ), (
        "[ERROR] Watermarked images (imgs_w) are unexpectedly identical.\n"
        "Expected: Different outputs because blender.scaling_w differs.\n"
        f"  • Max diff: {diff_w.max().item()}\n"
    )

    # ----------------------------------------------------------------------
    # 2. Message comparison
    # The embedded message should remain IDENTICAL because:
    #   • We pass the same `msgs` argument to both models.
    #   • Message embedding is independent of blender.scaling_w differences.
    # ----------------------------------------------------------------------
    diff_m = (outputs["msgs"] - adaptive_outputs["msgs"]).abs()
    assert torch.allclose(
        outputs["msgs"], adaptive_outputs["msgs"], atol=1e-6, rtol=1e-6
    ), (
        "[ERROR] Embedded messages (msgs) do not match.\n"
        "Expected: Same message since the input message is identical.\n"
        f"  • Max diff: {diff_m.max().item()}\n"
    )


#### --- stega stamp specific ---


def test_attributes_get_stega_stamp_videoseal(stega_stamp_videoseal_model):
    """Test that the videoseal model loads and can perform a basic embed pass."""
    model = stega_stamp_videoseal_model

    assert isinstance(model, Videoseal), (
        f"get_videoseal() returned a {type(model).__name__}, "
        f"but an instance of Videoseal was expected."
    )

    assert isinstance(model.embedder, StegaStampEmbedder), (
        "Invalid embedder: expected StegaStampEmbedder, "
        f"but got {model.embedder.__class__.__name__}. "
        "This usually means the wrong class was imported or reloaded."
    )

    assert isinstance(model.detector, StegaStampExtractor), (
        "Invalid detector: expected StegaStampExtractor, "
        f"but got {model.detector.__class__.__name__}."
    )

    assert model.img_size == STEGA_STAMP_IMAGE_WIDTH, (
        "Unexpected image size: "
        f"expected {STEGA_STAMP_IMAGE_WIDTH}, got {model.img_size}."
    )

    assert model.attenuation == STEGA_STAMP_ATTENUATION, (
        "Unexpected attenuation value: "
        f"expected {STEGA_STAMP_ATTENUATION}, got {model.attenuation}."
    )

    assert model.blender.scaling_i == STEGA_STAMP_SCALING_I, (
        "Invalid blender.scaling_i: "
        f"expected {STEGA_STAMP_SCALING_I}, got {model.blender.scaling_i}. "
        "Note: scaling_i must remain 1.0 in real usage."
    )

    assert model.blender.scaling_w == STEGA_STAMP_SCALING_W, (
        "Invalid blender.scaling_w: "
        f"expected {STEGA_STAMP_SCALING_W}, got {model.blender.scaling_w}. "
        "Detection accuracy drops if scaling_w < scaling_i."
    )
