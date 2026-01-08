import torch
from videoseal.models.extractor import Extractor
from hooks.utils import infer_input_type_by_module


def ecc_forward_hook(module: Extractor, input, output) -> torch.Tensor:
    """
    Post-forward hook that applies ecc decoding
    """
    probability = output[:, 0].unsqueeze(1)
    decoded_msgs, _ = apply_ecc_decoding(module, output[:, 1:])
    return torch.cat([probability, decoded_msgs], dim=1)


def apply_ecc_decoding(
    module: Extractor, raw_bits: torch.Tensor
) -> tuple[torch.Tensor, list[int]]:
    """Apply ECC decoding to extract and error-correct message bits.

    Args:
        raw_bits: (B, total_bits) tensor of soft bit values.

    Returns:
        decoded_msgs: (B, message_bits) tensor of corrected message bits.
        errors_list: List of error counts per batch item.
    """
    input_type = infer_input_type_by_module(module)
    decoded_tensor, errors_list = module.ecc_codec.decode(
        watermark_bits=raw_bits, input_type=input_type
    )
    return decoded_tensor, errors_list
