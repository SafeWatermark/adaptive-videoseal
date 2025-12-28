import torch


def apply_mac_encoding(module, msgs: torch.Tensor) -> torch.Tensor:
    """Apply MAC(Message Authenticated Code) encoding to message bits.

    Args:
        msgs: (B, message_bits) tensor of message bits.

    Returns:
        (B, message_bits + tag_bits) tensor with MAC-encoded bits.
    """
    encoded_tensor = (
        module.embedder.msg_processor.mac_provider.build_authenticated_payload_bits(
            msgs
        )
    )
    return encoded_tensor


def mac_forward_hook(module, input, output):
    """
    Post-forward hook that:
    1. Splits output into message_bits and tag_bits
    2. Verifies MAC tag
    3. Stores verification result
    4. Returns message_bits only (tag stripped)
    """
    probability = output[:, 0].unsqueeze(1)
    payload_bits = output[:, 1:]  # message || tag

    message_bits, _ = module.mac_provider.split_payload_bits(payload_bits)
    verification = module.mac_provider.verify_tag(payload_bits)

    module.mac_provider.verification_results.extend(verification.tolist())

    # TODO: Consider tune the probability following by verificatoin or not
    # For instance, model_weights* model_probability + mac_weights * mac_consistency
    return torch.cat([probability, message_bits], dim=1)
