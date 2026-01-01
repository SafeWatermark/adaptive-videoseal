import torch
from videoseal.models.extractor import Extractor


def mac_forward_hook(module: Extractor, input, output) -> torch.Tensor:
    """
    Post-forward hook that:
    1. Splits output into message_bits and tag_bits
    2. Verifies MAC tag
    3. Computes combined detection probability
    4. Returns combined_probability + message_bits (tag stripped)

    Combined probability formula:
        mac_prob = matching_tag_bits / total_tag_bits
        combined_prob = scaling_model * model_prob + scaling_mac * mac_prob
    """
    model_probability = output[:, 0].unsqueeze(1)
    payload_bits = output[:, 1:]  # message || tag

    message_bits, _ = module.mac_provider.split_payload_bits(payload_bits)
    verification = module.mac_provider.verify_tag(payload_bits)
    module.mac_provider.verification_results.extend(verification.tolist())

    # Compute MAC probability as bit-match ratio (e.g., 28/32 = 0.875)
    mac_probability = module.mac_provider.compute_tag_match_ratio(payload_bits)
    mac_probability = mac_probability.unsqueeze(1)

    # Combine probabilities using scaling factors
    scaling = module.mac_provider.detection_scaling
    combined_probability = (
        scaling.scaling_model * model_probability
        + scaling.scaling_mac * mac_probability
    )

    return torch.cat([combined_probability, message_bits], dim=1)
