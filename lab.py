import torch

latents = torch.load("convnext_extractor_latents.pt")  # F, C, H, W
latents_in_noisy = torch.load("convnext_extractor_latents_in_noisy.pt")  # F, C, H, W
masks = torch.load("convnext_extractor_mask.pt")
masks_in_noisy = torch.load("convnext_extractor_mask_in_noisy.pt")

adaptive_latents = torch.load("adaptive_convnext_extractor_latents.pt")  # F, C, H, W
adaptive_latents_in_noisy = torch.load(
    "adaptive_convnext_extractor_latents_in_noisy.pt"
)  # F, C, H, W

adaptive_masks_in_noisy = torch.load("convnext_extractor_mask_in_noisy.pt")


latents_diff = latents - latents_in_noisy

adaptive_latents_diff = adaptive_latents - adaptive_latents_in_noisy

dist = torch.sqrt(
    torch.mean((latents_diff[1:] - latents_diff[:-1]) ** 2, dim=(1, 2, 3))
)  # step size: 8

mask_dist = torch.sqrt(
    torch.mean((masks[1:] - masks[:-1]) ** 2, dim=-1)
)  # step size: 8
adaptive_dist = torch.sqrt(
    torch.mean(
        (adaptive_latents_diff[1:] - adaptive_latents_diff[:-1]) ** 2, dim=(1, 2, 3)
    )
)  # [2, 6, 8, 8, 4, 1, 1, 1, 1]: 32

print(dist)
print(adaptive_dist)
