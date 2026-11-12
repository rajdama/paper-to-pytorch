"""
ViT Step 01 - Patch embedding
Paper: Dosovitskiy et al., 2020, "An Image is Worth 16x16 Words", Section 3.1 (Eq. 1)

A Transformer eats a SEQUENCE of vectors. ViT turns an image into one by cutting it
into P x P patches, flattening each patch, and projecting it to d_model with a learned
matrix E:

    image (C, H, W)  ->  N = HW / P^2 patches, each of C * P * P numbers  ->  (N, d_model)

Two equivalent ways to compute it, both below:
  1. literally: unfold into patches, flatten, multiply by E          (patchify + Linear)
  2. efficiently: a Conv2d with kernel_size = stride = P             (what real code uses)
A convolution whose stride equals its kernel size looks at each patch exactly once,
which is exactly "apply the same linear map to every patch".
"""

import torch
import torch.nn as nn


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    images: (B, C, H, W) -> patches: (B, N, C * P * P), row-major order (left->right, top->bottom)
    """
    B, C, H, W = images.shape
    P = patch_size
    assert H % P == 0 and W % P == 0, f"image {H}x{W} is not divisible into {P}x{P} patches"
    x = images.reshape(B, C, H // P, P, W // P, P)       # split both axes into (blocks, within-block)
    x = x.permute(0, 2, 4, 1, 3, 5)                      # (B, rows, cols, C, P, P)
    return x.reshape(B, (H // P) * (W // P), C * P * P)


class PatchEmbedding(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_channels: int, d_model: int):
        super().__init__()
        assert image_size % patch_size == 0
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, C, H, W) -> (B, num_patches, d_model)"""
        x = self.proj(images)                    # (B, d_model, H/P, W/P): one vector per patch
        return x.flatten(2).transpose(1, 2)      # (B, N, d_model), same patch order as patchify

    def as_linear(self) -> nn.Linear:
        """The same projection written as the paper's matrix E (for the equivalence test)."""
        lin = nn.Linear(self.proj.in_channels * self.patch_size ** 2, self.proj.out_channels)
        with torch.no_grad():
            lin.weight.copy_(self.proj.weight.flatten(1))   # (d_model, C*P*P)
            lin.bias.copy_(self.proj.bias)
        return lin
