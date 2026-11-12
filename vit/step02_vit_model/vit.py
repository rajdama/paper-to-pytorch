"""
ViT Step 02 - The Vision Transformer
Paper: Dosovitskiy et al., 2020, Section 3.1, Eq. 1-4

    z_0 = [x_class ; x_p^1 E ; x_p^2 E ; ... ; x_p^N E] + E_pos        (Eq. 1)
    z_l = Transformer encoder layers                                  (Eq. 2-3)
    y   = LN(z_L^0)  -> MLP head -> class scores                      (Eq. 4)

The encoder is Step 08 from the first paper, unchanged, in its pre-norm arrangement
(ViT uses pre-norm). Only the input and output ends are new:
  - patches instead of word embeddings (ViT Step 01)
  - a learned [CLS] token prepended to the sequence; its final vector represents the image
  - learned 1-D position embeddings instead of sinusoids
"""

import torch
import torch.nn as nn

from step08_encoder.encoder import Encoder
from vit.step01_patch_embedding.patch_embedding import PatchEmbedding


class VisionTransformer(nn.Module):
    def __init__(self,
                 image_size: int = 28,
                 patch_size: int = 7,
                 in_channels: int = 1,
                 num_classes: int = 10,
                 d_model: int = 64,
                 num_heads: int = 4,
                 num_layers: int = 4,
                 d_ff: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, d_model)
        n = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))          # x_class
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, d_model))      # E_pos (learned)
        self.dropout = nn.Dropout(dropout)
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout, pre_norm=True)
        self.head = nn.Linear(d_model, num_classes)                        # MLP head (one layer for fine-tuning)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, C, H, W) -> class logits (B, num_classes)"""
        x = self.patch_embed(images)                                       # (B, N, d)
        cls = self.cls_token.expand(x.size(0), -1, -1)                     # one [CLS] per image
        x = torch.cat([cls, x], dim=1) + self.pos_embed                    # (B, N + 1, d)   Eq. 1
        x = self.encoder(self.dropout(x))                                  # no mask: every patch sees every patch
        return self.head(x[:, 0])                                          # read the [CLS] position  Eq. 4

    def attention_maps(self) -> list[torch.Tensor]:
        """Attention weights stored by each layer's last forward pass: (B, heads, N+1, N+1)."""
        return [layer.self_attn.attn_weights for layer in self.encoder.layers]


def attention_rollout(maps: list[torch.Tensor]) -> torch.Tensor:
    """
    Abnar & Zuidema (2020): multiply attention through the layers, accounting for the
    residual path (+ identity), to estimate how much each input patch flows into [CLS].
    maps: per layer (B, heads, T, T) -> (B, T-1) relevance of each patch for [CLS]
    """
    B, _, T, _ = maps[0].shape
    eye = torch.eye(T, device=maps[0].device)
    rollout = eye.expand(B, T, T)
    for a in maps:
        a = a.mean(dim=1)                                    # average the heads
        a = (a + eye) / 2                                    # residual connection
        a = a / a.sum(dim=-1, keepdim=True)
        rollout = a @ rollout
    return rollout[:, 0, 1:]                                 # [CLS] row, patch columns
