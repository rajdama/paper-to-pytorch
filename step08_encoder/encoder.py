"""
Step 08 - The Encoder
Paper: Section 3.1 "Encoder and Decoder Stacks"

One encoder layer = self-attention + feed-forward, each wrapped in Add & Norm.
The encoder is N = 6 of these layers stacked.

This is the first step that only ASSEMBLES earlier steps - no new math.
"""

import torch
import torch.nn as nn

from step05_multi_head_attention.multi_head_attention import MultiHeadAttention
from step06_feed_forward.feed_forward import PositionwiseFeedForward
from step07_add_and_norm.add_and_norm import LayerNorm, SublayerConnection


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 dropout: float = 0.1, pre_norm: bool = False):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout, pre_norm)
        self.sublayer2 = SublayerConnection(d_model, dropout, pre_norm)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        # 1. every source token looks at every (non-pad) source token
        x = self.sublayer1(x, lambda t: self.self_attn(t, t, t, src_mask))
        # 2. every token is processed on its own
        x = self.sublayer2(x, self.feed_forward)
        return x


class Encoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int,
                 dropout: float = 0.1, pre_norm: bool = False):
        super().__init__()
        self.layers = nn.ModuleList(
            EncoderLayer(d_model, num_heads, d_ff, dropout, pre_norm) for _ in range(num_layers)
        )
        # pre-norm stacks need one final norm (their outputs are never normalised otherwise)
        self.final_norm = LayerNorm(d_model) if pre_norm else nn.Identity()

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (batch, src_len, d_model) - embeddings + positional encoding (Steps 01-02)
        src_mask: (batch, 1, 1, src_len) padding mask (Step 04)
        returns the "memory": (batch, src_len, d_model), read by the decoder's cross-attention
        """
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.final_norm(x)
