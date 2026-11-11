"""
Step 09 - The Decoder
Paper: Section 3.1 "Encoder and Decoder Stacks"

One decoder layer has THREE sub-layers, each wrapped in Add & Norm:
  1. masked self-attention   - look at the target words produced so far
  2. cross-attention         - queries from the decoder, keys/values from the encoder
  3. feed-forward
The decoder is N = 6 of these layers stacked.
"""

import torch
import torch.nn as nn

from step05_multi_head_attention.multi_head_attention import MultiHeadAttention
from step06_feed_forward.feed_forward import PositionwiseFeedForward
from step07_add_and_norm.add_and_norm import LayerNorm, SublayerConnection


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 dropout: float = 0.1, pre_norm: bool = False):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout, pre_norm)
        self.sublayer2 = SublayerConnection(d_model, dropout, pre_norm)
        self.sublayer3 = SublayerConnection(d_model, dropout, pre_norm)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        """
        x:        (B, tgt_len, d_model) target embeddings (shifted right)
        memory:   (B, src_len, d_model) encoder output
        src_mask: (B, 1, 1, src_len)        padding mask for the source
        tgt_mask: (B, 1, tgt_len, tgt_len)  padding + causal mask for the target
        """
        # 1. masked self-attention: only earlier target tokens are visible
        x = self.sublayer1(x, lambda t: self.self_attn(t, t, t, tgt_mask))
        # 2. cross-attention: Q from the decoder, K and V from the encoder memory
        x = self.sublayer2(x, lambda t: self.cross_attn(t, memory, memory, src_mask))
        # 3. position-wise feed-forward
        x = self.sublayer3(x, self.feed_forward)
        return x


class Decoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int,
                 dropout: float = 0.1, pre_norm: bool = False):
        super().__init__()
        self.layers = nn.ModuleList(
            DecoderLayer(d_model, num_heads, d_ff, dropout, pre_norm) for _ in range(num_layers)
        )
        self.final_norm = LayerNorm(d_model) if pre_norm else nn.Identity()

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        """returns (B, tgt_len, d_model); Step 10 turns this into word probabilities"""
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.final_norm(x)
