"""
Step 05 - Multi-Head Attention
Paper: Section 3.2.2 "Multi-Head Attention" (Figure 2, right)

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)

Trick used by every real implementation: instead of h separate small Linear
layers per head, use ONE big Linear of size d_model and split its output into h heads.
"""

import torch
import torch.nn as nn

from step03_attention.attention import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_k = d_model // num_heads          # 512 / 8 = 64 in the paper

        self.w_q = nn.Linear(d_model, d_model)   # all heads' W^Q stacked together
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)   # W^O, mixes the heads back together

        self.attn_weights = None                 # kept for visualisation

    def _split_heads(self, x):
        # (B, T, d_model) -> (B, T, h, d_k) -> (B, h, T, d_k)
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x):
        # (B, h, T, d_k) -> (B, T, h, d_k) -> (B, T, d_model)
        B, _, T, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.d_k)

    def forward(self, query, key, value, mask=None):
        """
        query: (B, T_q, d_model)   key, value: (B, T_k, d_model)
        mask:  broadcastable to (B, h, T_q, T_k), True = may attend (Step 04)
        Self-attention:  query = key = value = x
        Cross-attention: query = decoder state, key = value = encoder output
        """
        q = self._split_heads(self.w_q(query))
        k = self._split_heads(self.w_k(key))
        v = self._split_heads(self.w_v(value))

        out, self.attn_weights = scaled_dot_product_attention(q, k, v, mask)  # each head in parallel
        return self.w_o(self._merge_heads(out))
