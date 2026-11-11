"""
Step 07 - Residual connection + Layer Normalization ("Add & Norm")
Paper: Section 3.1 ("LayerNorm(x + Sublayer(x))") and Section 5.4 (residual dropout)

Every attention and feed-forward block in the Transformer is wrapped like this.
"""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Normalise each token's vector to mean 0, variance 1, then let the model
    rescale (gamma) and shift (beta) it. Unlike BatchNorm, statistics are computed
    per token over the feature dimension, so batch size and sequence length don't matter.
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))    # learned scale
        self.beta = nn.Parameter(torch.zeros(d_model))    # learned shift
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta


class SublayerConnection(nn.Module):
    """
    post-norm (the paper):  LayerNorm(x + Dropout(Sublayer(x)))
    pre-norm  (GPT-2 on):   x + Dropout(Sublayer(LayerNorm(x)))
    Pre-norm trains more stably in deep models; the paper's original is post-norm.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, pre_norm: bool = False):
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = pre_norm

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        """sublayer: any function mapping (B, T, d_model) -> (B, T, d_model)"""
        if self.pre_norm:
            return x + self.dropout(sublayer(self.norm(x)))
        return self.norm(x + self.dropout(sublayer(x)))
