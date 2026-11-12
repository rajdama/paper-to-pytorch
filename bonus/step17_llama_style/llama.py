"""
Bonus Step 17 - From GPT-2 to LLaMA: three modern upgrades

LLaMA (Touvron et al., 2023), and most open models since, keep GPT's decoder-only
design but swap three parts. Everything else (causal attention, residuals, training
objective) is unchanged from Bonus 13.

  1. RMSNorm instead of LayerNorm           (Zhang & Sennrich, 2019)
       no mean subtraction, no bias: x / sqrt(mean(x^2) + eps) * g
  2. RoPE instead of learned positions        (Su et al., 2021, "RoFormer")
       rotate q and k by an angle proportional to their position, so q·k depends
       only on the DISTANCE between tokens
  3. SwiGLU instead of a ReLU MLP             (Shazeer, 2020, "GLU Variants Improve Transformer")
       FFN(x) = W2( SiLU(W1 x) * W3 x )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from bonus.step13_gpt_decoder_only.gpt import GPTConfig
from step03_attention.attention import scaled_dot_product_attention
from step04_masking.masks import make_causal_mask


# ---------------------------------------------------------------- 1. RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


# ---------------------------------------------------------------- 2. RoPE
def rope_cache(max_len: int, d_head: int, base: float = 10000.0):
    """cos and sin tables, each (max_len, d_head). Same frequencies as Step 02's sinusoids."""
    inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))   # (d_head/2,)
    angles = torch.outer(torch.arange(max_len).float(), inv_freq)            # (max_len, d_head/2)
    angles = torch.cat([angles, angles], dim=-1)                              # (max_len, d_head)
    return angles.cos(), angles.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Rotate each pair of dimensions (i, i + d/2) of x by its position's angle.
    x: (..., T, d_head); cos/sin: (T, d_head) for those T positions.
    """
    return x * cos + rotate_half(x) * sin


class RoPEAttention(nn.Module):
    """Step 05's multi-head attention, with q and k rotated by position before the dot product."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads, self.d_k = num_heads, d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model, bias=False)       # LLaMA uses no biases
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, cos, sin, mask):
        B, T, _ = x.shape
        split = lambda t: t.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        q, k, v = split(self.w_q(x)), split(self.w_k(x)), split(self.w_v(x))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)  # <- the only change to attention
        out, _ = scaled_dot_product_attention(q, k, v, mask)
        return self.w_o(out.transpose(1, 2).reshape(B, T, -1))


# ---------------------------------------------------------------- 3. SwiGLU
def swiglu_hidden(d_model: int) -> int:
    """LLaMA's sizing: 2/3 of 4*d (three matrices instead of two keep the parameter count equal), rounded up to a multiple of 8."""
    h = int(2 * 4 * d_model / 3)
    return 8 * ((h + 7) // 8)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden, bias=False)    # gate
        self.w3 = nn.Linear(d_model, hidden, bias=False)    # value
        self.w2 = nn.Linear(hidden, d_model, bias=False)    # back down
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


# ---------------------------------------------------------------- the model
class LlamaBlock(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = RoPEAttention(cfg.d_model, cfg.num_heads)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg.d_model, swiglu_hidden(cfg.d_model), cfg.dropout)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, mask):
        x = x + self.drop(self.attn(self.attn_norm(x), cos, sin, mask))   # pre-norm, as in GPT-2
        x = x + self.drop(self.ffn(self.ffn_norm(x)))
        return x


class TinyLlama(nn.Module):
    """Same interface as Bonus 13's GPT, so its training and sampling code works unchanged."""

    def __init__(self, cfg: GPTConfig, rope_max_len: int | None = None):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)       # NO position embedding table
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(LlamaBlock(cfg) for _ in range(cfg.num_layers))
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        cos, sin = rope_cache(rope_max_len or 4 * cfg.block_size, cfg.d_model // cfg.num_heads)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        self.lm_head.weight = self.tok_emb.weight                      # tied, to compare fairly with TinyGPT

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.rope_cos.size(0), "sequence longer than the RoPE table"
        x = self.drop(self.tok_emb(idx))
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        mask = make_causal_mask(T, idx.device)
        for block in self.blocks:
            x = block(x, cos, sin, mask)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
