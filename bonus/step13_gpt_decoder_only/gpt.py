"""
Bonus Step 13 - TinyGPT: a decoder-only language model

GPT (Radford et al., 2018/2019) keeps only the DECODER of the Transformer and drops
cross-attention, because there is no source sentence to read. It is trained on one
job: predict the next token.

Built from the same parts as the original Transformer:
    Step 04 causal mask, Step 05 multi-head attention, Step 06 feed-forward,
    Step 07 residual + layer norm (in the PRE-norm arrangement GPT-2 uses)

Differences from the 2017 decoder, all of which GPT-2 made:
    - no encoder, no cross-attention
    - learned position embeddings instead of sinusoids
    - pre-norm blocks plus one final LayerNorm
    - small N(0, 0.02) initialisation
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from step04_masking.masks import make_causal_mask
from step05_multi_head_attention.multi_head_attention import MultiHeadAttention
from step06_feed_forward.feed_forward import PositionwiseFeedForward
from step07_add_and_norm.add_and_norm import LayerNorm, SublayerConnection


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128      # context length: how many tokens the model can see at once
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 4
    d_ff: int | None = None    # defaults to 4 * d_model, as in GPT-2
    dropout: float = 0.1

    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model


class GPTBlock(nn.Module):
    """The Step 09 decoder layer with the cross-attention sub-layer removed."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.attn = MultiHeadAttention(cfg.d_model, cfg.num_heads)
        self.ffn = PositionwiseFeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.sublayer1 = SublayerConnection(cfg.d_model, cfg.dropout, pre_norm=True)
        self.sublayer2 = SublayerConnection(cfg.d_model, cfg.dropout, pre_norm=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.sublayer1(x, lambda t: self.attn(t, t, t, mask))   # masked self-attention
        x = self.sublayer2(x, self.ffn)                             # feed-forward
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)    # learned positions
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(GPTBlock(cfg) for _ in range(cfg.num_layers))
        self.final_norm = LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)
        self.lm_head.weight = self.tok_emb.weight                    # weight tying (as in Step 10)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """
        idx:     (B, T) token ids, T <= block_size
        targets: (B, T) the same sequence shifted left by one (the next tokens)
        returns: logits (B, T, vocab_size), and the loss if targets are given
        """
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        mask = make_causal_mask(T, idx.device)
        for block in self.blocks:
            x = block(x, mask)
        logits = self.lm_head(self.final_norm(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
