"""
Step 04 - Masking
Paper: Section 3.2.3 "Applications of Attention in our Model"

Convention used everywhere in this repo:
    mask value True  -> this query MAY attend to this key
    mask value False -> blocked (score set to -inf before softmax)

All masks are 4-D so they broadcast against attention scores of shape
(batch, heads, seq_q, seq_k) from Step 05 onwards.
"""

import torch


def make_padding_mask(tokens: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """
    Hide <pad> tokens so nobody attends to them.
    tokens: (batch, seq_len) integer ids
    returns: (batch, 1, 1, seq_len) bool - broadcasts over heads and queries
    """
    return (tokens != pad_idx)[:, None, None, :]


def make_causal_mask(seq_len: int, device=None) -> torch.Tensor:
    """
    Position i may only attend to positions 0..i (no looking into the future).
    returns: (1, 1, seq_len, seq_len) bool, lower-triangular
    """
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))[None, None]


def make_decoder_mask(tgt_tokens: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """
    The decoder's self-attention needs BOTH rules at once.
    returns: (batch, 1, seq_len, seq_len) bool
    """
    seq_len = tgt_tokens.size(1)
    return make_padding_mask(tgt_tokens, pad_idx) & make_causal_mask(seq_len, tgt_tokens.device)
