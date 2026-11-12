"""
Bonus Step 15 - KV cache for fast generation

Without a cache (Step 13's `generate`), producing token t re-runs the model on ALL
t previous tokens, so generating n tokens costs about n^2/2 token-forward-passes.

But with a causal mask, the keys and values of old tokens never change once computed:
token 5 can't see token 6, so adding token 6 doesn't alter token 5's K and V.
So we store them. Each new step then only processes ONE new token:
    - compute q, k, v for the new token
    - append k, v to the cache
    - attend from the new q to all cached keys/values (no mask needed: everything
      in the cache is in the past)

This file reuses the TinyGPT weights unchanged - only the forward pass differs.
"""

import torch

from bonus.step13_gpt_decoder_only.sampling import sample_next
from step03_attention.attention import scaled_dot_product_attention
from step04_masking.masks import make_causal_mask


class KVCache:
    """One (keys, values) pair per layer, each (B, heads, tokens_so_far, d_k)."""

    def __init__(self, num_layers: int):
        self.keys = [None] * num_layers
        self.values = [None] * num_layers

    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        if self.keys[layer] is None:
            self.keys[layer], self.values[layer] = k, v
        else:
            self.keys[layer] = torch.cat([self.keys[layer], k], dim=2)
            self.values[layer] = torch.cat([self.values[layer], v], dim=2)
        return self.keys[layer], self.values[layer]

    @property
    def length(self) -> int:
        return 0 if self.keys[0] is None else self.keys[0].size(2)

    def num_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.keys + self.values if t is not None)


def cached_attention(mha, x: torch.Tensor, cache: KVCache, layer: int) -> torch.Tensor:
    """Step 05's multi-head attention, but keys/values come from (and go into) the cache."""
    q = mha._split_heads(mha.w_q(x))                       # only the NEW tokens' queries
    k_new = mha._split_heads(mha.w_k(x))
    v_new = mha._split_heads(mha.w_v(x))
    past = cache.length
    k, v = cache.append(layer, k_new, v_new)               # all keys/values so far

    T_new, T_all = x.size(1), k.size(2)
    mask = None
    if T_new > 1:
        # prefill: the prompt's tokens still must not see each other's future.
        # query i (absolute position past + i) may see keys 0 .. past + i
        mask = make_causal_mask(T_all, x.device)[..., past:, :]
    out, _ = scaled_dot_product_attention(q, k, v, mask)
    return mha.w_o(mha._merge_heads(out))


@torch.no_grad()
def forward_with_cache(model, idx: torch.Tensor, cache: KVCache) -> torch.Tensor:
    """
    Run TinyGPT on `idx`, which continues the tokens already in the cache.
    idx: (B, T_new) -> logits (B, T_new, vocab)
    Mirrors GPT.forward + GPTBlock.forward exactly (pre-norm, dropout off in eval).
    """
    start = cache.length
    T_new = idx.size(1)
    assert start + T_new <= model.cfg.block_size, (
        "learned position embeddings stop at block_size; a longer context would need "
        "a sliding window (and recomputing the cache) or relative positions like RoPE")
    pos = torch.arange(start, start + T_new, device=idx.device)
    x = model.tok_emb(idx) + model.pos_emb(pos)
    for layer, block in enumerate(model.blocks):
        x = x + cached_attention(block.attn, block.sublayer1.norm(x), cache, layer)
        x = x + block.ffn(block.sublayer2.norm(x))
    return model.lm_head(model.final_norm(x))


@torch.no_grad()
def generate_with_kv_cache(model, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                           top_k: int | None = None, top_p: float | None = None,
                           generator: torch.Generator | None = None) -> torch.Tensor:
    """Same interface as Step 13's generate(), but each step processes one token."""
    model.eval()
    cache = KVCache(len(model.blocks))
    logits = forward_with_cache(model, idx, cache)          # 1. prefill: the whole prompt at once
    for step in range(max_new_tokens):
        next_tok = sample_next(logits[:, -1], temperature, top_k, top_p, generator)
        idx = torch.cat([idx, next_tok], dim=1)
        if step < max_new_tokens - 1:                       # (the last token never needs a forward pass)
            logits = forward_with_cache(model, next_tok, cache)   # 2. decode: ONE new token per step
    return idx
