"""Run from the repo root:  python -m step03_attention.test_attention"""

import torch
import torch.nn.functional as F

from step03_attention.attention import scaled_dot_product_attention

torch.manual_seed(0)


def test_shapes():
    q = torch.randn(2, 5, 8)        # (batch, seq_q, d_k)
    k = torch.randn(2, 7, 8)        # keys/values can have a different length (cross-attention!)
    v = torch.randn(2, 7, 16)
    out, w = scaled_dot_product_attention(q, k, v)
    assert out.shape == (2, 5, 16)
    assert w.shape == (2, 5, 7)


def test_rows_are_probabilities():
    q, k, v = (torch.randn(2, 6, 8) for _ in range(3))
    _, w = scaled_dot_product_attention(q, k, v)
    assert torch.all(w >= 0)
    assert torch.allclose(w.sum(-1), torch.ones(2, 6))


def test_matches_pytorch_builtin():
    q, k, v = (torch.randn(2, 6, 8) for _ in range(3))
    ours, _ = scaled_dot_product_attention(q, k, v)
    theirs = F.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(ours, theirs, atol=1e-5)


def test_mask_blocks_positions():
    q, k, v = (torch.randn(1, 4, 8) for _ in range(3))
    mask = torch.tril(torch.ones(4, 4))             # a causal mask (more in Step 04)
    out, w = scaled_dot_product_attention(q, k, v, mask)
    assert torch.all(w[0].triu(diagonal=1) == 0)    # nothing above the diagonal
    theirs = F.scaled_dot_product_attention(q, k, v, attn_mask=mask.bool())
    assert torch.allclose(out, theirs, atol=1e-5)


def test_works_with_head_dimension():
    q, k, v = (torch.randn(2, 4, 6, 8) for _ in range(3))   # (batch, heads, seq, d_k)
    out, w = scaled_dot_product_attention(q, k, v)
    assert out.shape == (2, 4, 6, 8) and w.shape == (2, 4, 6, 6)


def test_identical_keys_give_uniform_attention():
    q = torch.randn(1, 3, 8)
    k = torch.ones(1, 5, 8)                         # every key the same
    v = torch.randn(1, 5, 8)
    out, w = scaled_dot_product_attention(q, k, v)
    assert torch.allclose(w, torch.full_like(w, 1 / 5))
    assert torch.allclose(out[0, 0], v[0].mean(0), atol=1e-6)   # = plain average of values


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ✓ {name}")
    print("Step 03: all tests passed")
