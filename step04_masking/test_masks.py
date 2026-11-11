"""Run from the repo root:  python -m step04_masking.test_masks"""

import torch

from step03_attention.attention import scaled_dot_product_attention
from step04_masking.masks import make_causal_mask, make_decoder_mask, make_padding_mask

PAD = 0
torch.manual_seed(0)


def test_padding_mask_values():
    tokens = torch.tensor([[5, 6, 7, PAD, PAD],
                           [8, 9, PAD, PAD, PAD]])
    m = make_padding_mask(tokens, PAD)
    assert m.shape == (2, 1, 1, 5)
    assert m[0, 0, 0].tolist() == [True, True, True, False, False]
    assert m[1, 0, 0].tolist() == [True, True, False, False, False]


def test_causal_mask_values():
    m = make_causal_mask(4)
    assert m.shape == (1, 1, 4, 4)
    expected = [[1, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 0],
                [1, 1, 1, 1]]
    assert m[0, 0].int().tolist() == expected


def test_decoder_mask_combines_both():
    tokens = torch.tensor([[5, 6, PAD]])
    m = make_decoder_mask(tokens, PAD)[0, 0].int().tolist()
    assert m == [[1, 0, 0],
                 [1, 1, 0],
                 [1, 1, 0]]          # nobody looks at the pad, nobody looks ahead


def test_padding_gets_zero_attention():
    tokens = torch.tensor([[5, 6, 7, PAD, PAD]])
    x = torch.randn(1, 1, 5, 8)                      # (batch, heads, seq, d)
    _, w = scaled_dot_product_attention(x, x, x, make_padding_mask(tokens, PAD))
    assert torch.all(w[..., 3:] == 0)                # columns for the pads are zero
    assert torch.allclose(w.sum(-1), torch.ones(1, 1, 5))


def test_causal_output_ignores_the_future():
    """Changing a future token must not change earlier outputs."""
    x = torch.randn(1, 1, 6, 8)
    mask = make_causal_mask(6)
    out1, _ = scaled_dot_product_attention(x, x, x, mask)
    x2 = x.clone()
    x2[..., 4, :] = torch.randn(8)                   # change token 4
    out2, _ = scaled_dot_product_attention(x2, x2, x2, mask)
    assert torch.allclose(out1[..., :4, :], out2[..., :4, :])       # tokens 0-3 unchanged
    assert not torch.allclose(out1[..., 4, :], out2[..., 4, :])


def test_fully_masked_row_gives_nan():
    """A query with NOTHING to attend to: softmax over all -inf = 0/0 = NaN.
    This is why sequences always start with a real token like <bos>."""
    x = torch.randn(1, 1, 3, 4)
    mask = torch.zeros(1, 1, 3, 3, dtype=torch.bool)
    _, w = scaled_dot_product_attention(x, x, x, mask)
    assert torch.isnan(w).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ✓ {name}")
    print("Step 04: all tests passed")
